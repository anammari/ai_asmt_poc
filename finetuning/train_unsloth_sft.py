# finetuning/train_unsloth_sft.py
"""
Unsloth QLoRA fine-tuning for the dialect-specific Arabic ASMR scriptwriter.

DESIGNED TO RUN ON GOOGLE COLAB (free T4) OR A LINUX CUDA MACHINE.
Unsloth does not support macOS - do not run this on the Mac app host.
(The Mac only builds the dataset and later serves the exported GGUF
through Ollama.)

Data must live on Google Drive so it persists across Colab sessions.
See finetuning/README.md for the full workflow.

Quick start on Colab (Runtime -> T4 GPU):
    from google.colab import drive
    drive.mount('/content/drive')

    !pip install -q unsloth trl datasets

    !python /content/drive/MyDrive/arabic-asmr/train_unsloth_sft.py \\
        --dialect syria \\
        --dataset /content/drive/MyDrive/arabic-asmr/syria/arabic_asmr_sft.jsonl \\
        --output-dir /content/drive/MyDrive/arabic-asmr/outputs/syria \\
        --epochs 2

Outputs (default: <output-dir>/):
    <output-dir>/adapter/   LoRA adapter (small; can be pushed to HF Hub)
    <output-dir>/gguf/      q4_k_m GGUF for llama.cpp/Ollama
    <output-dir>/Modelfile  Ollama recipe; then on the Mac:
                               ollama create arabic-asmr-<dialect> -f <output-dir>/Modelfile
                             and set in the app .env:
                               LLM_PROVIDER=ollama
                               OLLAMA_MODEL=arabic-asmr-<dialect>:latest
"""
import argparse
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

DEFAULT_BASE_MODEL = "unsloth/Qwen2.5-7B-Instruct-unsloth-bnb-4bit"
MAX_SEQ_LENGTH = 2048
SUPPORTED_DIALECTS = {"syria", "egypt"}


def _default_dataset(dialect: str) -> str:
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data",
        "training",
        dialect,
        "arabic_asmr_sft.jsonl",
    )


def _derive_defaults(args) -> None:
    """Fill in dialect-dependent defaults after parsing so tests can inspect them."""
    if args.output_dir is None:
        args.output_dir = os.path.join("outputs", args.dialect)
    if args.ollama_name is None:
        args.ollama_name = f"arabic-asmr-{args.dialect}"
    if args.dataset is None:
        args.dataset = _default_dataset(args.dialect)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Unsloth QLoRA SFT for Arabic ASMR (dialect-specific)."
    )
    parser.add_argument(
        "--dialect",
        choices=sorted(SUPPORTED_DIALECTS),
        required=True,
        help="Dialect to train for (syria or egypt).",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path to Unsloth-compatible chat JSONL. Auto-derived from --dialect if omitted.",
    )
    parser.add_argument(
        "--base-model",
        default=os.getenv("FT_BASE_MODEL", DEFAULT_BASE_MODEL),
        help="Base model identifier or local path.",
    )
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Training output directory. Auto-derived from --dialect as outputs/<dialect>.",
    )
    parser.add_argument("--gguf-quant", default="q4_k_m")
    parser.add_argument(
        "--ollama-name",
        default=None,
        help="Ollama model name. Auto-derived from --dialect as arabic-asmr-<dialect>.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from the latest checkpoint if one exists (default: True).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        dest="force_no_resume",
        help="Ignore existing checkpoints and train from scratch.",
    )
    args = parser.parse_args(argv)
    _derive_defaults(args)
    return args


def find_latest_checkpoint(output_dir: str) -> str | None:
    """Return the newest checkpoint directory matching checkpoint-NNN under output_dir."""
    output_path = Path(output_dir)
    if not output_path.is_dir():
        return None
    checkpoints = [
        p for p in output_path.iterdir()
        if p.is_dir() and re.match(r"^checkpoint-\d+$", p.name)
    ]
    if not checkpoints:
        return None
    latest = max(checkpoints, key=lambda p: int(p.name.split("-")[1]))
    return str(latest)


def _format_duration(seconds: float) -> str:
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


def main(argv=None) -> int:
    args = parse_args(argv)

    output_dir = args.output_dir
    ollama_name = args.ollama_name
    dataset_path = args.dataset

    try:
        import torch  # noqa: F401
        from unsloth import FastLanguageModel
        from datasets import load_dataset
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        print(
            "error: training dependencies missing.\n"
            "This script runs on Colab/Linux CUDA, not macOS. Install with:\n"
            "  pip install unsloth trl datasets\n"
            f"Details: {exc}",
            file=sys.stderr,
        )
        return 2

    if not os.path.exists(dataset_path):
        print(f"error: dataset not found: {dataset_path}", file=sys.stderr)
        return 2

    print(f"[setup] dialect:    {args.dialect}")
    print(f"[setup] base model: {args.base_model}")
    print(f"[setup] dataset:    {dataset_path}")

    checkpoint_to_resume = None
    should_resume = args.resume and not args.force_no_resume
    if should_resume:
        checkpoint_to_resume = find_latest_checkpoint(output_dir)
        if checkpoint_to_resume:
            print(f"[setup] resuming from checkpoint: {checkpoint_to_resume}")

    print(f"[setup] output dir: {output_dir}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
    )

    dataset = load_dataset("json", data_files=dataset_path, split="train")
    print(f"[setup] {len(dataset)} training samples")

    def to_text(example):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False,
        )
        return {"text": text}

    dataset = dataset.map(to_text)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=SFTConfig(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            warmup_steps=5,
            logging_steps=10,
            optim="adamw_8bit",
            fp16=True,
            bf16=False,
            seed=42,
            output_dir=output_dir,
            report_to="none",
        ),
    )

    start_time = time.perf_counter()
    if checkpoint_to_resume:
        train_result = trainer.train(resume_from_checkpoint=True)
    else:
        train_result = trainer.train()
    elapsed = time.perf_counter() - start_time
    samples_per_second = train_result.metrics.get("train_samples_per_second")
    final_loss = train_result.metrics.get("train_loss")

    print("\n[stats] training finished")
    print(f"  elapsed time:       {_format_duration(elapsed)}")
    if samples_per_second is not None:
        print(f"  samples/second:     {samples_per_second:.3f}")
    if final_loss is not None:
        print(f"  final train loss:   {final_loss:.4f}")
    else:
        history = trainer.state.log_history
        if history and "loss" in history[-1]:
            print(f"  final logged loss:  {history[-1]['loss']:.4f}")

    adapter_dir = os.path.join(output_dir, "adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"[save] LoRA adapter -> {adapter_dir}")

    # GGUF export: merge + convert to f16 in a local temp directory to avoid
    # partial writes on Google Drive (a FUSE mount). Copy the f16 GGUF to
    # Drive before quantization — if the Colab session times out during the
    # 10-minute q4_k_m step, the f16 file is already saved.
    # The user can then quantize locally with llama.cpp or Ollama.
    gguf_drive_dir = os.path.join(output_dir, "gguf")
    os.makedirs(gguf_drive_dir, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="unsloth_gguf_") as tmp_gguf:
        print(f"[gguf] merging to local temp: {tmp_gguf}")
        print(f"[gguf] step 1: converting to f16 GGUF (fast, ~3 min)...")
        try:
            model.save_pretrained_gguf(
                tmp_gguf, tokenizer,
                quantization_method="f16",
            )
        except Exception as exc:
            print(f"[gguf] WARNING: f16 conversion failed: {exc}", file=sys.stderr)
            print("[gguf] The LoRA adapter was saved; the model can be merged manually.")
            return 1

        # Copy f16 GGUF to Drive (before quantization, so it survives session drop).
        f16_copied = False
        gguf_container = tmp_gguf + "_gguf"  # Unsloth creates this sibling dir
        if os.path.isdir(gguf_container):
            for fname in os.listdir(gguf_container):
                if fname.endswith(".gguf"):
                    src = os.path.join(gguf_container, fname)
                    dst = os.path.join(gguf_drive_dir, fname)
                    shutil.copy2(src, dst)
                    f16_copied = True
                    size_mb = os.path.getsize(dst) / (1024 * 1024)
                    print(f"[gguf] f16 copied to Drive: {fname} ({size_mb:.0f} MB)")

        if not f16_copied:
            print("[gguf] WARNING: no f16 GGUF produced", file=sys.stderr)
            return 1

    # Now quantize to q4_k_m from the f16 on Drive.
    # Use a local temp dir so the quantized file is built on local disk first.
    print(f"[gguf] step 2: quantizing f16 to {args.gguf_quant} (slow, ~10 min)...")
    with tempfile.TemporaryDirectory(prefix="unsloth_gguf_quant_") as tmp_quant:
        try:
            model.save_pretrained_gguf(
                tmp_quant, tokenizer,
                quantization_method=args.gguf_quant,
            )
        except Exception as exc:
            print(f"[gguf] WARNING: quantization failed: {exc}", file=sys.stderr)
            print(f"[gguf] The f16 GGUF is already on Drive at {gguf_drive_dir}/")
            print("[gguf] To quantize locally, install llama.cpp and run:")
            print(f"  ./quantize {gguf_drive_dir}/qwen2.5-7b-instruct.F16.gguf \\")
            print(f"    {gguf_drive_dir}/qwen2.5-7b-instruct.{args.gguf_quant.upper()}.gguf \\")
            print(f"    {args.gguf_quant}")
            return 1

        # Copy the quantized GGUF to Drive.
        quant_container = tmp_quant + "_gguf"
        quant_copied = 0
        if os.path.isdir(quant_container):
            for fname in os.listdir(quant_container):
                if fname.endswith(".gguf"):
                    src = os.path.join(quant_container, fname)
                    dst = os.path.join(gguf_drive_dir, fname)
                    shutil.copy2(src, dst)
                    quant_copied += 1
                    size_mb = os.path.getsize(dst) / (1024 * 1024)
                    print(f"[gguf] quantized copied to Drive: {fname} ({size_mb:.0f} MB)")

        if quant_copied == 0:
            print("[gguf] WARNING: no quantized GGUF produced", file=sys.stderr)
            return 1

    # Write the Ollama Modelfile pointing at the quantized GGUF.
    modelfile_path = os.path.join(output_dir, "Modelfile")
    gguf_files = [f for f in os.listdir(gguf_drive_dir) if f.endswith(".gguf")]
    # Prefer the quantized file (smaller) over f16.
    gguf_name = "model.gguf"
    for f in gguf_files:
        if args.gguf_quant.upper() in f.upper():
            gguf_name = f
            break
    if gguf_name == "model.gguf" and gguf_files:
        gguf_name = gguf_files[0]
    with open(modelfile_path, "w", encoding="utf-8") as fh:
        fh.write(
            f"FROM ./{gguf_name}\n"
            "PARAMETER temperature 0.7\n"
            "PARAMETER top_p 0.95\n"
        )
    print(f"[save] Ollama Modelfile -> {modelfile_path}")

    print(
        "\nNext steps on the Mac app host:\n"
        "  1. Download the outputs/<dialect>/gguf/ folder + outputs/<dialect>/Modelfile.\n"
        "  2. Place the Modelfile next to the .gguf file, then:\n"
        f"       ollama create {ollama_name} -f Modelfile\n"
        "  3. In the app .env set:\n"
        "       LLM_PROVIDER=ollama\n"
        f"       OLLAMA_MODEL={ollama_name}:latest\n"
        "  4. Run the app; Arabic scripts now come from the fine-tuned model.\n"
        "  5. Compare against the base model with:\n"
        f"       python finetuning/eval_ab.py --ft-model {ollama_name}:latest\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
