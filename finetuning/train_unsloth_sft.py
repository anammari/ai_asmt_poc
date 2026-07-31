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
import sys
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
    gguf_dir = os.path.join(output_dir, "gguf")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"[save] LoRA adapter -> {adapter_dir}")

    print(f"[save] exporting GGUF ({args.gguf_quant}) -> {gguf_dir}")
    model.save_pretrained_gguf(gguf_dir, tokenizer,
                             quantization_method=args.gguf_quant)

    modelfile_path = os.path.join(output_dir, "Modelfile")
    gguf_files = [f for f in os.listdir(gguf_dir) if f.endswith(".gguf")]
    gguf_name = gguf_files[0] if gguf_files else "model.gguf"
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
