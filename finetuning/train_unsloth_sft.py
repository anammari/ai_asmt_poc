# finetuning/train_unsloth_sft.py
"""
Unsloth QLoRA fine-tuning for the Arabic ASMR scriptwriter.

DESIGNED TO RUN ON GOOGLE COLAB (free T4) OR A LINUX CUDA MACHINE.
Unsloth does not support macOS - do not run this on the Mac app host.
(The Mac only builds the dataset and later serves the exported GGUF
through Ollama.)

Quick start on Colab (Runtime -> T4 GPU):
    !pip install -q unsloth trl datasets
    # upload finetuning/data/arabic_asmr_sft.jsonl next to this script, then:
    !python train_unsloth_sft.py --dataset arabic_asmr_sft.jsonl --epochs 2

Outputs (default: outputs/):
    outputs/adapter/   LoRA adapter (small; can be pushed to HF Hub)
    outputs/gguf/      q4_k_m GGUF for llama.cpp/Ollama
    outputs/Modelfile  Ollama recipe; then on the Mac:
                         ollama create arabic-asmr -f outputs/Modelfile
                       and set in the app .env:
                         LLM_PROVIDER=ollama
                         OLLAMA_MODEL=arabic-asmr:latest
"""
import argparse
import os
import sys

DEFAULT_DATASET = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "data", "arabic_asmr_sft.jsonl")

# Small instruct models with strong Arabic. Verified Unsloth 4-bit repos:
#   unsloth/Qwen2.5-7B-Instruct-unsloth-bnb-4bit  (default, best Arabic)
#   unsloth/Qwen3-4B-Instruct-2507-bnb-4bit       (lighter, faster)
#   unsloth/gemma-3-4b-it-unsloth-bnb-4bit        (alternative)
DEFAULT_BASE_MODEL = "unsloth/Qwen2.5-7B-Instruct-unsloth-bnb-4bit"
MAX_SEQ_LENGTH = 2048


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Unsloth QLoRA SFT for Arabic ASMR.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--base-model",
                        default=os.getenv("FT_BASE_MODEL", DEFAULT_BASE_MODEL))
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--gguf-quant", default="q4_k_m")
    parser.add_argument("--ollama-name", default="arabic-asmr")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

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

    if not os.path.exists(args.dataset):
        print(f"error: dataset not found: {args.dataset}", file=sys.stderr)
        return 2

    print(f"[setup] base model: {args.base_model}")
    print(f"[setup] dataset:    {args.dataset}")
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

    dataset = load_dataset("json", data_files=args.dataset, split="train")
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
            output_dir=args.output_dir,
            report_to="none",
        ),
    )
    trainer.train()

    adapter_dir = os.path.join(args.output_dir, "adapter")
    gguf_dir = os.path.join(args.output_dir, "gguf")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"[save] LoRA adapter -> {adapter_dir}")

    print(f"[save] exporting GGUF ({args.gguf_quant}) -> {gguf_dir}")
    model.save_pretrained_gguf(gguf_dir, tokenizer,
                               quantization_method=args.gguf_quant)

    modelfile_path = os.path.join(args.output_dir, "Modelfile")
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
        "  1. Download the outputs/gguf/ folder + outputs/Modelfile.\n"
        "  2. Place the Modelfile next to the .gguf file, then:\n"
        f"       ollama create {args.ollama_name} -f Modelfile\n"
        "  3. In the app .env set:\n"
        "       LLM_PROVIDER=ollama\n"
        f"       OLLAMA_MODEL={args.ollama_name}:latest\n"
        "  4. Run the app; Arabic scripts now come from the fine-tuned model.\n"
        "  5. Compare against the base model with:\n"
        f"       python finetuning/eval_ab.py --ft-model {args.ollama_name}:latest\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
