# Arabic ASMR Fine-Tuning (Unsloth)

Goal: improve the quality of the **Arabic ASMR transcripts** the app generates,
beyond what prompting alone achieves, by supervised fine-tuning (SFT) a small
Arabic-capable instruct model on curated Arabic ASMR scripts, then serving it
locally through Ollama (which the app already supports via `LLM_PROVIDER=ollama`).

## Why this design

- **Unsloth QLoRA**: trains on a free Colab T4 (Unsloth does not run on macOS).
  The Mac only builds the dataset and serves the result.
- **Chat-format SFT** on the app's actual system-prompt shape
  (`finetuning/build_dataset.py` mirrors `llm.rewrite_script` for Arabic), so
  the fine-tune matches the runtime distribution.
- **GGUF export + Ollama**: zero app code changes — just point
  `OLLAMA_MODEL` at the fine-tuned model.

## Files

| file | runs where | purpose |
|---|---|---|
| `build_dataset.py` | Mac (local) | Seeds + Gemini-augmented Arabic ASMR samples -> `data/arabic_asmr_sft.jsonl` (validated chat-format JSONL) |
| `train_unsloth_sft.py` | Colab / CUDA | QLoRA SFT with Unsloth; exports LoRA adapter + `q4_k_m` GGUF + Ollama `Modelfile` |
| `eval_ab.py` | Mac (local) | A/B: base Ollama model vs fine-tuned model -> `eval_report.md` |
| `data/arabic_asmr_sft.jsonl` | - | Generated dataset (5 built-in seeds + Gemini samples) |

## Developer action points (step by step)

1. **Build the dataset locally** (uses your `GEMINI_API_KEY`; ~5-15 min):
   ```bash
   uv run python finetuning/build_dataset.py --count 60
   ```
   (Smoke-test without Gemini: `uv run python finetuning/build_dataset.py --offline`)

2. **Train on Colab** (Runtime -> Change runtime type -> T4 GPU):
   ```python
   !pip install -q unsloth trl datasets
   # Upload train_unsloth_sft.py + arabic_asmr_sft.jsonl (Files panel), then:
   !python train_unsloth_sft.py --dataset arabic_asmr_sft.jsonl --epochs 2
   ```
   Default base model: `unsloth/Qwen2.5-7B-Instruct-unsloth-bnb-4bit` (strong
   Arabic). Lighter alternative: `--base-model unsloth/Qwen3-4B-Instruct-2507-bnb-4bit`.
   Expected runtime on T4: ~30-60 min for 60 samples x 2 epochs.

3. **Download the artifacts**: `outputs/gguf/*.gguf` and `outputs/Modelfile`.
   (Optional: push `outputs/adapter/` to HF Hub for versioning.)

4. **Serve locally via Ollama** (on the Mac, next to the `.gguf` file):
   ```bash
   ollama create arabic-asmr -f Modelfile
   ollama run arabic-asmr:latest "اختبرني"   # sanity check
   ```

5. **Point the app at the fine-tuned model** (`.env`):
   ```env
   LLM_PROVIDER=ollama
   OLLAMA_MODEL=arabic-asmr:latest
   ```

6. **Evaluate** (base vs fine-tuned side-by-side):
   ```bash
   uv run python finetuning/eval_ab.py --ft-model arabic-asmr:latest
   ```
   Fill the rubric in `finetuning/eval_report.md`. If quality is weak:
   add more/better seed exemplars to `SEED_SAMPLES` in `build_dataset.py`,
   raise `--count`, or train one more epoch — then repeat from step 1.

## Notes

- The dataset quality gate rejects non-Arabic, markdown-tainted, pause-less,
  or length-outlier samples, so a larger `--count` mostly costs Gemini quota.
- Keep `LLM_PROVIDER=gemini` if you only want the improved prompt path; the
  fine-tune is an opt-in alternative provider.
