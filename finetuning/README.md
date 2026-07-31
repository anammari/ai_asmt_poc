# Arabic ASMR Fine-Tuning (Unsloth)

Goal: fine-tune a dialect-specific Arabic ASMR scriptwriter using real
ASMR YouTube transcripts from Syria and Egypt. This produces two
specialised models, one per dialect, served locally through Ollama
(`LLM_PROVIDER=ollama`).

## Pipeline overview

1. **Ingest YouTube videos**: add URLs to `finetuning/data/metadata.csv`
   and run `ingest_YT_transcript.py`.
2. **Build the dataset**: run `build_dataset.py --dialect <dialect>`.
3. **Fine-tune on Colab**: run `train_unsloth_sft.py --dialect <dialect>`.

## Files

| file | runs where | purpose |
|---|---|---|
| `data/metadata.csv` | local | Curated list of dialect/video pairs to ingest (columns: `Dialect`, `URL`) |
| `ingest_YT_transcript.py` | local (CPU) | Downloads video title + Arabic transcript from YouTube into `data/ingested/<dialect>/` |
| `build_dataset.py` | local (CPU) | Maps `title -> user` and `transcript -> assistant`; writes `data/training/<dialect>/arabic_asmr_sft.jsonl` |
| `train_unsloth_sft.py` | Colab / CUDA | QLoRA SFT with Unsloth; exports LoRA adapter + `q4_k_m` GGUF + Ollama `Modelfile` |
| `eval_ab.py` | local | A/B: base Ollama model vs fine-tuned model -> `eval_report.md` |

## Data mapping principle (CRITICAL)

- **User prompt** = the YouTube video title (wrapped with the dialect label).
- **Assistant completion** = the full Arabic video transcript.

`build_dataset.py` enforces this 1-to-1 mapping in the chat-format JSONL
`messages` array.

## Step-by-step developer walkthrough

### 1. Add YouTube URLs to ingest

Edit `finetuning/data/metadata.csv`:

```csv
Dialect,URL
Syria,https://www.youtube.com/watch?v=SyriaSample1
Syria,https://www.youtube.com/watch?v=SyriaSample2
Egypt,https://www.youtube.com/watch?v=EgyptSample1
```

Use exactly `Syria` or `Egypt` in the `Dialect` column.

Run the ingestion script:

```bash
uv run python finetuning/ingest_YT_transcript.py
# or explicitly:
uv run python finetuning/ingest_YT_transcript.py \
    --metadata-csv finetuning/data/metadata.csv \
    --output-root finetuning/data/ingested
```

This creates JSON files like:

```
finetuning/data/ingested/syria/20250731_120000_SyriaSample1.json
finetuning/data/ingested/egypt/20250731_120100_EgyptSample1.json
```

Each file contains `dialect`, `video_id`, `url`, `title`, `transcript`
and `ingested_at`.

### 2. Build the training dataset

```bash
uv run python finetuning/build_dataset.py --dialect syria
uv run python finetuning/build_dataset.py --dialect egypt
```

Outputs:

```
finetuning/data/training/syria/arabic_asmr_sft.jsonl
finetuning/data/training/egypt/arabic_asmr_sft.jsonl
```

You can override paths:

```bash
uv run python finetuning/build_dataset.py --dialect syria \
    --ingestion-dir finetuning/data/ingested/syria \
    --out finetuning/data/training/syria/arabic_asmr_sft.jsonl
```

### 3. Upload prepared data to Google Drive

Before training on Colab, copy the prepared dataset to Google Drive so it
persists between Colab sessions:

```bash
# From your local machine, upload the training data and script to Drive.
# Option A — use the Google Drive web UI:
#   1. Open drive.google.com
#   2. Create folder: arabic-asmr/syria/
#   3. Upload finetuning/data/training/syria/arabic_asmr_sft.jsonl into it
#   4. Upload finetuning/train_unsloth_sft.py into arabic-asmr/
#
# Option B — use rclone (if your Drive is synced locally):
cp finetuning/data/training/syria/arabic_asmr_sft.jsonl ~/GoogleDrive/arabic-asmr/syria/
cp finetuning/train_unsloth_sft.py ~/GoogleDrive/arabic-asmr/
```

You only need to upload the JSONL file and the training script. The ingestion
and dataset builder run on your Mac, not on Colab.

### 4. Train on Google Colab

**Step 4a — start a Colab notebook:**

1. Go to https://colab.research.google.com/
2. Runtime → Change runtime type → T4 GPU
3. Create a new notebook

**Step 4b — mount Google Drive and set up the environment:**

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
!pip install -q unsloth trl datasets
```

**Step 4c — train with the dataset on Drive:**

```python
# Syrian dialect — dataset and outputs live on Drive
!python /content/drive/MyDrive/arabic-asmr/train_unsloth_sft.py \
    --dialect syria \
    --dataset /content/drive/MyDrive/arabic-asmr/syria/arabic_asmr_sft.jsonl \
    --output-dir /content/drive/MyDrive/arabic-asmr/outputs/syria \
    --epochs 2

# Egyptian dialect (once you have Egyptian data)
!python /content/drive/MyDrive/arabic-asmr/train_unsloth_sft.py \
    --dialect egypt \
    --dataset /content/drive/MyDrive/arabic-asmr/egypt/arabic_asmr_sft.jsonl \
    --output-dir /content/drive/MyDrive/arabic-asmr/outputs/egypt \
    --epochs 2
```

**Why Google Drive is important:**
| Without Drive (Files panel) | With Drive (mount) |
|---|---|
| Data lost on session disconnect | Data persists forever |
| Cannot resume after timeout | Checkpoints survive for resume |
| Must re-upload every time | Upload once, train many times |
| Outputs lost if session dies | GGUF + adapter survive for download |

Default base model: `unsloth/Qwen2.5-7B-Instruct-unsloth-bnb-4bit`
(strong Arabic). Lighter alternative:
`--base-model unsloth/Qwen3-4B-Instruct-2507-bnb-4bit`.

Expected output on Drive after training:

```
/content/drive/MyDrive/arabic-asmr/outputs/syria/adapter/
/content/drive/MyDrive/arabic-asmr/outputs/syria/gguf/
/content/drive/MyDrive/arabic-asmr/outputs/syria/Modelfile
/content/drive/MyDrive/arabic-asmr/outputs/syria/checkpoint-XXX/
```

### 5. Resume from an interrupted session

If your Colab session disconnects (common on the free tier), re-mount Drive
and re-run the same command. The script detects existing checkpoints:

```python
from google.colab import drive
drive.mount('/content/drive')
!pip install -q unsloth trl datasets

# The --output-dir already has checkpoints, so resume is automatic:
!python /content/drive/MyDrive/arabic-asmr/train_unsloth_sft.py \
    --dialect syria \
    --dataset /content/drive/MyDrive/arabic-asmr/syria/arabic_asmr_sft.jsonl \
    --output-dir /content/drive/MyDrive/arabic-asmr/outputs/syria \
    --epochs 2
```

To force a fresh run (ignore checkpoints):

```python
!python /content/drive/MyDrive/arabic-asmr/train_unsloth_sft.py \
    --dialect syria \
    --dataset /content/drive/MyDrive/arabic-asmr/syria/arabic_asmr_sft.jsonl \
    --output-dir /content/drive/MyDrive/arabic-asmr/outputs/syria \
    --epochs 2 \
    --no-resume
```

### 6. Download artifacts from Drive

After training completes, download the GGUF and Modelfile from Google Drive
to your Mac:

```bash
# Download the trained artifacts
# (from Google Drive web UI or rclone)

# Then serve locally via Ollama:
ollama create arabic-asmr-syria -f Modelfile
ollama run arabic-asmr-syria:latest "اختبرني"
```

### Point the app at the fine-tuned model

For the Syrian model:

```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=arabic-asmr-syria:latest
```

Or the Egyptian model:

```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=arabic-asmr-egypt:latest
```

### Evaluate

```bash
uv run python finetuning/eval_ab.py --ft-model arabic-asmr-syria:latest
```

Fill the rubric in `finetuning/eval_report.md`. If quality is weak,
add more high-quality YouTube URLs to `metadata.csv`, re-run ingestion
and dataset creation, then re-train.

## Notes

- The old Gemini-based synthetic generation has been removed. Every
  training sample must come from a real YouTube transcript curated in
  `metadata.csv`.
- Unsloth only runs on Linux with NVIDIA GPUs. The Colab T4 runtime is
  the intended training target.
- Keep `LLM_PROVIDER=gemini` in the app if you only want the prompt
  path; the fine-tuned models are an opt-in provider.
