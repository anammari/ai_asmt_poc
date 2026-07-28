import inspect
import os

import torchaudio
from f5_tts.infer.utils_infer import infer_process, load_model
from f5_tts.model import DiT

# Inputs aligned with tts.py
reference_audio = "input/test_ahmad_2.wav"
reference_text = (
    "خذ نفس عميق، ريح بالك كل شي تمام، خلي عيونك مغمضة واسمعني، "
    "اليوم كان طويل صح؟ هلق صار الوقت ترتاح، انسى كل الهم والقلق"
)
target_text = "مرحباً بكم في عالم الاسترخاء. خذ نفساً عميقاً واستمع إلى هذا الصوت الهادئ."
output_path = "output/silma_asmr_whisper.wav"
model_ckpt = "silma-ai/silma-tts"


def _build_load_model_kwargs() -> dict:
    sig = inspect.signature(load_model)
    kwargs = {
        "model_cls": DiT,
        "mel_spec_type": "vocos",
        "vocab_file": None,
    }
    if "ckpt_path" in sig.parameters:
        kwargs["ckpt_path"] = model_ckpt
    elif "chkp_path" in sig.parameters:
        kwargs["chkp_path"] = model_ckpt
    elif "ckpt_file" in sig.parameters:
        kwargs["ckpt_file"] = model_ckpt
    elif "model_ckpt" in sig.parameters:
        kwargs["model_ckpt"] = model_ckpt
    return kwargs


if __name__ == "__main__":
    if not os.path.exists(reference_audio):
        raise FileNotFoundError(
            f"Reference audio is missing at '{reference_audio}'. "
            "Place your Arabic voice sample there before running this script."
        )

    print("Loading SILMA TTS Model...")
    model = load_model(**_build_load_model_kwargs())

    print("Generating high-quality whispered audio...")
    wav, sample_rate, _ = infer_process(
        ref_audio=reference_audio,
        ref_text=reference_text,
        gen_text=target_text,
        model_obj=model,
        mel_spec_type="vocos",
        speed=0.85,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torchaudio.save(output_path, wav.cpu(), sample_rate, bits_per_sample=16)

    print(f"Success! High-quality ASMR WAV file saved at: {output_path}")
