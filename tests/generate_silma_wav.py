import os
import sys
from pathlib import Path

import soundfile as sf

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tts import _ARABIC_REF_AUDIO, generate_f5_audio

# Inputs aligned with tts.py
reference_audio = _ARABIC_REF_AUDIO
target_text = "مرحباً بكم في عالم الاسترخاء. خذ نفساً عميقاً واستمع إلى هذا الصوت الهادئ."
output_path = "output/silma_asmr_whisper.wav"


if __name__ == "__main__":
    if not os.path.exists(reference_audio):
        raise FileNotFoundError(
            f"Reference audio is missing at '{reference_audio}'. "
            "Place your Arabic voice sample there before running this script."
        )

    print("Loading SILMA TTS Model...")
    print("Generating high-quality whispered audio...")
    wav = generate_f5_audio(target_text, speed=0.85)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, wav, 24000, subtype="PCM_16")

    print(f"Success! High-quality ASMR WAV file saved at: {output_path}")
