# stt.py
import io
import re
import soundfile as sf
import numpy as np
import os
from typing import Any

def _disable_torchcodec_for_asr():
    """
    Hard-disables torchcodec inside transformers to prevent the 
    'Could not load libtorchcodec' crash reported in logs.
    """
    try:
        import transformers.pipelines.automatic_speech_recognition as asr
        asr.is_torchcodec_available = lambda: False
    except ImportError:
        pass

def create_stt_pipeline():
    """Creates the whisper STT pipeline."""
    _disable_torchcodec_for_asr()
    from transformers import pipeline
    model_name = os.getenv("STT_MODEL", "openai/whisper-small")
    return pipeline("automatic-speech-recognition", model=model_name)

def _extract_text(output: Any) -> str:
    if isinstance(output, dict):
        return str(output.get("text", "")).strip()
    if isinstance(output, str):
        return output.strip()
    return ""

def _build_generate_kwargs(language: str | None) -> dict[str, str]:
    kwargs = {"task": "transcribe"}
    if not language:
        return kwargs

    normalized = language.strip().lower()
    if normalized in {"arabic", "ar", "العربية"}:
        kwargs["language"] = "arabic"
    elif normalized in {"english", "en"}:
        kwargs["language"] = "english"
    return kwargs

def _is_likely_bad_auto_transcript(text: str) -> bool:
    tokens = [tok for tok in text.lower().split() if tok]
    if len(tokens) < 20:
        return False
    unique_ratio = len(set(tokens)) / len(tokens)
    return unique_ratio < 0.35

def _contains_arabic_chars(text: str) -> bool:
    return any("\u0600" <= ch <= "\u06ff" for ch in text)

def _cleanup_transcript(text: str) -> str:
    cleaned = text.strip()
    while True:
        updated = re.sub(r"^\s*transcribed instructions:\s*", "", cleaned, flags=re.IGNORECASE)
        if updated == cleaned:
            break
        cleaned = updated.strip()
    return cleaned

def transcribe(pipeline, audio_input: bytes | dict[str, Any], language: str = "Auto") -> str:
    """
    Transcribes audio bytes safely without requiring an ffmpeg binary installation.
    """
    _disable_torchcodec_for_asr()

    if isinstance(audio_input, (bytes, bytearray)):
        with io.BytesIO(audio_input) as buf:
            data, samplerate = sf.read(buf)
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        inputs: dict[str, Any] = {"array": data, "sampling_rate": samplerate}
    elif isinstance(audio_input, dict):
        inputs = audio_input
    else:
        raise TypeError("audio_input must be bytes/bytearray or a dict with array/sampling_rate")

    generate_kwargs = _build_generate_kwargs(language)
    first_output = pipeline(inputs, batch_size=4, generate_kwargs=generate_kwargs)
    first_text = _cleanup_transcript(_extract_text(first_output))

    if language.strip().lower() == "auto" and _is_likely_bad_auto_transcript(first_text):
        arabic_output = pipeline(
            inputs,
            batch_size=4,
            generate_kwargs={"task": "transcribe", "language": "arabic"},
        )
        arabic_text = _cleanup_transcript(_extract_text(arabic_output))
        if _contains_arabic_chars(arabic_text):
            return arabic_text

    return first_text