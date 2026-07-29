# stt.py
import io
import re
import difflib
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


def _normalize_for_compare(text: str) -> str:
    """Comparison-normalizes Arabic text: no tashkeel, unified alefs, letters only."""
    try:
        import preprocess_arabic
        text = preprocess_arabic.strip_diacritics(text)
    except ImportError:
        text = re.sub(r"[ً-ْٰ]", "", text)
    text = re.sub(r"[أإآٱ]", "ا", text)
    text = re.sub(r"[^ء-ي0-9a-zA-Z ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def verify_audible_speech(
    audio_input: bytes | dict[str, Any],
    expected_text: str | None = None,
    language: str = "arabic",
    stt_pipeline=None,
    min_rms: float = 0.005,
    min_similarity: float = 0.3,
) -> dict[str, Any]:
    """
    Verifies that generated audio actually contains audible human speech
    (regression guard against "silent/noise-only" TTS outputs).

    Primary signal: whisper transcription must yield non-empty text in the
    expected language; when `expected_text` is given, the diacritics-stripped,
    alef-unified character similarity must reach `min_similarity`
    (lenient on purpose - whisper models transcribe whispered speech roughly).
    Secondary signal: overall RMS energy above `min_rms` (not silence).

    Returns a dict with: ok, rms, transcript, similarity, reason.
    """
    if isinstance(audio_input, (bytes, bytearray)):
        with io.BytesIO(audio_input) as buf:
            data, samplerate = sf.read(buf)
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        inputs: dict[str, Any] = {"array": data, "sampling_rate": samplerate}
    elif isinstance(audio_input, dict):
        inputs = audio_input
        data = np.asarray(audio_input["array"], dtype=np.float32)
    else:
        raise TypeError("audio_input must be bytes/bytearray or a dict with array/sampling_rate")

    rms = float(np.sqrt(np.mean(np.square(data)))) if len(data) else 0.0

    if stt_pipeline is None:
        stt_pipeline = create_stt_pipeline()
    transcript = transcribe(stt_pipeline, inputs, language=language)
    has_arabic = _contains_arabic_chars(transcript)

    similarity = None
    if expected_text is not None:
        ref = _normalize_for_compare(expected_text)
        hyp = _normalize_for_compare(transcript)
        similarity = (
            difflib.SequenceMatcher(None, ref, hyp).ratio() if ref and hyp else 0.0
        )

    reasons = []
    if rms < min_rms:
        reasons.append(f"rms {rms:.4f} < {min_rms} (near-silence)")
    if not transcript:
        reasons.append("empty transcript")
    elif language.strip().lower() in {"arabic", "ar", "العربية"} and not has_arabic:
        reasons.append("transcript contains no Arabic characters")
    if expected_text is not None and (similarity or 0.0) < min_similarity:
        reasons.append(f"similarity {similarity:.2f} < {min_similarity}")

    return {
        "ok": not reasons,
        "rms": round(rms, 4),
        "transcript": transcript,
        "similarity": round(similarity, 3) if similarity is not None else None,
        "reason": "; ".join(reasons) if reasons else "ok",
    }