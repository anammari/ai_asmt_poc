# tts.py
import os
import re
import io
import numpy as np
import soundfile as sf


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip())
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip())
    except (TypeError, ValueError):
        return default


def _resample_linear(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio.astype(np.float32)
    if len(audio) == 0:
        return audio.astype(np.float32)
    new_len = max(1, int(round(len(audio) * (target_sr / orig_sr))))
    x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def _validate_output(wav_array: np.ndarray) -> np.ndarray:
    if wav_array.size == 0:
        raise RuntimeError("TTS returned an empty audio segment.")
    if not np.isfinite(wav_array).all():
        raise RuntimeError("TTS returned non-finite audio samples (NaN/Inf).")
    clipping_ratio = float(np.mean(np.abs(wav_array) >= 1.0))
    if clipping_ratio > 0.001:
        print(f"[Warning] TTS output clipping on {clipping_ratio:.1%} of samples; clamping.")
        wav_array = np.clip(wav_array, -1.0, 1.0)
    return wav_array


_FISH_AUDIO_VOICE_ID = os.getenv("FISH_AUDIO_VOICE_ID", "").strip() or None


def generate_fish_audio(
    text: str,
    speed: float = 0.85,
    vocal_tone: str = "Soft Spoken",
    voice_id: str | None = None,
) -> np.ndarray:
    if not text or not text.strip():
        raise ValueError("text must not be empty")
    try:
        from fish_audio_tts import FishAudioTTS
    except ImportError as exc:
        raise ImportError(
            "Fish Audio backend requires fish_audio_tts.py and 'requests'."
        ) from exc

    # Strip URL artifacts that cause noise.
    text = re.sub(r'https?://\S+', '', text)

    # Normalize Arabic orthography BEFORE converting silence markers,
    # so normalize_arabic() preserves the <<SILENCE>> markers and does not
    # strip the [break]/[long-break] tags that replace them.
    try:
        from preprocess_arabic import normalize_arabic
        text = normalize_arabic(text)
    except ImportError:
        pass

    # Convert <<SILENCE>> markers to Fish Audio S2 natural-language tags.
    # Fish Audio does NOT support millisecond-exact silence tags.
    # Supported tags: [pause], [short pause], [long pause], [long-break].
    # Multiple [pause] tags sequentially create longer gaps.
    def _silence_to_fish_tags(ms: int) -> str:
        if ms < 500:
            return "[pause]"
        elif ms < 1500:
            return "[pause] [pause]"
        elif ms < 3000:
            return "[long pause]"
        elif ms < 5000:
            return "[long pause] [pause]"
        else:
            return "[long pause] [long pause]"

    text = re.sub(
        r"<<SILENCE(\d+)MS>>",
        lambda m: _silence_to_fish_tags(int(m.group(1))),
        text,
    )

    prefix = "[whispering] " if vocal_tone == "Whispering" else "[soft] "
    effective_voice_id = voice_id or _FISH_AUDIO_VOICE_ID

    client = FishAudioTTS()
    result = client.synthesize(
        text=prefix + text.strip(),
        voice_id=effective_voice_id,
        fmt="wav",
        speed=max(0.5, min(2.0, speed)),
        sample_rate=44100,
        name="app-arabic",
    )

    data, sample_rate = sf.read(io.BytesIO(result.audio_bytes), dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    data = _validate_output(data.astype(np.float32))
    return _resample_linear(data, int(sample_rate), 24000)


def synthesize(text: str, voices=None, speed=0.85, vocal_tone="Soft Spoken") -> bytes:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not text.strip():
        raise ValueError("text must not be empty")

    if voices is None:
        voices = ["fish_0de68eaa0cc5438389b82bba728c8e39"]
    voices = [voices] if isinstance(voices, str) else voices
    if not voices:
        voices = ["fish_0de68eaa0cc5438389b82bba728c8e39"]

    if vocal_tone == "Whispering":
        speed = max(0.65, speed - 0.1)

    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    all_audio = []
    sample_rate = 24000

    for i, paragraph in enumerate(paragraphs):
        current_voice = voices[i % len(voices)]
        extracted_id = current_voice[5:] if current_voice.startswith("fish_") else None
        all_audio.append(generate_fish_audio(
            paragraph, speed=speed,
            vocal_tone=vocal_tone,
            voice_id=extracted_id or None,
        ))

    if not all_audio:
        raise RuntimeError("TTS failed to produce any audio segments.")

    full_audio = np.concatenate(all_audio)
    buffer = io.BytesIO()
    sf.write(buffer, full_audio, sample_rate, format='WAV')
    buffer.seek(0)
    return buffer.read()
