# tts.py
import os
import re
import io
import numpy as np
import soundfile as sf

try:
    from kokoro import KPipeline
    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False

_SILMA_MODEL = None
_ARABIC_REF_AUDIO = "input/test_ahmad_2.wav"
_ARABIC_REF_TEXT = (
    "خذ نفس عميق، ريح بالك كل شي تمام، خلي عيونك مغمضة واسمعني، "
    "اليوم كان طويل صح؟ هلق صار الوقت ترتاح، انسى كل الهم والقلق"
)


def _resample_linear(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio.astype(np.float32)
    if len(audio) == 0:
        return audio.astype(np.float32)
    new_len = max(1, int(round(len(audio) * (target_sr / orig_sr))))
    x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def _load_silma_model():
    global _SILMA_MODEL
    if _SILMA_MODEL is not None:
        return _SILMA_MODEL

    try:
        from f5_tts.model import DiT
        from f5_tts.infer.utils_infer import load_model
    except ImportError as exc:
        raise ImportError(
            "Arabic SILMA/F5-TTS requires 'f5-tts'. Install dependencies and retry."
        ) from exc

    _SILMA_MODEL = load_model(
        model_cls=DiT,
        ckpt_path="silma-ai/silma-tts",
        mel_spec_type="vocos",
        vocab_file=None,
    )
    return _SILMA_MODEL


def generate_f5_audio(text: str, speed: float = 0.85) -> np.ndarray:
    if not os.path.exists(_ARABIC_REF_AUDIO):
        raise FileNotFoundError(
            f"Arabic reference audio is required for SILMA/F5-TTS at '{_ARABIC_REF_AUDIO}'."
        )

    try:
        from f5_tts.infer.utils_infer import infer_process
    except ImportError as exc:
        raise ImportError(
            "Arabic SILMA/F5-TTS requires 'f5-tts'. Install dependencies and retry."
        ) from exc

    model = _load_silma_model()
    wav_tensor, sample_rate, _ = infer_process(
        ref_audio=_ARABIC_REF_AUDIO,
        ref_text=_ARABIC_REF_TEXT,
        gen_text=text,
        model_obj=model,
        mel_spec_type="vocos",
        speed=speed,
    )

    if hasattr(wav_tensor, "detach"):
        wav_array = wav_tensor.squeeze().detach().cpu().numpy()
    else:
        wav_array = np.asarray(wav_tensor).squeeze()

    wav_array = wav_array.astype(np.float32)
    return _resample_linear(wav_array, int(sample_rate), 24000)

def create_tts_pipeline(lang_code="a"):
    """Initializes the Kokoro TTS pipeline into memory."""
    if KOKORO_AVAILABLE:
        # Relies on the HF_HOME cache preloaded during setup
        return KPipeline(lang_code=lang_code)
    return None

def synthesize(pipeline, text: str, voices=None, speed=0.85, vocal_tone="Soft Spoken") -> bytes:
    """
    Synthesizes text into a WAV file buffer. 
    Supports alternating multiple voices and injecting absolute silence.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not text.strip():
        raise ValueError("text must not be empty")

    if voices is None:
        voices = ["af_bella"]
        
    # Ensure voices is always a list for paragraph alternation
    voices = [voices] if isinstance(voices, str) else voices
    if not voices:
        voices = ["af_bella"]
        
    # Adjust speed slightly if whispering is requested to simulate slower, breathier articulation
    if vocal_tone == "Whispering":
        speed = max(0.65, speed - 0.1)
        
    # Split the incoming script into paragraphs to swap voices on line breaks
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    all_audio = []
    sample_rate = 24000
    
    # Regex to extract our explicitly defined silence markers from sanitize_text
    silence_regex = re.compile(r'<<SILENCE(\d+)MS>>')

    for i, paragraph in enumerate(paragraphs):
        # Rotate voices sequentially
        current_voice = voices[i % len(voices)]
        is_arabic_voice = current_voice.startswith("ar_")

        if not is_arabic_voice and (not KOKORO_AVAILABLE or pipeline is None):
            raise ImportError("Kokoro TTS is required but missing. Run `uv sync`.")
        
        # Split paragraph chunks around the silence markers
        parts = silence_regex.split(paragraph)
        
        for j, part in enumerate(parts):
            if j % 2 == 1:
                # Matched group (odd index) represents the silence duration in MS
                duration_ms = int(part)
                num_samples = int((duration_ms / 1000.0) * sample_rate)
                if num_samples > 0:
                    # Inject an array of zeros (absolute digital silence)
                    all_audio.append(np.zeros(num_samples, dtype=np.float32))
            else:
                # Normal text portion to synthesize
                text_part = part.strip()
                if text_part:
                    if is_arabic_voice:
                        all_audio.append(generate_f5_audio(text_part, speed=speed))
                    else:
                        generator = pipeline(text_part, voice=current_voice, speed=speed, split_pattern=r"\n+")
                        for _, _, audio in generator:
                            if audio is None:
                                raise RuntimeError("TTS engine returned an empty audio chunk.")
                            all_audio.append(audio)

    if not all_audio:
        raise RuntimeError("TTS failed to produce any audio segments.")

    full_audio = np.concatenate(all_audio)
    
    buffer = io.BytesIO()
    # Write as WAV format
    sf.write(buffer, full_audio, sample_rate, format='WAV')
    buffer.seek(0)
    
    return buffer.read()