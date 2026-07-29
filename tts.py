# tts.py
import os
import re
import io
import inspect
import numpy as np
import soundfile as sf
import yaml
from huggingface_hub import hf_hub_download

try:
    from kokoro import KPipeline
    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False

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


_SILMA_MODEL = None
_SILMA_VOCODER = None
_SILMA_REPO_ID = "silma-ai/silma-tts"
_F5_DEVICE = os.getenv("F5_TTS_DEVICE", "cpu")

# F5-TTS inference tuning (see f5_asmr_inference.py for before/after runs).
# nfe_step: diffusion steps; higher = cleaner whisper transients.
# cfg_strength: higher = stricter text adherence (fights hallucinations).
# sway_sampling_coef: -1.0 disables Sway Sampling; a positive value enables it.
# Defaults mirror the stock F5 profile proven stable with the Arabic voice;
# tune via env vars only after validating with f5_asmr_inference.py --sweep.
_F5_NFE_STEP = _env_int("F5_NFE_STEP", 32)
_F5_CFG_STRENGTH = _env_float("F5_CFG_STRENGTH", 2.0)
_F5_SWAY_SAMPLING_COEF = _env_float("F5_SWAY_SAMPLING_COEF", -1.0)
_F5_TARGET_RMS = _env_float("F5_TARGET_RMS", 0.1)

# Arabic TTS backend: "silma" (local SILMA/F5-TTS) or "fish" (Fish Audio API).
_ARABIC_TTS_BACKEND = os.getenv("ARABIC_TTS_BACKEND", "silma").strip().lower()

# Arabic reference audio engineering: ideally a 5-10s, mono, >=24kHz dry
# studio whisper clip (use prepare_ref_audio.py to produce one).
_ARABIC_REF_AUDIO = os.getenv("ARABIC_REF_AUDIO", "input/test_ahmad_2.wav")
_ARABIC_REF_TEXT_PATH = os.getenv("ARABIC_REF_TEXT_PATH", "input/test_ahmad_2.txt")
_ARABIC_REF_TEXT = (
    "خذ نفس عميق، ريح بالك كل شي تمام، خلي عيونك مغمضة واسمعني، "
    "اليوم كان طويل صح؟ هلق صار الوقت ترتاح، انسى كل الهم والقلق"
)
_ARABIC_REF_TEXT_CACHE: str | None = None
_REF_AUDIO_WARNED = False
_PREPROCESS_WARNED = False


def _resolve_silma_assets() -> tuple[dict, str, str]:
    config_path = hf_hub_download(repo_id=_SILMA_REPO_ID, filename="config.yaml")
    ckpt_path = hf_hub_download(repo_id=_SILMA_REPO_ID, filename="model.pt")
    vocab_path = hf_hub_download(repo_id=_SILMA_REPO_ID, filename="vocab.txt")

    with open(config_path, "r", encoding="utf-8") as cfg_file:
        config = yaml.safe_load(cfg_file)
    model_cfg = config["model"]["arch"]
    return model_cfg, ckpt_path, vocab_path


def _build_f5_load_model_kwargs(load_model, DiT, model_cfg: dict, ckpt_path: str, vocab_path: str):
    sig = inspect.signature(load_model)
    kwargs = {
        "model_cls": DiT,
        "mel_spec_type": "vocos",
    }
    if "ckpt_path" in sig.parameters:
        kwargs["ckpt_path"] = ckpt_path
    elif "chkp_path" in sig.parameters:
        kwargs["chkp_path"] = ckpt_path
    elif "ckpt_file" in sig.parameters:
        kwargs["ckpt_file"] = ckpt_path
    elif "model_ckpt" in sig.parameters:
        kwargs["model_ckpt"] = ckpt_path

    if "vocab_file" in sig.parameters:
        kwargs["vocab_file"] = vocab_path
    if "model_cfg" in sig.parameters:
        kwargs["model_cfg"] = model_cfg
    if "device" in sig.parameters:
        kwargs["device"] = _F5_DEVICE
    return kwargs


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

    model_cfg, ckpt_path, vocab_path = _resolve_silma_assets()
    _SILMA_MODEL = load_model(
        **_build_f5_load_model_kwargs(load_model, DiT, model_cfg, ckpt_path, vocab_path)
    )
    return _SILMA_MODEL


def _load_silma_vocoder():
    global _SILMA_VOCODER
    if _SILMA_VOCODER is not None:
        return _SILMA_VOCODER

    try:
        from f5_tts.infer.utils_infer import load_vocoder
    except ImportError as exc:
        raise ImportError(
            "Arabic SILMA/F5-TTS requires 'f5-tts'. Install dependencies and retry."
        ) from exc

    _SILMA_VOCODER = load_vocoder(
        vocoder_name="vocos",
        is_local=False,
        local_path="",
        device=_F5_DEVICE,
    )
    return _SILMA_VOCODER


def _load_arabic_ref_text() -> str:
    """
    Reference transcript for the Arabic F5 voice. Prefers the exact text file
    at ARABIC_REF_TEXT_PATH (use prepare_ref_audio.py to diacritize it);
    falls back to the built-in inline transcript.
    """
    global _ARABIC_REF_TEXT_CACHE
    if _ARABIC_REF_TEXT_CACHE is not None:
        return _ARABIC_REF_TEXT_CACHE
    if os.path.exists(_ARABIC_REF_TEXT_PATH):
        with open(_ARABIC_REF_TEXT_PATH, encoding="utf-8") as fh:
            text = fh.read().strip()
            if text:
                _ARABIC_REF_TEXT_CACHE = text
                return text
    return _ARABIC_REF_TEXT


def _validate_ref_audio(path: str) -> dict:
    """
    Checks the reference clip against the F5 best-practice spec
    (5-10s, mono, >=24kHz). Warnings print once per process; never fatal.
    """
    global _REF_AUDIO_WARNED
    info: dict = {"duration_s": None, "sample_rate": None, "channels": None, "warnings": []}
    try:
        meta = sf.info(path)
        info["duration_s"] = round(meta.frames / float(meta.samplerate), 2)
        info["sample_rate"] = meta.samplerate
        info["channels"] = meta.channels
    except Exception as exc:
        info["warnings"].append(f"Could not inspect reference audio: {exc}")

    duration = info["duration_s"]
    if duration is not None and not (5.0 <= duration <= 10.0):
        info["warnings"].append(
            f"Reference audio is {duration}s; F5 style transfer works best with 5-10s. "
            "Trim it with prepare_ref_audio.py."
        )
    if info["channels"] is not None and info["channels"] != 1:
        info["warnings"].append(
            f"Reference audio has {info['channels']} channels; mono is recommended."
        )
    if info["sample_rate"] is not None and info["sample_rate"] < 24000:
        info["warnings"].append(
            f"Reference audio is {info['sample_rate']}Hz; 24kHz+ is recommended."
        )

    if not _REF_AUDIO_WARNED:
        for warning in info["warnings"]:
            print(f"[Warning] Arabic F5 reference audio: {warning}")
        _REF_AUDIO_WARNED = True
    return info


def _apply_arabic_preprocessing(text: str) -> str:
    """
    Normalizes + diacritizes Arabic text before F5 inference (reduces
    hallucinations and mispronunciation). Soft-imports preprocess_arabic;
    degrades gracefully (never raises) if the diacritizer is unavailable.
    """
    global _PREPROCESS_WARNED
    if not text or not text.strip():
        return text
    try:
        import preprocess_arabic
    except ImportError:
        if not _PREPROCESS_WARNED:
            print("[Warning] preprocess_arabic.py not found; sending raw text to F5-TTS.")
            _PREPROCESS_WARNED = True
        return text

    normalized = preprocess_arabic.normalize_arabic(text)
    try:
        return preprocess_arabic.diacritize(normalized)
    except (ImportError, ValueError) as exc:
        if not _PREPROCESS_WARNED:
            print(f"[Warning] Arabic diacritization skipped: {exc}")
            _PREPROCESS_WARNED = True
        return normalized
    except Exception as exc:
        if not _PREPROCESS_WARNED:
            print(f"[Warning] Arabic diacritization failed ({exc}); using normalized text.")
            _PREPROCESS_WARNED = True
        return normalized


def _validate_f5_output(wav_array: np.ndarray) -> np.ndarray:
    """Sanity-checks F5 output: non-empty, finite, and clip-safe."""
    if wav_array.size == 0:
        raise RuntimeError("F5-TTS returned an empty audio segment.")
    if not np.isfinite(wav_array).all():
        raise RuntimeError("F5-TTS returned non-finite audio samples (NaN/Inf).")
    clipping_ratio = float(np.mean(np.abs(wav_array) >= 1.0))
    if clipping_ratio > 0.001:
        print(f"[Warning] F5-TTS output clipping on {clipping_ratio:.1%} of samples; clamping.")
        wav_array = np.clip(wav_array, -1.0, 1.0)
    return wav_array


def generate_f5_audio(
    text: str,
    speed: float = 0.85,
    *,
    use_preprocessing: bool = True,
    nfe_step: int | None = None,
    cfg_strength: float | None = None,
    sway_sampling_coef: float | None = None,
    target_rms: float | None = None,
    ref_audio: str | None = None,
    ref_text: str | None = None,
) -> np.ndarray:
    ref_audio = ref_audio or _ARABIC_REF_AUDIO
    if not os.path.exists(ref_audio):
        raise FileNotFoundError(
            f"Arabic reference audio is required for SILMA/F5-TTS at '{ref_audio}'."
        )

    try:
        from f5_tts.infer.utils_infer import infer_process
        import f5_tts.infer.utils_infer as infer_utils
        import torch
    except ImportError as exc:
        raise ImportError(
            "Arabic SILMA/F5-TTS requires 'f5-tts'. Install dependencies and retry."
        ) from exc

    _validate_ref_audio(ref_audio)

    # CRITICAL: the reference text must stay the EXACT transcript of the
    # reference audio. F5-TTS relies on precise character-to-phoneme
    # alignment with the ref clip; algorithmically altering it (e.g. guessed
    # diacritics) breaks alignment and can collapse the output to noise.
    # Diacritize the ref transcript OFFLINE (prepare_ref_audio.py) and
    # manually verify it instead.
    effective_ref_text = ref_text if ref_text is not None else _load_arabic_ref_text()
    if use_preprocessing:
        text = _apply_arabic_preprocessing(text)

    # Explicit inference params beat env-configured module defaults.
    tuning = {
        "nfe_step": nfe_step if nfe_step is not None else _F5_NFE_STEP,
        "cfg_strength": cfg_strength if cfg_strength is not None else _F5_CFG_STRENGTH,
        "sway_sampling_coef": (
            sway_sampling_coef if sway_sampling_coef is not None else _F5_SWAY_SAMPLING_COEF
        ),
        "target_rms": target_rms if target_rms is not None else _F5_TARGET_RMS,
    }
    # Only forward params supported by the installed f5-tts version.
    supported = inspect.signature(infer_process).parameters
    tuning = {k: v for k, v in tuning.items() if k in supported}

    model = _load_silma_model()
    vocoder = _load_silma_vocoder()

    def _soundfile_audio_loader(path: str):
        audio_np, sample_rate = sf.read(path, dtype="float32")
        if audio_np.ndim == 1:
            audio_np = np.expand_dims(audio_np, axis=0)
        else:
            audio_np = audio_np.T
        return torch.from_numpy(audio_np), sample_rate

    original_torchaudio_load = infer_utils.torchaudio.load
    infer_utils.torchaudio.load = _soundfile_audio_loader
    try:
        wav_tensor, sample_rate, _ = infer_process(
            ref_audio=ref_audio,
            ref_text=effective_ref_text,
            gen_text=text,
            model_obj=model,
            vocoder=vocoder,
            mel_spec_type="vocos",
            speed=speed,
            device=_F5_DEVICE,
            **tuning,
        )
    finally:
        infer_utils.torchaudio.load = original_torchaudio_load

    if hasattr(wav_tensor, "detach"):
        wav_array = wav_tensor.squeeze().detach().cpu().numpy()
    else:
        wav_array = np.asarray(wav_tensor).squeeze()

    wav_array = wav_array.astype(np.float32)
    wav_array = _validate_f5_output(wav_array)
    return _resample_linear(wav_array, int(sample_rate), 24000)


def generate_fish_audio(text: str, speed: float = 0.85, vocal_tone: str = "Soft Spoken") -> np.ndarray:
    """
    Arabic ASMR via the Fish Audio API (model from FISH_AUDIO_MODEL,
    default s2.1-pro-free) using S2 inline direction tags for delivery
    control. Requests WAV (soundfile-decodable; no ffmpeg needed) and
    returns 24kHz float32 mono samples, matching generate_f5_audio.
    """
    if not text or not text.strip():
        raise ValueError("text must not be empty")
    try:
        from fish_arabic_asmr_test import FishAudioTTS
    except ImportError as exc:
        raise ImportError(
            "Fish Audio backend requires fish_arabic_asmr_test.py and 'requests'."
        ) from exc

    # Defensive: convert any residual app pause markers to S2 pause tags
    # (synthesize() normally splits them out before this point).
    text = re.sub(
        r"<<SILENCE(\d+)MS>>",
        lambda m: "[break]" if int(m.group(1)) < 1500 else "[long-break]",
        text,
    )

    # Inline delivery control: whisper vs soft-spoken opening cue.
    prefix = "[whispering][soft tone] " if vocal_tone == "Whispering" else "[soft tone] "

    client = FishAudioTTS()  # reads FISH_AUDIO_API_KEY / FISH_AUDIO_MODEL env
    voice_id = os.getenv("FISH_AUDIO_VOICE_ID", "").strip() or None
    result = client.synthesize(
        text=prefix + text.strip(),
        voice_id=voice_id,
        fmt="wav",
        speed=max(0.5, min(2.0, speed)),
        sample_rate=44100,
        name="app-arabic",
    )

    data, sample_rate = sf.read(io.BytesIO(result.audio_bytes), dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    data = _validate_f5_output(data.astype(np.float32))
    return _resample_linear(data, int(sample_rate), 24000)

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
                        if _ARABIC_TTS_BACKEND == "fish":
                            all_audio.append(generate_fish_audio(text_part, speed=speed, vocal_tone=vocal_tone))
                        else:
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