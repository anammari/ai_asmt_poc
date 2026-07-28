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

_SILMA_MODEL = None
_SILMA_VOCODER = None
_SILMA_REPO_ID = "silma-ai/silma-tts"
_F5_DEVICE = os.getenv("F5_TTS_DEVICE", "cpu")
_ARABIC_REF_AUDIO = "input/test_ahmad_2.wav"
_ARABIC_REF_TEXT = (
    "خذ نفس عميق، ريح بالك كل شي تمام، خلي عيونك مغمضة واسمعني، "
    "اليوم كان طويل صح؟ هلق صار الوقت ترتاح، انسى كل الهم والقلق"
)


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


def generate_f5_audio(text: str, speed: float = 0.85) -> np.ndarray:
    if not os.path.exists(_ARABIC_REF_AUDIO):
        raise FileNotFoundError(
            f"Arabic reference audio is required for SILMA/F5-TTS at '{_ARABIC_REF_AUDIO}'."
        )

    try:
        from f5_tts.infer.utils_infer import infer_process
        import f5_tts.infer.utils_infer as infer_utils
        import torch
    except ImportError as exc:
        raise ImportError(
            "Arabic SILMA/F5-TTS requires 'f5-tts'. Install dependencies and retry."
        ) from exc

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
            ref_audio=_ARABIC_REF_AUDIO,
            ref_text=_ARABIC_REF_TEXT,
            gen_text=text,
            model_obj=model,
            vocoder=vocoder,
            mel_spec_type="vocos",
            speed=speed,
            device=_F5_DEVICE,
        )
    finally:
        infer_utils.torchaudio.load = original_torchaudio_load

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