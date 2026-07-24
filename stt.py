# stt.py
import io
import soundfile as sf
import numpy as np

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
    # Tiny model keeps local processing fast
    return pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

def transcribe(pipeline, audio_bytes: bytes) -> str:
    """
    Transcribes audio bytes safely without requiring an ffmpeg binary installation.
    """
    _disable_torchcodec_for_asr()
    
    # Decode audio purely in python memory via soundfile to bypass Transformers' ffmpeg path
    with io.BytesIO(audio_bytes) as buf:
        data, samplerate = sf.read(buf)
        
    # Convert stereo to mono if needed (Transformers expects 1D arrays)
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
        
    inputs = {"array": data, "sampling_rate": samplerate}
    outputs = pipeline(inputs)
    
    return outputs.get("text", "").strip()