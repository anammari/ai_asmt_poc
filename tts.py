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
    if voices is None:
        voices = ["af_bella"]
        
    # Ensure voices is always a list for paragraph alternation
    voices = [voices] if isinstance(voices, str) else voices
    if not voices:
        voices = ["af_bella"]
        
    if not KOKORO_AVAILABLE or pipeline is None:
        raise ImportError("Kokoro TTS is required but missing. Run `uv sync`.")
    
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
                    generator = pipeline(text_part, voice=current_voice, speed=speed, split_pattern=r"\n+")
                    for _, _, audio in generator:
                        all_audio.append(audio)

    if not all_audio:
        raise RuntimeError("TTS failed to produce any audio segments.")

    full_audio = np.concatenate(all_audio)
    
    buffer = io.BytesIO()
    # Write as WAV format
    sf.write(buffer, full_audio, sample_rate, format='WAV')
    buffer.seek(0)
    
    return buffer.read()