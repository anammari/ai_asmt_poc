# tests/test_f5_voice_regression.py
"""
Integration regression test for the "no audible voice" F5-TTS bug.

Generates a short clip through the REAL optimized pipeline (exact ref text,
diacritized gen text, current env-tuned params) and asserts the output
contains audible Arabic speech via stt.verify_audible_speech
(whisper transcription + normalized similarity + RMS energy gate).

Runtime: ~2-4 min on CPU (SILMA model + whisper-small, both HF-cached).
Auto-skips when the reference assets or model dependencies are unavailable
(e.g. Docker/CI without input/ assets).
"""
import io
import os
import unittest

import numpy as np
import soundfile as sf

import stt
import tts

_REF_AUDIO = os.getenv("ARABIC_REF_AUDIO", "input/ref_asmr_tn_24k.wav")
_REF_TEXT = os.getenv("ARABIC_REF_TEXT_PATH", "input/test_asmr_tn_3.exact.txt")
_GEN_TEXT = "خذ نفساً عميقاً واستمع إلى صوت المطر الهادئ على النافذة."


def _assets_available() -> bool:
    if not (os.path.exists(_REF_AUDIO) and os.path.exists(_REF_TEXT)):
        return False
    try:
        import f5_tts  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


@unittest.skipUnless(_assets_available(), "F5 ref assets or model deps missing")
class TestF5VoiceRegression(unittest.TestCase):
    def test_optimized_pipeline_produces_audible_arabic_speech(self):
        with open(_REF_TEXT, encoding="utf-8") as fh:
            ref_text = fh.read().strip()

        wav = tts.generate_f5_audio(
            _GEN_TEXT,
            speed=0.85,
            use_preprocessing=True,
            ref_audio=_REF_AUDIO,
            ref_text=ref_text,
        )
        self.assertGreater(len(wav), 24000, "output too short to contain speech")

        buf = io.BytesIO()
        sf.write(buf, wav, 24000, format="WAV")
        verdict = stt.verify_audible_speech(
            buf.getvalue(), expected_text=_GEN_TEXT, language="arabic",
        )
        self.assertTrue(
            verdict["ok"],
            f"F5 output failed the voice check: {verdict['reason']} "
            f"(transcript={verdict['transcript']!r}, rms={verdict['rms']}, "
            f"similarity={verdict['similarity']})",
        )
        self.assertGreaterEqual(verdict["similarity"], 0.3)


if __name__ == "__main__":
    unittest.main()
