import io
import unittest
import wave

import numpy as np

import stt
import transformers.pipelines.automatic_speech_recognition as asr_pipeline


class TestSttSmoke(unittest.TestCase):
    def _make_wav_bytes(self, sample_rate: int = 16000, duration_s: float = 0.1) -> bytes:
        t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
        tone = 0.2 * np.sin(2 * np.pi * 220 * t)
        pcm16 = (tone * 32767).astype(np.int16)

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm16.tobytes())
        return buf.getvalue()

    def test_transcribe_decodes_bytes_without_ffmpeg(self):
        captured = {}

        def fake_pipe(inp, **kwargs):
            captured["input"] = inp
            captured["kwargs"] = kwargs
            return {"text": "hello whisper"}

        wav_bytes = self._make_wav_bytes()
        text = stt.transcribe(fake_pipe, wav_bytes)

        self.assertEqual(text, "hello whisper")
        self.assertIsInstance(captured["input"], dict)
        self.assertIn("array", captured["input"])
        self.assertIn("sampling_rate", captured["input"])
        self.assertEqual(captured["input"]["sampling_rate"], 16000)
        self.assertEqual(captured["kwargs"]["batch_size"], 4)

    def test_transcribe_passes_through_non_bytes_input(self):
        captured = {}

        def fake_pipe(inp, **kwargs):
            captured["input"] = inp
            return {"text": "ok"}

        sentinel = {"array": [0.0], "sampling_rate": 16000}
        text = stt.transcribe(fake_pipe, sentinel)

        self.assertEqual(text, "ok")
        self.assertIs(captured["input"], sentinel)

    def test_torchcodec_guard_is_forced_off(self):
        stt._disable_torchcodec_for_asr()
        self.assertFalse(asr_pipeline.is_torchcodec_available())


if __name__ == "__main__":
    unittest.main()
