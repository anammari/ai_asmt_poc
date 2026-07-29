# tests/test_fish_tts_smoke.py
import io
import os
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

import tts
from fish_arabic_asmr_test import FishAudioError, FishAudioTTS


class _FakeResponse:
    def __init__(self, status_code=200, chunks=None, text=""):
        self.status_code = status_code
        self._chunks = chunks if chunks is not None else [b"audio-bytes"]
        self.text = text

    def iter_content(self, chunk_size=8192):
        return iter(self._chunks)

    def close(self):
        pass


def _wav_bytes(sr=44100, seconds=0.1) -> bytes:
    data = np.full(int(sr * seconds), 0.05, dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="WAV")
    return buf.getvalue()


class TestFishAudioTTSClient(unittest.TestCase):
    def test_missing_api_key_raises(self):
        client = FishAudioTTS(api_key="")
        with patch.dict(os.environ, {}, clear=True):
            client.api_key = ""
            with self.assertRaises(FishAudioError):
                client.synthesize("مرحبا")

    def test_payload_headers_and_metrics(self):
        captured = {}

        def fake_post(url, json, headers, stream, timeout):
            captured.update(url=url, json=json, headers=headers, stream=stream)
            return _FakeResponse(200, chunks=[b"chunk1", b"chunk2"])

        with patch("fish_arabic_asmr_test.requests.post", side_effect=fake_post):
            result = FishAudioTTS(api_key="k", model="s2.1-pro-free").synthesize(
                "[whispering] مرحبا بالعالم",
                voice_id="voice-123",
                fmt="mp3",
                speed=0.9,
                name="case1",
            )

        self.assertEqual(captured["headers"]["Authorization"], "Bearer k")
        self.assertEqual(captured["headers"]["model"], "s2.1-pro-free")
        self.assertEqual(captured["json"]["text"], "[whispering] مرحبا بالعالم")
        self.assertEqual(captured["json"]["reference_id"], "voice-123")
        self.assertEqual(captured["json"]["format"], "mp3")
        self.assertEqual(captured["json"]["prosody"], {"speed": 0.9})
        self.assertTrue(captured["stream"])
        self.assertEqual(result.audio_bytes, b"chunk1chunk2")
        self.assertGreaterEqual(result.ttfb_s, 0.0)
        self.assertFalse(result.fallback_used)
        self.assertEqual(result.fmt, "mp3")

    def test_voice_fallback_on_rejected_reference(self):
        calls = []

        def fake_post(url, json, headers, stream, timeout):
            calls.append(dict(json))
            if len(calls) == 1:
                return _FakeResponse(404, text="voice not found")
            return _FakeResponse(200)

        with patch("fish_arabic_asmr_test.requests.post", side_effect=fake_post):
            result = FishAudioTTS(api_key="k").synthesize("مرحبا", voice_id="bad-voice")

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].get("reference_id"), "bad-voice")
        self.assertNotIn("reference_id", calls[1])
        self.assertTrue(result.fallback_used)
        self.assertIsNone(result.voice_id)

    def test_retries_on_429_then_success(self):
        responses = [_FakeResponse(429, text="rate limited"),
                     _FakeResponse(429, text="rate limited"),
                     _FakeResponse(200)]
        calls = []

        def fake_post(url, json, headers, stream, timeout):
            calls.append(1)
            return responses[len(calls) - 1]

        with patch("fish_arabic_asmr_test.requests.post", side_effect=fake_post), \
             patch("fish_arabic_asmr_test.time.sleep") as mock_sleep:
            result = FishAudioTTS(api_key="k").synthesize("مرحبا")

        self.assertEqual(len(calls), 3)
        self.assertEqual([c.args[0] for c in mock_sleep.call_args_list], [1, 2])
        self.assertEqual(result.audio_bytes, b"audio-bytes")

    def test_unrecoverable_4xx_raises_without_retry(self):
        calls = []

        def fake_post(url, json, headers, stream, timeout):
            calls.append(1)
            return _FakeResponse(401, text="invalid api key")

        with patch("fish_arabic_asmr_test.requests.post", side_effect=fake_post):
            with self.assertRaises(FishAudioError) as ctx:
                FishAudioTTS(api_key="bad").synthesize("مرحبا")

        self.assertEqual(len(calls), 1)
        self.assertIn("401", str(ctx.exception))

    def test_network_error_exhausts_retries(self):
        import requests as real_requests

        with patch("fish_arabic_asmr_test.requests.post",
                   side_effect=real_requests.ConnectionError("down")), \
             patch("fish_arabic_asmr_test.time.sleep"):
            with self.assertRaises(FishAudioError) as ctx:
                FishAudioTTS(api_key="k", max_retries=2).synthesize("مرحبا")
        self.assertIn("network error", str(ctx.exception))


class TestFishBackendInApp(unittest.TestCase):
    def test_synthesize_routes_arabic_voice_to_fish_when_enabled(self):
        with patch.object(tts, "_ARABIC_TTS_BACKEND", "fish"), \
             patch("tts.generate_fish_audio",
                   return_value=np.zeros(24000, dtype=np.float32)) as mock_fish, \
             patch("tts.generate_f5_audio") as mock_f5:
            out = tts.synthesize(pipeline=None, text="مرحباً بكم", voices=["ar_ahmad"])

        mock_fish.assert_called_once()
        mock_f5.assert_not_called()
        self.assertTrue(out.startswith(b"RIFF"))

    def test_synthesize_defaults_to_silma_backend(self):
        self.assertEqual(tts._ARABIC_TTS_BACKEND, "silma")

    def test_generate_fish_audio_decodes_wav_and_rewrites_markers(self):
        captured = {}

        def fake_post(url, json, headers, stream, timeout):
            captured["json"] = json
            return _FakeResponse(200, chunks=[_wav_bytes(sr=44100, seconds=0.1)])

        env = {"FISH_AUDIO_API_KEY": "k", "FISH_AUDIO_MODEL": "s2.1-pro-free",
               "FISH_AUDIO_VOICE_ID": ""}
        with patch.dict(os.environ, env, clear=False), \
             patch("fish_arabic_asmr_test.requests.post", side_effect=fake_post):
            wav = tts.generate_fish_audio(
                "مرحباً بك <<SILENCE2000MS>> استمع بهدوء",
                speed=0.85,
                vocal_tone="Whispering",
            )

        # Marker conversion + whisper delivery prefix + WAV sample rate request
        text_sent = captured["json"]["text"]
        self.assertIn("[long-break]", text_sent)
        self.assertNotIn("<<SILENCE", text_sent)
        self.assertTrue(text_sent.startswith("[whispering][soft tone] "))
        self.assertEqual(captured["json"]["format"], "wav")
        self.assertEqual(captured["json"]["sample_rate"], 44100)

        # 0.1s @ 44.1k -> 0.1s @ 24k float32 mono
        self.assertIsInstance(wav, np.ndarray)
        self.assertEqual(wav.dtype, np.float32)
        self.assertAlmostEqual(len(wav) / 24000, 0.1, places=2)

    def test_generate_fish_audio_soft_spoken_prefix(self):
        captured = {}

        def fake_post(url, json, headers, stream, timeout):
            captured["json"] = json
            return _FakeResponse(200, chunks=[_wav_bytes(sr=44100, seconds=0.05)])

        with patch.dict(os.environ, {"FISH_AUDIO_API_KEY": "k"}, clear=False), \
             patch("fish_arabic_asmr_test.requests.post", side_effect=fake_post):
            tts.generate_fish_audio("مرحبا", vocal_tone="Soft Spoken")

        self.assertTrue(captured["json"]["text"].startswith("[soft tone] "))


if __name__ == "__main__":
    unittest.main()
