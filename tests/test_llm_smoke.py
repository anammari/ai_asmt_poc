import unittest
from unittest.mock import patch
import os

import llm


class _FakeResponse:
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text

    @property
    def ok(self):
        return 200 <= self.status_code < 300

    def json(self):
        if isinstance(self._json_data, Exception):
            raise self._json_data
        return self._json_data


class TestLlmSmoke(unittest.TestCase):
    def test_get_gemini_candidate_models_deduplicates(self):
        with patch.dict(os.environ, {"GEMINI_FALLBACK_MODELS": "gemini-1.5-flash, gemini-2.0-flash, gemini-1.5-flash"}, clear=False):
            models = llm._get_gemini_candidate_models("gemini-2.5-flash-lite")
        self.assertEqual(models[0], "gemini-2.5-flash-lite")
        self.assertEqual(len(models), len(set(models)))

    def test_normalize_model_name_strips_models_prefix(self):
        self.assertEqual(llm._normalize_model_name("models/gemini-2.5-flash"), "gemini-2.5-flash")

    def test_rewrite_script_returns_local_fallback_for_unknown_provider(self):
        out = llm.rewrite_script(
            context_text="",
            user_prompt="ocean winds",
            provider="unknown-provider",
        )
        self.assertIn("ocean winds", out)
        self.assertIn("[pause:2s]", out)

    def test_call_ollama_uses_chat_endpoint_when_available(self):
        def fake_post(url, json, timeout):
            self.assertEqual(timeout, 90)
            self.assertTrue(url.endswith("/api/chat"))
            self.assertIn("messages", json)
            return _FakeResponse(200, {"message": {"content": "soft script [pause]"}})

        with patch("llm.requests.post", side_effect=fake_post):
            out = llm._call_ollama("sys", "user")

        self.assertIn("soft script", out)

    def test_call_ollama_falls_back_to_v1_chat_completions(self):
        calls = []

        def fake_post(url, json, timeout):
            calls.append(url)
            if url.endswith("/api/chat"):
                return _FakeResponse(404, {"error": "not found"})
            if url.endswith("/v1/chat/completions"):
                return _FakeResponse(200, {"choices": [{"message": {"content": "chat script [pause]"}}]})
            return _FakeResponse(404, {"error": "not found"})

        with patch("llm.requests.post", side_effect=fake_post):
            out = llm._call_ollama("sys", "aurora")

        self.assertIn("chat script", out)
        self.assertTrue(any(url.endswith("/api/chat") for url in calls))
        self.assertTrue(any(url.endswith("/v1/chat/completions") for url in calls))

    def test_call_gemini_returns_text(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "abc123", "GEMINI_MODEL": "gemini-1.5-flash"}, clear=False):
            with patch("llm.requests.post", return_value=_FakeResponse(200, {"candidates": [{"content": {"parts": [{"text": "hello"}]}}]})):
                out = llm._call_gemini("sys", "user")
        self.assertEqual(out, "hello")

    def test_call_gemini_uses_exponential_backoff_for_503(self):
        responses = [
            _FakeResponse(503, {"error": "unavailable"}, text="temporary"),
            _FakeResponse(503, {"error": "unavailable"}, text="temporary"),
            _FakeResponse(200, {"candidates": [{"content": {"parts": [{"text": "hello"}]}}]}),
        ]
        with patch.dict(os.environ, {"GEMINI_API_KEY": "abc123", "GEMINI_MODEL": "gemini-1.5-flash"}, clear=False):
            with patch("llm.requests.post", side_effect=responses), patch("llm.time.sleep") as mock_sleep:
                out = llm._call_gemini("sys", "user")
        self.assertEqual(out, "hello")
        self.assertEqual([c.args[0] for c in mock_sleep.call_args_list], [1, 2])

    def test_call_gemini_falls_back_to_stable_model_after_primary_503(self):
        calls = []

        def fake_post(url, json, headers, timeout):
            calls.append(url)
            if "gemini-2.5-flash-lite" in url:
                return _FakeResponse(503, {"error": "unavailable"}, text="temporary")
            return _FakeResponse(200, {"candidates": [{"content": {"parts": [{"text": "stable model response"}]}}]})

        with patch.dict(
            os.environ,
            {
                "GEMINI_API_KEY": "abc123",
                "GEMINI_MODEL": "gemini-2.5-flash-lite",
                "GEMINI_FALLBACK_MODELS": "gemini-1.5-flash",
            },
            clear=False,
        ):
            with patch("llm.requests.post", side_effect=fake_post), patch("llm.time.sleep"):
                out = llm._call_gemini("sys", "user")
        self.assertEqual(out, "stable model response")
        self.assertTrue(any("gemini-2.5-flash-lite" in url for url in calls))
        self.assertTrue(any("gemini-1.5-flash" in url for url in calls))

    def test_call_gemini_skips_bad_fallback_model_and_uses_next(self):
        calls = []

        def fake_post(url, json, headers, timeout):
            calls.append(url)
            if "gemini-2.5-flash-lite" in url:
                return _FakeResponse(503, {"error": "unavailable"}, text="temporary")
            if "gemini-1.5-flash" in url:
                return _FakeResponse(404, {"error": "not found"}, text="not found")
            if "gemini-2.0-flash" in url:
                return _FakeResponse(200, {"candidates": [{"content": {"parts": [{"text": "second fallback ok"}]}}]})
            return _FakeResponse(503, {"error": "unavailable"}, text="temporary")

        with patch.dict(
            os.environ,
            {
                "GEMINI_API_KEY": "abc123",
                "GEMINI_MODEL": "gemini-2.5-flash-lite",
                "GEMINI_FALLBACK_MODELS": "gemini-1.5-flash,gemini-2.0-flash",
            },
            clear=False,
        ):
            with patch("llm.requests.post", side_effect=fake_post), patch("llm.time.sleep"):
                out = llm._call_gemini("sys", "user")
        self.assertEqual(out, "second fallback ok")
        self.assertTrue(any("gemini-1.5-flash" in url for url in calls))
        self.assertTrue(any("gemini-2.0-flash" in url for url in calls))

    def test_call_gemini_redacts_key_in_errors(self):
        bad_key = "secret-key-xyz"
        text = f"boom {bad_key}"
        with patch.dict(os.environ, {"GEMINI_API_KEY": bad_key, "GEMINI_MODEL": "gemini-1.5-flash"}, clear=False):
            with patch("llm.requests.post", return_value=_FakeResponse(403, {"error": "forbidden"}, text=text)):
                with self.assertRaises(RuntimeError) as ctx:
                    llm._call_gemini("sys", "user")
        self.assertIn("***REDACTED***", str(ctx.exception))
        self.assertNotIn(bad_key, str(ctx.exception))

    def test_validate_gemini_api_key_reports_placeholder(self):
        result = llm.validate_gemini_api_key(api_key="your_gemini_api_key_here")
        self.assertFalse(result["ok"])
        self.assertTrue(result["needs_new_key"])
        self.assertEqual(result["reason"], "missing_or_placeholder")

    def test_validate_gemini_api_key_reports_invalid_key(self):
        with patch("llm.requests.post", return_value=_FakeResponse(403, {"error": "forbidden"}, text="forbidden")):
            result = llm.validate_gemini_api_key(api_key="abc123", model="gemini-1.5-flash")
        self.assertFalse(result["ok"])
        self.assertTrue(result["needs_new_key"])
        self.assertEqual(result["reason"], "invalid_key_or_permission")

    def test_validate_gemini_api_key_reports_service_unavailable(self):
        with patch("llm.requests.post", return_value=_FakeResponse(503, {"error": "unavailable"}, text="unavailable")):
            result = llm.validate_gemini_api_key(api_key="abc123", model="gemini-1.5-flash")
        self.assertFalse(result["ok"])
        self.assertFalse(result["needs_new_key"])
        self.assertEqual(result["reason"], "provider_unavailable")

    def test_validate_gemini_api_key_reports_model_unavailable(self):
        with patch("llm.requests.post", return_value=_FakeResponse(404, {"error": "not found"}, text="not found")):
            result = llm.validate_gemini_api_key(api_key="abc123", model="gemini-1.5-flash")
        self.assertFalse(result["ok"])
        self.assertFalse(result["needs_new_key"])
        self.assertEqual(result["reason"], "model_unavailable")

    def test_rewrite_script_falls_back_to_ollama_when_gemini_503(self):
        with patch("llm._call_gemini", side_effect=RuntimeError("Gemini API call failed with HTTP 503")), patch(
            "llm._call_ollama", return_value="ollama fallback script"
        ):
            out = llm.rewrite_script(
                context_text="rain",
                user_prompt="calm rain",
                provider="gemini",
                language="Arabic (العربية)",
            )
        self.assertEqual(out, "ollama fallback script")

    def test_rewrite_script_falls_back_to_local_when_gemini_and_ollama_fail(self):
        with patch("llm._call_gemini", side_effect=RuntimeError("Gemini API call failed with HTTP 503")), patch(
            "llm._call_ollama", side_effect=RuntimeError("ollama down")
        ):
            out = llm.rewrite_script(
                context_text="rain",
                user_prompt="calm rain",
                provider="gemini",
            )
        self.assertIn("[pause:2s]", out)
        self.assertIn("calm rain", out)

    def test_rewrite_script_raises_on_non_transient_gemini_errors(self):
        with patch("llm._call_gemini", side_effect=RuntimeError("Gemini API call failed with HTTP 403")):
            with self.assertRaises(RuntimeError):
                llm.rewrite_script(context_text="rain", user_prompt="calm rain", provider="gemini")

    def test_arabic_respects_arabic_llm_provider(self):
        with patch("llm._call_gemini", return_value="gemini script") as mock_gemini, \
             patch("llm._call_ollama") as mock_ollama, \
             patch.dict(os.environ, {"ARABIC_LLM_PROVIDER": "gemini"}):
            out = llm.rewrite_script(
                context_text="",
                user_prompt="صوت المطر",
                language="Arabic (العربية)",
            )
        mock_gemini.assert_called_once()
        mock_ollama.assert_not_called()
        self.assertEqual(out, "gemini script")

    def test_english_respects_ollama_provider(self):
        with patch("llm._call_gemini") as mock_gemini, \
             patch("llm._call_ollama", return_value="ollama script") as mock_ollama:
            out = llm.rewrite_script(
                context_text="",
                user_prompt="rain sounds",
                provider="ollama",
                language="English",
            )
        mock_ollama.assert_called_once()
        mock_gemini.assert_not_called()
        self.assertEqual(out, "ollama script")

    def test_arabic_prompt_has_structure_rules(self):
        with patch("llm._call_gemini", return_value="ok") as mock_gemini:
            llm.rewrite_script(
                context_text="",
                user_prompt="test",
                provider="gemini",
                language="Arabic (العربية)",
            )
        args, _ = mock_gemini.call_args
        system_msg = args[0]
        self.assertIn("4-6 paragraphs", system_msg)
        self.assertIn("VARY PAUSE DURATIONS", system_msg)
        self.assertIn("[pause:2s]", system_msg)
        self.assertIn("[pause:3s]", system_msg)
        self.assertIn("[pause:4s]", system_msg)

    def test_call_openai_raises_not_implemented(self):
        with self.assertRaises(NotImplementedError) as ctx:
            llm._call_openai("sys", "user")
        self.assertIn("ARABIC_LLM_PROVIDER", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
