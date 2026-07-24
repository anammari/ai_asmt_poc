import unittest
from unittest.mock import patch

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
    def test_generate_uses_api_generate_when_available(self):
        def fake_post(url, json, timeout):
            self.assertTrue(url.endswith("/api/generate"))
            self.assertEqual(json["model"], llm.MODEL_NAME)
            return _FakeResponse(200, {"response": "soft script [pause]"})

        with patch("llm.requests.post", side_effect=fake_post):
            out = llm.generate_asmr_script("ocean winds")

        self.assertIn("soft script", out)

    def test_generate_falls_back_to_chat_endpoint(self):
        calls = []

        def fake_post(url, json, timeout):
            calls.append(url)
            if url.endswith("/api/generate"):
                return _FakeResponse(404, {"error": "not found"})
            if url.endswith("/api/chat"):
                return _FakeResponse(200, {"message": {"content": "chat script [pause]"}})
            return _FakeResponse(404, {"error": "not found"})

        with patch("llm.requests.post", side_effect=fake_post):
            out = llm.generate_asmr_script("aurora")

        self.assertIn("chat script", out)
        self.assertTrue(any(url.endswith("/api/generate") for url in calls))
        self.assertTrue(any(url.endswith("/api/chat") for url in calls))


if __name__ == "__main__":
    unittest.main()
