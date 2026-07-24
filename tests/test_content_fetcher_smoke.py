import unittest
from unittest.mock import patch

import content_fetcher


class TestContentFetcherSmoke(unittest.TestCase):
    def test_context_falls_back_when_loader_raises(self):
        class FailingLoader:
            def __init__(self, *args, **kwargs):
                pass

            def load(self):
                raise ValueError("JSON decode failed")

        with patch.object(content_fetcher, "WikipediaLoader", FailingLoader):
            result = content_fetcher.fetch_content("context", "aurora borealis")

        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)
        self.assertIn("aurora borealis", result.lower())

    def test_context_falls_back_when_no_docs(self):
        class EmptyLoader:
            def __init__(self, *args, **kwargs):
                pass

            def load(self):
                return []

        with patch.object(content_fetcher, "WikipediaLoader", EmptyLoader):
            result = content_fetcher.fetch_content("context", "deep ocean")

        self.assertTrue(result)
        self.assertIn("deep ocean", result.lower())


if __name__ == "__main__":
    unittest.main()
