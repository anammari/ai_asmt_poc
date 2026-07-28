import unittest
from unittest.mock import patch

import content_fetcher


class TestContentFetcherSmoke(unittest.TestCase):
    def test_fetch_content_routes_movie_prefix(self):
        with patch.object(content_fetcher, "fetch_movie_info", return_value="movie context") as mock_fetch:
            result = content_fetcher.fetch_content("movie: interstellar")

        self.assertEqual(result, "movie context")
        mock_fetch.assert_called_once_with("interstellar")

    def test_fetch_content_routes_wiki_prefix(self):
        with patch.object(content_fetcher, "fetch_wikipedia", return_value="wiki context") as mock_fetch:
            result = content_fetcher.fetch_content("wiki: aurora borealis")

        self.assertEqual(result, "wiki context")
        mock_fetch.assert_called_once_with("aurora borealis")

    def test_fetch_content_routes_wikipedia_url(self):
        with patch.object(content_fetcher, "fetch_wikipedia", return_value="url wiki context") as mock_fetch:
            result = content_fetcher.fetch_content("https://en.wikipedia.org/wiki/Deep_ocean")

        self.assertEqual(result, "url wiki context")
        mock_fetch.assert_called_once_with("Deep ocean")

    def test_fetch_content_returns_empty_for_blank_source(self):
        self.assertEqual(content_fetcher.fetch_content("   "), "")

    def test_fetch_webpage_returns_fallback_on_request_error(self):
        with patch("content_fetcher.requests.get", side_effect=Exception("network error")):
            result = content_fetcher.fetch_webpage("https://example.com")

        self.assertIn("Background contextual fetch failed", result)


if __name__ == "__main__":
    unittest.main()
