# tests/test_preprocess_arabic_smoke.py
import os
import unittest
from unittest.mock import MagicMock, patch

import preprocess_arabic as pa


def _fake_backend(text: str) -> str:
    """Deterministic stand-in diacritizer: tags each word, drops nothing."""
    return " ".join(word + "*" for word in text.split())


class TestNormalizeArabic(unittest.TestCase):
    def test_unifies_alef_variants(self):
        self.assertEqual(pa.normalize_arabic("أحمد إبراهيم ٱلله"), "احمد ابراهيم الله")

    def test_keeps_phonemic_forms(self):
        out = pa.normalize_arabic("القرآن فتاة ليلى")
        self.assertIn("آ", out)  # alef madda preserved
        self.assertIn("ة", out)  # ta marbuta preserved
        self.assertIn("ى", out)  # alef maqsura preserved

    def test_maps_variant_letters(self):
        out = pa.normalize_arabic("ۀ ـہ ی ک")
        self.assertNotIn("ۀ", out)
        self.assertNotIn("ی", out)
        self.assertNotIn("ک", out)

    def test_removes_tatweel_quranic_marks_and_symbols(self):
        out = pa.normalize_arabic("بكـــتُم 🌙 الْقُرْآنۖ هنا")
        self.assertNotIn("ـ", out)
        self.assertNotIn("🌙", out)
        self.assertNotIn("ۖ", out)
        self.assertIn("الْقُرْآن", out)  # core tashkeel kept

    def test_collapses_whitespace(self):
        self.assertEqual(pa.normalize_arabic("مرحبا    بالعالم"), "مرحبا بالعالم")

    def test_preserves_silence_markers(self):
        out = pa.normalize_arabic("أهلا <<SILENCE2000MS>> بالعالم")
        self.assertIn("<<SILENCE2000MS>>", out)
        self.assertIn("اهلا", out)

    def test_empty_and_non_arabic_passthrough(self):
        self.assertEqual(pa.normalize_arabic(""), "")
        self.assertEqual(pa.normalize_arabic("hello world"), "hello world")


class TestDiacritize(unittest.TestCase):
    def test_none_backend_passthrough(self):
        text = "مرحبا بالعالم"
        self.assertEqual(pa.diacritize(text, backend="none"), text)

    def test_non_arabic_passthrough(self):
        self.assertEqual(pa.diacritize("hello world"), "hello world")

    def test_unknown_backend_raises(self):
        with self.assertRaises(ValueError):
            pa.diacritize("مرحبا", backend="bogus-backend")

    def test_markers_and_sentence_delimiters_survive(self):
        with patch.dict(pa._BACKENDS, {"fake": _fake_backend}):
            out = pa.diacritize("الجملة الاولى. الجملة الثانية <<SILENCE500MS>> الثالثة",
                                backend="fake")
        self.assertIn("<<SILENCE500MS>>", out)
        self.assertIn(".", out)  # period delimiter preserved verbatim
        self.assertIn("الجملة* الاولى*", out)  # body words went through backend

    def test_space_before_punctuation_tightened(self):
        def spaced_backend(text: str) -> str:
            return text.replace("،", " ،")

        with patch.dict(pa._BACKENDS, {"spaced": spaced_backend}):
            out = pa.diacritize("مرحبا، بالعالم", backend="spaced")
        self.assertIn("مرحبا،", out)
        self.assertNotIn(" ،", out)

    def test_backend_env_precedence(self):
        with patch.dict(os.environ, {"AR_DIACRITIZER": "camel"}, clear=False):
            self.assertEqual(pa.resolve_diacritizer_backend(), "camel")
        with patch.dict(os.environ, {"AR_DIACRITIZER": "camel"}, clear=False):
            self.assertEqual(pa.resolve_diacritizer_backend("none"), "none")
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(pa.resolve_diacritizer_backend(), "mishkal")

    def test_missing_backend_import_error(self):
        def missing(_text: str) -> str:
            raise ImportError("not installed")

        with patch.dict(pa._BACKENDS, {"missing": missing}):
            with self.assertRaises(ImportError):
                pa.diacritize("مرحبا", backend="missing")

    def test_already_diacritized_text_skips_backend(self):
        # Exact/hand-verified transcripts must pass through untouched:
        # re-diacritizing correct vowels corrupts F5 alignment.
        spy = MagicMock(side_effect=_fake_backend)
        already = "خُذْ نَفْسًا عَمِيقًا وَاِسْتَرْخِ تَمَامًا"
        self.assertGreaterEqual(pa.diacritization_coverage(already),
                                pa.ALREADY_DIACRITIZED_THRESHOLD)
        with patch.dict(pa._BACKENDS, {"spy": spy}):
            out = pa.diacritize(already, backend="spy")
        self.assertEqual(out, already)
        spy.assert_not_called()

    def test_force_rediacritizes_when_requested(self):
        spy = MagicMock(side_effect=_fake_backend)
        already = "خُذْ نَفْسًا عَمِيقًا"
        with patch.dict(pa._BACKENDS, {"spy": spy}):
            pa.diacritize(already, backend="spy", force=True)
        spy.assert_called_once()


class TestStripHelpers(unittest.TestCase):
    def test_strip_diacritics(self):
        self.assertEqual(pa.strip_diacritics("مَرْحَبًا بِالْعَالَمِ"), "مرحبا بالعالم")
        self.assertEqual(pa.strip_diacritics("hello"), "hello")

    def test_strip_fish_tags(self):
        raw = ("[soft] هَيْ... [whispering] تَعَالِي [pause] [laughing] "
               "لاَ تَقْلَقِي <<SILENCE500MS>> أَنَا هُنَا.")
        out = pa.strip_fish_tags(raw)
        self.assertNotIn("[", out)
        self.assertIn("<<SILENCE500MS>>", out)  # app markers are not Fish tags
        self.assertIn("هَيْ...", out)
        self.assertIn("تَعَالِي", out)

    def test_strip_fish_tags_collapses_whitespace(self):
        out = pa.strip_fish_tags("[soft]   مرحبا   بالعالم")
        self.assertEqual(out, "مرحبا بالعالم")


class TestMetrics(unittest.TestCase):
    def test_is_arabic(self):
        self.assertTrue(pa.is_arabic("مرحبا"))
        self.assertFalse(pa.is_arabic("hello"))
        self.assertFalse(pa.is_arabic("123"))

    def test_diacritization_coverage(self):
        self.assertEqual(pa.diacritization_coverage("مرحبا بالعالم"), 0.0)
        cov = pa.diacritization_coverage("مَرْحَبًا بِالْعَالَمِ")
        self.assertGreater(cov, 0.7)
        self.assertEqual(pa.diacritization_coverage("no letters"), 0.0)


class TestMishkalIntegration(unittest.TestCase):
    def setUp(self):
        try:
            import mishkal  # noqa: F401
        except ImportError:
            self.skipTest("mishkal not installed")

    def test_mishkal_raises_coverage_on_raw_text(self):
        raw = "خذ نفسا عميقا واستمع الى هذا الصوت الهادئ"
        out = pa.diacritize(pa.normalize_arabic(raw), backend="mishkal")
        self.assertGreater(pa.diacritization_coverage(out), 0.5)

    def test_mishkal_preserves_sentence_boundaries(self):
        out = pa.diacritize("الجملة الاولى هنا. الجملة الثانية هناك", backend="mishkal")
        self.assertIn(". ", out)


if __name__ == "__main__":
    unittest.main()
