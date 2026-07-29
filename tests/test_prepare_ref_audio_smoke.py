# tests/test_prepare_ref_audio_smoke.py
import os
import tempfile
import unittest

import numpy as np
import soundfile as sf

import prepare_ref_audio as pra


class TestSplitSentences(unittest.TestCase):
    TRANSCRIPT = ("هَيْ... لِمَاذَا تَجْلِسِينَ وَحْدَكِ؟ "
                  "تَعَالِي... دَعِينِي أَجْلِسُ بِجَانِبِكِ. مَاذَا بِكِ؟ "
                  "أَنْتِ هَادِئَةٌ جِدًّا. هَلْ بِسَبَبِ مَا حَدَث؟ "
                  "لاَ تَقْلَقِي... اسْتَرْخِي مَعِي. أَنَا هُنَا.")

    def test_splits_on_sentence_enders(self):
        sentences = pra.split_sentences(self.TRANSCRIPT)
        self.assertEqual(len(sentences), 7)
        self.assertTrue(sentences[0].endswith("؟"))
        self.assertIn("مَاذَا بِكِ؟", sentences[2])

    def test_ellipsis_is_not_a_sentence_boundary(self):
        sentences = pra.split_sentences("تَعَالِي... دَعِينِي أَجْلِسُ بِجَانِبِكِ.")
        self.assertEqual(len(sentences), 1)
        self.assertIn("...", sentences[0])


class TestParseSentenceRange(unittest.TestCase):
    def test_range_and_single(self):
        self.assertEqual(pra._parse_sentence_range("0-2", 7), (0, 2))
        self.assertEqual(pra._parse_sentence_range("3", 7), (3, 3))

    def test_invalid_specs(self):
        for bad in ("x", "2-1", "0-9", "-1", "1-2-3"):
            with self.assertRaises(ValueError, msg=bad):
                pra._parse_sentence_range(bad, 7)


class TestPrepareReferenceText(unittest.TestCase):
    def _write(self, tmp, content):
        path = os.path.join(tmp, "ref.txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
        return path

    def test_exact_path_strips_tags_selects_sentences_no_rewriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = self._write(
                tmp,
                "[soft] هَيْ... لِمَاذَا تَجْلِسِينَ وَحْدَكِ؟ "
                "[whispering] تَعَالِي... دَعِينِي أَجْلِسُ بِجَانِبِكِ. "
                "[emphasis] مَاذَا بِكِ؟ [breathy] أَنَا هُنَا.\n",
            )
            out = pra.prepare_reference_text(
                src, strip_tags=True, sentences="0-2", no_diacritize=True,
            )
            with open(out, encoding="utf-8") as fh:
                result = fh.read().strip()

        self.assertTrue(out.endswith(".exact.txt"))
        # Exact preservation: hamzated alefs and tashkeel untouched
        self.assertIn("هَيْ... لِمَاذَا تَجْلِسِينَ وَحْدَكِ؟", result)
        self.assertIn("تَعَالِي... دَعِينِي أَجْلِسُ بِجَانِبِكِ.", result)
        self.assertIn("مَاذَا بِكِ؟", result)
        self.assertNotIn("أَنَا هُنَا", result)  # sentence 3 excluded
        self.assertNotIn("[", result)

    def test_out_of_range_sentences_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = self._write(tmp, "جملة واحدة فقط.")
            with self.assertRaises(ValueError):
                pra.prepare_reference_text(src, sentences="0-5", no_diacritize=True)


class TestPrepareReferenceAudio(unittest.TestCase):
    def test_produces_spec_compliant_clip(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src.wav")
            t = np.linspace(0, 12, 12 * 44100, endpoint=False)
            stereo = np.stack([0.1 * np.sin(2 * np.pi * 220 * t)] * 2, axis=1)
            sf.write(src, stereo.astype(np.float32), 44100)

            out = os.path.join(tmp, "nested", "ref_24k.wav")
            report = pra.prepare_reference_audio(src, out, start_s=1.0, duration_s=8.0)

            self.assertEqual(report["duration_s"], 8.0)
            self.assertEqual(report["sample_rate"], 24000)
            self.assertEqual(report["channels"], 1)
            data, sr = sf.read(out)
            self.assertEqual(sr, 24000)
            self.assertAlmostEqual(float(np.max(np.abs(data))), 0.95, places=2)

    def test_rejects_too_short_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "short.wav")
            sf.write(src, np.zeros(3 * 24000, dtype=np.float32), 24000)
            with self.assertRaises(ValueError):
                pra.prepare_reference_audio(src, os.path.join(tmp, "out.wav"))

    def test_duration_clamped_to_spec(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "long.wav")
            sf.write(src, np.full(20 * 24000, 0.05, dtype=np.float32), 24000)
            out = os.path.join(tmp, "out.wav")
            report = pra.prepare_reference_audio(src, out, start_s=0.0, duration_s=15.0)
            self.assertEqual(report["duration_s"], 10.0)


if __name__ == "__main__":
    unittest.main()
