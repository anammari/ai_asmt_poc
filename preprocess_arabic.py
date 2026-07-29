# preprocess_arabic.py
"""
Arabic text preprocessing for the SILMA/F5-TTS ASMR pipeline.

F5-TTS fails to infer stable character-to-phoneme alignments for Arabic
unless the input text is clean and (ideally) fully diacritized. Without
explicit *tashkeel* the model hallucinates words and mispronounces MSA.

This module provides:
  - normalize_arabic():        orthographic normalization (uniform Alifs,
                               Ta Marbuta variants, tatweel/symbol cleanup)
  - diacritize():              automatic diacritization via a pluggable
                               backend ("mishkal" default, "camel", "none")
  - is_arabic():               Arabic script detection
  - diacritization_coverage(): share of Arabic letters carrying tashkeel

Backend selection order: explicit `backend` arg > AR_DIACRITIZER env > "mishkal".
All public functions preserve the app's <<SILENCE<n>MS>> pause markers intact.

Optional dependencies (soft imports, never required by the core app):
  - mishkal:      uv pip install mishkal
  - camel_tools:  uv pip install camel-tools  (+ `camel-tools download` data)
"""
import argparse
import os
import re
import sys

# ---------------------------------------------------------------------------
# Character classes
# ---------------------------------------------------------------------------
# Core Arabic letters (Hamza 0621 .. Yeh 064A), incl. Alef Madda 0622,
# hamzated alefs 0623/0625, Ta Marbuta 0629, Alef Maqsura 0649.
def _is_arabic_letter(ch: str) -> bool:
    return "ء" <= ch <= "ي"

# Tashkeel: Fathatan..Sukun (064B-0652) + superscript Alef (0670).
def _is_diacritic(ch: str) -> bool:
    return ("ً" <= ch <= "ْ") or ch == "ٰ"

_TATWEEL = "ـ"          # U+0640 kashida
# Quranic annotation marks (U+06D6..U+06ED) - not needed for TTS.
_QURANIC_MARKS_RE = re.compile(r"[ۖ-ۭ]")

# Orthographic normalization map.
# Phonemically-meaningful characters (آ madda, ة, ى) are kept intact:
# the diacritizer needs them to assign correct vowels.
_CHARS_MAP = {
    "أ": "ا",  # Alef with hamza above -> bare Alef (uniform Alifs)
    "إ": "ا",  # Alef with hamza below -> bare Alef
    "ٱ": "ا",  # Alef wasla -> bare Alef
    "ۀ": "ة",  # Ae (U+06C0) -> Ta Marbuta
    "ہ": "ة",  # Urdu goal heh -> Ta Marbuta
    "ی": "ي",  # Farsi Yeh -> Arabic Yeh
    "ک": "ك",  # Farsi Kaf -> Arabic Kaf
    "‍": "",   # ZWJ removed
    "‌": "",   # ZWNJ removed
    "‎": "",   # LRM removed
    "‏": "",   # RLM removed
    "﻿": "",   # BOM/ZWNBSP removed
}
_CHARS_MAP_TABLE = str.maketrans(_CHARS_MAP)

# Punctuation allowed to survive normalization (in addition to Arabic
# letters, tashkeel, ASCII alphanumerics and whitespace).
_KEEP_PUNCT = set(" .,!?;:،؛؟…()[]«»\"'’‘-–—_\n\t")

# App pause markers (see app.sanitize_text) that must pass through untouched.
_MARKER_RE = re.compile(r"(<<SILENCE\d+MS>>)")

# Sentence-delimiting punctuation. Some backends (mishkal) drop or mangle
# these, which destroys F5's sentence batching/prosody - so they are carved
# out around the diacritizer call and re-attached verbatim.
_SENT_DELIM_RE = re.compile(r"([.!؟?…\n]+)")


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------
def is_arabic(text: str) -> bool:
    """True if the text contains at least one Arabic letter."""
    return any(_is_arabic_letter(ch) for ch in text)


def strip_diacritics(text: str) -> str:
    """Removes all tashkeel marks (keeps letters, punctuation, markers)."""
    return "".join(ch for ch in text if not _is_diacritic(ch))


# Fish Audio S2 inline direction tags, e.g. [soft], [whispering], [laughing].
_FISH_TAG_RE = re.compile(r"\[[^\[\]]*\]")


def strip_fish_tags(text: str) -> str:
    """
    Removes Fish Audio S2 inline tags ([soft], [whispering], [pause], ...)
    from a transcript, leaving only the spoken words. Used when adapting a
    Fish Audio transcript into an exact F5 reference transcript.
    <<SILENCE<n>MS>> markers are NOT Fish tags and are preserved.
    """
    parts = _MARKER_RE.split(text)
    out = [
        part if i % 2 == 1 else _FISH_TAG_RE.sub("", part)
        for i, part in enumerate(parts)
    ]
    return re.sub(r"[ \t]+", " ", "".join(out)).strip()


def diacritization_coverage(text: str) -> float:
    """Share (0..1) of Arabic letters immediately followed by a tashkeel mark."""
    letters = 0
    marked = 0
    for i, ch in enumerate(text):
        if _is_arabic_letter(ch):
            letters += 1
            if i + 1 < len(text) and _is_diacritic(text[i + 1]):
                marked += 1
    return (marked / letters) if letters else 0.0


def normalize_arabic(text: str) -> str:
    """
    Normalizes Arabic orthography for TTS consumption:
      - unifies Alif variants (أ إ ٱ -> ا) and Ta Marbuta variants (-> ة)
      - maps Farsi Yeh/Kaf to Arabic forms
      - removes tatweel, Quranic annotation marks, bidi/zero-width controls
      - strips unexpected non-Arabic symbols (emoji, stray control chars)
      - collapses repeated whitespace
    <<SILENCE<n>MS>> markers are preserved exactly.
    """
    if not text:
        return text

    parts = _MARKER_RE.split(text)
    out_parts: list[str] = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            out_parts.append(part)  # pause marker: keep verbatim
            continue
        segment = part.translate(_CHARS_MAP_TABLE)
        segment = segment.replace(_TATWEEL, "")
        segment = _QURANIC_MARKS_RE.sub("", segment)
        cleaned_chars: list[str] = []
        for ch in segment:
            if _is_arabic_letter(ch) or _is_diacritic(ch):
                cleaned_chars.append(ch)
            elif ch.isascii() and (ch.isalnum() or ch in _KEEP_PUNCT):
                cleaned_chars.append(ch)
            elif ch in _KEEP_PUNCT:
                cleaned_chars.append(ch)
            # everything else is dropped silently
        segment = "".join(cleaned_chars)
        segment = re.sub(r"[ \t]+", " ", segment)
        segment = re.sub(r"\n{3,}", "\n\n", segment)
        out_parts.append(segment)

    return "".join(out_parts).strip()


# ---------------------------------------------------------------------------
# Diacritization backends
# ---------------------------------------------------------------------------
_MISHKAL_VOCALIZER = None
_CAMEL_DISAMBIGUATOR = None


def _diacritize_mishkal(text: str) -> str:
    global _MISHKAL_VOCALIZER
    try:
        from mishkal.tashkeel import TashkeelClass
    except ImportError as exc:
        raise ImportError(
            "Diacritizer 'mishkal' is not installed. "
            "Run `uv pip install mishkal`, or set AR_DIACRITIZER=camel|none."
        ) from exc
    if _MISHKAL_VOCALIZER is None:
        _MISHKAL_VOCALIZER = TashkeelClass()
    return _MISHKAL_VOCALIZER.tashkeel(text)


def _diacritize_camel(text: str) -> str:
    global _CAMEL_DISAMBIGUATOR
    try:
        from camel_tools.disambig.mle import MLEDisambiguator
    except ImportError as exc:
        raise ImportError(
            "Diacritizer 'camel' is not installed. "
            "Run `uv pip install camel-tools` and `camel-tools download light`, "
            "or set AR_DIACRITIZER=mishkal|none."
        ) from exc
    if _CAMEL_DISAMBIGUATOR is None:
        _CAMEL_DISAMBIGUATOR = MLEDisambiguator.pretrained()

    # Disambiguate sentence-by-sentence; keep tokens the analyzer can't
    # resolve (numbers, punctuation, foreign words) verbatim.
    diacritized_lines: list[str] = []
    for line in text.split("\n"):
        words = line.split()
        if not words or not any(is_arabic(w) for w in words):
            diacritized_lines.append(line)
            continue
        try:
            disambiguated = _CAMEL_DISAMBIGUATOR.disambiguate(words)
            resolved = [
                d.analyses[0].analysis.get("diac", d.word) if d.analyses else d.word
                for d in disambiguated
            ]
            diacritized_lines.append(" ".join(resolved))
        except Exception:
            diacritized_lines.append(line)
    return "\n".join(diacritized_lines)


_BACKENDS = {
    "mishkal": _diacritize_mishkal,
    "camel": _diacritize_camel,
    "camel_tools": _diacritize_camel,
}

# If the text already carries this much tashkeel, it is treated as an exact
# (e.g. hand-verified) transcript and the backend is NOT run - re-diacritizing
# correct vowels with a best-guess algorithm would corrupt them.
ALREADY_DIACRITIZED_THRESHOLD = 0.5


def resolve_diacritizer_backend(backend: str | None = None) -> str:
    """Effective backend name: arg > AR_DIACRITIZER env > 'mishkal'."""
    resolved = (backend or os.getenv("AR_DIACRITIZER", "mishkal") or "mishkal").strip().lower()
    return resolved or "mishkal"


def diacritize(text: str, backend: str | None = None, force: bool = False) -> str:
    """
    Fully diacritizes Arabic text using the selected backend.
    Returns the input unchanged when:
      - backend is "none"/"off" (explicit opt-out), or
      - the text contains no Arabic letters, or
      - the text is already diacritized (coverage >=
        ALREADY_DIACRITIZED_THRESHOLD) unless force=True - protecting
        exact/hand-verified transcripts from re-diacritization drift.
    Raises ImportError (with install guidance) if the backend is missing.
    <<SILENCE<n>MS>> markers pass through untouched.
    """
    if not text:
        return text

    resolved = resolve_diacritizer_backend(backend)
    if resolved in ("none", "off", "disabled"):
        return text
    if not is_arabic(text):
        return text
    if not force and diacritization_coverage(text) >= ALREADY_DIACRITIZED_THRESHOLD:
        return text

    diacritizer = _BACKENDS.get(resolved)
    if diacritizer is None:
        raise ValueError(
            f"Unknown diacritizer backend '{resolved}'. "
            "Supported: mishkal, camel, none."
        )

    parts = _MARKER_RE.split(text)
    out_parts: list[str] = []
    for i, part in enumerate(parts):
        if i % 2 == 1 or not part.strip():
            out_parts.append(part)
            continue
        # Diacritize sentence bodies only; keep delimiters (periods, newlines)
        # verbatim so F5 keeps its sentence batching and pause prosody.
        subparts = _SENT_DELIM_RE.split(part)
        out_parts.append(
            "".join(
                sub if (j % 2 == 1 or not sub.strip()) else diacritizer(sub)
                for j, sub in enumerate(subparts)
            )
        )
    result = "".join(out_parts)
    # mishkal tends to insert a space before punctuation; tighten it back.
    result = re.sub(r"\s+([،؛؟.!:…])", r"\1", result)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Normalize and diacritize Arabic text for SILMA/F5-TTS."
    )
    parser.add_argument("input", nargs="?", help="Input text file (default: stdin).")
    parser.add_argument("-o", "--output", help="Output file (default: stdout).")
    parser.add_argument(
        "--backend",
        default=None,
        help="Diacritizer backend: mishkal (default) | camel | none.",
    )
    parser.add_argument(
        "--no-diacritize",
        action="store_true",
        help="Only normalize; skip diacritization.",
    )
    args = parser.parse_args(argv)

    raw = open(args.input, encoding="utf-8").read() if args.input else sys.stdin.read()

    normalized = normalize_arabic(raw)
    if args.no_diacritize:
        result = normalized
    else:
        result = diacritize(normalized, backend=args.backend)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(result + "\n")

    sys.stdout.write(result + "\n")
    print(
        f"[stats] diacritization coverage: {diacritization_coverage(result):.1%} "
        f"(backend={resolve_diacritizer_backend(args.backend)})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
