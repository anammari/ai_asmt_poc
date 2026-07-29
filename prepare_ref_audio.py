# prepare_ref_audio.py
"""
Reference-audio preparation tool for the Arabic SILMA/F5-TTS voice.

F5-TTS transfers speaking style (e.g. whispering) strictly via the reference
audio prompt. Best-practice reference specs for stable Arabic ASMR output:

  - Length:    strictly 5-10 seconds (too short -> weak style transfer;
               too long -> alignment degradation / hallucinations)
  - Channels:  mono
  - Rate:      >= 24 kHz WAV
  - Acoustics: dry studio, soft whisper, minimal background noise
  - Text:      the ref_text passed to F5 must be the EXACT, fully
               diacritized transcript of this clip

This tool converts any input WAV into that spec (mono, 24 kHz, trimmed
5-10 s window, peak-normalized) using soundfile + numpy only (no ffmpeg).
Optionally it also normalizes + diacritizes the matching transcript file.

Usage:
  python prepare_ref_audio.py input/test_ahmad_2.wav \
      --start 2.0 --duration 8.0 \
      -o input/ref_arabic_whisper_24k.wav \
      --ref-text-file input/test_ahmad_2.txt
"""
import argparse
import os
import re
import sys

import numpy as np
import soundfile as sf

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from tts import _resample_linear

MIN_WINDOW_S = 5.0
MAX_WINDOW_S = 10.0
TARGET_SR = 24000
PEAK_TARGET = 0.95


def _audio_report(path: str, data: np.ndarray, sr: int, channels: int) -> dict:
    duration = len(data) / float(sr) if sr else 0.0
    peak = float(np.max(np.abs(data))) if len(data) else 0.0
    rms = float(np.sqrt(np.mean(np.square(data)))) if len(data) else 0.0
    return {
        "path": path,
        "duration_s": round(duration, 2),
        "sample_rate": sr,
        "channels": channels,
        "peak": round(peak, 4),
        "rms": round(rms, 4),
    }


def _print_report(title: str, report: dict) -> None:
    print(f"[{title}]")
    for key, value in report.items():
        print(f"  {key}: {value}")


def prepare_reference_audio(
    input_path: str,
    output_path: str,
    start_s: float = 0.0,
    duration_s: float = 8.0,
    target_sr: int = TARGET_SR,
) -> dict:
    """
    Trims/downmixes/resamples/normalizes `input_path` into an F5-ready
    reference clip at `output_path`. Returns the output audio report.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input audio not found: '{input_path}'")

    data, sr = sf.read(input_path, dtype="float32")
    channels = 1 if data.ndim == 1 else data.shape[1]
    src_report = _audio_report(input_path, data, sr, channels)
    _print_report("source", src_report)

    # 1) Mono downmix
    if data.ndim > 1:
        print(f"  -> downmixing {channels} channels to mono (mean)")
        data = np.mean(data, axis=1)

    # 2) Window selection, clamped to the 5-10s F5 spec
    requested = duration_s
    duration_s = min(MAX_WINDOW_S, max(MIN_WINDOW_S, duration_s))
    if requested != duration_s:
        print(f"  -> duration clamped from {requested}s to {duration_s}s (spec: 5-10s)")

    total_s = len(data) / float(sr)
    start_s = max(0.0, min(start_s, total_s))
    available_s = total_s - start_s
    if available_s < MIN_WINDOW_S:
        raise ValueError(
            f"Only {available_s:.1f}s of audio available from start={start_s}s; "
            f"the F5 reference window needs at least {MIN_WINDOW_S}s. "
            "Record a longer whisper clip (aim for 8-10s)."
        )
    window_s = min(duration_s, available_s)
    start_idx = int(start_s * sr)
    end_idx = int((start_s + window_s) * sr)
    data = data[start_idx:end_idx]
    print(f"  -> trimmed window: start={start_s:.1f}s, duration={window_s:.1f}s")

    # 3) Resample to target rate
    if sr != target_sr:
        print(f"  -> resampling {sr} Hz -> {target_sr} Hz")
        data = _resample_linear(data, sr, target_sr)
        sr = target_sr

    # 4) Peak normalization (keep whisper dynamics, just avoid clipping)
    peak = float(np.max(np.abs(data))) if len(data) else 0.0
    if peak > 0:
        data = (data / peak * PEAK_TARGET).astype(np.float32)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    sf.write(output_path, data, sr, subtype="PCM_16")

    out_report = _audio_report(output_path, data, sr, 1)
    _print_report("output", out_report)
    return out_report


def split_sentences(text: str) -> list[str]:
    """
    Splits a transcript into sentences on Arabic/Latin sentence-ending
    punctuation (؟ . ! ?). Ellipses ("...", "…") are treated as intra-sentence
    pauses, not sentence boundaries.
    """
    protected = text.replace("...", "\x00").replace("…", "\x00")
    parts = re.split(r"(?<=[؟.!?])\s+", protected)
    return [p.replace("\x00", "...").strip() for p in parts if p.strip()]


def _parse_sentence_range(spec: str, total: int) -> tuple[int, int]:
    """Parses '0-2' (inclusive) or '3' into a (start, end) sentence range."""
    match = re.fullmatch(r"(\d+)(?:-(\d+))?", spec.strip())
    if not match:
        raise ValueError(f"Invalid --sentences value '{spec}'. Use e.g. '0-2' or '3'.")
    start = int(match.group(1))
    end = int(match.group(2)) if match.group(2) is not None else start
    if start > end or end >= total:
        raise ValueError(
            f"Sentence range {start}-{end} out of bounds for {total} sentences."
        )
    return start, end


def prepare_reference_text(
    text_path: str,
    backend: str | None = None,
    strip_tags: bool = False,
    sentences: str | None = None,
    no_diacritize: bool = False,
) -> str:
    """
    Builds the EXACT reference transcript next to the audio:
      - optionally strips Fish Audio S2 inline tags ([soft], [whispering]...)
      - optionally selects the sentence range matching the audio window
      - normalizes + diacritizes (skipped when --no-diacritize or when the
        text is already diacritized; manual verification is still advised)
    Writes '<stem>.diacritized.txt' (or '<stem>.exact.txt' when diacritization
    is disabled) and returns its path.
    """
    import preprocess_arabic

    with open(text_path, encoding="utf-8") as fh:
        raw_text = fh.read().strip()

    if strip_tags:
        raw_text = preprocess_arabic.strip_fish_tags(raw_text)

    if sentences is not None:
        all_sentences = split_sentences(raw_text)
        start, end = _parse_sentence_range(sentences, len(all_sentences))
        selected = all_sentences[start : end + 1]
        print(f"[ref-text] selected sentences {start}-{end} of {len(all_sentences)}:")
        for i, sentence in enumerate(selected, start=start):
            print(f"  [{i}] {sentence}")
        raw_text = " ".join(selected)

    if no_diacritize:
        # Exact-transcript path: NO normalization, NO diacritization. Even
        # alef unification would alter phonemically-meaningful characters
        # (hamzated alefs) and break F5's character-to-phoneme alignment.
        prepared = re.sub(r"\s+", " ", raw_text).strip()
        coverage = preprocess_arabic.diacritization_coverage(prepared)
        suffix = ".exact"
    else:
        normalized = preprocess_arabic.normalize_arabic(raw_text)
        prepared = preprocess_arabic.diacritize(normalized, backend=backend)
        coverage = preprocess_arabic.diacritization_coverage(prepared)
        suffix = ".diacritized"

    stem, _ = os.path.splitext(text_path)
    out_path = f"{stem}{suffix}.txt"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(prepared + "\n")

    print(f"[ref-text] transcript written to: {out_path}")
    print(f"[ref-text] diacritization coverage: {coverage:.1%}")
    if coverage < 0.5 and no_diacritize:
        print("[ref-text] WARNING: coverage is low and diacritization was disabled; "
              "F5 alignment quality may suffer.")
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare an F5-TTS-ready Arabic whisper reference clip "
        "(mono, 24 kHz, 5-10s window, peak-normalized)."
    )
    parser.add_argument("input_wav", help="Source WAV file (any rate/channels).")
    parser.add_argument(
        "-o", "--output",
        help="Output WAV path (default: '<input_stem>_24k_ref.wav').",
    )
    parser.add_argument("--start", type=float, default=0.0,
                        help="Window start in seconds (default: 0).")
    parser.add_argument("--duration", type=float, default=8.0,
                        help="Window duration in seconds; clamped to 5-10s (default: 8).")
    parser.add_argument("--target-sr", type=int, default=TARGET_SR,
                        help=f"Target sample rate (default: {TARGET_SR}).")
    parser.add_argument("--ref-text-file",
                        help="Optional transcript file to prepare alongside the audio.")
    parser.add_argument("--strip-tags", action="store_true",
                        help="Strip Fish Audio S2 inline tags ([soft], [whispering]...) "
                             "from the transcript, keeping only spoken words.")
    parser.add_argument("--sentences",
                        help="Sentence range of the transcript matching the audio "
                             "window, e.g. '0-2' (inclusive) or '3'.")
    parser.add_argument("--no-diacritize", action="store_true",
                        help="Do not run the diacritizer (use for transcripts that "
                             "are already exactly diacritized).")
    parser.add_argument("--diacritizer", default=None,
                        help="Diacritizer backend for --ref-text-file: "
                             "mishkal (default) | camel | none.")
    args = parser.parse_args(argv)

    output_path = args.output
    if not output_path:
        stem, _ = os.path.splitext(args.input_wav)
        output_path = f"{stem}_24k_ref.wav"

    prepare_reference_audio(
        input_path=args.input_wav,
        output_path=output_path,
        start_s=args.start,
        duration_s=args.duration,
        target_sr=args.target_sr,
    )

    if args.ref_text_file:
        prepare_reference_text(
            args.ref_text_file,
            backend=args.diacritizer,
            strip_tags=args.strip_tags,
            sentences=args.sentences,
            no_diacritize=args.no_diacritize,
        )

    print(
        "\nChecklist:\n"
        "  1. Listen to the output clip: it must contain ONLY the whispered voice.\n"
        "  2. Point ARABIC_REF_AUDIO at the output file in your .env.\n"
        "  3. Point ARABIC_REF_TEXT_PATH at the EXACT diacritized transcript of\n"
        "     the spoken words in that clip (mismatches cause hallucinations)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
