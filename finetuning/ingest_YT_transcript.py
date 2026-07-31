# finetuning/ingest_YT_transcript.py
"""
Ingest real YouTube ASMR videos into dialect-specific JSON files.

Input:
    finetuning/data/metadata.csv
    Columns: Dialect (Syria|Egypt), URL (YouTube link)

Processing:
    - Extracts the video title using yt-dlp.
    - Extracts the Arabic transcript using youtube-transcript-api
      (falls back to yt-dlp captions if the API is unavailable).
    - Writes one JSON file per video.

Output:
    finetuning/data/ingested/<dialect>/<timestamp>_<video_id>.json

Example:
    python finetuning/ingest_YT_transcript.py
"""
import argparse
import csv
import json
import os
import re
import sys
import shutil
from datetime import datetime, timezone
from typing import Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

DEFAULT_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "metadata.csv"
)
DEFAULT_OUTPUT_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "ingested"
)

DIALECTS = {"syria": "syria", "egypt": "egypt", "syrian": "syria"}


def _detect_js_runtime() -> dict[str, dict[str, Any]] | None:
    """Return a yt-dlp compatible js_runtimes dict if node/deno/bun is available."""
    for name in ("node", "deno", "bun"):
        path = shutil.which(name)
        if path:
            return {name: {"timeout": 30000, "path": path}}
    return None


def _video_id(url: str) -> str | None:
    """Return the YouTube video ID from a URL, or None."""
    url = url.strip()
    if not url or "youtube.com" not in url.lower() and "youtu.be" not in url.lower():
        return None
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"(?:embed\/)([0-9A-Za-z_-]{11})",
        r"youtu\.be\/([0-9A-Za-z_-]{11})",
    ]
    for pat in patterns:
        m = re.search(pat, url)
        if m:
            return m.group(1)
    return None


def _normalize_dialect(raw: str) -> str | None:
    return DIALECTS.get(raw.strip().lower())


def _fetch_title_ytdlp(url: str, cookiefile: str | None = None) -> str | None:
    """Fetch video title with yt-dlp, preferring lightweight extraction."""
    import yt_dlp

    js_runtime = _detect_js_runtime()

    common_opts = {
        "quiet": True,
        "skip_download": True,
        "cookiefile": cookiefile,
    }
    if not js_runtime:
        print("[warn] no JS runtime found; yt-dlp YouTube extraction may fail")
    else:
        common_opts["js_runtimes"] = js_runtime

    # Fast, metadata-only extraction that does not need JS runtime.
    light_opts = {
        **common_opts,
        "extract_flat": True,
        "playlist_items": "1",
    }
    with yt_dlp.YoutubeDL(light_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False, process=False)
            title = info.get("title") or info.get("fulltitle")
            if title:
                return title.strip()
        except Exception:
            pass

    # Fallback to full extraction if the lightweight path fails.
    full_opts = {
        **common_opts,
        "forcetitle": True,
    }
    with yt_dlp.YoutubeDL(full_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
        except Exception:
            return None
        return info.get("title") or info.get("fulltitle")


def _fetch_transcript_yta(video_id: str) -> str | None:
    """Fetch Arabic transcript with youtube-transcript-api."""
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled

    api = YouTubeTranscriptApi()

    # Try Arabic explicitly — this video may not have English captions,
    # and the default fetch() targets English.
    try:
        transcript_list = api.list(video_id)
    except (TranscriptsDisabled, NoTranscriptFound):
        return None
    except Exception:
        return None

    # Try manual Arabic first.
    for lang in ("ar", "ar-SA", "ar-EG"):
        try:
            transcript = transcript_list.find_transcript([lang])
            segments = transcript.fetch()
            text = " ".join(seg.text for seg in segments if seg.text)
            if text.strip():
                return text.strip()
        except Exception:
            continue

    # Fall back to auto-generated Arabic.
    try:
        generated = transcript_list.find_generated_transcript(["ar", "ar-SA", "ar-EG"])
        segments = generated.fetch()
        text = " ".join(seg.text for seg in segments if seg.text)
        return text.strip() or None
    except Exception:
        return None


def _fetch_transcript_ytdlp(url: str, cookiefile: str | None = None) -> str | None:
    """Fallback transcript extraction using yt-dlp captions."""
    import yt_dlp

    js_runtime = _detect_js_runtime()
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "writesubtitles": False,
        "writeautomaticsub": False,
        "cookiefile": cookiefile,
        "extractor_args": {
            "youtube": {
                "player_client": ["web"],
                "player_skip": ["webpage"],
            }
        },
    }
    if js_runtime:
        ydl_opts["js_runtimes"] = js_runtime

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
        except Exception:
            return None
        subtitles = info.get("subtitles") or {}
        auto_captions = info.get("automatic_captions") or {}
        auto_captions = info.get("automatic_captions") or {}

        # Prefer Arabic manual/auto captions.
        for lang in ("ar", "ar-SA", "ar-EG"):
            if lang in subtitles:
                return _caption_text(subtitles[lang])
            if lang in auto_captions:
                return _caption_text(auto_captions[lang])

        # Fall back to first available Arabic code.
        for key, tracks in {**subtitles, **auto_captions}.items():
            if key.lower().startswith("ar"):
                return _caption_text(tracks)
    return None


def _caption_text(tracks: list[dict[str, Any]]) -> str | None:
    """Extract plain text from a yt-dlp caption track list."""
    for track in tracks:
        text_or_url = track.get("data") or track.get("url")
        if text_or_url:
            text = text_or_url.strip()
            if "<" in text or ".sbv" in track.get("ext", ""):
                # Very basic strip of HTML/sbv timing in emergency fallback.
                text = re.sub(r"<[^>]+>", "", text)
                text = re.sub(r"\d{1,2}:\d{2}:\d{2}[,.\s]+\d+\s*-->.*", "", text)
            return re.sub(r"\s+", " ", text).strip()
    return None


def fetch_video_data(url: str, cookiefile: str | None = None) -> dict[str, Any]:
    """Return a dict with title, transcript, and video_id for a YouTube URL."""
    video_id = _video_id(url)
    if not video_id:
        raise ValueError(f"Could not parse YouTube video ID from URL: {url}")

    title = _fetch_title_ytdlp(url, cookiefile=cookiefile)
    if not title:
        raise RuntimeError(f"Could not fetch title for {url}")

    transcript = _fetch_transcript_yta(video_id)
    if not transcript:
        transcript = _fetch_transcript_ytdlp(url, cookiefile=cookiefile)
    if not transcript:
        raise RuntimeError(f"Could not fetch transcript for {url}")

    return {
        "video_id": video_id,
        "url": url,
        "title": title.strip(),
        "transcript": transcript.strip(),
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }


def _extract_video_info(url: str) -> dict[str, Any]:
    """Convenience alias for fetch_video_data; kept for testability."""
    return fetch_video_data(url)


def _read_metadata(csv_path: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(csv_path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append({k.strip(): (v or "").strip() for k, v in row.items()})
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--metadata-csv",
        default=DEFAULT_CSV,
        help="CSV with Dialect and URL columns.",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory under which dialect sub-folders are created.",
    )
    parser.add_argument(
        "--cookies",
        default=None,
        help="Optional Netscape-format cookie file to pass to yt-dlp.",
    )
    args = parser.parse_args(argv)

    if not os.path.exists(args.metadata_csv):
        print(f"error: metadata CSV not found: {args.metadata_csv}", file=sys.stderr)
        return 2

    rows = _read_metadata(args.metadata_csv)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    skipped = 0

    for i, row in enumerate(rows):
        dialect = _normalize_dialect(row.get("Dialect", ""))
        url = row.get("URL", "")
        if not dialect:
            print(f"[skip {i+1}] unknown dialect '{row.get('Dialect')}'")
            skipped += 1
            continue
        if not url:
            print(f"[skip {i+1}] missing URL for dialect {dialect}")
            skipped += 1
            continue

        try:
            data = fetch_video_data(url, cookiefile=args.cookies)
        except Exception as exc:
            print(f"[error] row {i+1}: {exc}", file=sys.stderr)
            skipped += 1
            continue

        out_dir = os.path.join(args.output_root, dialect)
        os.makedirs(out_dir, exist_ok=True)
        filename = f"{timestamp}_{data['video_id']}.json"
        out_path = os.path.join(out_dir, filename)

        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "dialect": dialect,
                    "video_id": data["video_id"],
                    "url": data["url"],
                    "title": data["title"],
                    "transcript": data["transcript"],
                    "ingested_at": data["ingested_at"],
                },
                fh,
                ensure_ascii=False,
                indent=2,
            )

        print(
            f"[ok] {dialect}: {data['title'][:60]}{'...' if len(data['title']) > 60 else ''} "
            f"-> {out_path}"
        )

    total = len(rows)
    processed = total - skipped
    print(f"[done] {processed}/{total} rows processed; {skipped} skipped")
    return 0 if processed > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
