# fish_arabic_asmr_test.py
"""
Fish Audio (s2.1-pro-free) evaluation harness for Arabic whispering/ASMR TTS.

Alternative TTS path to SILMA/F5-TTS, using Fish Audio's free developer tier:
  - Model:    s2.1-pro-free (S2.1 Pro quality at $0; no TTFA/DPA SLAs)
  - Controls: inline natural-language direction in square brackets, e.g.
              [soft], [whispering], [emphasis], [break], [long-break]
              (S2 syntax; works with Arabic input, UTF-8 safe)
  - Voices:   optional community voice via reference_id (FISH_AUDIO_VOICE_ID),
              with automatic fallback to the default voice when unavailable

Measured per case: time-to-first-byte (TTFB), total latency, payload size.
Outputs:
  - output/output_asmr_arabic.mp3 (+ one file per extra test case)
  - fish_audio_assessment.md (latency table + quality observation template
    + free->pro transition recommendations)

Auth: set FISH_AUDIO_API_KEY in your environment or .env.

Usage:
  python fish_arabic_asmr_test.py                      # run all built-in cases
  python fish_arabic_asmr_test.py --case inline_tags_whisper
  python fish_arabic_asmr_test.py --text "[whispering] مرحباً بالعالم" --format wav

Docs: https://docs.fish.audio/features/text-to-speech
"""
import argparse
import os
import sys
import time
from dataclasses import dataclass, field

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API_URL = "https://api.fish.audio/v1/tts"
DEFAULT_MODEL = "s2.1-pro-free"

# QA-provided inline-tag sample plus app-realistic ASMR variants.
TEST_CASES = [
    {
        "name": "inline_tags_whisper",
        "text": "[soft] مرحباً بك... [whispering] استمع إلى هذا الصوت الناعم والمريح... "
                "[emphasis] هل تشعر بالاسترخاء الآن؟",
    },
    {
        "name": "asmr_rain_pauses",
        "text": "[whispering] خذ نفساً عميقاً [break] واستمع إلى صوت المطر الخفيف "
                "على النافذة [long-break] هل تشعر بالهدوء الآن؟",
    },
    {
        "name": "plain_arabic_control",
        "text": "مرحباً بك في جلسة استرخاء هادئة ومريحة للأعصاب.",
    },
]

# Statuses worth retrying with exponential backoff (transient).
_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
# Statuses that justify dropping a custom voice_id and retrying default.
_VOICE_FALLBACK_STATUSES = {400, 404, 422}


class FishAudioError(RuntimeError):
    """Raised for unrecoverable Fish Audio API failures."""


@dataclass
class FishResult:
    name: str
    text: str
    voice_id: str | None
    fallback_used: bool
    ttfb_s: float
    total_s: float
    audio_bytes: bytes
    fmt: str
    output_path: str = ""
    error: str = ""
    extra: dict = field(default_factory=dict)


class FishAudioTTS:
    """Minimal requests-based wrapper for the Fish Audio /v1/tts endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str = API_URL,
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        self.api_key = (api_key or os.getenv("FISH_AUDIO_API_KEY", "")).strip()
        self.model = (model or os.getenv("FISH_AUDIO_MODEL", DEFAULT_MODEL)).strip()
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # Fish Audio selects the model via the `model` HTTP header.
            "model": self.model,
        }

    def synthesize(
        self,
        text: str,
        voice_id: str | None = None,
        fmt: str = "mp3",
        speed: float = 1.0,
        latency: str = "normal",
        chunk_length: int = 200,
        sample_rate: int | None = None,
        name: str = "adhoc",
    ) -> FishResult:
        if not self.api_key:
            raise FishAudioError(
                "Fish Audio API key missing. Set FISH_AUDIO_API_KEY in your .env."
            )
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        payload = {
            "text": text,
            "format": fmt,
            "prosody": {"speed": speed},
            "latency": latency,
            "chunk_length": chunk_length,
        }
        if sample_rate:
            payload["sample_rate"] = sample_rate
        if voice_id:
            payload["reference_id"] = voice_id

        fallback_used = False
        last_error: str | None = None

        for attempt in range(self.max_retries):
            try:
                t_start = time.perf_counter()
                resp = requests.post(
                    self.base_url,
                    json=payload,
                    headers=self._headers(),
                    stream=True,
                    timeout=self.timeout,
                )
                ttfb = time.perf_counter() - t_start  # response headers received
            except requests.RequestException as exc:
                last_error = f"network error: {exc}"
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise FishAudioError(
                    f"Fish Audio request failed after {self.max_retries} attempts. "
                    f"Last error: {last_error}"
                ) from exc

            if resp.status_code == 200:
                chunks = [chunk for chunk in resp.iter_content(chunk_size=8192) if chunk]
                total = time.perf_counter() - t_start
                return FishResult(
                    name=name,
                    text=text,
                    voice_id=payload.get("reference_id"),
                    fallback_used=fallback_used,
                    ttfb_s=round(ttfb, 3),
                    total_s=round(total, 3),
                    audio_bytes=b"".join(chunks),
                    fmt=fmt,
                )

            body = resp.text[:300]
            resp.close()

            # Custom voice rejected/unavailable -> drop it and retry once
            # with the platform default voice.
            if (
                voice_id
                and "reference_id" in payload
                and resp.status_code in _VOICE_FALLBACK_STATUSES
            ):
                print(f"  [warn] voice_id '{voice_id}' rejected "
                      f"(HTTP {resp.status_code}); falling back to default voice.")
                payload.pop("reference_id")
                fallback_used = True
                continue

            if resp.status_code in _RETRYABLE_STATUSES and attempt < self.max_retries - 1:
                wait = 2 ** attempt
                print(f"  [warn] HTTP {resp.status_code}; retrying in {wait}s "
                      f"(attempt {attempt + 1}/{self.max_retries})...")
                time.sleep(wait)
                continue

            last_error = f"HTTP {resp.status_code}: {body}"
            break

        raise FishAudioError(
            f"Fish Audio synthesis failed for case '{name}'. {last_error}"
        )


def run_cases(
    client: FishAudioTTS,
    cases: list[dict],
    voice_id: str | None,
    fmt: str,
    speed: float,
    output_dir: str,
) -> list[FishResult]:
    results: list[FishResult] = []
    os.makedirs(output_dir, exist_ok=True)

    for index, case in enumerate(cases):
        name = case["name"]
        text = case["text"]
        print(f"[case] {name} ({len(text)} chars, utf-8: {len(text.encode('utf-8'))} bytes)")
        try:
            result = client.synthesize(
                text=text, voice_id=voice_id, fmt=fmt, speed=speed, name=name,
            )
        except FishAudioError as exc:
            print(f"  [error] {exc}")
            results.append(FishResult(
                name=name, text=text, voice_id=voice_id, fallback_used=False,
                ttfb_s=0.0, total_s=0.0, audio_bytes=b"", fmt=fmt, error=str(exc),
            ))
            continue

        # QA deliverable filename for the primary case; per-case names otherwise.
        filename = "output_asmr_arabic" if index == 0 else f"fish_{name}"
        out_path = os.path.join(output_dir, f"{filename}.{fmt}")
        with open(out_path, "wb") as fh:
            fh.write(result.audio_bytes)
        result.output_path = out_path
        print(f"  -> {out_path} ({len(result.audio_bytes)} bytes, "
              f"ttfb={result.ttfb_s}s, total={result.total_s}s)")
        results.append(result)

    return results


def write_assessment_report(
    path: str,
    results: list[FishResult],
    model: str,
    voice_id: str | None,
) -> str:
    lines = [
        "# Fish Audio Arabic ASMR TTS - Technical Assessment",
        "",
        f"- Model: `{model}` (free tier: no TTFA/DPA SLAs)",
        f"- Voice: `{voice_id or 'platform default'}`",
        f"- API: POST https://api.fish.audio/v1/tts (model via HTTP header)",
        "",
        "## Latency measurements",
        "",
        "| case | chars | fmt | voice fallback | TTFB (s) | total (s) | bytes | status |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        status = r.error if r.error else "ok"
        lines.append(
            f"| {r.name} | {len(r.text)} | {r.fmt} | "
            f"{'yes' if r.fallback_used else 'no'} | {r.ttfb_s} | {r.total_s} | "
            f"{len(r.audio_bytes)} | {status} |"
        )

    lines += [
        "",
        "## Voice quality observations (fill after listening)",
        "",
        "- [ ] Whisper/soft delivery is convincing (vs normal speaking energy)",
        "- [ ] Arabic pronunciation is correct (MSA)",
        "- [ ] No hallucinated words vs the input text",
        "- [ ] Inline tags ([soft], [whispering], [emphasis], [break]) are honored",
        "- [ ] Preferred community voice candidate: ______",
        "",
        "## Recommendations: s2.1-pro-free -> s2.1-pro",
        "",
        "- The free tier shares the S2.1 Pro model weights; production upgrade",
        "  buys guaranteed TTFA latency and DPA/SLA coverage, not new quality.",
        "- Keep `s2.1-pro-free` for prototyping; switch the FISH_AUDIO_MODEL env",
        "  var to `s2.1-pro` for production traffic.",
        "- If rate limits (HTTP 429) appear in the latency table above, they are",
        "  expected on the free tier; the wrapper already backs off and retries.",
        "",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fish Audio s2.1-pro-free Arabic ASMR TTS evaluation harness."
    )
    parser.add_argument("--text", help="Custom Arabic text (single case).")
    parser.add_argument("--case", help="Run only the named built-in case.")
    parser.add_argument("--voice-id", default=os.getenv("FISH_AUDIO_VOICE_ID", ""),
                        help="reference_id of a community voice (env FISH_AUDIO_VOICE_ID).")
    parser.add_argument("--format", choices=["mp3", "wav"], default="mp3",
                        help="Audio format (default: mp3).")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Prosody speed 0.5-2.0 (default: 1.0).")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--report", default="fish_audio_assessment.md")
    args = parser.parse_args(argv)

    if args.text:
        cases = [{"name": "custom", "text": args.text.strip()}]
    elif args.case:
        cases = [c for c in TEST_CASES if c["name"] == args.case]
        if not cases:
            print(f"error: unknown case '{args.case}'. "
                  f"Available: {[c['name'] for c in TEST_CASES]}", file=sys.stderr)
            return 2
    else:
        cases = TEST_CASES

    voice_id = args.voice_id.strip() or None
    client = FishAudioTTS()
    results = run_cases(client, cases, voice_id, args.format, args.speed, args.output_dir)
    report_path = write_assessment_report(args.report, results, client.model, voice_id)
    print(f"[done] assessment report written to: {report_path}")

    failures = [r for r in results if r.error]
    return 1 if failures and len(failures) == len(results) else 0


if __name__ == "__main__":
    sys.exit(main())
