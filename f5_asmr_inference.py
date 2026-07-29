# f5_asmr_inference.py
"""
Standalone SILMA/F5-TTS Arabic ASMR inference wrapper + before/after harness.

Runs the project's Arabic voice in two configurations:

  - baseline:  the legacy behavior (raw text, no diacritization, f5-tts
               stock defaults: nfe_step=32, cfg_strength=2.0, sway sampling off)
  - optimized: the new pipeline (Arabic normalization + diacritization,
               tuned nfe_step/cfg_strength/sway sampling, validated ref clip)

Outputs (default: output/):
  - f5_baseline.wav / f5_optimized.wav  (24 kHz PCM16)
  - f5_comparison_log.md                (params, audio validation stats,
                                         diacritization coverage, and a
                                         human listening checklist for
                                         fidelity / pronunciation / whisper)

Usage:
  python f5_asmr_inference.py --text "خذ نفساً عميقاً واسترخِ تماماً"
  python f5_asmr_inference.py --text-file my_script.txt --mode compare
  python f5_asmr_inference.py --text "..." --ref-audio input/ref_arabic_whisper_24k.wav
"""
import argparse
import io
import os
import sys
from datetime import datetime

import numpy as np
import soundfile as sf

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import tts

SAMPLE_RATE = 24000

# Legacy behavior snapshot (what the app did before the optimization work).
BASELINE_PARAMS = {
    "use_preprocessing": False,
    "nfe_step": 32,
    "cfg_strength": 2.0,
    "sway_sampling_coef": -1.0,
    "target_rms": 0.1,
}

# Optimized behavior: preprocessing on; remaining params come from the
# env-configured defaults in tts.py (F5_NFE_STEP, F5_CFG_STRENGTH, ...).
OPTIMIZED_PARAMS = {
    "use_preprocessing": True,
}

DEFAULT_TEST_TEXT = (
    "خذ نفساً عميقاً، ودع كتفيك يسترخيان ببطء. "
    "استمع إلى صوت المطر الخفيف وهو يلمس النافذة بهدوء تام."
)

# Evidence grid for the --sweep mode: (nfe_step, sway_sampling_coef) combos.
# cfg_strength stays fixed at 2.0; all combos run with optimized preprocessing.
SWEEP_GRID = [(32, -1.0), (32, 0.5), (40, -1.0), (40, 0.5)]


def audio_stats(wav: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    """Audio output validation metrics for the comparison log."""
    if wav.size == 0:
        return {"duration_s": 0.0, "rms": 0.0, "peak": 0.0,
                "clipping_ratio": 0.0, "silence_ratio": 0.0, "finite": False}
    abs_wav = np.abs(wav)
    return {
        "duration_s": round(len(wav) / float(sr), 2),
        "rms": round(float(np.sqrt(np.mean(np.square(wav)))), 4),
        "peak": round(float(np.max(abs_wav)), 4),
        "clipping_ratio": round(float(np.mean(abs_wav >= 0.999)), 5),
        "silence_ratio": round(float(np.mean(abs_wav < 1e-4)), 4),
        "finite": bool(np.isfinite(wav).all()),
    }


def _ref_audio_spec(path: str) -> dict:
    try:
        meta = sf.info(path)
        return {
            "duration_s": round(meta.frames / float(meta.samplerate), 2),
            "sample_rate": meta.samplerate,
            "channels": meta.channels,
        }
    except Exception as exc:
        return {"error": str(exc)}


def _effective_params(mode: str) -> dict:
    if mode == "baseline":
        return dict(BASELINE_PARAMS)
    return {
        "use_preprocessing": True,
        "nfe_step": tts._F5_NFE_STEP,
        "cfg_strength": tts._F5_CFG_STRENGTH,
        "sway_sampling_coef": tts._F5_SWAY_SAMPLING_COEF,
        "target_rms": tts._F5_TARGET_RMS,
    }


def run_mode(
    mode: str,
    text: str,
    speed: float,
    ref_audio: str | None,
    ref_text: str | None,
    output_dir: str,
) -> tuple[np.ndarray, dict]:
    """Generates one configuration and returns (wav, report-context)."""
    params = _effective_params(mode)

    # The baseline reproduces the legacy app exactly: raw inline ref text,
    # no normalization/diacritization anywhere.
    baseline_ref_text = ref_text if ref_text is not None else tts._ARABIC_REF_TEXT
    gen_kwargs = dict(
        speed=speed,
        use_preprocessing=params["use_preprocessing"],
        nfe_step=params["nfe_step"],
        cfg_strength=params["cfg_strength"],
        sway_sampling_coef=params["sway_sampling_coef"],
        target_rms=params["target_rms"],
        ref_audio=ref_audio,
        ref_text=baseline_ref_text if mode == "baseline" else ref_text,
    )
    wav = tts.generate_f5_audio(text, **{k: v for k, v in gen_kwargs.items() if v is not None})

    out_path = os.path.join(output_dir, f"f5_{mode}.wav")
    os.makedirs(output_dir, exist_ok=True)
    sf.write(out_path, wav, SAMPLE_RATE, subtype="PCM_16")

    return wav, {"params": params, "path": out_path}


def _diacritization_report(text: str) -> dict:
    try:
        import preprocess_arabic
    except ImportError:
        return {"available": False}
    try:
        processed = preprocess_arabic.diacritize(preprocess_arabic.normalize_arabic(text))
        return {
            "available": True,
            "backend": preprocess_arabic.resolve_diacritizer_backend(),
            "coverage_before": round(preprocess_arabic.diacritization_coverage(text), 3),
            "coverage_after": round(preprocess_arabic.diacritization_coverage(processed), 3),
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def run_sweep(
    text: str,
    speed: float,
    ref_audio: str | None,
    ref_text: str | None,
    output_dir: str,
    grid: list[tuple[int, float]] | None = None,
) -> list[dict]:
    """
    Runs the (nfe_step x sway_sampling_coef) evidence grid with optimized
    preprocessing. Each combo is validated for audible speech via whisper STT
    (stt.verify_audible_speech) so the winning config is chosen by evidence.
    """
    import stt as stt_module

    grid = grid or SWEEP_GRID
    try:
        stt_pipe = stt_module.create_stt_pipeline()
    except Exception as exc:
        print(f"[warn] STT pipeline unavailable ({exc}); voice checks will be skipped.")
        stt_pipe = None

    rows: list[dict] = []
    for nfe, sway in grid:
        label = f"nfe{nfe}_sway{sway}"
        print(f"[sweep] {label}: synthesizing ...")
        wav = tts.generate_f5_audio(
            text,
            speed=speed,
            use_preprocessing=True,
            nfe_step=nfe,
            sway_sampling_coef=sway,
            ref_audio=ref_audio,
            ref_text=ref_text,
        )
        out_path = os.path.join(output_dir, f"f5_sweep_{label}.wav")
        os.makedirs(output_dir, exist_ok=True)
        sf.write(out_path, wav, SAMPLE_RATE, subtype="PCM_16")

        voice: dict = {"ok": None, "reason": "stt unavailable"}
        if stt_pipe is not None:
            buf = io.BytesIO()
            sf.write(buf, wav, SAMPLE_RATE, format="WAV")
            voice = stt_module.verify_audible_speech(
                buf.getvalue(), expected_text=text, stt_pipeline=stt_pipe
            )

        row = {
            "label": label, "nfe": nfe, "sway": sway, "path": out_path,
            "stats": audio_stats(wav), "voice": voice,
        }
        rows.append(row)
        print(f"  -> {out_path} ({row['stats']['duration_s']}s, "
              f"rms={row['stats']['rms']}) | voice check: "
              f"{voice.get('ok')} (similarity={voice.get('similarity')}, "
              f"{voice.get('reason')})")
    return rows


def recommend_sweep_winner(rows: list[dict]) -> dict | None:
    """Best passing combo: highest STT similarity, then higher nfe_step."""
    passing = [r for r in rows if r["voice"].get("ok")]
    if not passing:
        return None
    return max(passing, key=lambda r: (r["voice"].get("similarity") or 0.0, r["nfe"]))


def write_sweep_log(
    output_dir: str,
    text: str,
    ref_audio: str,
    rows: list[dict],
    winner: dict | None,
) -> str:
    lines = [
        "# F5-TTS Arabic ASMR - Parameter Sweep Log",
        "",
        f"- Date: {datetime.now().isoformat(timespec='seconds')}",
        f"- Reference audio: `{ref_audio}`",
        "- Preprocessing: optimized (normalize + diacritize) for all combos",
        "- Voice check: whisper STT transcription + normalized similarity (>=0.3) + RMS gate",
        "",
        "## Input text",
        "",
        "```",
        text.strip(),
        "```",
        "",
        "## Sweep results",
        "",
        "| combo | duration_s | rms | peak | clipping | voice ok | similarity | transcript (whisper) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        stats = r["stats"]
        voice = r["voice"]
        transcript = (voice.get("transcript") or "").replace("|", "/")[:60]
        lines.append(
            f"| {r['label']} | {stats['duration_s']} | {stats['rms']} | {stats['peak']} "
            f"| {stats['clipping_ratio']} | {voice.get('ok')} | {voice.get('similarity')} "
            f"| {transcript} |"
        )
    lines += ["", "## Recommendation", ""]
    if winner:
        lines += [
            f"Winning combo: **{winner['label']}**",
            "",
            "```env",
            f"F5_NFE_STEP={winner['nfe']}",
            f"F5_SWAY_SAMPLING_COEF={winner['sway']}",
            "```",
        ]
    else:
        lines += [
            "No combo passed the voice check. Keep the stock profile "
            "(F5_NFE_STEP=32, F5_SWAY_SAMPLING_COEF=-1.0) and review the "
            "reference audio/text alignment.",
        ]
    lines.append("")

    log_path = os.path.join(output_dir, "f5_sweep_log.md")
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return log_path



def write_comparison_log(
    output_dir: str,
    text: str,
    speed: float,
    ref_audio: str,
    ref_text_source: str,
    results: dict[str, dict],
) -> str:
    lines = [
        "# F5-TTS Arabic ASMR - Before/After Comparison Log",
        "",
        f"- Date: {datetime.now().isoformat(timespec='seconds')}",
        f"- Reference audio: `{ref_audio}`",
        f"- Reference text source: {ref_text_source}",
        f"- Speed: {speed}",
        f"- Diacritizer: {(results.get('diacritization') or {}).get('backend', 'n/a')}",
        "",
        "## Input text",
        "",
        "```",
        text.strip(),
        "```",
        "",
        "## Reference audio spec (target: 5-10s, mono, >=24kHz)",
        "",
        "| spec | value |",
        "|---|---|",
    ]
    for key, value in _ref_audio_spec(ref_audio).items():
        lines.append(f"| {key} | {value} |")

    lines += [
        "",
        "## Generation parameters",
        "",
        "| param | baseline | optimized |",
        "|---|---|---|",
    ]
    param_keys = sorted(results["baseline"]["params"].keys())
    for key in param_keys:
        lines.append(
            f"| {key} | {results['baseline']['params'][key]} "
            f"| {results.get('optimized', {}).get('params', {}).get(key, 'n/a')} |"
        )

    diac = results.get("diacritization")
    if diac and diac.get("available"):
        lines += [
            "",
            "## Diacritization (tashkeel) coverage of gen text",
            "",
            f"- Before preprocessing: **{diac['coverage_before']:.1%}**",
            f"- After preprocessing (backend={diac['backend']}): **{diac['coverage_after']:.1%}**",
        ]
    elif diac:
        lines += ["", f"_Diacritizer unavailable: {diac.get('error', 'not installed')}_"]

    lines += [
        "",
        "## Audio output validation",
        "",
        "| metric | baseline | optimized |",
        "|---|---|---|",
    ]
    metric_keys = ["duration_s", "rms", "peak", "clipping_ratio", "silence_ratio", "finite"]
    for key in metric_keys:
        base = results["baseline"]["stats"].get(key, "n/a")
        opt = results.get("optimized", {}).get("stats", {}).get(key, "n/a")
        lines.append(f"| {key} | {base} | {opt} |")

    lines += [
        "",
        "## Human listening checklist (fill after comparing both WAVs)",
        "",
        "- [ ] Whisper/ASMR delivery (soft, breathy, low energy) improved",
        "- [ ] Pronunciation accuracy for MSA words improved",
        "- [ ] Text fidelity: no hallucinated/added words vs the input text",
        "- [ ] Text fidelity: no dropped or repeated phrases",
        "- [ ] No metallic artifacts / unstable breaths",
        "",
        "Notes:",
        "",
        "- baseline file: `f5_baseline.wav`",
        "- optimized file: `f5_optimized.wav`",
        "",
    ]

    log_path = os.path.join(output_dir, "f5_comparison_log.md")
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return log_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="F5-TTS Arabic ASMR inference wrapper (baseline vs optimized)."
    )
    parser.add_argument("--text", help="Arabic text to synthesize.")
    parser.add_argument("--text-file", help="Read the Arabic text from a file.")
    parser.add_argument("--mode", choices=["baseline", "optimized", "compare"],
                        default="compare", help="Which configuration(s) to run.")
    parser.add_argument("--sweep", action="store_true",
                        help="Run the (nfe_step x sway) evidence grid with STT voice "
                             "checks instead of baseline/optimized modes.")
    parser.add_argument("--speed", type=float, default=0.85,
                        help="Synthesis speed (0.85-0.90 recommended for ASMR).")
    parser.add_argument("--ref-audio", default=None,
                        help="Reference whisper clip (default: ARABIC_REF_AUDIO / tts.py).")
    parser.add_argument("--ref-text-file", default=None,
                        help="Exact (ideally diacritized) transcript of the ref clip.")
    parser.add_argument("--output-dir", default="output",
                        help="Directory for WAVs and the comparison log.")
    args = parser.parse_args(argv)

    if args.text_file:
        with open(args.text_file, encoding="utf-8") as fh:
            text = fh.read().strip()
    elif args.text:
        text = args.text.strip()
    else:
        text = DEFAULT_TEST_TEXT
        print(f"[info] no --text/--text-file given; using built-in sample text.")

    if not text:
        print("error: empty input text.", file=sys.stderr)
        return 2

    ref_audio = args.ref_audio or tts._ARABIC_REF_AUDIO
    if not os.path.exists(ref_audio):
        print(f"error: reference audio not found at '{ref_audio}'.", file=sys.stderr)
        return 2

    ref_text = None
    ref_text_source = "inline fallback in tts.py"
    if args.ref_text_file:
        with open(args.ref_text_file, encoding="utf-8") as fh:
            ref_text = fh.read().strip()
        ref_text_source = args.ref_text_file
    elif os.path.exists(tts._ARABIC_REF_TEXT_PATH):
        ref_text_source = tts._ARABIC_REF_TEXT_PATH

    if args.sweep:
        rows = run_sweep(text, args.speed, ref_audio, ref_text, args.output_dir)
        winner = recommend_sweep_winner(rows)
        log_path = write_sweep_log(args.output_dir, text, ref_audio, rows, winner)
        print(f"[done] sweep log written to: {log_path}")
        if winner:
            print(f"[recommendation] set in .env: F5_NFE_STEP={winner['nfe']} "
                  f"F5_SWAY_SAMPLING_COEF={winner['sway']}")
        else:
            print("[recommendation] no combo passed the voice check; keep the "
                  "stock profile and review ref audio/text alignment.")
        return 0

    modes = ["baseline", "optimized"] if args.mode == "compare" else [args.mode]
    results: dict[str, dict] = {}
    for mode in modes:
        print(f"[run] {mode}: synthesizing {len(text)} chars ...")
        wav, ctx = run_mode(mode, text, args.speed, ref_audio, ref_text, args.output_dir)
        ctx["stats"] = audio_stats(wav)
        results[mode] = ctx
        print(f"  -> {ctx['path']} ({ctx['stats']['duration_s']}s, "
              f"rms={ctx['stats']['rms']}, peak={ctx['stats']['peak']})")

    results["diacritization"] = _diacritization_report(text)
    log_path = write_comparison_log(
        args.output_dir, text, args.speed, ref_audio, ref_text_source, results
    )
    print(f"[done] comparison log written to: {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
