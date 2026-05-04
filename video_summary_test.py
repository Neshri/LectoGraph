from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from lectograph.config import Config
from openscenesense_ollama.models import AnalysisPrompts
from lectograph.transcriber import FasterWhisperAdapter
from openscenesense_ollama.analyzer import OllamaVideoAnalyzer
from openscenesense_ollama.frame_selectors import DynamicFrameSelector


def main() -> None:
    parser = argparse.ArgumentParser(description="Test video summarization.")
    parser.add_argument("video_path", help="Path to the video file to analyze")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger(__name__)

    try:
        cfg = Config.from_yaml(Path(args.config))
    except Exception as exc:
        log.error("Failed to load config: %s", exc)
        sys.exit(1)

    transcriber = FasterWhisperAdapter(
        model_name=cfg.whisper_model,
        device=cfg.whisper_device,
        initial_prompt=cfg.whisper_initial_prompt or None,
        hotwords=cfg.whisper_hotwords or None,
    )

    custom_prompts = AnalysisPrompts(
        frame_analysis=cfg.frame_analysis_prompt,
        detailed_summary=cfg.detailed_summary_prompt,
        brief_summary=cfg.brief_summary_prompt,
    )

    analyzer = OllamaVideoAnalyzer(
        frame_analysis_model=cfg.frame_analysis_model,
        summary_model=cfg.summary_model,
        min_frames=cfg.min_frames,
        max_frames=cfg.max_frames,
        frames_per_minute=cfg.frames_per_minute,
        frame_selector=DynamicFrameSelector(threshold=cfg.frame_threshold),
        audio_transcriber=transcriber,
        prompts=custom_prompts,
        request_timeout=cfg.request_timeout,
        request_retries=cfg.request_retries,
        log_level=logging.INFO,
    )

    from lectograph.pipeline import (
        _KNOWN_BAD_TERMS,
        _summaries_are_clean,
        _transcript_needs_correction,
        format_knowledge_doc,
    )

    # ── Analyze ───────────────────────────────────────────────────────────────
    try:
        results = analyzer.analyze_video_structured(args.video_path)
    except Exception as exc:
        log.error("analyze_video_structured raised an exception: %s", exc, exc_info=True)
        raise

    video_path_obj = Path(args.video_path)
    output_file = "summary_output.txt"

    # ── Bad-term detection ────────────────────────────────────────────────────
    transcript_dirty = _transcript_needs_correction(results.summary.transcript)
    summaries_clean  = _summaries_are_clean(results.summary.brief, results.summary.detailed)

    if transcript_dirty and not summaries_clean:
        # Summaries are also contaminated — cannot use them as a correction reference.
        bad_terms_found = [
            term for term in _KNOWN_BAD_TERMS
            if term.lower() in (results.summary.brief + " " + results.summary.detailed).lower()
        ]
        reason = (
            f"Hallucinated terms detected in BOTH the transcript and the summaries: "
            f"{bad_terms_found}. "
            f"Summaries cannot be used as a correction reference. "
            f"A more complex solution is required."
        )
        log.error("FAILED — %s", reason)

        with open(output_file, "w", encoding="utf-8") as fh:
            fh.write("STATUS: FAILED\n")
            fh.write(f"REASON: {reason}\n")
            fh.write("\n" + "=" * 72 + "\n")
            fh.write("RAW OUTPUT (for diagnosis)\n")
            fh.write("=" * 72 + "\n\n")
            fh.write(format_knowledge_doc(video_path_obj, results))

        print(f"\nFAILED — raw output written to {output_file} for diagnosis.")
        sys.exit(1)

    # ── Write output ──────────────────────────────────────────────────────────
    doc = format_knowledge_doc(video_path_obj, results)

    with open(output_file, "w", encoding="utf-8") as fh:
        if transcript_dirty:
            # Summaries are clean → LLM correction is feasible; flag for awareness.
            bad_terms_found = [
                t for t in _KNOWN_BAD_TERMS
                if t.lower() in results.summary.transcript.lower()
            ]
            log.warning(
                "Known bad terms found in transcript but summaries are clean — "
                "LLM correction will be needed before ingestion. Bad terms: %s",
                bad_terms_found,
            )
            fh.write(f"WARNING: Transcript contains known bad terms: {bad_terms_found}\n")
            fh.write("Summaries are clean — LLM correction is feasible before ingestion.\n")
            fh.write("=" * 72 + "\n\n")

        fh.write(doc)

    print(f"\nResults have been written to {output_file}")


if __name__ == "__main__":
    main()