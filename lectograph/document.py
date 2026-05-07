from __future__ import annotations

from pathlib import Path

# ─── Knowledge document formatter ────────────────────────────────────────────

def _readable_title(video_path: Path) -> str:
    """Turn a filename like 'windows7_kontrollpanel.mp4' into 'Windows7 Kontrollpanel'."""
    return video_path.stem.replace("_", " ").replace("-", " ").title()


def format_knowledge_doc(video_path: Path, results) -> str:
    """
    Convert OpenSceneSense results into a knowledge document optimised for
    LightRAG's entity/relationship extraction pipeline.

    Design rationale:
    - No operational metadata (duration, frame counts) — these become junk
      graph triples and dilute extraction quality.
    - Brief summary opens the document as a factual context paragraph. This
      seeds LightRAG's global-mode community summaries without duplicating
      the detailed content.
    - Detailed walkthrough is the primary extraction surface — dense,
      structured, factual prose yields the best entity/relationship recall.
    - Transcript is appended last as a supplementary signal. Raw speech is
      noisy, but qwen3:32b can still mine real entities from it and it
      preserves the instructor's exact instructions.
    - Brief summary is NOT repeated as its own section to avoid duplicate
      graph edges from redundant extraction.
    """
    title = _readable_title(video_path)

    return (
        f"# {title}\n"
        f"\n"
        f"{results.summary.brief}\n"
        f"\n"
        f"## Detaljerad genomgång\n"
        f"\n"
        f"{results.summary.detailed}\n"
        f"\n"
        f"## Transkription (vad som sades)\n"
        f"\n"
        f"{results.summary.transcript}\n"
    )
