"""
Core ingestion pipeline.

Responsibilities:
  - Build the OpenSceneSense analyzer (once, shared across all videos)
  - Build the LightRAG instance (once, shared across all videos)
  - Loop over pending videos, analyze → format → save doc → ingest
  - Honour a threading.Event stop signal between videos for clean shutdown
  - Report per-video success/failure back to the caller via StateDB
"""

from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from .config import Config
from .state import StateDB


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


# ─── Analyzer factory ─────────────────────────────────────────────────────────

def build_analyzer(cfg: Config, logger: logging.Logger):
    """
    Construct and return a fully-initialised OllamaVideoAnalyzer.
    Loading Whisper is the expensive part — do this once before the loop.
    """
    from openscenesense_ollama.models import AnalysisPrompts
    from lectograph.transcriber import FasterWhisperAdapter
    from openscenesense_ollama.analyzer import OllamaVideoAnalyzer
    from openscenesense_ollama.frame_selectors import DynamicFrameSelector

    logger.info(f"Loading Whisper model: {cfg.whisper_model} (device={cfg.whisper_device})")

    custom_prompts = AnalysisPrompts(
        frame_analysis=cfg.frame_analysis_prompt,
        detailed_summary=cfg.detailed_summary_prompt,
        brief_summary=cfg.brief_summary_prompt,
    )

    transcriber = FasterWhisperAdapter(
        model_name=cfg.whisper_model,
        device=cfg.whisper_device,
        initial_prompt=cfg.whisper_initial_prompt or None,
        hotwords=cfg.whisper_hotwords or None,
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

    logger.info("Analyzer ready.")
    return analyzer


# ─── LightRAG factory ─────────────────────────────────────────────────────────

async def build_rag(cfg: Config, logger: logging.Logger):
    """Construct and initialise a LightRAG instance."""
    from lightrag import LightRAG
    from lightrag.llm.ollama import ollama_model_complete, ollama_embed
    from lightrag.utils import wrap_embedding_func_with_attrs

    logger.info(
        f"Initialising LightRAG at {cfg.working_dir}  "
        f"(llm={cfg.rag_llm_model}, embed={cfg.rag_embedding_model})"
    )

    @wrap_embedding_func_with_attrs(
        embedding_dim=cfg.rag_embedding_dim,
        max_token_size=8192,
        model_name=cfg.rag_embedding_model,
    )
    async def embedding_func(texts: list[str]) -> np.ndarray:
        return await ollama_embed.func(
            texts,
            embed_model=cfg.rag_embedding_model,
            host=cfg.ollama_url,
        )

    rag = LightRAG(
        working_dir=str(cfg.working_dir),
        llm_model_func=ollama_model_complete,
        llm_model_name=cfg.rag_llm_model,
        llm_model_max_async=1,
        embedding_func=embedding_func,
        enable_llm_cache=False,
        default_embedding_timeout=int(cfg.rag_request_timeout),
        default_llm_timeout=int(cfg.rag_request_timeout),
        llm_model_kwargs={
            "host": cfg.ollama_url,
            "options": {
                "num_ctx": cfg.rag_llm_num_ctx,
                "temperature": cfg.rag_llm_temperature,
            },
            "think": cfg.rag_llm_think,
        },
    )

    await rag.initialize_storages()
    logger.info("LightRAG initialised.")
    return rag


# ─── Main ingestion loop ──────────────────────────────────────────────────────

async def run_ingestion(
    cfg: Config,
    rag,
    analyzer,
    state: StateDB,
    logger: logging.Logger,
    stop_event: threading.Event,
    limit: Optional[int] = None,
    only: Optional[set] = None,
) -> tuple[list[str], list[str]]:
    """
    Process all pending videos.

    Returns (ingested_list, failed_list).

    Checks *stop_event* between videos — if set, exits cleanly after the
    current video finishes so the DB is never left in a dirty state.

    If *only* is provided, restricts processing to that set of filenames,
    ignoring any other pending videos. Used by the reingest flow.
    """
    pending = state.get_pending()
    if only is not None:
        pending = [f for f in pending if f in only]
    if limit is not None:
        pending = pending[:limit]

    total = len(pending)
    ingested: list[str] = []
    failed:   list[str] = []

    if total == 0:
        logger.info("No pending videos — nothing to do.")
        return ingested, failed

    logger.info(f"Starting ingestion of {total} video(s).")

    cfg.docs_dir.mkdir(parents=True, exist_ok=True)

    for idx, filename in enumerate(pending, start=1):
        # ── Check for stop signal before starting a new video ──────────────
        if stop_event.is_set():
            logger.info("Stop signal received. Halting before next video.")
            break

        video_path = cfg.input_folder / filename
        logger.info(f"[{idx}/{total}] ── {filename}")

        if not video_path.exists():
            msg = f"File not found on disk: {video_path}"
            logger.warning(msg)
            state.mark_failed(filename, msg)
            failed.append(filename)
            continue

        # ── Step 1: Analyze ────────────────────────────────────────────────
        state.mark_analyzing(filename)
        logger.info(f"  Analyzing video...")
        try:
            results = analyzer.analyze_video_structured(str(video_path))
        except Exception as exc:
            msg = f"Analysis error: {exc}"
            logger.error(f"  {msg}", exc_info=True)
            state.mark_failed(filename, msg)
            failed.append(filename)
            continue

        # ── Step 1.5: Correct transcript if it contains known-bad terms ───────
        if _transcript_needs_correction(results.summary.transcript):
            if _summaries_are_clean(results.summary.brief, results.summary.detailed):
                logger.info("  Transcript contains known-bad terms; summaries are clean — correcting...")
                results.summary.transcript = await correct_transcript(
                    results.summary.transcript,
                    results.summary.brief,
                    results.summary.detailed,
                    cfg,
                    logger,
                )
            else:
                bad_terms = [
                    t for t in _KNOWN_BAD_TERMS
                    if t.lower() in
                    (results.summary.brief + " " + results.summary.detailed).lower()
                ]
                msg = (
                    f"Bad terms {bad_terms} found in both transcript and summaries — "
                    "summaries unusable as correction reference."
                )
                logger.warning("  %s", msg)
                state.mark_failed(filename, msg)
                failed.append(filename)
                continue

        # ── Step 2: Format + save knowledge document ───────────────────────
        doc = format_knowledge_doc(video_path, results)
        doc_path = cfg.docs_dir / (video_path.stem + "_ingested.txt")
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write(doc)
        logger.info(f"  Document saved → {doc_path}  ({len(doc):,} chars)")

        # ── Step 3: Ingest into LightRAG ───────────────────────────────────
        # Use the video stem as a stable, predictable doc ID so we can
        # reliably delete/replace this document in the future.
        doc_id = video_path.stem
        logger.info(f"  Inserting into LightRAG knowledge graph (id='{doc_id}')...")
        try:
            await rag.ainsert(doc, ids=[doc_id])

            doc_status = await rag.doc_status.get_by_id(doc_id)
            if doc_status and doc_status.get("status") == "failed":
                msg = f"LightRAG ingestion failed internally: {doc_status.get('error_msg', 'Unknown error')}"
                logger.error(f"  {msg}")
                state.mark_failed(filename, msg)
                failed.append(filename)
                continue

        except Exception as exc:
            msg = f"Ingestion error: {exc}"
            logger.error(f"  {msg}", exc_info=True)
            state.mark_failed(filename, msg)
            failed.append(filename)
            continue

        # ── Commit success ─────────────────────────────────────────────────
        state.mark_ingested(filename, len(doc))
        ingested.append(filename)
        logger.info(f"  ✓ Ingested successfully.")

    return ingested, failed


# ─── Faulty document detection ────────────────────────────────────────────────

# CJK Unified Ideographs (basic block, covers virtually all common Chinese chars)
_CJK_RANGE = range(0x4E00, 0xA000)

def _contains_cjk(text: str) -> bool:
    return any(ord(ch) in _CJK_RANGE for ch in text)


# Known Whisper mishearings for this course.
# Key = bad term Whisper wrote, value = what was actually said (documentation only).
# Add new entries here as they are discovered.
_KNOWN_BAD_TERMS: dict[str, str] = {
    "DOCP":   "DHCP",    # teacher's skånska accent
    "comfig": "config",  # teacher's skånska accent
}

# Pre-compiled whole-word, case-insensitive patterns (built once at import time).
_BAD_TERM_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
    for term in _KNOWN_BAD_TERMS
]


def _is_faulty(text: str) -> bool:
    """Return True if *text* contains CJK characters or any known-bad Whisper term."""
    if _contains_cjk(text):
        return True
    return any(p.search(text) for p in _BAD_TERM_PATTERNS)


def _transcript_needs_correction(transcript: str) -> bool:
    """Return True if the raw Whisper transcript contains any known-bad term."""
    return any(p.search(transcript) for p in _BAD_TERM_PATTERNS)


def _summaries_are_clean(brief: str, detailed: str) -> bool:
    """Return True if neither summary contains a known-bad term.

    A clean summary can be used as a trusted reference to correct the transcript.
    If the summaries are also contaminated the LLM correction path is not safe to use.
    """
    combined = brief + " " + detailed
    return not any(p.search(combined) for p in _BAD_TERM_PATTERNS)


async def correct_transcript(
    transcript: str,
    brief: str,
    detailed: str,
    cfg: Config,
    logger: logging.Logger,
) -> str:
    """
    Ask the summary LLM to find and fix Whisper mishearings in *transcript*,
    using *brief* and *detailed* summaries as a trusted factual reference.

    The LLM is prompted to return a JSON object with a single ``replacements``
    list of ``{"wrong": "...", "right": "..."}`` pairs.  Each pair is applied
    as a whole-word, case-insensitive substitution.

    Fail-safe: if the LLM call fails or returns malformed JSON the original
    transcript is returned unchanged so ingestion is never blocked.
    """
    from lightrag.llm.ollama import ollama_model_complete

    prompt = cfg.transcript_correction_prompt.format(
        brief=brief,
        detailed=detailed,
        transcript=transcript,
    )

    try:
        raw: str = await ollama_model_complete(
            prompt,
            model=cfg.summary_model,
            host=cfg.ollama_url,
            options={"temperature": 0.0},
        )
    except Exception as exc:
        logger.warning(
            "correct_transcript: LLM call failed (%s) — using original transcript.", exc
        )
        return transcript

    # Strip markdown code fences if the model wrapped its JSON.
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()

    try:
        data = json.loads(raw)
        replacements: list[dict] = data.get("replacements", [])
    except (json.JSONDecodeError, AttributeError) as exc:
        logger.warning(
            "correct_transcript: could not parse LLM response as JSON (%s) — using original.",
            exc,
        )
        logger.debug("Raw LLM response: %s", raw)
        return transcript

    if not replacements:
        logger.info("  Transcript correction: no replacements suggested.")
        return transcript

    corrected = transcript
    for item in replacements:
        wrong = item.get("wrong", "")
        right = item.get("right", "")
        if not wrong or not right:
            continue
        pattern = re.compile(r"\b" + re.escape(wrong) + r"\b", re.IGNORECASE)
        corrected, n = pattern.subn(right, corrected)
        if n:
            logger.info(
                "  Transcript correction: '%s' → '%s' (%d occurrence(s))", wrong, right, n
            )

    return corrected


def detect_faulty_docs(docs_dir: Path, state: StateDB) -> list[str]:
    """
    Scan every saved knowledge-document in *docs_dir* for known quality issues:
    - CJK (Chinese) characters — LLM hallucination artefact
    - Known Whisper mishearings (see ``_KNOWN_BAD_TERMS``)

    Returns a list of video *filenames* (e.g. 'cisco23.mp4') whose saved
    ``_ingested.txt`` is faulty.  Only files tracked in the state DB as
    'ingested' are considered.
    """
    faulty: list[str] = []

    if not docs_dir.exists():
        return faulty

    # Build a quick lookup: stem → filename for all ingested rows.
    all_rows = state.get_all()
    stem_to_filename = {
        Path(r["filename"]).stem: r["filename"]
        for r in all_rows
        if r["status"] == "ingested"
    }

    for doc_path in sorted(docs_dir.glob("*_ingested.txt")):
        # Strip the "_ingested" suffix to recover the video stem.
        stem = doc_path.stem[: -len("_ingested")]
        if stem not in stem_to_filename:
            continue  # not in DB (orphan file), skip

        try:
            content = doc_path.read_text(encoding="utf-8")
        except OSError:
            continue

        if _is_faulty(content):
            faulty.append(stem_to_filename[stem])

    return faulty

# ─── Re-ingest pipeline ───────────────────────────────────────────────────────

async def run_reingest(
    cfg: Config,
    rag,
    analyzer,
    state: StateDB,
    logger: logging.Logger,
    filenames: list[str],
    stop_event: threading.Event,
) -> tuple[list[str], list[str]]:
    """
    Delete *filenames* from the LightRAG knowledge graph and re-queue them.

    For each video:
      1. Read the saved ``docs/<stem>_ingested.txt`` file.
      2. Derive the LightRAG doc ID via the same MD5 hash LightRAG uses.
      3. Call ``rag.adelete_by_doc_id()`` — LightRAG rebuilds the KG automatically.
      4. Remove the stale txt file so the next analysis writes a fresh one.
      5. Call ``state.reset_to_pending()`` to re-queue the video.

    Returns (deleted_list, skipped_list) from the deletion phase.
    After this call, run the normal ``run_ingestion`` loop to process pending videos.
    """
    deleted:  list[str] = []
    skipped:  list[str] = []

    for filename in filenames:
        if stop_event.is_set():
            logger.info("Stop signal received during deletion phase. Halting.")
            break

        stem     = Path(filename).stem
        doc_path = cfg.docs_dir / f"{stem}_ingested.txt"

        if not doc_path.exists():
            logger.warning(f"  [reingest] No saved doc found for {filename} — skipping deletion, but resetting to pending.")
            state.reset_to_pending(filename)
            skipped.append(filename)
            continue

        content = doc_path.read_text(encoding="utf-8")
        # The doc ID is the video stem — set explicitly at insert time.
        doc_id  = Path(filename).stem

        logger.info(f"  [reingest] Deleting doc '{doc_id}' for {filename} from LightRAG…")
        try:
            await rag.adelete_by_doc_id(doc_id)
        except Exception as exc:
            logger.error(f"  [reingest] Delete failed for {filename}: {exc}", exc_info=True)
            skipped.append(filename)
            continue

        # Remove stale saved document so pipeline writes a fresh one.
        doc_path.unlink(missing_ok=True)
        logger.info(f"  [reingest] Removed stale doc file: {doc_path.name}")

        # Reset DB entry so the normal ingestion loop picks it up.
        state.reset_to_pending(filename)
        logger.info(f"  [reingest] Reset '{filename}' → pending.")
        deleted.append(filename)

    return deleted, skipped

