"""
Core ingestion pipeline.

Responsibilities:
  - Loop over pending videos, analyze → save doc (always persisted) → correct → ingest
  - Honour a threading.Event stop signal between videos for clean shutdown
  - Report per-video success/failure back to the caller via StateDB
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

from .config import Config
from .state import StateDB
from .document import format_knowledge_doc
from .quality import (
    _transcript_needs_correction,
    _summaries_are_clean,
    correct_transcript,
    get_bad_terms,
)


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

        # ── Step 2: Format + save knowledge document ───────────────────────
        doc = format_knowledge_doc(video_path, results)
        doc_path = cfg.docs_dir / (video_path.stem + "_ingested.txt")
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write(doc)
        logger.info(f"  Document saved → {doc_path.name}  ({len(doc):,} chars)")

        # ── Step 3: Correct transcript if it contains known-bad terms ───────
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
                # Re-format and re-save corrected document
                doc = format_knowledge_doc(video_path, results)
                with open(doc_path, "w", encoding="utf-8") as f:
                    f.write(doc)
                logger.info(f"  Document updated (corrected) → {doc_path.name}")
            else:
                bad_terms = get_bad_terms(results.summary.brief + " " + results.summary.detailed)
                msg = (
                    f"Bad terms {bad_terms} found in both transcript and summaries — "
                    "summaries unusable as correction reference."
                )
                logger.warning("  %s", msg)
                state.mark_failed(filename, msg)
                failed.append(filename)
                continue

        # ── Step 4: Ingest into LightRAG ───────────────────────────────────
        # Use the video stem as a stable, predictable doc ID so we can
        # reliably delete/replace this document in the future.
        doc_id = video_path.stem
        logger.info(f"  Inserting into LightRAG knowledge graph (id='{doc_id}')...")
        try:
            # Check if LightRAG already has a record for this doc_id (e.g. from
            # a previous failed attempt) and clear it so we don't get 
            # 'Duplicate document detected' warnings that block ingestion.
            old_status = await rag.doc_status.get_by_id(doc_id)
            if old_status:
                status_str = old_status.get("status", "unknown")
                logger.info(f"  Clearing existing LightRAG record for '{doc_id}' (previous status: {status_str})")
                await rag.adelete_by_doc_id(doc_id)

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
