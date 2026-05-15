#!/usr/bin/env python3
"""
LectoGraph — batch video → LightRAG ingestion CLI.

Usage examples:
  python ingest.py                      # process all pending videos
  python ingest.py --status             # show DB state table and exit
  python ingest.py --rag-status         # inspect LightRAG doc_status (non-processed entries)
  python ingest.py --rag-purge-failed   # delete all FAILED entries from LightRAG doc_status
  python ingest.py --dry-run            # discover videos, show plan, don't process
  python ingest.py --retry-failed       # re-queue failed videos, then process
  python ingest.py --limit 2            # process at most 2 videos (useful for testing)
  python ingest.py --config other.yaml  # use a different config file

Press Ctrl+C at any time. The current video will finish, then ingestion stops
cleanly — all completed videos remain in the DB and will be skipped next run.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
import threading
from datetime import datetime
from pathlib import Path


# ─── CLI argument parsing ─────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch-ingest lecture videos into a LightRAG knowledge graph.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--config", default="./config.yaml", metavar="PATH",
        help="Path to config.yaml  (default: ./config.yaml)",
    )
    p.add_argument(
        "--status", action="store_true",
        help="Print processing state table and exit.",
    )
    p.add_argument(
        "--retry-failed", action="store_true",
        help="Re-queue all failed videos as pending, then run ingestion.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Discover videos and show the ingestion plan without processing anything.",
    )
    p.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Process at most N videos this run.",
    )
    p.add_argument(
        "--reset-db", action="store_true",
        help=(
            "Delete the ingestion state database and start completely fresh. "
            "Does NOT delete the LightRAG knowledge graph — use this when you want "
            "to re-ingest everything into an existing or new graph."
        ),
    )
    # ── Re-ingest faulty docs ──────────────────────────────────────────────────
    p.add_argument(
        "--detect-faulty", action="store_true",
        help=(
            "Scan docs/ for saved documents containing known quality issues "
            "(Chinese characters, known Whisper mishearings like DOCP/comfig). "
            "Fast — does not start LightRAG."
        ),
    )
    p.add_argument(
        "--reingest", nargs="+", metavar="FILE",
        help=(
            "Delete the named video(s) from LightRAG and re-ingest them. "
            "Example: --reingest cisco23.mp4 cisco24.mp4"
        ),
    )
    p.add_argument(
        "--reingest-faulty", action="store_true",
        help=(
            "Automatically detect documents with known quality issues (Chinese "
            "characters, known Whisper mishearings), remove them from LightRAG, "
            "and re-ingest them."
        ),
    )
    p.add_argument(
        "--rag-status", action="store_true",
        help=(
            "Query LightRAG's internal doc_status store. Shows all non-processed "
            "entries (pending, failed, processing) with full error messages. "
            "Useful for inspecting failures invisible to --status."
        ),
    )
    p.add_argument(
        "--rag-purge-failed", action="store_true",
        help=(
            "Delete all FAILED entries from LightRAG's doc_status store. "
            "Safe to run: these are stale ghost records (e.g. duplicate detection "
            "artifacts) that no longer serve a purpose."
        ),
    )
    return p.parse_args()


# ─── Logging setup ────────────────────────────────────────────────────────────

def setup_logging(logs_dir: Path) -> logging.Logger:
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = logs_dir / f"ingest_{timestamp}.log"

    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(fmt)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)

    logger = logging.getLogger("ingest")
    logger.info(f"Log file: {log_file.resolve()}")
    return logger


# ─── Status table printer ─────────────────────────────────────────────────────

def print_status(state, working_dir: Path | None = None) -> None:
    rows = state.get_all()
    counts = state.get_counts()

    # ── LightRAG cross-reference ──────────────────────────────────────────────
    rag_doc_ids: set[str] | None = None
    if working_dir is not None:
        doc_store = working_dir / "kv_store_full_docs.json"
        if doc_store.exists():
            import json
            rag_doc_ids = set(json.loads(doc_store.read_text(encoding="utf-8")).keys())

    col_w = 55
    print()
    print(f"{'Filename':<{col_w}}  {'Status':<10}  {'Ingested at':<20}  {'Doc chars':>10}  {'In LightRAG':<12}  Error")
    print("─" * 135)
    for r in rows:
        err = (r["error_message"] or "")[:60]
        ingested_at = r["ingested_at"] or ""
        doc_chars = f"{r['doc_char_count']:,}" if r["doc_char_count"] else ""

        if rag_doc_ids is None:
            in_rag = "?"
        elif r["status"] != "ingested":
            in_rag = "-"
        else:
            stem = Path(r["filename"]).stem
            in_rag = "✓" if stem in rag_doc_ids else "⚠ MISSING"

        print(f"{r['filename']:<{col_w}}  {r['status']:<10}  {ingested_at:<20}  {doc_chars:>10}  {in_rag:<12}  {err}")

    print("─" * 135)
    print(
        f"  Total: {sum(counts.values())}  |  "
        + "  ".join(f"{s}: {n}" for s, n in sorted(counts.items()))
    )
    if rag_doc_ids is not None:
        print(f"  LightRAG docs in store: {len(rag_doc_ids)}")
    print()


# ─── LightRAG doc_status inspector & maintenance ─────────────────────────────

async def print_rag_status(rag) -> None:
    """Query LightRAG's internal doc_status store and print non-processed entries.

    Skips PROCESSED entries — they're working correctly and would flood the output.
    get_docs_by_status returns dict[str, DocProcessingStatus] dataclass objects.
    """
    from lightrag.base import DocStatus

    # Fetch all non-processed statuses; values are DocProcessingStatus dataclass instances
    status_groups: dict[str, list] = {}
    for status in DocStatus:
        if status == DocStatus.PROCESSED:
            continue
        docs = await rag.doc_status.get_docs_by_status(status)
        if docs:
            status_groups[status.value] = list(docs.items())  # [(doc_id, DocProcessingStatus)]

    # Fetch the processed count separately just for the summary line
    processed_docs = await rag.doc_status.get_docs_by_status(DocStatus.PROCESSED)
    processed_count = len(processed_docs)

    if not status_groups and processed_count == 0:
        print("\nLightRAG doc_status store is empty.\n")
        return

    shown = sum(len(v) for v in status_groups.values())
    print(f"\nLightRAG doc_status — {shown + processed_count} total  ({processed_count} processed, {shown} non-processed)\n")

    if not status_groups:
        print("  No non-processed entries found.\n")
        return

    for status_name, entries in sorted(status_groups.items()):
        print(f"  {'─' * 70}")
        print(f"  STATUS: {status_name.upper()}  ({len(entries)} entries)")
        print(f"  {'─' * 70}")
        for doc_id, doc in entries:
            file_path  = getattr(doc, "file_path", None) or "<unknown source>"
            summary    = getattr(doc, "content_summary", None) or ""
            error      = getattr(doc, "error_msg", None) or ""
            updated_at = getattr(doc, "updated_at", None) or ""
            metadata   = getattr(doc, "metadata", None) or {}
            is_dup     = metadata.get("is_duplicate", False)

            print(f"    ID      : {doc_id}")
            print(f"    File    : {file_path}")
            if summary:
                print(f"    Summary : {summary}")
            if updated_at:
                print(f"    Updated : {updated_at}")
            if is_dup:
                orig = metadata.get("original_doc_id", "?")
                print(f"    Kind    : duplicate ghost (original: {orig})")
            if error:
                print(f"    Error   : {error}")
            print()

    counts = {s: len(e) for s, e in status_groups.items()}
    counts["processed"] = processed_count
    print("  Summary: " + "  ".join(
        f"{s}: {n}" for s, n in sorted(counts.items())
    ) + "\n")


async def purge_rag_failed(rag) -> int:
    """Delete all FAILED entries from LightRAG's doc_status store.

    Returns the number of entries deleted.
    """
    from lightrag.base import DocStatus

    failed_docs = await rag.doc_status.get_docs_by_status(DocStatus.FAILED)
    if not failed_docs:
        print("\nNo FAILED entries in LightRAG doc_status — nothing to purge.\n")
        return 0

    ids = list(failed_docs.keys())
    await rag.doc_status.delete(ids)
    await rag.doc_status.index_done_callback()
    print(f"\nPurged {len(ids)} FAILED entries from LightRAG doc_status.\n")
    return len(ids)


# ─── Main async entry point ───────────────────────────────────────────────────

async def main_async(args: argparse.Namespace) -> int:
    # ── Load config ───────────────────────────────────────────────────────────
    from lectograph.config import Config
    try:
        cfg = Config.from_yaml(Path(args.config))
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    # ── Logging (needs logs_dir from config) ──────────────────────────────────
    logger = setup_logging(cfg.logs_dir)
    logger.info("LectoGraph ingestion pipeline starting.")
    logger.info("Configuration:\n" + cfg.summary())

    # ── Open state DB ─────────────────────────────────────────────────────────
    from lectograph.state import StateDB
    cfg.working_dir.mkdir(parents=True, exist_ok=True)
    state = StateDB(cfg.working_dir / "ingestion_state.db")

    # ── Register any new videos in the input folder ───────────────────────────
    if not cfg.input_folder.exists():
        logger.error(f"Input folder does not exist: {cfg.input_folder.resolve()}")
        state.close()
        return 1

    new_count = state.register_new_videos(cfg.input_folder, cfg.video_extensions_set)
    if new_count:
        logger.info(f"Registered {new_count} new video(s) in the database.")

    # ── Reset videos stuck mid-analysis from a previous killed run ────────────
    stuck = state.reset_stuck_analyzing()
    if stuck:
        logger.warning(
            f"Reset {stuck} video(s) that were stuck in 'analyzing' state "
            f"(the process was killed mid-flight last time)."
        )

    # ── --reset-db ────────────────────────────────────────────────────────────
    if args.reset_db:
        db_path = cfg.working_dir / "ingestion_state.db"
        state.close()
        if db_path.exists():
            db_path.unlink()
            logger.info(f"Deleted ingestion state DB: {db_path}")
        else:
            logger.info("No state DB found — nothing to reset.")
        # Re-open a fresh DB so the rest of the run works normally
        state = StateDB(cfg.working_dir / "ingestion_state.db")
        logger.info("Fresh state DB created. All videos will be re-ingested.")

    # ── --retry-failed ────────────────────────────────────────────────────────
    if args.retry_failed:
        requeued = state.reset_failed()
        logger.info(f"Re-queued {requeued} previously failed video(s).")

    # ── --detect-faulty ───────────────────────────────────────────────────────
    if args.detect_faulty:
        from lectograph.quality import detect_faulty_docs
        faulty = detect_faulty_docs(cfg.docs_dir, state)
        if faulty:
            print(f"\nFound {len(faulty)} document(s) with quality issues (CJK chars or known mishearings):")
            for name in faulty:
                print(f"  • {name}")
            print("\nRun with --reingest-faulty to delete and re-ingest them.\n")
        else:
            print("\nNo faulty documents detected (no CJK characters or known mishearings found).\n")
        state.close()
        return 0

    # ── --reingest / --reingest-faulty ────────────────────────────────────────
    reingest_targets: list[str] | None = None

    if args.reingest_faulty:
        from lectograph.quality import detect_faulty_docs
        reingest_targets = detect_faulty_docs(cfg.docs_dir, state)

        if not reingest_targets:
            logger.info("No faulty documents detected — nothing to re-ingest.")
            state.close()
            return 0
        logger.info(
            f"Detected {len(reingest_targets)} faulty document(s): "
            + ", ".join(reingest_targets)
        )
    elif args.reingest:
        reingest_targets = args.reingest
        logger.info(f"Re-ingest requested for: {', '.join(reingest_targets)}")


    # ── --status ──────────────────────────────────────────────────────────────
    if args.status:
        print_status(state, cfg.working_dir)
        state.close()
        return 0

    # ── --rag-status ──────────────────────────────────────────────────────────
    if args.rag_status:
        from lectograph.factories import build_rag
        try:
            rag = await build_rag(cfg, logger)
        except Exception as e:
            logger.error(f"Failed to initialise LightRAG: {e}", exc_info=True)
            state.close()
            return 1
        await print_rag_status(rag)
        state.close()
        return 0

    # ── --rag-purge-failed ────────────────────────────────────────────────────
    if args.rag_purge_failed:
        from lectograph.factories import build_rag
        try:
            rag = await build_rag(cfg, logger)
        except Exception as e:
            logger.error(f"Failed to initialise LightRAG: {e}", exc_info=True)
            state.close()
            return 1
        await purge_rag_failed(rag)
        state.close()
        return 0

    # ── --dry-run ─────────────────────────────────────────────────────────────
    if args.dry_run:
        pending = state.get_pending()
        counts = state.get_counts()
        print()
        print(f"Dry run — {len(pending)} video(s) would be processed:")
        for name in pending:
            print(f"  • {name}")
        print()
        print("Current state:", "  ".join(f"{s}={n}" for s, n in sorted(counts.items())))
        print()
        state.close()
        return 0

    # ── Bail early if nothing to do ───────────────────────────────────────────
    # Skip this check when reingest_targets is set: the videos will be reset to
    # 'pending' by run_reingest (inside the deletion phase), which hasn't run yet.
    pending = state.get_pending()
    if not pending and not reingest_targets:
        logger.info("No pending videos. Run with --retry-failed to requeue failures.")
        print_status(state, cfg.working_dir)
        state.close()
        return 0

    if pending:
        if reingest_targets:
            logger.info(f"{len(pending)} other video(s) already pending (will not be touched by reingest).")
        else:
            logger.info(f"{len(pending)} video(s) pending.")


    # ── Set up Ctrl+C / SIGTERM handler ───────────────────────────────────────
    stop_event = threading.Event()

    def _request_stop(signum, frame):  # called from the OS signal thread
        if not stop_event.is_set():
            print(
                "\n[!] Stop requested. Finishing the current video, then exiting...\n"
                "    Press Ctrl+C again to force quit immediately.\n",
                flush=True,
            )
            stop_event.set()
            # Restore the default SIGINT handler so a second Ctrl+C actually
            # kills the process instead of being swallowed.
            signal.signal(signal.SIGINT, signal.SIG_DFL)

    signal.signal(signal.SIGINT, _request_stop)
    # SIGTERM is not reliably available on Windows; ignore if unsupported.
    try:
        signal.signal(signal.SIGTERM, _request_stop)
    except (OSError, ValueError):
        pass

    # ── Build analyzer (loads Whisper — takes a moment) ───────────────────────
    from lectograph.factories import build_analyzer, build_rag
    from lectograph.pipeline import run_ingestion, run_reingest
    try:
        analyzer = build_analyzer(cfg, logger)
    except Exception as e:
        logger.error(f"Failed to build analyzer: {e}", exc_info=True)
        state.close()
        return 1

    # ── Build LightRAG ────────────────────────────────────────────────────────
    try:
        rag = await build_rag(cfg, logger)
    except Exception as e:
        logger.error(f"Failed to initialise LightRAG: {e}", exc_info=True)
        state.close()
        return 1

    # ── Run deletion phase (reingest only) ────────────────────────────────────
    if reingest_targets:
        logger.info("=" * 70)
        logger.info(f"Phase 1/2 — Deleting {len(reingest_targets)} document(s) from LightRAG…")
        deleted, skipped = await run_reingest(
            cfg=cfg,
            rag=rag,
            analyzer=analyzer,
            state=state,
            logger=logger,
            filenames=reingest_targets,
            stop_event=stop_event,
        )
        logger.info(
            f"Deletion phase complete.  Deleted: {len(deleted)}  Skipped: {len(skipped)}"
        )
        if skipped:
            logger.warning("Skipped (delete failed): " + ", ".join(skipped))
        logger.info("=" * 70)
        logger.info("Phase 2/2 — Re-ingesting newly queued video(s)…")

    # ── Run the ingestion loop ─────────────────────────────────────────────────
    logger.info("=" * 70)
    ingested, failed = await run_ingestion(
        cfg=cfg,
        rag=rag,
        analyzer=analyzer,
        state=state,
        logger=logger,
        stop_event=stop_event,
        limit=args.limit,
        only=set(deleted) if reingest_targets else None,
    )

    # ── Final summary ─────────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info(f"Run complete.  Ingested: {len(ingested)}  Failed: {len(failed)}")
    if failed:
        logger.warning("Failed videos (run with --retry-failed to requeue):")
        for name in failed:
            logger.warning(f"  ✗ {name}")

    print_status(state, cfg.working_dir)
    state.close()

    return 0 if not failed else 2  # exit code 2 = partial failure


# ─── Entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    args = parse_args()
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
