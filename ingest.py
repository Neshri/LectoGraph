#!/usr/bin/env python3
"""
LectoGraph — batch video → LightRAG ingestion CLI.

Usage examples:
  python ingest.py                      # process all pending videos
  python ingest.py --status             # show DB state table and exit
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

def print_status(state) -> None:
    rows = state.get_all()
    counts = state.get_counts()

    col_w = 55  # filename column width
    print()
    print(f"{'Filename':<{col_w}}  {'Status':<10}  {'Ingested at':<20}  {'Doc chars':>10}  Error")
    print("─" * 120)
    for r in rows:
        err = (r["error_message"] or "")[:60]
        ingested_at = r["ingested_at"] or ""
        doc_chars = f"{r['doc_char_count']:,}" if r["doc_char_count"] else ""
        print(f"{r['filename']:<{col_w}}  {r['status']:<10}  {ingested_at:<20}  {doc_chars:>10}  {err}")

    print("─" * 120)
    print(
        f"  Total: {sum(counts.values())}  |  "
        + "  ".join(f"{s}: {n}" for s, n in sorted(counts.items()))
    )
    print()


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


    # ── --status ──────────────────────────────────────────────────────────────
    if args.status:
        print_status(state)
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
    pending = state.get_pending()
    if not pending:
        logger.info("No pending videos. Run with --retry-failed to requeue failures.")
        print_status(state)
        state.close()
        return 0

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
    from lectograph.pipeline import build_analyzer, build_rag, run_ingestion
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
    )

    # ── Final summary ─────────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info(f"Run complete.  Ingested: {len(ingested)}  Failed: {len(failed)}")
    if failed:
        logger.warning("Failed videos (run with --retry-failed to requeue):")
        for name in failed:
            logger.warning(f"  ✗ {name}")

    print_status(state)
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
