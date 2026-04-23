#!/usr/bin/env python3
"""
repair_faulty_docs.py — Remove documents containing Chinese characters from LightRAG.

Reads the actual doc IDs directly from kv_store_full_docs.json (LightRAG's
own storage), so there is no hash-guessing involved.

Usage:
  python repair_faulty_docs.py            # dry-run: list faulty doc IDs
  python repair_faulty_docs.py --delete   # actually delete them
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from lectograph.pipeline import _is_faulty

# ─── Main ──────────────────────────────────────────────────────────────────────

async def main(delete: bool) -> int:
    from lectograph.config import Config
    from lectograph.pipeline import build_rag

    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("repair")

    cfg = Config.from_yaml(Path("./config.yaml"))

    # ── Read full_docs store directly ──────────────────────────────────────────
    full_docs_path = cfg.working_dir / "kv_store_full_docs.json"
    if not full_docs_path.exists():
        logger.error(f"Not found: {full_docs_path}")
        return 1

    with open(full_docs_path, encoding="utf-8") as f:
        full_docs: dict = json.load(f)

    logger.info(f"Loaded {len(full_docs)} documents from {full_docs_path.name}")

    # ── Find faulty doc IDs ────────────────────────────────────────────────────
    faulty: list[tuple[str, str]] = []  # (doc_id, content_preview)
    for doc_id, entry in full_docs.items():
        # LightRAG stores either the raw string or a dict with a 'content' key
        content = entry if isinstance(entry, str) else entry.get("content", "")
        if _is_faulty(content):
            preview = content[:80].replace("\n", " ")
            faulty.append((doc_id, preview))

    if not faulty:
        print("\nNo faulty documents found in LightRAG. Nothing to do.\n")
        return 0

    print(f"\nFound {len(faulty)} faulty document(s):\n")
    for doc_id, preview in faulty:
        print(f"  ID : {doc_id}")
        print(f"       {preview!r}\n")

    if not delete:
        print("Dry run — run with --delete to remove them.\n")
        return 0

    # ── Delete from LightRAG ───────────────────────────────────────────────────
    logger.info("Initialising LightRAG…")
    rag = await build_rag(cfg, logger)

    deleted = 0
    for doc_id, preview in faulty:
        logger.info(f"Deleting {doc_id}…")
        try:
            await rag.adelete_by_doc_id(doc_id)
            deleted += 1
            logger.info(f"  ✓ Deleted")
        except Exception as exc:
            logger.error(f"  ✗ Failed: {exc}", exc_info=True)

    await rag.finalize_storages()

    print(f"\nDeleted {deleted}/{len(faulty)} faulty documents from LightRAG.\n")

    # ── Reset state DB entries ─────────────────────────────────────────────────
    # Cross-reference by scanning docs/ for txt files that contained CJK
    # (already deleted from disk), or by matching to the DB via the doc content.
    # Since txt files are gone, report which videos need manual re-queue.
    from lectograph.state import StateDB
    state = StateDB(cfg.working_dir / "ingestion_state.db")
    all_rows = state.get_all()

    # Build a set of stems that are already pending (reset earlier by reingest)
    pending_stems = {
        Path(r["filename"]).stem
        for r in all_rows
        if r["status"] == "pending"
    }
    ingested_stems = {
        Path(r["filename"]).stem: r["filename"]
        for r in all_rows
        if r["status"] == "ingested"
    }

    if pending_stems:
        print(
            "The following videos were already reset to pending (by the earlier "
            "--reingest-faulty run) and will be re-ingested on the next run:\n"
        )
        for stem in sorted(pending_stems):
            print(f"  • {stem}")
        print()

    state.close()
    return 0 if deleted == len(faulty) else 1


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--delete", action="store_true",
        help="Actually delete the faulty documents. Without this flag, only prints them.",
    )
    return p.parse_args()


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    args = _parse_args()
    sys.exit(asyncio.run(main(args.delete)))
