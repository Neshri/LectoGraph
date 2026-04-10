"""
SQLite-backed processing state tracker.

One row per video file. Tracks status through the pipeline:
  pending → analyzing → ingested
                      ↘ failed

On startup, any rows stuck at "analyzing" (from a killed process) are
automatically reset to "pending" so they get re-processed.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional


# ─── Status constants ─────────────────────────────────────────────────────────

STATUS_PENDING   = "pending"
STATUS_ANALYZING = "analyzing"
STATUS_INGESTED  = "ingested"
STATUS_FAILED    = "failed"


# ─── State DB ────────────────────────────────────────────────────────────────

class StateDB:
    """Thread-safe SQLite wrapper for tracking video ingestion state."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False is safe here because we only ever use one
        # coroutine at a time (llm_model_max_async=1).
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")  # safer under process kills
        self._init_schema()

    # ── Schema ────────────────────────────────────────────────────────────────

    def _init_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS videos (
                id             INTEGER  PRIMARY KEY AUTOINCREMENT,
                filename       TEXT     UNIQUE NOT NULL,
                status         TEXT     NOT NULL DEFAULT 'pending',
                error_message  TEXT,
                added_at       TEXT     DEFAULT (datetime('now')),
                analyzed_at    TEXT,
                ingested_at    TEXT,
                doc_char_count INTEGER
            )
        """)
        self._conn.commit()

    # ── Discovery ─────────────────────────────────────────────────────────────

    def register_new_videos(self, folder: Path, extensions: set) -> int:
        """
        Scan *folder* and INSERT any video files not already tracked.
        Returns the number of newly registered files.
        """
        new_count = 0
        for p in sorted(folder.iterdir()):
            if p.is_file() and p.suffix.lower() in extensions:
                cur = self._conn.execute(
                    "INSERT OR IGNORE INTO videos (filename) VALUES (?)",
                    (p.name,),
                )
                if cur.rowcount:
                    new_count += 1
        self._conn.commit()
        return new_count

    # ── Recovery ─────────────────────────────────────────────────────────────

    def reset_stuck_analyzing(self) -> int:
        """
        Videos left in 'analyzing' state mean the process was killed mid-flight
        before LightRAG insertion completed. Reset them to 'pending'.
        """
        cur = self._conn.execute(
            "UPDATE videos SET status=?, error_message='Reset: process was killed during analysis' "
            "WHERE status=?",
            (STATUS_PENDING, STATUS_ANALYZING),
        )
        self._conn.commit()
        return cur.rowcount

    def reset_failed(self) -> int:
        """Re-queue all failed videos so they are retried on the next run."""
        cur = self._conn.execute(
            "UPDATE videos SET status=?, error_message=NULL WHERE status=?",
            (STATUS_PENDING, STATUS_FAILED),
        )
        self._conn.commit()
        return cur.rowcount

    def reset_to_pending(self, filename: str) -> bool:
        """
        Reset a single video back to 'pending', clearing timestamps and error.
        Used by the re-ingest flow after the old LightRAG document has been deleted.
        Returns True if the row was found and updated.
        """
        cur = self._conn.execute(
            """UPDATE videos
               SET status=?, error_message=NULL,
                   analyzed_at=NULL, ingested_at=NULL, doc_char_count=NULL
               WHERE filename=?""",
            (STATUS_PENDING, filename),
        )
        self._conn.commit()
        return cur.rowcount > 0

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_pending(self) -> List[str]:
        """Return filenames of all videos queued for processing, sorted by name."""
        rows = self._conn.execute(
            "SELECT filename FROM videos WHERE status=? ORDER BY filename",
            (STATUS_PENDING,),
        ).fetchall()
        return [r["filename"] for r in rows]

    def get_counts(self) -> dict:
        """Return a dict of {status: count} for all tracked videos."""
        rows = self._conn.execute(
            "SELECT status, COUNT(*) AS count FROM videos GROUP BY status"
        ).fetchall()
        return {r["status"]: r["count"] for r in rows}

    def get_all(self) -> list:
        """Return all rows for display in a status table."""
        return self._conn.execute(
            "SELECT filename, status, error_message, ingested_at, doc_char_count "
            "FROM videos ORDER BY filename"
        ).fetchall()

    def get_failed(self) -> list:
        """Return only failed rows."""
        return self._conn.execute(
            "SELECT filename, error_message FROM videos WHERE status=? ORDER BY filename",
            (STATUS_FAILED,),
        ).fetchall()

    # ── State transitions ─────────────────────────────────────────────────────

    def mark_analyzing(self, filename: str) -> None:
        self._conn.execute(
            "UPDATE videos SET status=? WHERE filename=?",
            (STATUS_ANALYZING, filename),
        )
        self._conn.commit()

    def mark_ingested(self, filename: str, doc_char_count: int) -> None:
        now = _utcnow()
        self._conn.execute(
            "UPDATE videos SET status=?, analyzed_at=?, ingested_at=?, doc_char_count=? "
            "WHERE filename=?",
            (STATUS_INGESTED, now, now, doc_char_count, filename),
        )
        self._conn.commit()

    def mark_failed(self, filename: str, error: str) -> None:
        self._conn.execute(
            "UPDATE videos SET status=?, error_message=? WHERE filename=?",
            (STATUS_FAILED, error[:2000], filename),
        )
        self._conn.commit()

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def close(self) -> None:
        self._conn.close()


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
