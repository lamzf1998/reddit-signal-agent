"""Novelty registry — the thing that makes "new" mean new.

A tiny SQLite table of already-surfaced entities (tool / model / workflow
names) and posts. Without it the digest re-surfaces the same tool forever.
"""
from __future__ import annotations

import re
import sqlite3
from contextlib import closing

from . import config

_NORM_RE = re.compile(r"[^a-z0-9]+")


def _connect() -> sqlite3.Connection:
    config.STATE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.SEEN_DB)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS seen (
               key TEXT PRIMARY KEY,
               kind TEXT,
               first_seen TEXT,
               seen_count INTEGER DEFAULT 1
           )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS processed (
               raw_id TEXT PRIMARY KEY,
               processed_at TEXT
           )"""
    )
    return conn


def normalize(name: str) -> str:
    return _NORM_RE.sub("-", name.lower()).strip("-")


def is_new(key: str) -> bool:
    key = normalize(key)
    if not key:
        return False
    with closing(_connect()) as conn:
        row = conn.execute("SELECT 1 FROM seen WHERE key = ?", (key,)).fetchone()
        return row is None


def is_processed(raw_id: str) -> bool:
    """True if this post has already been through the extractor (any outcome)."""
    if not raw_id:
        return False
    with closing(_connect()) as conn:
        return conn.execute(
            "SELECT 1 FROM processed WHERE raw_id = ?", (raw_id,)
        ).fetchone() is not None


def mark_processed(raw_id: str, date: str) -> None:
    if not raw_id:
        return
    with closing(_connect()) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO processed (raw_id, processed_at) VALUES (?, ?)",
            (raw_id, date),
        )
        conn.commit()


def record(key: str, kind: str, date: str) -> None:
    key = normalize(key)
    if not key:
        return
    with closing(_connect()) as conn:
        conn.execute(
            """INSERT INTO seen (key, kind, first_seen, seen_count)
               VALUES (?, ?, ?, 1)
               ON CONFLICT(key) DO UPDATE SET seen_count = seen_count + 1""",
            (key, kind, date),
        )
        conn.commit()
