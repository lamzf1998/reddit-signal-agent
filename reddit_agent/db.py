"""Persistent store of every analysed post.

Reuses the same SQLite file as the seen/processed registries. One row per post
(a post is analysed once), capturing the extraction result and whether it was
sent to Telegram. Powers the local dashboard.
"""
from __future__ import annotations

import datetime
import json
import sqlite3
from contextlib import closing

from . import config


def _connect() -> sqlite3.Connection:
    config.STATE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.SEEN_DB)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE IF NOT EXISTS analyses (
               raw_id        TEXT PRIMARY KEY,
               track         TEXT,
               subreddit     TEXT,
               title         TEXT,
               permalink     TEXT,
               source_url    TEXT,
               relevant      INTEGER,
               confidence    REAL,
               importance    TEXT,
               matches_prefs INTEGER,
               sent          INTEGER,
               corpus_date   TEXT,
               analysed_at   TEXT,
               extraction    TEXT
           )"""
    )
    return conn


def save_analysis(post, ex, confidence: float, sent: bool, corpus_date: str) -> None:
    """Upsert one analysed post + its extraction."""
    data = ex.model_dump()
    with closing(_connect()) as conn:
        conn.execute(
            """INSERT INTO analyses
               (raw_id, track, subreddit, title, permalink, source_url,
                relevant, confidence, importance, matches_prefs, sent,
                corpus_date, analysed_at, extraction)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(raw_id) DO UPDATE SET
                 sent = MAX(analyses.sent, excluded.sent),
                 confidence = excluded.confidence,
                 extraction = excluded.extraction""",
            (
                post.raw_id, post.track, post.subreddit, post.title, post.permalink,
                (post.links[0] if post.links else ""),
                int(bool(data.get("relevant"))), float(confidence),
                data.get("importance", ""), int(bool(data.get("matches_prefs"))),
                int(bool(sent)), corpus_date,
                datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                json.dumps(data, ensure_ascii=False),
            ),
        )
        conn.commit()


def fetch_analyses(limit: int = 500, track: str | None = None,
                   sent_only: bool = False) -> list[dict]:
    """Return analysed posts, newest first, with the extraction inlined."""
    q = "SELECT * FROM analyses"
    where, args = [], []
    if track:
        where.append("track = ?"); args.append(track)
    if sent_only:
        where.append("sent = 1")
    if where:
        q += " WHERE " + " AND ".join(where)
    q += " ORDER BY analysed_at DESC LIMIT ?"; args.append(limit)

    with closing(_connect()) as conn:
        rows = conn.execute(q, args).fetchall()

    out = []
    for r in rows:
        d = dict(r)
        d["extraction"] = json.loads(d["extraction"]) if d["extraction"] else {}
        d["relevant"] = bool(d["relevant"])
        d["matches_prefs"] = bool(d["matches_prefs"])
        d["sent"] = bool(d["sent"])
        out.append(d)
    return out


def export_json(path) -> None:
    """Write a static snapshot (for GitHub Pages / any static host)."""
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "relevance_threshold": config.RELEVANCE_THRESHOLD,
        "signals": fetch_analyses(limit=2000),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def stats() -> dict:
    with closing(_connect()) as conn:
        total = conn.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
        sent = conn.execute("SELECT COUNT(*) FROM analyses WHERE sent = 1").fetchone()[0]
        by_track = {
            row["track"]: row["n"] for row in conn.execute(
                "SELECT track, COUNT(*) n FROM analyses WHERE sent = 1 GROUP BY track"
            ).fetchall()
        }
    return {
        "total_analysed": total, "total_sent": sent, "sent_by_track": by_track,
        "relevance_threshold": config.RELEVANCE_THRESHOLD,
    }
