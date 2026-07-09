"""Load the daily_data corpus and join comments to their parent posts.

The corpus layout is:  daily_data/<YYYY-MM-DD>/<Sub>_posts.json
                       daily_data/<YYYY-MM-DD>/<Sub>_comments.json

Posts and comments share one object shape. Comments link to a post via
`postParentID` == the post's `postRawID`.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import config

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _is_reddit(url: str) -> bool:
    return "reddit.com" in url or "redd.it" in url


def _clean_url(url: str) -> str:
    return "".join(url.split())  # strip embedded whitespace/newlines that break links


@dataclass
class Comment:
    raw_id: str
    author: str
    body: str
    score: int


@dataclass
class Post:
    raw_id: str
    subreddit: str        # bare name, e.g. "OpenAI"
    track: str
    title: str
    body: str
    author: str
    permalink: str
    links: list[str]      # off-reddit urls from urlsList
    score: int
    num_comments: int
    flair: str
    nsfw: bool
    date: str = ""            # which daily_data folder it came from
    comments: list[Comment] = field(default_factory=list)


def _latest_score(counter_data: list[dict]) -> tuple[int, int]:
    """Return (upvotes, num_comments) from the most recent counter snapshot."""
    if not counter_data:
        return 0, 0
    last = counter_data[-1]
    return int(last.get("postNumUpvotes") or last.get("postScore") or 0), int(
        last.get("postNumComments") or 0
    )


def available_dates() -> list[str]:
    if not config.DATA_DIR.exists():
        return []
    return sorted(
        d.name for d in config.DATA_DIR.iterdir() if d.is_dir() and _DATE_RE.match(d.name)
    )


def resolve_date(date: str | None) -> str:
    dates = available_dates()
    if not dates:
        raise FileNotFoundError(f"No dated folders under {config.DATA_DIR}")
    if date is None:
        return dates[-1]
    if date not in dates:
        raise FileNotFoundError(f"{date} not found. Available: {dates[-1]} (latest)")
    return date


def _sub_of(path: Path, suffix: str) -> str:
    return path.name[: -len(suffix)]


def _track_of(sub: str) -> str | None:
    low = sub.lower()
    for track, subs in config.TRACK_SUBS.items():
        if any(low == s.lower() for s in subs):
            return track
    return None


def _load_json(path: Path) -> list[dict]:
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def _load_sub(track: str, sub: str, day_dir: Path) -> list[Post]:
    """Load one subreddit's posts (with joined top comments) from a day folder."""
    posts_path = day_dir / f"{sub}_posts.json"
    if not posts_path.exists():
        return []

    comments_by_parent: dict[str, list[Comment]] = {}
    for c in _load_json(day_dir / f"{sub}_comments.json"):
        score, _ = _latest_score(c.get("postCounterData") or [])
        comment = Comment(
            raw_id=c.get("postRawID", ""),
            author=c.get("redditorName", ""),
            body=(c.get("postContent") or "").strip(),
            score=score,
        )
        if comment.body and comment.body not in ("[removed]", "[deleted]"):
            comments_by_parent.setdefault(c.get("postParentID", ""), []).append(comment)

    out: list[Post] = []
    for p in _load_json(posts_path):
        raw_id = p.get("postRawID", "")
        score, num_comments = _latest_score(p.get("postCounterData") or [])
        post = Post(
            raw_id=raw_id,
            subreddit=sub,
            track=track,
            title=(p.get("postTitle") or "").strip(),
            body=(p.get("postContent") or "").strip(),
            author=p.get("redditorName", ""),
            permalink=_clean_url(p.get("postThreadUrl") or p.get("postUrl") or ""),
            links=[_clean_url(u) for u in (p.get("urlsList") or []) if u and not _is_reddit(u)],
            score=score,
            num_comments=num_comments,
            flair=p.get("postFlairText") or "",
            nsfw=bool(p.get("postOver18") or p.get("nsfw")),
            date=day_dir.name,
        )
        comments = sorted(
            comments_by_parent.get(raw_id, []), key=lambda c: c.score, reverse=True
        )
        post.comments = comments[: config.MAX_COMMENTS_PER_POST]
        out.append(post)
    return out


def load_day(date: str | None = None) -> tuple[str, dict[str, list[Post]]]:
    """Load one specific day's corpus (all subs from that folder), grouped by track."""
    date = resolve_date(date)
    day_dir = config.DATA_DIR / date
    by_track: dict[str, list[Post]] = {t: [] for t in config.TRACK_SUBS}
    for posts_path in sorted(day_dir.glob("*_posts.json")):
        sub = _sub_of(posts_path, "_posts.json")
        track = _track_of(sub)
        if track:
            by_track[track].extend(_load_sub(track, sub, day_dir))
    return date, by_track


def _latest_dir_for_sub(sub: str) -> str | None:
    """Newest date folder that contains this subreddit's posts file."""
    for d in reversed(available_dates()):
        if (config.DATA_DIR / d / f"{sub}_posts.json").exists():
            return d
    return None


def load_latest() -> tuple[str, dict[str, list[Post]]]:
    """Load each configured sub from ITS OWN most-recent folder.

    Handles a multi-source corpus: e.g. AI/content-gen from the external
    collector's latest day, financial from today's self-collected folder.
    """
    by_track: dict[str, list[Post]] = {t: [] for t in config.TRACK_SUBS}
    dates_used: set[str] = set()
    for track, subs in config.TRACK_SUBS.items():
        for sub in subs:
            d = _latest_dir_for_sub(sub)
            if d:
                by_track[track].extend(_load_sub(track, sub, config.DATA_DIR / d))
                dates_used.add(d)
    label = (min(dates_used) + "…" + max(dates_used)) if len(dates_used) > 1 else \
        (next(iter(dates_used)) if dates_used else "n/a")
    return label, by_track


def select_candidates(posts: list[Post]) -> tuple[list[Post], int]:
    """Top-N posts per subreddit by score. Returns (kept, dropped_count)."""
    by_sub: dict[str, list[Post]] = {}
    for p in posts:
        by_sub.setdefault(p.subreddit, []).append(p)

    per_sub: list[list[Post]] = []
    dropped = 0
    for sub_posts in by_sub.values():
        sub_posts.sort(key=lambda p: (p.score, p.num_comments), reverse=True)
        per_sub.append(sub_posts[: config.MAX_POSTS_PER_SUB])
        dropped += max(0, len(sub_posts) - config.MAX_POSTS_PER_SUB)

    # Round-robin across subs so a --limit sample (and the digest) spans subs
    # instead of clustering on whichever sub sorts first.
    kept: list[Post] = []
    for i in range(max((len(s) for s in per_sub), default=0)):
        for s in per_sub:
            if i < len(s):
                kept.append(s[i])
    return kept, dropped
