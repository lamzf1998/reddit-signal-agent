"""Self-collect the financial subreddits via PRAW (free Reddit script app).

Writes into daily_data/<today>/<Sub>_posts.json + _comments.json in the same
schema the external collector uses, so data_loader treats them identically.

The other tracks (AI, content-gen) come from your external collector; this only
fills the financial track. No-ops with a clear message if creds are missing.
"""
from __future__ import annotations

import datetime
import json
import time
from pathlib import Path

from . import config


def _counter(score: int, num_comments: int = 0) -> list[dict]:
    return [{"postScore": score, "postNumUpvotes": score, "postNumComments": num_comments}]


def _map_post(sub: str, s) -> dict:
    return {
        "postType": "Forum_Thread",
        "postRawID": s.id,
        "redditorName": str(s.author) if s.author else "[deleted]",
        "postTitle": s.title or "",
        "postThreadUrl": "https://reddit.com" + s.permalink,
        "postUrl": s.url or "",
        "postContent": s.selftext or "",
        "postFlairText": s.link_flair_text or "",
        "postParentID": "",
        "postSourceName": f"r/{sub}",
        "postCounterData": _counter(int(s.score), int(s.num_comments)),
        "urlsList": [] if s.is_self else ([s.url] if s.url else []),
        "postOver18": bool(s.over_18),
    }


def _map_comment(sub: str, parent_id: str, c) -> dict:
    return {
        "postType": "Forum_Comment",
        "postRawID": c.id,
        "redditorName": str(c.author) if c.author else "[deleted]",
        "postTitle": "",
        "postContent": c.body or "",
        "postParentID": parent_id,
        "postSourceName": f"r/{sub}",
        "postCounterData": _counter(int(c.score)),
        "urlsList": [],
    }


def _all_subs() -> list[str]:
    """Every configured subreddit across all tracks (de-duplicated, order-preserving)."""
    seen, out = set(), []
    for subs in config.TRACK_SUBS.values():
        for s in subs:
            if s.lower() not in seen:
                seen.add(s.lower())
                out.append(s)
    return out


def collect(subs: list[str] | None = None, limit: int | None = None) -> str | None:
    """Pull all configured subs into today's daily_data folder. Returns the date, or None."""
    subs = subs or _all_subs()
    if config.COLLECT_BACKEND == "arctic":
        return _collect_arctic(subs)
    return _collect_praw(subs, limit)


def _collect_arctic(subs: list[str] | None = None) -> str | None:
    """Collect via your Arctic-shift collector (tools/arctic_subs.py), same schema."""
    import datetime
    import importlib.util

    path = config.PROJECT_ROOT / "tools" / "arctic_subs.py"
    if not path.exists():
        print(f"  arctic collection skipped: {path} not found")
        return None
    spec = importlib.util.spec_from_file_location("arctic_subs", path)
    arctic = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arctic)

    subs = subs or _all_subs()
    now = datetime.datetime.now(datetime.timezone.utc)
    before = int(now.timestamp() * 1000)
    after = int((now - datetime.timedelta(hours=config.COLLECT_WINDOW_HOURS)).timestamp() * 1000)

    today = datetime.date.today().isoformat()
    day_dir = config.DATA_DIR / today
    day_dir.mkdir(parents=True, exist_ok=True)
    print(f"  arctic-shift window: last {config.COLLECT_WINDOW_HOURS}h "
          "(note: archive may lag the most recent hours)")

    import time
    for sub in subs:
        posts = comments = None
        for attempt in range(3):  # arctic-shift occasionally drops a stream mid-response
            try:
                posts = arctic.parse_arctic_posts(arctic.get_subreddit_posts(sub, after, before))
                comments = arctic.parse_arctic_comments(arctic.get_subreddit_comments(sub, after, before))
                break
            except Exception as e:
                print(f"  r/{sub} attempt {attempt + 1}/3 failed: {str(e)[:60]}")
                time.sleep(1.5)
        if posts is None:
            print(f"  ! r/{sub} arctic collection failed after retries")
            continue
        _write(day_dir / f"{sub}_posts.json", posts)
        _write(day_dir / f"{sub}_comments.json", comments)
        print(f"  collected r/{sub}: {len(posts)} posts, {len(comments)} comments")
    return today


def _collect_praw(subs: list[str] | None = None, limit: int | None = None) -> str | None:
    if not (config.REDDIT_CLIENT_ID and config.REDDIT_CLIENT_SECRET):
        print("  finance collection skipped: set REDDIT_CLIENT_ID/SECRET in .env")
        return None
    try:
        import praw
    except ImportError:
        print("  finance collection skipped: `pip install praw`")
        return None

    subs = subs or _all_subs()
    limit = limit or config.COLLECT_POSTS_PER_SUB

    reddit = praw.Reddit(
        client_id=config.REDDIT_CLIENT_ID,
        client_secret=config.REDDIT_CLIENT_SECRET,
        user_agent=config.REDDIT_USER_AGENT,
    )
    reddit.read_only = True

    today = datetime.date.today().isoformat()
    day_dir = config.DATA_DIR / today
    day_dir.mkdir(parents=True, exist_ok=True)

    hot = config.COLLECT_SORT == "hot"
    cutoff = time.time() - config.COLLECT_RECENCY_HOURS * 3600
    if hot:
        print(f"  praw: {config.COLLECT_HOT_LIMIT} hottest posts from the last "
              f"{config.COLLECT_RECENCY_HOURS:g}h per sub")

    for sub in subs:
        posts: list[dict] = []
        comments: list[dict] = []
        try:
            if hot:
                scanned = reddit.subreddit(sub).hot(limit=config.COLLECT_HOT_SCAN)
                # hot order preserved; keep only recently-created, cap at HOT_LIMIT
                selected = [s for s in scanned if s.created_utc >= cutoff][: config.COLLECT_HOT_LIMIT]
            else:
                selected = list(reddit.subreddit(sub).top(time_filter="day", limit=limit))

            for s in selected:
                posts.append(_map_post(sub, s))
                s.comments.replace_more(limit=0)  # only need top-level, no expansion
                top = sorted(s.comments, key=lambda c: c.score, reverse=True)
                for c in top[: config.MAX_COMMENTS_PER_POST]:
                    comments.append(_map_comment(sub, s.id, c))
        except Exception as e:  # one bad sub shouldn't sink the rest
            print(f"  ! r/{sub} collection failed: {e}")
            continue

        _write(day_dir / f"{sub}_posts.json", posts)
        _write(day_dir / f"{sub}_comments.json", comments)
        print(f"  collected r/{sub}: {len(posts)} posts, {len(comments)} comments")

    return today


def _write(path: Path, data: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
