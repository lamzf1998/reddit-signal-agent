"""Orchestrate one digest run: load -> filter+extract -> novelty -> digest -> send.

Usage:
    python -m reddit_agent.main [--date YYYY-MM-DD] [--dry-run] [--track ai]
"""
from __future__ import annotations

import argparse
import sys

from . import collect, config, data_loader, db, seen, telegram
from .extract import extract_post, unload_model
from .telegram import esc


def _render_artifact(post, ex) -> str:
    link = post.links[0] if post.links else post.permalink
    recv = {
        "positive": "👍 positive", "negative": "👎 negative",
        "mixed": "🤷 mixed", "neutral": "😐 neutral", "unknown": "❔ unknown",
    }.get(ex.reception, ex.reception)
    flag = ("🔴 <b>BREAKING</b> · " if ex.importance == "breaking" else "")
    lines = [
        f"🆕 <b>NEW</b> · {flag}<b>{esc(ex.name)}</b>  <i>({esc(ex.kind)})</i>",
        f"- {esc(ex.one_liner)}",
        f"- Reception: {recv} — {esc(ex.reception_summary)}",
        f'- r/{esc(post.subreddit)} · <a href="{esc(link)}">source</a> · '
        f'<a href="{esc(post.permalink)}">thread</a>',
    ]
    return "\n".join(lines)


def _render_financial(post, ex) -> str:
    sent = {
        "bullish": "🟢 bullish", "bearish": "🔴 bearish",
        "mixed": "🟡 mixed", "neutral": "⚪ neutral",
    }.get(ex.sentiment, ex.sentiment)
    flag = ("🔴 <b>BREAKING</b> · " if ex.importance == "breaking" else "")
    lines = [f"🆕 <b>NEW</b> · {flag}<b>{esc(post.title)}</b>", f"- AI angle: {esc(ex.ai_angle)}"]
    if ex.tickers:
        lines.append("- Tickers: " + ", ".join(esc(t) for t in ex.tickers))
    lines.append(f"- Sentiment: {sent}")
    for op in ex.key_opinions[:4]:
        lines.append(f"- {esc(op)}")
    if ex.implication:
        lines.append(f"- Implication: {esc(ex.implication)}")
    lines.append(f'- r/{esc(post.subreddit)} · <a href="{esc(post.permalink)}">thread</a>')
    return "\n".join(lines)


def _build_messages(items: list[tuple[str, str, str]], date: str) -> list[str]:
    """Turn (track, subreddit, rendered) tuples into Telegram messages."""
    if not items:
        return [f"📡 <b>Reddit Signal Digest</b> — {esc(date)}\n\n"
                "Nothing net-new cleared the relevance bar today."]

    if config.GROUP_BY == "subreddit":
        order: list[str] = []
        groups: dict[str, tuple[str, list[str]]] = {}
        for track, sub, text in items:
            if sub not in groups:
                groups[sub] = (track, [])
                order.append(sub)
            groups[sub][1].append(text)
        messages = []
        for sub in order:
            track, texts = groups[sub]
            header = f"📡 <b>r/{esc(sub)}</b> · {esc(date)}"
            messages.append(header + "\n\n" + "\n\n".join(texts))
        return messages

    # "digest": one message, grouped by track
    sections = [f"📡 <b>Reddit Signal Digest</b> — {esc(date)}"]
    for track in config.TRACK_SUBS:
        texts = [t for tr, _, t in items if tr == track]
        if texts:
            sections.append(f"\n<b>{esc(config.TRACK_LABELS[track])}</b>\n\n"
                            + "\n\n".join(texts))
    return ["\n".join(sections)]


def run(date: str | None, only_track: str | None, dry_run: bool,
        limit: int = 0, ignore_seen: bool = False) -> None:
    # Self-collect all tracks' subs live (hottest recent posts) before filtering.
    if config.COLLECT_LIVE and not date:
        print("Collecting live (hot) posts across all tracks...")
        try:
            collect.collect()
        except Exception as e:
            print(f"  ! live collection failed: {e}", file=sys.stderr)

    if date:
        resolved, by_track = data_loader.load_day(date)
    else:
        resolved, by_track = data_loader.load_latest()
    print(f"Loaded corpus ({resolved})")

    items: list[tuple[str, str, str]] = []  # (track, subreddit, rendered)

    for track, posts in by_track.items():
        if only_track and track != only_track:
            continue
        if not posts:
            continue

        candidates, dropped = data_loader.select_candidates(posts)
        if limit:
            candidates = candidates[:limit]
        print(
            f"[{track}] {len(posts)} posts -> {len(candidates)} candidates "
            f"({dropped} dropped by top-N cap)"
        )

        kept = 0
        skipped = 0
        for post in candidates:
            # Skip posts already extracted on a prior run — keeps hourly cheap.
            if not ignore_seen and seen.is_processed(post.raw_id):
                skipped += 1
                continue
            try:
                ex = extract_post(post)
            except Exception as e:  # never let one post kill the run
                print(f"  ! extraction failed for {post.raw_id}: {e}", file=sys.stderr)
                continue  # not marked processed → retried next run
            if not dry_run:
                seen.mark_processed(post.raw_id, resolved)
            conf = ex.confidence / 100 if ex.confidence > 1 else ex.confidence
            if dry_run:  # show the model's verdict for every candidate
                tag = getattr(ex, "name", None) or getattr(ex, "sentiment", "")
                print(
                    f"    · r/{post.subreddit}: relevant={ex.relevant} "
                    f"conf={conf:.2f} imp={ex.importance} "
                    f"pref={ex.matches_prefs} [{tag}] — {post.title[:60]}"
                )
            # Evaluate the send gates (relevance → preferences → novelty).
            passed = ex.relevant and conf >= config.RELEVANCE_THRESHOLD
            if passed and config.USER_PREFERENCES:
                is_breaking = config.SEND_BREAKING and ex.importance == "breaking"
                passed = ex.matches_prefs or is_breaking
            novelty_key = post.raw_id if track == "financial" else ex.name
            if passed:
                passed = ignore_seen or seen.is_new(novelty_key)

            # Persist every analysed post (real runs only), with its send outcome.
            if not dry_run:
                db.save_analysis(post, ex, conf, sent=passed, corpus_date=resolved)

            if not passed:
                continue

            if track == "financial":
                items.append((track, post.subreddit, _render_financial(post, ex)))
                if not dry_run:
                    seen.record(post.raw_id, "financial_post", resolved)
            else:
                items.append((track, post.subreddit, _render_artifact(post, ex)))
                if not dry_run:
                    seen.record(ex.name, ex.kind, resolved)
            kept += 1

        print(f"[{track}] kept {kept} net-new relevant items "
              f"({skipped} already-processed skipped before extraction)")

    messages = _build_messages(items, resolved)

    if dry_run:
        print(f"\n----- {len(messages)} MESSAGE(S) (dry run, not sent) -----")
        for m in messages:
            print("\n" + "─" * 40 + "\n" + m)
    elif items:
        for m in messages:
            telegram.send(m)
        print(f"\nSent {len(messages)} message(s) to Telegram ({len(items)} items).")
    else:
        print("\nNothing net-new to send; skipped Telegram.")

    # Export a static snapshot for the GitHub Pages dashboard.
    if not dry_run:
        db.export_json(config.PROJECT_ROOT / "docs" / "signals.json")
        print("Exported docs/signals.json")

    # Free the GPU right after the run so the laptop isn't kept warm between runs.
    if config.OLLAMA_UNLOAD_AFTER_RUN and unload_model():
        print("Unloaded model from GPU.")


def main() -> None:
    # Windows consoles default to cp1252 and choke on emoji in the digest.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description="Reddit Signal Agent digest")
    ap.add_argument("--date", help="YYYY-MM-DD (default: latest available)")
    ap.add_argument("--track", choices=list(config.TRACK_SUBS), help="limit to one track")
    ap.add_argument("--dry-run", action="store_true", help="print instead of sending")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap candidates per track (0 = no cap; useful for testing)")
    ap.add_argument("--ignore-seen", action="store_true",
                    help="bypass the novelty registry (for testing/tuning)")
    ap.add_argument("--unload", action="store_true",
                    help="free the model from the GPU and exit (cools the laptop)")
    args = ap.parse_args()
    if args.unload:
        print("Unloaded model from GPU." if unload_model() else "Nothing to unload.")
        return
    run(args.date, args.track, args.dry_run, args.limit, args.ignore_seen)


if __name__ == "__main__":
    main()
