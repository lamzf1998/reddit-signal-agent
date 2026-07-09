# Reddit Signal Agent → Telegram

Reads your existing `daily_data/` corpus, filters each subreddit vertical for the
signal you care about, extracts structured records + community reception with
Claude, and pushes a precision-favoring digest to Telegram.

It does **not** collect Reddit data — that's your existing pipeline (PRAW /
Arctic-shift). This bot is the filter → extract → digest → deliver half.

## Pipeline

```
daily_data/<date>/  →  load + join comments  →  top-N per sub  →  Claude filter+extract
                                                                    (+ reception, + link fetch)
                                                 →  novelty check  →  digest  →  Telegram
```

Three tracks (see `config.py`):

| Track | Keeps | Extracts |
|-------|-------|----------|
| `ai` | new tools / frameworks / model releases | name, what-it-does, reception |
| `content_gen` | new workflows / prompt guides / models | artifact, base model, reception |
| `financial` | posts *about AI* | AI angle, tickers, sentiment, opinions |

> The current corpus has no financial subreddits, so that track stays dormant
> until financial data appears. AI and content-gen are live.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env      # fill in Telegram token + chat id
```

**Extraction backend.** Defaults to a **local Ollama model** (free, private, GPU) —
no API key. Install [Ollama](https://ollama.com), then `ollama pull qwen2.5:14b`.
Configure via `LLM_BACKEND` / `OLLAMA_MODEL` in `.env`.

To use Claude instead: `pip install anthropic`, set `LLM_BACKEND=anthropic`, and
provide `ANTHROPIC_API_KEY` (a Console key — a claude.ai Pro/Team plan does **not**
grant API access).

## Run

```bash
# Preview the latest day's digest without sending (recommended first):
python -m reddit_agent.main --dry-run

# One track only:
python -m reddit_agent.main --track ai --dry-run

# A specific day:
python -m reddit_agent.main --date 2026-02-28 --dry-run

# Send to Telegram:
python -m reddit_agent.main
```

## Live collection (all tracks, PRAW hot)

Each run self-collects **every configured subreddit** into `daily_data/<today>/`
before filtering, so the whole agent is real-time and self-contained. Strategy
(`.env`): the **`COLLECT_HOT_LIMIT`** (5) hottest posts created in the last
**`COLLECT_RECENCY_HOURS`** (3) hours per sub, plus their top-upvoted comments for
reception. `load_latest` reads each sub from its newest folder.

Needs a free Reddit "script" app: create one at https://www.reddit.com/prefs/apps
(type **script**, redirect `http://localhost:8080`), put the **client id** and
**secret** in `.env` (`REDDIT_CLIENT_ID`/`REDDIT_CLIENT_SECRET`, read-only, no
password), and `pip install praw`. Quiet subs may return 0 in a 3h window —
raise `COLLECT_RECENCY_HOURS` for fuller sweeps.

### Legacy / backfill (Arctic-shift)

`tools/arctic_subs.py` collects via the Arctic-shift archive (no credentials,
handles the 1000-item cap, historical date ranges). It's retired from the live
path but kept for **backfill / historical scraping**: run it standalone, or set
`COLLECT_BACKEND=arctic` (`COLLECT_WINDOW_HOURS` sets the lookback) for a bulk pull.

## What gets sent (preferences + breaking)

After the relevance filter, an item is sent **only if** it matches your interests
**or** the model flags it `breaking`:

- Edit `preferences.txt` (plain English, one interest per line) to say what you care
  about. The model matches each item against it.
- `breaking` = a major/anticipated release or a post with very high engagement.
  Sent even if it doesn't match your interests, so you never miss a big drop
  (`SEND_BREAKING=0` in `.env` to disable). Breaking items are tagged 🔴 in the digest.
- If `preferences.txt` is empty, every relevant item is sent (no interest gate).

## Database & live dashboard

Every analysed post is persisted to an `analyses` table in
`reddit_agent/.state/seen.sqlite3` (one row per post: metadata, the full
extraction, whether it was sent, timestamp). Real runs write it; dry runs don't.

A local web dashboard reads that DB:

```bash
pip install flask         # already in requirements
python -m reddit_agent.server     # or double-click run_server.bat
# open http://127.0.0.1:8765
```

It serves on `127.0.0.1` only (your machine). Filter by track; three view tabs —
**Sent** (pushed to Telegram), **Hidden** (analysed but filtered, each tagged with
the reason: noise / off-interest / dup-low-conf), and **All**; plus search. The
page auto-refreshes every 5 min, so it stays in sync with each scheduled run.
JSON API: `/api/signals`, `/api/stats`.

**Keep it always-on:** register it to start at login —
`schtasks /Create /TN "RedditSignalDashboard" /TR "C:\Users\userAdmin\Documents\reddit\run_server.bat" /SC ONLOGON /F`.

### Hosted dashboard (GitHub Pages, phone-accessible)

Each run also writes a static snapshot to `docs/signals.json`; `docs/index.html`
is a static build of the dashboard that reads it (all tabs/filters work in the
browser). GitHub Pages serves `docs/` — no Actions needed. One-time setup:

1. Create a **PRIVATE** GitHub repo (the snapshot holds your digest data; private
   Pages needs GitHub Pro). `.gitignore` already excludes `.env`, `daily_data/`,
   and local state.
2. `git init && git add . && git commit -m "init" && git remote add origin <url> && git push -u origin main`
3. Repo **Settings → Pages → Deploy from a branch → `main` / `/docs`**.
4. `run_digest.bat` commits & pushes `docs/signals.json` every run (the git block
   only activates once the folder is a repo), so the Pages dashboard refreshes
   each 6h run. Telegram is still sent from the local run.

**In sync with Telegram:** each 3-hourly agent run writes the DB and sends
Telegram in the same pass, so the dashboard and your Telegram never diverge.

## Message grouping

`GROUP_BY` in `.env` controls how output is split:
- `subreddit` (default) — one Telegram message per subreddit that has items.
- `digest` — a single message, grouped by track.

## Turning it off / GPU heat

The model only occupies the GPU during a run. To keep the laptop cool:

- **Auto-unload after each run** is on by default (`OLLAMA_UNLOAD_AFTER_RUN=1`) —
  the model is freed from VRAM the moment a run finishes, not 5 min later.
- **Free the GPU now:** `python -m reddit_agent.main --unload` (or `ollama stop qwen2.5:14b`).
- **Pause the schedule:** `schtasks /Change /TN "RedditSignalAgent" /DISABLE`
  (re-enable with `/ENABLE`). Same for `RedditSignalDashboard`.
- **Fully off:** disable both scheduled tasks and close the dashboard server.
- `OLLAMA_KEEP_ALIVE` (default `5m`) controls how long the model lingers in VRAM
  after activity if auto-unload is off.

## Notes & knobs

- **Precision over recall.** `RELEVANCE_THRESHOLD` (default 0.6) gates what ships;
  raise it if the digest is noisy.
- **Novelty.** A SQLite registry (`.state/seen.sqlite3`) suppresses tools/models
  already surfaced on prior runs. Delete it to reset.
- **Cost.** Only the top `MAX_POSTS_PER_SUB` posts per sub hit the classifier;
  dropped counts are logged, never hidden. The default extraction model is
  `claude-opus-4-8` — set `EXTRACT_MODEL=claude-haiku-4-5` (or sonnet) to cut cost.
- **Link enrichment** (`ENRICH_LINKS=1`) fetches & summarizes the first off-Reddit
  link; it degrades to link-only on any failure.
- **Scheduling (every 6 hours).** Register `run_digest.bat` with Windows Task Scheduler:
  `schtasks /Create /TN "RedditSignalAgent" /TR "C:\Users\userAdmin\Documents\reddit\run_digest.bat" /SC HOURLY /MO 6 /F`
  (collection window `COLLECT_RECENCY_HOURS=6` matches the 6h cadence).
  Each run collects the hottest recent posts; the "processed posts" registry skips
  any post already analysed in a prior run, so only genuinely new posts are sent
  (each tagged 🆕 NEW). Logs to `reddit_agent\.state\run.log`.
```
