"""Configuration for the Reddit Signal Agent.

Everything tunable lives here or in environment variables (.env). Nothing
secret is hard-coded.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (no dependency). Existing env vars win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


_load_dotenv(PROJECT_ROOT / ".env")

# --- paths -----------------------------------------------------------------
DATA_DIR = Path(os.getenv("REDDIT_DATA_DIR", PROJECT_ROOT / "daily_data"))
STATE_DIR = Path(os.getenv("REDDIT_STATE_DIR", PROJECT_ROOT / "reddit_agent" / ".state"))
SEEN_DB = STATE_DIR / "seen.sqlite3"

# --- extraction backend ----------------------------------------------------
# "ollama" (local, free, private) or "anthropic" (Claude API — needs a key).
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")

# Ollama (local model) settings.
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")
# How long Ollama keeps the model in VRAM after a request (kept loaded across a
# run's calls, then released). And whether to free the GPU right after each run.
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "5m")
OLLAMA_UNLOAD_AFTER_RUN = os.getenv("OLLAMA_UNLOAD_AFTER_RUN", "1") == "1"

# Anthropic settings (only used when LLM_BACKEND=anthropic).
EXTRACT_MODEL = os.getenv("EXTRACT_MODEL", "claude-opus-4-8")
EXTRACT_MAX_TOKENS = int(os.getenv("EXTRACT_MAX_TOKENS", "1024"))

# --- Telegram --------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# --- pipeline knobs --------------------------------------------------------
# Only the top-N posts per subreddit (by upvotes) are sent to the classifier,
# to control cost. Dropped counts are logged, never silently hidden.
MAX_POSTS_PER_SUB = int(os.getenv("MAX_POSTS_PER_SUB", "15"))
# Top-K comments (by upvotes) attached to each post for reception analysis.
MAX_COMMENTS_PER_POST = int(os.getenv("MAX_COMMENTS_PER_POST", "8"))
# A post is kept only if the classifier's confidence clears this bar.
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.6"))


def _load_preferences() -> str:
    """User interests from preferences.txt (comments/blank lines stripped)."""
    path = PROJECT_ROOT / "preferences.txt"
    if not path.exists():
        return os.getenv("USER_PREFERENCES", "").strip()
    lines = [
        ln.strip()
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    return "\n".join(lines)


USER_PREFERENCES = _load_preferences()
# "Breaking" items are sent even if they don't match your interests.
SEND_BREAKING = os.getenv("SEND_BREAKING", "1") == "1"

# How to split Telegram output:
#   "subreddit" — one message per subreddit (default)
#   "digest"    — a single message, grouped by track
GROUP_BY = os.getenv("GROUP_BY", "subreddit")
# Fetch & summarize off-Reddit links (GitHub/HF/Civitai/arXiv). Degrades to
# link-only on any failure.
ENRICH_LINKS = os.getenv("ENRICH_LINKS", "1") == "1"
ENRICH_TIMEOUT = int(os.getenv("ENRICH_TIMEOUT", "12"))
ENRICH_MAX_CHARS = int(os.getenv("ENRICH_MAX_CHARS", "4000"))

# --- track → subreddit mapping --------------------------------------------
# Matched case-insensitively against the `<Sub>` in `<Sub>_posts.json`.
# Financial subs are listed but absent from the current corpus; the track
# stays dormant until their data appears.
_DEFAULT_TRACK_SUBS = {
    "ai": [
        "ArtificialInteligence", "Artificial", "OpenAI", "Singularity",
        "agi", "technology", "Futurology", "automation", "vibecoding",
    ],
    "content_gen": [
        "StableDiffusion", "Midjourney", "ConceptArt", "DigitalArt",
        "Illustration",
    ],
    "financial": [
        "stocks", "investing", "wallstreetbets", "StockMarket", "NVDA_Stock",
    ],
}

# Subreddit lists are editable via the local dashboard; persisted here.
SUBS_FILE = PROJECT_ROOT / "config_subs.json"
PREFERENCES_FILE = PROJECT_ROOT / "preferences.txt"


def _load_track_subs() -> dict:
    """Track→subreddits from config_subs.json, falling back to the defaults."""
    if SUBS_FILE.exists():
        try:
            import json
            data = json.loads(SUBS_FILE.read_text(encoding="utf-8"))
            return {t: [s for s in data.get(t, _DEFAULT_TRACK_SUBS[t])] for t in _DEFAULT_TRACK_SUBS}
        except Exception:
            pass
    return {t: list(v) for t, v in _DEFAULT_TRACK_SUBS.items()}


TRACK_SUBS = _load_track_subs()


def reload() -> None:
    """Re-read the editable config files (preferences + subreddits) in-process."""
    global USER_PREFERENCES, TRACK_SUBS
    USER_PREFERENCES = _load_preferences()
    TRACK_SUBS = _load_track_subs()

# --- live self-collection --------------------------------------------------
# Self-collect ALL tracks' subs before each run (fully real-time, self-contained).
COLLECT_LIVE = os.getenv("COLLECT_LIVE", os.getenv("COLLECT_FINANCIAL", "1")) == "1"
# Live backend is "praw" (real-time hot ranking). "arctic" is kept for LEGACY /
# historical backfill only (run tools/arctic_subs.py, or set this for a bulk pull).
COLLECT_BACKEND = os.getenv("COLLECT_BACKEND", "praw")
# Arctic-shift lookback window (hours) per run.
COLLECT_WINDOW_HOURS = int(os.getenv("COLLECT_WINDOW_HOURS", "24"))

# PRAW collection strategy: "hot" (momentum ranking, real-time) or "top" (top of day).
COLLECT_SORT = os.getenv("COLLECT_SORT", "hot")
# For "hot": only keep posts created within this many hours (rising, recent).
COLLECT_RECENCY_HOURS = float(os.getenv("COLLECT_RECENCY_HOURS", "3"))
# Keep at most this many of the hottest recent posts per sub.
COLLECT_HOT_LIMIT = int(os.getenv("COLLECT_HOT_LIMIT", "10"))
# Scan this many hot posts before applying the recency filter.
COLLECT_HOT_SCAN = int(os.getenv("COLLECT_HOT_SCAN", "60"))
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.getenv(
    "REDDIT_USER_AGENT", "windows:reddit-signal-agent:0.1 (personal digest bot)"
)
COLLECT_POSTS_PER_SUB = int(os.getenv("COLLECT_POSTS_PER_SUB", "25"))

TRACK_LABELS = {
    "ai": "🧠 AI — new tools & updates",
    "content_gen": "🎨 Content generation — workflows, guides & models",
    "financial": "📈 Financial — AI-related market opinion",
}
