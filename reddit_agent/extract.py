"""Per-track relevance filtering + structured extraction.

Two interchangeable backends:
  * "ollama"    — a local model (free, private) via the Ollama HTTP API,
                  using its JSON-schema structured-output mode.
  * "anthropic" — Claude API (needs ANTHROPIC_API_KEY).

Each track has its own filter (what to keep) and extraction schema (what to
pull out), plus community reception read from the post's top comments.
"""
from __future__ import annotations

from typing import Literal

import requests
from pydantic import BaseModel

from . import config, enrich
from .data_loader import Post


# --- extraction schemas ----------------------------------------------------
class ArtifactExtraction(BaseModel):
    relevant: bool
    confidence: float
    kind: Literal["tool", "model", "workflow", "prompt_guide", "other"]
    name: str
    one_liner: str
    reception: Literal["positive", "mixed", "negative", "neutral", "unknown"]
    reception_summary: str
    importance: Literal["breaking", "notable", "minor"]
    matches_prefs: bool
    pref_reason: str


class FinancialExtraction(BaseModel):
    relevant: bool
    confidence: float
    ai_angle: str
    tickers: list[str]
    sentiment: Literal["bullish", "bearish", "mixed", "neutral"]
    key_opinions: list[str]
    implication: str
    importance: Literal["breaking", "notable", "minor"]
    matches_prefs: bool
    pref_reason: str


_SYSTEM = {
    "ai": (
        "You screen Reddit posts from AI subreddits. KEEP a post if it is substantive "
        "and on-topic for real AI developments: a tool/model/framework release, OR "
        "notable news or analysis about AI companies, models, coding assistants, "
        "agents, AI chips, or datacenters. REJECT memes, rage-bait, low-effort hot "
        "takes, personal anecdotes, and vague questions. In `name` put a short topic "
        "or tool name; in `one_liner` summarize what it is; set `kind` (use 'other' "
        "for news/analysis). Summarize community reception from the comments. When "
        "unsure, set relevant=false. Respond only with the requested JSON."
    ),
    "content_gen": (
        "You screen Reddit posts from image/video generation subreddits. KEEP a post "
        "if it introduces or substantively discusses a new workflow, prompt guide, "
        "model, or generation tool/technique. REJECT personal art showcases and images "
        "posted just to show off with no reusable artifact. In `name` put the "
        "artifact/topic; in `one_liner` summarize it. Summarize community reception "
        "from the comments. When unsure, set relevant=false. Respond only with the "
        "requested JSON."
    ),
    "financial": (
        "You screen Reddit posts from finance subreddits. KEEP a post only if it is "
        "about AI (AI stocks, AI capex/datacenter buildout, 'AI bubble' discourse, "
        "AI-driven earnings or market moves). Extract the AI angle, tickers, overall "
        "sentiment, the key opinions with their reasoning, and any financial "
        "implication. Report opinion only — never give advice. When unsure, set "
        "relevant=false. Respond only with the requested JSON."
    ),
}


_IMPORTANCE = (
    "\n\nALSO rate `importance`: 'breaking' = a major/highly-anticipated release or "
    "announcement, or a post with very high community engagement for its subreddit; "
    "'notable' = worth knowing; 'minor' = niche or low-impact. Use the ENGAGEMENT "
    "numbers as one signal."
)

_PREFS = (
    "\n\nThe user cares about these interests:\n{prefs}\n"
    "Set `matches_prefs`=true only if the item clearly matches one of them, and give "
    "the reason in `pref_reason`. If it matches none, set matches_prefs=false."
)


def _system_for(track: str) -> str:
    system = _SYSTEM[track] + _IMPORTANCE
    if config.USER_PREFERENCES:
        system += _PREFS.format(prefs=config.USER_PREFERENCES)
    else:
        system += "\n\nNo user interests are configured; set matches_prefs=true."
    return system


def _model_for(track: str) -> type[BaseModel]:
    return FinancialExtraction if track == "financial" else ArtifactExtraction


def _post_prompt(post: Post, source_text: str) -> str:
    lines = [
        f"SUBREDDIT: r/{post.subreddit}",
        f"ENGAGEMENT: {post.score} upvotes, {post.num_comments} comments",
        f"TITLE: {post.title}",
    ]
    if post.flair:
        lines.append(f"FLAIR: {post.flair}")
    if post.body:
        lines.append(f"BODY: {post.body[:2000]}")
    if post.links:
        lines.append(f"LINKS: {', '.join(post.links[:5])}")
    if source_text:
        lines.append(f"\nLINKED SOURCE:\n{source_text}")
    if post.comments:
        lines.append("\nTOP COMMENTS (for reception):")
        for c in post.comments:
            lines.append(f"- ({c.score:+d}) {c.body[:400]}")
    return "\n".join(lines)


# --- backends --------------------------------------------------------------
def _extract_ollama(system: str, prompt: str, schema: type[BaseModel]):
    resp = requests.post(
        f"{config.OLLAMA_HOST}/api/chat",
        json={
            "model": config.OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "format": schema.model_json_schema(),  # structured JSON output
            "stream": False,
            "keep_alive": config.OLLAMA_KEEP_ALIVE,  # release VRAM after idle
            "options": {"temperature": 0},
        },
        timeout=180,
    )
    resp.raise_for_status()
    content = resp.json()["message"]["content"]
    return schema.model_validate_json(content)


_anthropic_client = None


def _extract_anthropic(system: str, prompt: str, schema: type[BaseModel]):
    global _anthropic_client
    if _anthropic_client is None:
        from anthropic import Anthropic

        _anthropic_client = Anthropic()
    resp = _anthropic_client.messages.parse(
        model=config.EXTRACT_MODEL,
        max_tokens=config.EXTRACT_MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": prompt}],
        output_format=schema,
    )
    return resp.parsed_output


class _SubIdea(BaseModel):
    name: str
    track: Literal["ai", "content_gen", "financial"]
    reason: str


class _SubSuggestion(BaseModel):
    reply: str
    suggestions: list[_SubIdea]


_SUGGEST_SYS = (
    "You help configure 'Reddictator', a Reddit monitoring agent with three tracks: "
    "ai (AI tools & news), content_gen (image/video generation models & workflows), "
    "financial (markets & stocks). Given the user's interests and message, suggest real, "
    "active subreddits (bare names, no 'r/') that match, each tagged with the best-fit "
    "track and a one-line reason. Do NOT suggest subreddits already in their watch list. "
    "Then ask ONE short follow-up question to uncover more of their interests. Keep 'reply' "
    "to 1-2 sentences. Respond only with the requested JSON."
)


def suggest_subreddits(interests, watching, message="", history=None) -> dict:
    """Chatbot: propose subreddits matching the user's interests, ask for more."""
    lines = ["USER INTERESTS:"] + ([f"- {i}" for i in interests] or ["- (none yet)"])
    lines.append("\nALREADY WATCHING: " + (", ".join(watching) or "(none)"))
    lines.append(f"\nUSER MESSAGE: {message}" if message
                 else "\nSuggest subreddits that match these interests.")
    msgs = [{"role": "system", "content": _SUGGEST_SYS}]
    for h in (history or [])[-6:]:
        msgs.append({"role": h.get("role", "user"), "content": h.get("content", "")})
    msgs.append({"role": "user", "content": "\n".join(lines)})

    resp = requests.post(
        f"{config.OLLAMA_HOST}/api/chat",
        json={
            "model": config.OLLAMA_MODEL,
            "messages": msgs,
            "format": _SubSuggestion.model_json_schema(),
            "stream": False,
            "keep_alive": config.OLLAMA_KEEP_ALIVE,
            "options": {"temperature": 0.4},
        },
        timeout=180,
    )
    resp.raise_for_status()
    return _SubSuggestion.model_validate_json(resp.json()["message"]["content"]).model_dump()


def unload_model() -> bool:
    """Free the local model from the GPU (Ollama keep_alive=0). No-op for other backends."""
    if config.LLM_BACKEND != "ollama":
        return False
    try:
        requests.post(
            f"{config.OLLAMA_HOST}/api/generate",
            json={"model": config.OLLAMA_MODEL, "keep_alive": 0},
            timeout=30,
        )
        return True
    except Exception:
        return False


def extract_post(post: Post):
    """Run relevance + extraction for one post. Returns the parsed model."""
    source_text = enrich.fetch_source(post.links[0]) if post.links else ""
    system = _system_for(post.track)
    prompt = _post_prompt(post, source_text)
    schema = _model_for(post.track)

    if config.LLM_BACKEND == "anthropic":
        return _extract_anthropic(system, prompt, schema)
    return _extract_ollama(system, prompt, schema)
