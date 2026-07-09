"""Best-effort off-Reddit link enrichment.

Fetches the linked source (GitHub / Hugging Face / Civitai / arXiv / …) and
returns a plain-text blob (title + readable text, truncated). Degrades to an
empty string on any failure — the caller then falls back to link-only.
"""
from __future__ import annotations

import re

from . import config

try:
    import requests
except ImportError:  # requests is optional; enrichment simply no-ops without it
    requests = None  # type: ignore

_TAG_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_ANGLE_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)

_HEADERS = {"User-Agent": "reddit-signal-agent/0.1 (+digest bot)"}


def fetch_source(url: str) -> str:
    """Return a text summary blob for `url`, or '' if it can't be fetched."""
    if not (config.ENRICH_LINKS and requests and url.startswith(("http://", "https://"))):
        return ""
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=config.ENRICH_TIMEOUT)
        resp.raise_for_status()
    except Exception:
        return ""

    ctype = resp.headers.get("content-type", "")
    if "html" not in ctype and "text" not in ctype:
        return ""  # skip binaries (images, model weights, pdfs)

    html = resp.text
    title_match = _TITLE_RE.search(html)
    title = _clean(title_match.group(1)) if title_match else ""

    body = _TAG_RE.sub(" ", html)
    body = _ANGLE_RE.sub(" ", body)
    body = _clean(body)[: config.ENRICH_MAX_CHARS]

    parts = []
    if title:
        parts.append(f"TITLE: {title}")
    if body:
        parts.append(f"TEXT: {body}")
    return "\n".join(parts)


def _clean(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()
