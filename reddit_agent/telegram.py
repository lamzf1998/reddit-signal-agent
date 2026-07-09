"""Deliver the digest to Telegram via the Bot API (no heavy dependency)."""
from __future__ import annotations

import html
import time

from . import config

try:
    import requests
except ImportError:
    requests = None  # type: ignore

_API = "https://api.telegram.org/bot{token}/sendMessage"
_LIMIT = 4000  # Telegram hard limit is 4096; leave headroom


def esc(text: str) -> str:
    return html.escape(text or "")


def _chunks(text: str) -> list[str]:
    out, buf = [], ""
    for line in text.split("\n"):
        if len(buf) + len(line) + 1 > _LIMIT:
            if buf:
                out.append(buf)
            # a single over-long line gets hard-split
            while len(line) > _LIMIT:
                out.append(line[:_LIMIT])
                line = line[_LIMIT:]
            buf = line
        else:
            buf = f"{buf}\n{line}" if buf else line
    if buf:
        out.append(buf)
    return out


def send(text: str) -> None:
    """Send `text` (HTML) to the configured chat, splitting as needed."""
    if not (config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID):
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set (see .env.example)."
        )
    if requests is None:
        raise RuntimeError("The 'requests' package is required to send to Telegram.")

    url = _API.format(token=config.TELEGRAM_BOT_TOKEN)
    for i, chunk in enumerate(_chunks(text)):
        resp = requests.post(
            url,
            json={
                "chat_id": config.TELEGRAM_CHAT_ID,
                "text": chunk,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=30,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Telegram API {resp.status_code}: {resp.text}")
        if i:  # gentle pacing across chunks
            time.sleep(0.4)
