"""Local web server: a live dashboard + JSON API over the analyses DB.

Run:  python -m reddit_agent.server   (then open http://127.0.0.1:8765)

Serves your own machine only (127.0.0.1). It reads what the agent has already
analysed — run the agent (or the scheduled task) to populate/refresh the DB.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from flask import Flask, Response, jsonify, request

from . import db

app = Flask(__name__)
_DASHBOARD = Path(__file__).parent / "dashboard.html"


@app.route("/")
def index() -> Response:
    return Response(_DASHBOARD.read_text(encoding="utf-8"), mimetype="text/html")


@app.route("/api/signals")
def signals():
    track = request.args.get("track") or None
    sent_only = request.args.get("sent") == "1"
    limit = min(int(request.args.get("limit", "500")), 2000)
    return jsonify(db.fetch_analyses(limit=limit, track=track, sent_only=sent_only))


@app.route("/api/stats")
def stats():
    return jsonify(db.stats())


def main() -> None:
    for stream in (sys.stdout, sys.stderr):  # Windows cp1252 safety
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    port = int(os.getenv("DASHBOARD_PORT", "8765"))
    print(f"Reddit Signal Agent dashboard -> http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
