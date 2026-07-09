"""Local web server: a live dashboard + JSON API over the analyses DB.

Run:  python -m reddit_agent.server   (then open http://127.0.0.1:8765)

Serves your own machine only (127.0.0.1). It reads what the agent has already
analysed — run the agent (or the scheduled task) to populate/refresh the DB.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory

from . import config, db

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


@app.route("/api/interests", methods=["POST"])
def set_interests():
    prefs = [str(p).strip() for p in (request.get_json(force=True).get("preferences") or [])
             if str(p).strip()]
    body = "# Your interests — one per line. Managed from the dashboard.\n" + "\n".join(prefs) + "\n"
    config.PREFERENCES_FILE.write_text(body, encoding="utf-8")
    config.reload()
    return jsonify(db.config_summary())


@app.route("/api/subreddits", methods=["POST"])
def set_subreddits():
    incoming = request.get_json(force=True).get("subreddits") or {}
    clean = {}
    for track in config.TRACK_SUBS:
        seen, out = set(), []
        for s in incoming.get(track, config.TRACK_SUBS[track]):
            s = str(s).strip().lstrip("/").removeprefix("r/").strip("/").strip()
            if s and s.lower() not in seen:
                seen.add(s.lower()); out.append(s)
        clean[track] = out
    config.SUBS_FILE.write_text(json.dumps(clean, indent=2), encoding="utf-8")
    config.reload()
    return jsonify(db.config_summary())


@app.route("/api/suggest", methods=["POST"])
def suggest():
    from . import extract
    config.reload()
    body = request.get_json(force=True)
    interests = [ln for ln in config.USER_PREFERENCES.splitlines() if ln.strip()]
    watching = [s for subs in config.TRACK_SUBS.values() for s in subs]
    try:
        return jsonify(extract.suggest_subreddits(
            interests, watching, body.get("message", ""), body.get("history")))
    except Exception as e:
        return jsonify({"error": str(e)[:200]}), 502


@app.route("/<path:fname>")
def asset(fname):
    """Serve static files (e.g. mbappe.jpg) from the docs/ folder."""
    docs = Path(__file__).parent.parent / "docs"
    if (docs / fname).is_file():
        return send_from_directory(docs, fname)
    return Response("not found", status=404)


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
