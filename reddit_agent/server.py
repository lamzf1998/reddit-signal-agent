"""Local web server: a live dashboard + JSON API over the analyses DB.

Run:  python -m reddit_agent.server   (then open http://127.0.0.1:8765)

Serves your own machine only (127.0.0.1). It reads what the agent has already
analysed — run the agent (or the scheduled task) to populate/refresh the DB.
"""
from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory

from . import config, db

app = Flask(__name__)
_DASHBOARD = Path(__file__).parent / "dashboard.html"

# In-process extraction-run state (a manual run triggered from the dashboard).
_RUN = {"running": False, "started_at": None, "finished_at": None,
        "error": None, "subs": []}
_RUN_LOCK = threading.Lock()


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


def _clean_sub(s: str) -> str:
    s = str(s).strip().lstrip("/")
    if s.lower().startswith("r/"):
        s = s[2:]
    return s.strip("/").strip()


def _write_interests(new_lines):
    """Append unique interests to preferences.txt."""
    cur = [ln for ln in config.USER_PREFERENCES.splitlines() if ln.strip()]
    low = {c.lower() for c in cur}
    for i in new_lines:
        i = str(i).strip()
        if i and i.lower() not in low:
            cur.append(i); low.add(i.lower())
    config.PREFERENCES_FILE.write_text(
        "# Your interests — one per line. Managed from the dashboard.\n" + "\n".join(cur) + "\n",
        encoding="utf-8")
    return cur


@app.route("/api/subreddits", methods=["POST"])
def set_subreddits():
    incoming = request.get_json(force=True).get("subreddits") or {}
    cats = {k: dict(v, subreddits=list(v["subreddits"])) for k, v in config.CATEGORIES.items()}
    for key, meta in cats.items():
        seen, out = set(), []
        for s in incoming.get(key, meta["subreddits"]):
            s = _clean_sub(s)
            if s and s.lower() not in seen:
                seen.add(s.lower()); out.append(s)
        meta["subreddits"] = out
    config.save_categories(cats)
    config.reload()
    return jsonify(db.config_summary())


@app.route("/api/apply", methods=["POST"])
def apply_suggestions():
    """Chatbot approvals: add interests + subreddits, creating new categories as needed."""
    body = request.get_json(force=True)
    if body.get("interests"):
        _write_interests(body["interests"])
    cats = {k: dict(v, subreddits=list(v["subreddits"])) for k, v in config.CATEGORIES.items()}
    for a in (body.get("additions") or []):
        name = _clean_sub(a.get("name", ""))
        if not name:
            continue
        key = (str(a.get("category", "")).strip() or "misc").lower().replace(" ", "_")
        if key not in cats:
            cats[key] = {
                "label": a.get("category_label") or key.replace("_", " ").title(),
                "color": config._PALETTE[len(cats) % len(config._PALETTE)],
                "description": a.get("description", "") or (a.get("category_label") or key),
                "subreddits": [],
            }
        if not any(s.lower() == name.lower() for s in cats[key]["subreddits"]):
            cats[key]["subreddits"].append(name)
    config.save_categories(cats)
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


def _now():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _do_run():
    from . import main
    try:
        config.reload()  # pick up any interests/subreddits added since startup
        main.run(None, None, False)
    except Exception as e:  # noqa: BLE001 — surface to the dashboard, don't crash the server
        _RUN["error"] = str(e)[:300]
    finally:
        _RUN["running"] = False
        _RUN["finished_at"] = _now()


@app.route("/api/run", methods=["POST"])
def api_run():
    """Kick off one extraction run in the background (collect → analyse → export)."""
    with _RUN_LOCK:
        if _RUN["running"]:
            return jsonify(dict(_RUN, status="running"))
        config.reload()
        _RUN.update(running=True, started_at=_now(), finished_at=None,
                    error=None, subs=db.new_subreddits())
        threading.Thread(target=_do_run, daemon=True).start()
    return jsonify(dict(_RUN, status="started"))


@app.route("/api/run-status")
def api_run_status():
    return jsonify(dict(_RUN, new_subreddits=db.new_subreddits()))


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
