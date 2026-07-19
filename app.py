"""
app.py — minimal Flask app exposing the fine-tuned MedBot.

Routes:
  GET  /            chat UI (templates/index.html)
  POST /api/chat    JSON API: {"question": "..."} -> {"answer", "warning"}
  GET  /health      liveness/readiness info (model load status)

The model is loaded lazily (first request) or eagerly in a background
thread when WARMUP=1, so the container binds its port immediately and
hosting platforms don't kill it during the ~5-10 min weight download.
"""

import os
import threading

from flask import Flask, jsonify, render_template, request

import inference

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html", mock=inference.MOCK_MODE)


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    question = str(data.get("question", "")).strip()

    if not question:
        return jsonify({"error": "Please enter a question."}), 400
    if len(question) > 500:
        return jsonify({"error": "Question too long (max 500 characters)."}), 400

    try:
        result = inference.ask_medbot(question)
    except Exception as e:
        return jsonify({"error": f"Model error: {e}"}), 503

    return jsonify(result)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": inference.model_status()})


def _warmup():
    try:
        inference.load_model()
    except Exception:
        pass  # error is reported on /health


if os.getenv("WARMUP", "1") == "1" and not inference.MOCK_MODE:
    threading.Thread(target=_warmup, daemon=True).start()


if __name__ == "__main__":
    # Local development only; in Docker we run gunicorn (see Dockerfile).
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "7860")), debug=False)
