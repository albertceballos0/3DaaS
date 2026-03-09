"""
flows/notify.py
===============
Console logging and webhook notification helpers.
"""

from __future__ import annotations

from datetime import datetime

import requests


def _ts() -> str:
    return datetime.utcnow().strftime("%H:%M:%S")


def notify(msg: str, level: str = "INFO") -> None:
    icons = {"INFO": "ℹ", "OK": "✅", "WARN": "⚠", "ERROR": "❌", "START": "🚀", "WAIT": "⏳"}
    icon = icons.get(level, "•")
    print(f"[{_ts()}] {icon}  {msg}")


def notify_job_url(job_id: str, vertex_console: str) -> None:
    print(f"[{_ts()}]    Monitor → {vertex_console}")
    print(f"[{_ts()}]    Job ID  → {job_id}")


def post_webhook(url: str, payload: dict) -> None:
    """Fire-and-forget HTTP POST. Never raises — pipeline must not fail due to webhook."""
    if not url:
        return
    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
    except Exception as exc:
        print(f"[{_ts()}] ⚠  Webhook POST failed: {exc}")
