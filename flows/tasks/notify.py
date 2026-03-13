"""
flows/tasks/notify.py
=====================
Webhook notifications for pipeline events. Non-fatal on failure.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import requests

from flows.config import WEBHOOK_URL

_log = logging.getLogger("worker.notify")


def send_webhook(event: str, payload: dict) -> None:
    """POST a JSON event to WEBHOOK_URL. Silently ignored if URL not configured."""
    if not WEBHOOK_URL:
        return
    try:
        resp = requests.post(
            WEBHOOK_URL,
            json={
                "event": event,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **payload,
            },
            timeout=10,
        )
        _log.debug(f"Webhook '{event}' enviado → {resp.status_code}")
    except Exception as exc:
        _log.warning(f"Webhook '{event}' fallido: {exc}")
