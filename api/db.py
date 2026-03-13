"""
api/db.py
=========
Firestore data-access layer for pipeline run records.

Collection: pipeline_runs
Document ID: run_id (UUID)

Schema per document:
  run_id        str       — unique run identifier
  dataset       str       — GCS dataset name
  status        str       — queued | running | done | failed
  stage         str|null  — current stage label
  started_at    str       — ISO-8601 UTC
  completed_at  str|null  — ISO-8601 UTC
  ply_uri       str|null  — gs:// URI of exported .ply
  error         str|null  — error message on failure
  params        dict      — full copy of all pipeline parameters

Usage:
  from api.db import create_run, update_run, get_run, list_runs, delete_run
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from google.cloud import firestore

_APP_ENV = os.environ.get("APP_ENV", "production")
COLLECTION = "pipeline_runs" if _APP_ENV == "production" else "pipeline_runs_dev"


@lru_cache(maxsize=1)
def _get_client() -> firestore.Client:
    """Return a cached Firestore client. Thread-safe via lru_cache."""
    return firestore.Client()


def _col():
    return _get_client().collection(COLLECTION)


# ── Write operations ──────────────────────────────────────────────────────────

def create_run(run_id: str, data: dict) -> None:
    """Insert a new run document. Raises if document already exists."""
    _col().document(run_id).set(data)


def update_run(run_id: str, updates: dict) -> None:
    """Partial update of an existing run document."""
    _col().document(run_id).update(updates)


def delete_run(run_id: str) -> None:
    """Delete a run document from Firestore."""
    _col().document(run_id).delete()


# ── Read operations ───────────────────────────────────────────────────────────

def get_run(run_id: str) -> Optional[dict]:
    """Return run document as dict, or None if not found."""
    doc = _col().document(run_id).get()
    return doc.to_dict() if doc.exists else None


def list_runs(limit: int = 100) -> list[dict]:
    """Return up to `limit` runs ordered by started_at descending."""
    docs = (
        _col()
        .order_by("started_at", direction=firestore.Query.DESCENDING)
        .limit(limit)
        .stream()
    )
    return [d.to_dict() for d in docs]
