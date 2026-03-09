"""
api/main.py
===========
FastAPI gateway for the 3DaaS Gaussian Splatting pipeline.

Endpoints:
  POST /pipeline/start          → trigger a new run (returns run_id immediately)
  GET  /pipeline/{run_id}       → get run status + result
  GET  /pipeline                → list runs (optional ?limit=N, default 100)
  GET  /health                  → health check

Persistence: every run is stored in Firestore (collection: pipeline_runs).

Run:
  uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
import sys
import uuid
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from dotenv import load_dotenv

# ── Bootstrap ─────────────────────────────────────────────────────────────────
load_dotenv(Path(__file__).parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent))

from flows.gaussian_pipeline import gaussian_pipeline  # noqa: E402
import api.db as db                                    # noqa: E402

app = FastAPI(title="3DaaS Pipeline API", version="2.0.0")


# ── Models ────────────────────────────────────────────────────────────────────

class PipelineRequest(BaseModel):
    dataset: str
    poll_interval: int = 30
    webhook_url: Optional[str] = None
    # Stage control
    skip_preprocess: bool = False
    # Preprocessing (COLMAP) — ignored when skip_preprocess=True
    matching_method: str = "vocab_tree"
    sfm_tool: str = "colmap"
    feature_type: str = "sift"
    matcher_type: str = "NN"
    num_downscales_preprocess: int = 3
    skip_colmap: bool = False
    # Training (splatfacto)
    max_iters: int = 30000
    sh_degree: int = 3
    num_random: int = 100000
    num_downscales_train: int = 0
    res_schedule: int = 250
    densify_grad: float = 0.0002
    densify_size: float = 0.01
    n_split: int = 2
    refine_every: int = 100
    cull_scale: float = 0.5
    reset_alpha: int = 30


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/pipeline/start", status_code=202)
def start_pipeline(req: PipelineRequest):
    run_id = str(uuid.uuid4())
    run_data = {
        "run_id":       run_id,
        "dataset":      req.dataset,
        "status":       "queued",
        "stage":        None,
        "started_at":   datetime.utcnow().isoformat() + "Z",
        "completed_at": None,
        "ply_uri":      None,
        "error":        None,
        "params": req.model_dump(exclude={"dataset", "webhook_url"}),
    }

    db.create_run(run_id, run_data)

    threading.Thread(
        target=_execute_pipeline,
        args=(run_id, req),
        daemon=True,
    ).start()

    return db.get_run(run_id)


@app.get("/pipeline", response_model=List[Dict[str, Any]])
def list_runs(limit: int = Query(default=100, ge=1, le=500)):
    return db.list_runs(limit=limit)


@app.get("/pipeline/{run_id}")
def get_run(run_id: str):
    run = db.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


# ── Background worker ─────────────────────────────────────────────────────────

def _execute_pipeline(run_id: str, req: PipelineRequest) -> None:
    db.update_run(run_id, {"status": "running"})
    try:
        ply_uri = gaussian_pipeline(
            run_id=run_id,
            webhook_url=req.webhook_url or os.environ.get("WEBHOOK_URL", ""),
            **req.model_dump(exclude={"webhook_url"}),
        )
        db.update_run(run_id, {
            "status":       "done",
            "ply_uri":      ply_uri,
            "completed_at": datetime.utcnow().isoformat() + "Z",
        })
    except Exception as exc:
        db.update_run(run_id, {
            "status":       "failed",
            "error":        str(exc),
            "completed_at": datetime.utcnow().isoformat() + "Z",
        })
