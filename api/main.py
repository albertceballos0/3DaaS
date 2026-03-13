"""
api/main.py
===========
FastAPI gateway para el pipeline 3DaaS Gaussian Splatting.

Endpoints públicos:
  POST   /pipeline/start    → dispara un Prefect flow run (retorna run_id)
  GET    /pipeline/{run_id} → estado del run (leído desde Firestore)
  GET    /pipeline          → lista de runs (paginada)
  DELETE /pipeline/{run_id} → cancela el flow run en Prefect y elimina de Firestore
  GET    /health            → health check

Todas las respuestas siguen el envelope estándar ApiResponse[T]:
  { "success": true,  "data": <T>,   "error": null }
  { "success": false, "data": null,  "error": { "code": "...", "message": "..." } }

Run:
  uvicorn api.main:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Generic, List, Optional, TypeVar

import requests
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
from dotenv import load_dotenv

# ── Bootstrap ─────────────────────────────────────────────────────────────────
load_dotenv(Path(__file__).parent.parent / ".env")

from utils.logger import get_logger, setup_logging
setup_logging("api")
logger = get_logger("api")

import api.db as db

app = FastAPI(title="3DaaS Pipeline API", version="5.0.0")

PREFECT_API_URL  = os.environ.get("PREFECT_API_URL", "")
PREFECT_API_KEY  = os.environ.get("PREFECT_API_KEY", "")
PREFECT_DEPLOYMENT = os.environ.get("PREFECT_DEPLOYMENT", "gaussian-pipeline/prod")
WANDB_API_KEY    = os.environ.get("WANDB_API_KEY", "")

_WANDB_RUN_RE = re.compile(r'https://wandb\.ai/([^/\s]+)/([^/\s]+)/runs/([^/\s?#]+)')

# Stages en los que tiene sentido consultar métricas en vivo
_LIVE_METRIC_STAGES = {"training", "preprocessing"}


# ── Response envelope ──────────────────────────────────────────────────────────

T = TypeVar("T")


class ErrorDetail(BaseModel):
    code: str
    message: str


class ApiResponse(BaseModel, Generic[T]):
    success: bool
    data: Optional[T] = None
    error: Optional[ErrorDetail] = None


def ok(data: T) -> ApiResponse[T]:
    return ApiResponse(success=True, data=data)


def err(code: str, message: str) -> ApiResponse:
    return ApiResponse(success=False, error=ErrorDetail(code=code, message=message))


# ── Domain models ──────────────────────────────────────────────────────────────

class TrainingProgress(BaseModel):
    step: int
    total_steps: int
    pct: float
    train_loss: Optional[float] = None


class PipelineRun(BaseModel):
    model_config = ConfigDict(extra="ignore")

    run_id: str
    dataset: str
    status: str                          # queued | running | done | failed | cancelled
    stage: Optional[str] = None
    dataset_type: Optional[str] = None
    image_count: Optional[int] = None
    vertex_job_id: Optional[str] = None
    vertex_job_state: Optional[str] = None
    progress: Optional[TrainingProgress] = None
    recent_logs: Optional[List[str]] = None
    started_at: str
    completed_at: Optional[str] = None
    ply_uri: Optional[str] = None
    wandb_url: Optional[str] = None
    wandb_metrics: Optional[dict] = None
    error: Optional[str] = None
    params: dict
    prefect_flow_run_id: Optional[str] = None


class CancelResult(BaseModel):
    run_id: str
    cancelled_in_prefect: bool


class HealthResponse(BaseModel):
    status: str
    version: str


# ── Request models ─────────────────────────────────────────────────────────────

class PipelineRequest(BaseModel):
    dataset: str
    # Dataset type — controla qué stages se ejecutan y qué dataparser usa nerfstudio
    # "auto"        → detectar automáticamente la estructura GCS
    # "raw"         → pipeline completo: COLMAP + training (imágenes en <dataset>/raw/)
    # "nerfstudio"  → solo training, datos en <dataset>/data/ (formato nerfstudio)
    # "dnerf"       → solo training, datos en <dataset>/data/ (formato dnerf)
    # "blender"     → solo training, datos en <dataset>/data/ (formato blender)
    # cualquier otro tipo válido de nerfstudio → solo training con "<type>-data"
    dataset_type: str = "auto"
    # Stage control (ignorado si dataset_type != "raw")
    skip_preprocess: bool = False
    # Preprocessing (COLMAP)
    matching_method: str = "vocab_tree"
    sfm_tool: str = "colmap"
    feature_type: str = "sift"
    matcher_type: str = "NN"
    num_downscales_preprocess: int = 3
    skip_colmap: bool = False
    # Training (splatfacto)
    dataparser: str = ""       # override explícito del dataparser; vacío = auto por tipo
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
    use_wandb: bool = False


# ── Exception handlers ─────────────────────────────────────────────────────────

@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException):
    code = f"HTTP_{exc.status_code}"
    return JSONResponse(
        status_code=exc.status_code,
        content=err(code, str(exc.detail)).model_dump(),
    )


@app.exception_handler(Exception)
async def generic_exception_handler(_: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=err("INTERNAL_ERROR", "An unexpected error occurred.").model_dump(),
    )


# ── Public routes ──────────────────────────────────────────────────────────────

@app.get("/health", response_model=ApiResponse[HealthResponse])
def health():
    return ok(HealthResponse(status="ok", version=app.version))


@app.post("/pipeline/start", status_code=201, response_model=ApiResponse[PipelineRun])
def start_pipeline(req: PipelineRequest):
    run_id = str(uuid.uuid4())
    run_data = {
        "run_id":       run_id,
        "dataset":      req.dataset,
        "status":       "queued",
        "stage":        None,
        "started_at":   datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "ply_uri":      None,
        "error":        None,
        "params":       req.model_dump(exclude={"dataset"}),
    }

    db.create_run(run_id, run_data)
    logger.info(f"Run creado: {run_id} dataset={req.dataset} type={req.dataset_type}")
    prefect_run_id = _trigger_prefect_flow(run_id, req)
    db.update_run(run_id, {"prefect_flow_run_id": prefect_run_id})
    logger.debug(f"Flow run de Prefect creado: {prefect_run_id} para run {run_id}")

    run = db.get_run(run_id)
    return ok(PipelineRun.model_validate(run))


@app.get("/pipeline", response_model=ApiResponse[List[PipelineRun]])
def list_runs(limit: int = Query(default=100, ge=1, le=500)):
    runs = [PipelineRun.model_validate(r) for r in db.list_runs(limit=limit)]
    return ok(runs)


@app.get("/pipeline/{run_id}", response_model=ApiResponse[PipelineRun])
def get_run(run_id: str):
    run = db.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    if run.get("stage") in _LIVE_METRIC_STAGES:
        run = _enrich_with_live_metrics(run)
    return ok(PipelineRun.model_validate(run))


@app.delete("/pipeline/{run_id}", status_code=200, response_model=ApiResponse[CancelResult])
def cancel_run(run_id: str):
    run = db.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

    prefect_flow_run_id = run.get("prefect_flow_run_id")
    cancelled_in_prefect = False
    if prefect_flow_run_id and run.get("status") not in ("done", "failed", "cancelled"):
        cancelled_in_prefect = _cancel_prefect_flow(prefect_flow_run_id)

    db.update_run(run_id, {
        "status":       "cancelled",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "error":        "Cancelled by user",
    })
    logger.info(f"Run {run_id} cancelado (cancelled_in_prefect={cancelled_in_prefect})")
    return ok(CancelResult(run_id=run_id, cancelled_in_prefect=cancelled_in_prefect))


# ── Live metrics enrichment ────────────────────────────────────────────────────

def _fetch_live_wandb_metrics(wandb_url: str) -> Optional[dict]:
    if not WANDB_API_KEY:
        return None
    m = _WANDB_RUN_RE.match(wandb_url)
    if not m:
        return None
    entity, project, run_id = m.groups()
    try:
        import wandb
        api = wandb.Api(api_key=WANDB_API_KEY, timeout=10)
        run = api.run(f"{entity}/{project}/{run_id}")
        source = dict(run.summary)
        if not source:
            hist = run.history(last=1, pandas=False)
            source = hist[0] if hist else {}
        return {
            k: round(v, 6) if isinstance(v, float) else v
            for k, v in source.items()
            if not k.startswith("_") and isinstance(v, (int, float))
        } or None
    except Exception as exc:
        logger.warning(f"No se pudieron obtener métricas de wandb ({wandb_url}): {exc}")
        return None


def _enrich_with_live_metrics(run: dict) -> dict:
    wandb_url = run.get("wandb_url")
    if wandb_url:
        live = _fetch_live_wandb_metrics(wandb_url)
        if live:
            run = {**run, "wandb_metrics": live}
    return run


# ── Prefect trigger ────────────────────────────────────────────────────────────

def _trigger_prefect_flow(run_id: str, req: PipelineRequest) -> str:
    """Dispara un Prefect flow run via Prefect API. Retorna el flow run ID."""
    if not PREFECT_API_URL:
        raise HTTPException(
            status_code=503,
            detail="PREFECT_API_URL is not configured.",
        )

    headers = {"Content-Type": "application/json"}
    if PREFECT_API_KEY:
        headers["Authorization"] = f"Bearer {PREFECT_API_KEY}"

    dep_resp = requests.get(
        f"{PREFECT_API_URL}/deployments/name/{PREFECT_DEPLOYMENT}",
        headers=headers,
        timeout=10,
    )
    if dep_resp.status_code != 200:
        logger.error(f"Deployment '{PREFECT_DEPLOYMENT}' no encontrado: {dep_resp.status_code} {dep_resp.text}")
        raise HTTPException(
            status_code=502,
            detail=f"Could not retrieve deployment '{PREFECT_DEPLOYMENT}'.",
        )
    deployment_id = dep_resp.json()["id"]

    params = req.model_dump(exclude={"dataset"})
    preprocess_params = {
        "matching_method":  params.pop("matching_method"),
        "sfm_tool":         params.pop("sfm_tool"),
        "feature_type":     params.pop("feature_type"),
        "matcher_type":     params.pop("matcher_type"),
        "num_downscales":   params.pop("num_downscales_preprocess"),
        "skip_colmap":      params.pop("skip_colmap"),
    }
    train_params = {
        "dataparser":     params.pop("dataparser"),
        "max_iters":      params.pop("max_iters"),
        "sh_degree":      params.pop("sh_degree"),
        "num_random":     params.pop("num_random"),
        "num_downscales": params.pop("num_downscales_train"),
        "res_schedule":   params.pop("res_schedule"),
        "densify_grad":   params.pop("densify_grad"),
        "densify_size":   params.pop("densify_size"),
        "n_split":        params.pop("n_split"),
        "refine_every":   params.pop("refine_every"),
        "cull_scale":     params.pop("cull_scale"),
        "reset_alpha":    params.pop("reset_alpha"),
        "use_wandb":      params.pop("use_wandb"),
    }

    run_resp = requests.post(
        f"{PREFECT_API_URL}/deployments/{deployment_id}/create_flow_run",
        headers=headers,
        json={
            "name": f"run-{run_id[:8]}",
            "parameters": {
                "run_id":            run_id,
                "dataset":           req.dataset,
                "dataset_type":      params.get("dataset_type", "auto"),
                "skip_preprocess":   params.get("skip_preprocess", False),
                "preprocess_params": preprocess_params,
                "train_params":      train_params,
            },
        },
        timeout=10,
    )
    if run_resp.status_code not in (200, 201):
        logger.error(f"Error al crear flow run en Prefect: {run_resp.status_code} {run_resp.text}")
        raise HTTPException(
            status_code=502,
            detail="Could not create Prefect flow run.",
        )

    return run_resp.json()["id"]


def _cancel_prefect_flow(prefect_flow_run_id: str) -> bool:
    """Envía señal de cancelación a un Prefect flow run. Retorna True si se canceló."""
    if not PREFECT_API_URL:
        return False
    headers = {"Content-Type": "application/json"}
    if PREFECT_API_KEY:
        headers["Authorization"] = f"Bearer {PREFECT_API_KEY}"
    try:
        resp = requests.post(
            f"{PREFECT_API_URL}/flow_runs/{prefect_flow_run_id}/set_state",
            headers=headers,
            json={"state": {"type": "CANCELLING"}, "force": True},
            timeout=10,
        )
        return resp.status_code in (200, 201)
    except Exception:
        return False
