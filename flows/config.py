"""
flows/config.py
===============
Environment configuration, parameter dataclasses, and data-structure schemas
for the 3DaaS pipeline.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATA STRUCTURES — where each piece of information lives
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─ FIRESTORE (DB) ─────────────────────────────────────────────────────────────
│  Collection: pipeline_runs / Document: <run_id>
│
│  run_id            str       UUID generado por la API al crear el run
│  dataset           str       Nombre del dataset en GCS
│  status            str       queued | running | done | failed
│  stage             str|null  Estado granular del stage actual (ver STAGES)
│  dataset_type      str       raw | nerfstudio | dnerf | blender | auto
│  image_count       int       Nº imágenes encontradas en raw/ (solo si raw)
│  vertex_job_id     str       ID numérico del último Vertex AI Custom Job
│  vertex_job_state  str       Estado actual del job en Vertex AI, actualizado
│                              en cada poll: JOB_STATE_QUEUED | JOB_STATE_PENDING
│                              | JOB_STATE_RUNNING | JOB_STATE_SUCCEEDED | ...
│  progress          dict      Solo durante training (ver TrainingProgress)
│  recent_logs       list[str] Últimas 10 líneas de log del job actual
│  started_at        str       ISO-8601 UTC — momento en que se creó el run
│  completed_at      str|null  ISO-8601 UTC — momento de finalización
│  ply_uri           str|null  gs:// URI del .ply exportado (solo si done)
│  wandb_url         str|null  URL pública del run en wandb.ai (si use_wandb=true)
│  wandb_metrics     dict|null Métricas escalares del summary de wandb, actualizado
│                              en cada poll: train/loss, val/psnr, system/gpu.0.gpu…
│  error             str|null  Mensaje de error (solo si failed)
│  params            dict      Copia completa de todos los parámetros del run
│  prefect_flow_run_id str     ID del flow run en Prefect
│
│  STAGES:
│    validating_dataset → detecting_dataset → validating_raw → preprocessing
│    → validating_processed → training → validating_exported → null (done/failed)
│
│  TrainingProgress:
│    step        int    Paso actual de entrenamiento
│    total_steps int    Total de pasos (= max_iters)
│    pct         float  Porcentaje completado (0.0–100.0)
│    train_loss  float  Loss más reciente (si disponible en logs)
│
└──────────────────────────────────────────────────────────────────────────────

┌─ API ────────────────────────────────────────────────────────────────────────
│  POST /pipeline/start   → acepta PipelineRequest, retorna el doc Firestore
│  GET  /pipeline/{id}    → retorna el doc Firestore completo (todos los campos)
│  GET  /pipeline         → lista de docs Firestore (limit 100 por defecto)
│  GET  /health           → {"status": "ok"}
│
│  El doc retornado incluye todos los campos de Firestore, incluyendo
│  progress y recent_logs si el job está en curso.
│
└──────────────────────────────────────────────────────────────────────────────

┌─ WEBHOOK EVENTS ─────────────────────────────────────────────────────────────
│  Todos los eventos incluyen: event (str), timestamp (ISO-8601 UTC)
│
│  pipeline_started
│    run_id, dataset, dataset_type
│
│  stage_changed
│    run_id, dataset, stage
│    + image_count (si stage=preprocessing)
│    + max_iters   (si stage=training)
│
│  job_submitted
│    run_id, job_type (preprocess|train), vertex_job_id, vertex_console
│
│  job_heartbeat                          ← cada poll_interval segundos (30s)
│    run_id, job_type, vertex_job_id
│    state       str   nombre del JobState de Vertex AI (ej. JOB_STATE_RUNNING)
│    progress    dict|null   TrainingProgress (solo job_type=train, si hay logs)
│    recent_logs list[str]   últimas 5 líneas de log nuevas en este ciclo
│
│  wandb_url_available                    ← una sola vez, al detectar la URL
│    run_id, job_type, vertex_job_id, wandb_url
│
│  job_heartbeat (con wandb activo añade además):
│    wandb_metrics  dict|null  mismo contenido que Firestore wandb_metrics
│
│  job_completed
│    run_id, job_type, vertex_job_id
│
│  job_failed
│    run_id, job_type, vertex_job_id, state
│
│  pipeline_done
│    run_id, dataset, ply_uri
│
│  pipeline_failed
│    run_id, dataset, error
│
└──────────────────────────────────────────────────────────────────────────────

┌─ PREFECT ────────────────────────────────────────────────────────────────────
│  Flow: gaussian-pipeline/prod
│  Params: run_id, dataset, dataset_type, skip_preprocess,
│          preprocess_params (dict → PreprocessParams),
│          train_params (dict → TrainParams)
│
│  El worker actualiza Firestore directamente — no usa callbacks API.
│  Los logs de Prefect se leen desde Cloud Logging (resource.type=ml_job).
│
└──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")


def _require(var: str) -> str:
    value = os.environ.get(var)
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{var}' is not set. "
            f"Check your .env file."
        )
    return value


PROJECT_ID       = _require("GCP_PROJECT_ID")
REGION           = _require("GCP_REGION")
BUCKET           = _require("GCS_BUCKET")
SERVICE_ACCOUNT  = _require("GCP_SERVICE_ACCOUNT")
IMAGE_PREPROCESS = _require("IMAGE_PREPROCESS")
IMAGE_TRAIN      = _require("IMAGE_TRAIN")
WEBHOOK_URL             = os.environ.get("WEBHOOK_URL", "")
WANDB_API_KEY           = os.environ.get("WANDB_API_KEY", "")
PREBUILT_DATA_SUBFOLDER = os.environ.get("PREBUILT_DATA_SUBFOLDER", "data")

_APP_ENV = os.environ.get("APP_ENV", "production")
GCS_DATASET_PREFIX = "pipeline_runs" if _APP_ENV == "production" else "pipeline_runs_dev"

API_ENDPOINT   = f"{REGION}-aiplatform.googleapis.com"
VERTEX_CONSOLE = (
    f"https://console.cloud.google.com/vertex-ai/training/custom-jobs"
    f"?project={PROJECT_ID}"
)


@dataclass
class PreprocessParams:
    matching_method: str  = "vocab_tree"  # vocab_tree | exhaustive | sequential
    sfm_tool:        str  = "colmap"      # colmap | hloc
    feature_type:    str  = "sift"        # sift | superpoint | superpoint_aachen
    matcher_type:    str  = "NN"          # NN | superglue | supergluefast
    num_downscales:  int  = 3
    skip_colmap:     bool = False


@dataclass
class TrainParams:
    dataparser:     str   = ""     # vacío = nerfstudio default; "dnerf-data", "blender-data"...
    max_iters:      int   = 30000
    sh_degree:      int   = 3
    num_random:     int   = 100000
    num_downscales: int   = 0
    res_schedule:   int   = 250
    densify_grad:   float = 0.0002
    densify_size:   float = 0.01
    n_split:        int   = 2
    refine_every:   int   = 100
    cull_scale:     float = 0.5
    reset_alpha:    int   = 30
    use_wandb:      bool  = False
