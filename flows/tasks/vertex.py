"""
flows/tasks/vertex.py
=====================
Prefect tasks para submit y polling de Vertex AI Custom Jobs.
"""

from __future__ import annotations

import re
import time
from datetime import datetime
from typing import Optional

from prefect import task
from google.cloud import aiplatform
from google.cloud.aiplatform_v1 import JobServiceClient
from google.cloud.aiplatform_v1.types import JobState
from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import ServiceUnavailable, DeadlineExceeded, InternalServerError

import api.db as db
from utils.logger import get_logger
from flows.config import (
    PROJECT_ID, REGION, BUCKET, SERVICE_ACCOUNT,
    IMAGE_PREPROCESS, IMAGE_TRAIN,
    API_ENDPOINT, VERTEX_CONSOLE, WANDB_API_KEY,
    GCS_DATASET_PREFIX,
)
from flows.config import PreprocessParams, TrainParams
from flows.tasks.notify import send_webhook

_log = get_logger("worker.vertex")


def _job_service_client() -> JobServiceClient:
    return JobServiceClient(
        client_options=ClientOptions(api_endpoint=API_ENDPOINT)
    )


_TRANSIENT_ERRORS = (ServiceUnavailable, DeadlineExceeded, InternalServerError)

# Regex para parsear líneas de log de nerfstudio.
# Formato real de Cloud Logging: "4350 (14.50%)  68.353 ms  29 m, 13 s  9.36 M"
_STEP_PCT_RE = re.compile(r'\b(\d+)\s+\((\d+\.?\d*)%\)')
# Formato alternativo: "step 1000/30000" o "[step: 001000/030000]"
_STEP_RE = re.compile(r'step[:\s]*(\d+)\s*/\s*(\d+)', re.IGNORECASE)
# Loss: "train_loss=0.0123" o "Loss: 0.0123"
_LOSS_RE = re.compile(r'(?:train_loss|Loss)[=:\s]*([\d.eE+\-]+)')
# ANSI escape codes (colores de rich/nerfstudio)
_ANSI_RE = re.compile(r'\x1b\[[0-9;]*[mGKHF]')
# URL de wandb run: "wandb: 🚀 View run at https://wandb.ai/..."
_WANDB_URL_RE = re.compile(r'https://wandb\.ai/\S+')
# Parsea entity/project/run_id de una URL de wandb
_WANDB_RUN_RE = re.compile(r'https://wandb\.ai/([^/\s]+)/([^/\s]+)/runs/([^/\s?#]+)')


def _fetch_vertex_logs(
    job_id: str,
    since: Optional[datetime] = None,
) -> tuple[list[str], Optional[datetime]]:
    """Lee las entradas nuevas de Cloud Logging para un Vertex AI Custom Job.

    Returns:
        (log_lines, latest_timestamp) — listas vacías si falla o no hay entradas.
    """
    try:
        from google.cloud import logging as gcp_logging
        client = gcp_logging.Client(project=PROJECT_ID)
        filter_parts = [
            'resource.type="ml_job"',
            f'resource.labels.job_id="{job_id}"',
        ]
        if since:
            filter_parts.append(f'timestamp > "{since.strftime("%Y-%m-%dT%H:%M:%S.%fZ")}"')

        entries = list(client.list_entries(
            filter_=" AND ".join(filter_parts),
            order_by=gcp_logging.ASCENDING,
            page_size=100,
        ))

        lines: list[str] = []
        latest = since
        for e in entries:
            payload = e.payload
            raw = (
                payload.get("message", "") if isinstance(payload, dict) else str(payload)
            ).strip()
            text = _ANSI_RE.sub("", raw).strip()
            if text:
                lines.append(text)
            if e.timestamp and (latest is None or e.timestamp > latest):
                latest = e.timestamp
        return lines, latest
    except Exception:
        return [], since

def _fetch_wandb_metrics(wandb_url: str) -> Optional[dict]:
    """Obtiene las métricas actuales de un run de wandb a partir de su URL.

    Parsea entity/project/run_id de la URL y consulta el summary (actualizado
    en tiempo real, contiene PSNR, SSIM, LPIPS, gaussian_count, etc.).
    Fallback al último row del historial si summary está vacío.
    Returns None si falta WANDB_API_KEY, la URL es inválida, o hay error.
    """
    if not WANDB_API_KEY:
        return None
    m = _WANDB_RUN_RE.match(wandb_url)
    if not m:
        return None
    entity, project, run_id = m.groups()
    try:
        import wandb
        api = wandb.Api(api_key=WANDB_API_KEY, timeout=15)
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
        print(f"Error al obtener métricas de wandb: {exc}")
        return None


def _parse_train_progress(logs: list[str]) -> Optional[dict]:
    """Extrae el progreso de training más reciente de una lista de líneas de log.

    Soporta dos formatos:
      - Nerfstudio rich output: "4350 (14.50%)  68.353 ms  ..."
      - Formato explícito:      "step 1000/30000 Loss: 0.01"

    Returns dict con step, total_steps, pct, (train_loss si presente), o None.
    """
    for line in reversed(logs):
        clean = _ANSI_RE.sub("", line)

        # Formato nerfstudio rich: "N (P%)"
        m = _STEP_PCT_RE.search(clean)
        if m:
            step = int(m.group(1))
            pct = round(float(m.group(2)), 1)
            total = int(round(step * 100 / pct)) if pct > 0 else 0
            progress: dict = {"step": step, "total_steps": total, "pct": pct}
            lm = _LOSS_RE.search(clean)
            if lm:
                progress["train_loss"] = float(lm.group(1))
            return progress

        # Formato explícito: "step N/M"
        m = _STEP_RE.search(clean)
        if m:
            step, total = int(m.group(1)), int(m.group(2))
            pct = round(step / total * 100, 1) if total else 0.0
            progress = {"step": step, "total_steps": total, "pct": pct}
            lm = _LOSS_RE.search(clean)
            if lm:
                progress["train_loss"] = float(lm.group(1))
            return progress

    return None


def check_vertex_job_state(job_id: str) -> str:
    """Devuelve el nombre del estado actual de un Vertex AI Custom Job.

    Retorna 'JOB_STATE_UNSPECIFIED' si el job no se encuentra o hay un error.
    """
    resource = build_job_resource_name(job_id)
    try:
        client = _job_service_client()
        job = client.get_custom_job(name=resource)
        return job.state.name
    except Exception as exc:
        print(f"No se pudo obtener estado del job {job_id}: {exc}")
        return "JOB_STATE_UNSPECIFIED"


def build_job_resource_name(job_id: str) -> str:
    """Construye el resource name completo de un Vertex AI Custom Job."""
    return f"projects/{PROJECT_ID}/locations/{REGION}/customJobs/{job_id}"


@task(name="poll-vertex-job", log_prints=True)
def poll_vertex_job(
    job_resource_name: str,
    run_id: Optional[str] = None,
    job_type: str = "generic",
    poll_interval: int = 30,
    is_resume: bool = False,
) -> None:
    """Bloquea hasta que el Vertex AI Custom Job llegue a un estado terminal.

    Si se provee run_id, actualiza Firestore con progreso de logs y envía
    webhooks en cambios de estado y milestones de entrenamiento (cada 10%).

    Crea un JobServiceClient fresco en cada poll para evitar conexiones gRPC
    stale (timeout ~40 min). Los errores transitorios se reintentan silenciosamente.
    """
    terminal = {
        JobState.JOB_STATE_SUCCEEDED,
        JobState.JOB_STATE_FAILED,
        JobState.JOB_STATE_CANCELLED,
        JobState.JOB_STATE_PAUSED,
    }
    state = JobState.JOB_STATE_UNSPECIFIED
    job_id = job_resource_name.split("/")[-1]
    last_log_ts: Optional[datetime] = None
    wandb_url: Optional[str] = None

    if run_id:
        db.update_run(run_id, {"vertex_job_id": job_id})
        if not is_resume:
            send_webhook("job_submitted", {
                "run_id": run_id,
                "job_type": job_type,
                "vertex_job_id": job_id,
                "vertex_console": VERTEX_CONSOLE,
            })

    while True:
        try:
            client = _job_service_client()
            job = client.get_custom_job(name=job_resource_name)
            state = job.state
        except _TRANSIENT_ERRORS as exc:
            _log.warning(f"Error transitorio ({exc.__class__.__name__}) en job {job_id}, reintentando en {poll_interval}s")
            print(f"Error transitorio ({exc.__class__.__name__}), reintentando en {poll_interval}s...")
            time.sleep(poll_interval)
            continue

        new_lines: list[str] = []
        progress: Optional[dict] = None
        wandb_metrics: Optional[dict] = None

        if run_id:
            db.update_run(run_id, {"vertex_job_state": state.name})

        if state == JobState.JOB_STATE_RUNNING:
            print(f"Job RUNNING — próximo check en {poll_interval}s...")

            if run_id:
                new_lines, last_log_ts = _fetch_vertex_logs(job_id, last_log_ts)

                # Métricas de wandb (si ya tenemos la URL del run)
                if wandb_url:
                    wandb_metrics = _fetch_wandb_metrics(wandb_url)

                update: dict = {}
                if new_lines:
                    update["recent_logs"] = new_lines[-10:]
                    if job_type == "train":
                        progress = _parse_train_progress(new_lines)
                        if progress:
                            update["progress"] = progress
                            print(
                                f"Training: {progress['step']}/{progress['total_steps']}"
                                f" ({progress['pct']}%)"
                                + (f" loss={progress['train_loss']:.4f}" if "train_loss" in progress else "")
                            )
                        # Extraer URL de wandb la primera vez que aparezca en los logs
                        if not wandb_url:
                            for line in new_lines:
                                m = _WANDB_URL_RE.search(line)
                                if m:
                                    wandb_url = m.group(0).rstrip(")")
                                    update["wandb_url"] = wandb_url
                                    print(f"wandb URL: {wandb_url}")
                                    send_webhook("wandb_url_available", {
                                        "run_id": run_id,
                                        "job_type": job_type,
                                        "vertex_job_id": job_id,
                                        "wandb_url": wandb_url,
                                    })
                                    break

                if wandb_metrics:
                    update["wandb_metrics"] = wandb_metrics
                    print(f"wandb metrics: {list(wandb_metrics.keys())}")

                if update:
                    db.update_run(run_id, update)

        elif state in (JobState.JOB_STATE_QUEUED, JobState.JOB_STATE_PENDING):
            print(f"Job PENDING — próximo check en {poll_interval}s...")

        # Heartbeat en cada poll (incluyendo QUEUED/PENDING)
        if run_id and state not in terminal:
            send_webhook("job_heartbeat", {
                "run_id": run_id,
                "job_type": job_type,
                "vertex_job_id": job_id,
                "state": state.name,
                "progress": progress,
                "recent_logs": new_lines[-5:],
                "wandb_metrics": wandb_metrics,
            })

        if state in terminal:
            break

        time.sleep(poll_interval)

    if state != JobState.JOB_STATE_SUCCEEDED:
        _log.error(f"Job {job_id} (type={job_type}) terminó con estado: {state.name}")
        if run_id:
            send_webhook("job_failed", {
                "run_id": run_id,
                "job_type": job_type,
                "vertex_job_id": job_id,
                "state": state.name,
            })
        raise RuntimeError(
            f"Vertex AI job terminó con estado: {state.name}\n"
            f"Job: {job_resource_name}\n"
            f"Ver logs: {VERTEX_CONSOLE}"
        )

    if run_id:
        send_webhook("job_completed", {
            "run_id": run_id,
            "job_type": job_type,
            "vertex_job_id": job_id,
        })
    print(f"Job SUCCEEDED: {job_resource_name}")


@task(name="submit-preprocess-job", log_prints=True)
def submit_preprocess_job(dataset: str, p: PreprocessParams) -> str:
    """Construye comando COLMAP, submite Vertex AI Custom Job, retorna resource name."""
    job_name = f"preprocess_{dataset}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    skip_flag = "--skip-colmap" if p.skip_colmap else ""

    gcs_prefix = f"{GCS_DATASET_PREFIX}/{dataset}"
    command = (
        f"mkdir -p /tmp/processed && "
        f"ns-process-data images "
        f"  --data /gcs/{BUCKET}/{gcs_prefix}/raw "
        f"  --output-dir /tmp/processed "
        f"  --no-gpu "
        f"  --matching-method {p.matching_method} "
        f"  --sfm-tool {p.sfm_tool} "
        f"  --feature-type {p.feature_type} "
        f"  --matcher-type {p.matcher_type} "
        f"  --num-downscales {p.num_downscales} "
        f"  {skip_flag} && "
        f"gcloud storage cp -r /tmp/processed/* gs://{BUCKET}/{gcs_prefix}/processed/"
    )

    aiplatform.init(project=PROJECT_ID, location=REGION)
    job = aiplatform.CustomJob(
        display_name=job_name,
        worker_pool_specs=[{
            "machine_spec": {"machine_type": "n1-highmem-16"},
            "replica_count": 1,
            "container_spec": {
                "image_uri": IMAGE_PREPROCESS,
                "command": ["bash", "-c", command],
                "env": [{"name": "QT_QPA_PLATFORM", "value": "offscreen"}],
            },
        }],
        staging_bucket=f"gs://{BUCKET}",
    )

    job.submit(service_account=SERVICE_ACCOUNT)
    resource_name = job.resource_name
    job_id = resource_name.split("/")[-1]

    _log.info(f"Preprocess job submiteado: {job_name} (job_id={job_id})")
    print(f"Preprocess job submiteado: {job_name}")
    print(f"Monitor → {VERTEX_CONSOLE} | Job ID: {job_id}")
    return resource_name


@task(name="submit-train-job", log_prints=True)
def submit_train_job(dataset: str, t: TrainParams, data_path: str | None = None) -> str:
    """Construye comando splatfacto, submite Vertex AI Custom Job, retorna resource name.

    Args:
        dataset:   Nombre del dataset en GCS.
        t:         Parámetros de training. t.dataparser si no es vacío se inserta
                   justo después de ``splatfacto`` (p.ej. ``dnerf-data``).
        data_path: Ruta absoluta al dataset dentro del contenedor. Si None, usa
                   ``/gcs/<BUCKET>/<dataset>/processed`` (pipeline estándar post-COLMAP).
    """
    job_name = f"train_{dataset}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    gcs_prefix = f"{GCS_DATASET_PREFIX}/{dataset}"

    resolved_path = data_path or f"/gcs/{BUCKET}/{gcs_prefix}/processed"
    dataparser_arg = f"{t.dataparser} " if t.dataparser else ""
    vis_arg = "--vis wandb " if t.use_wandb else ""

    command = (
        f"export TORCH_COMPILE_DISABLE=1 && "
        f"export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/nvidia/lib64 && "
        f"ns-train splatfacto "
        f"  --data {resolved_path} "
        f"  --output-dir /tmp/outputs "
        f"  {vis_arg}"
        f"  --viewer.quit-on-train-completion True "
        f"  --project-name {dataset} "
        f"  --max-num-iterations {t.max_iters} "
        f"  --pipeline.model.sh-degree {t.sh_degree} "
        f"  --pipeline.model.sh-degree-interval 1000 "
        f"  --pipeline.model.num-random {t.num_random} "
        f"  --pipeline.model.num-downscales {t.num_downscales} "
        f"  --pipeline.model.resolution-schedule {t.res_schedule} "
        f"  --pipeline.model.densify-grad-thresh {t.densify_grad} "
        f"  --pipeline.model.densify-size-thresh {t.densify_size} "
        f"  --pipeline.model.n-split-samples {t.n_split} "
        f"  --pipeline.model.refine-every {t.refine_every} "
        f"  --pipeline.model.cull-scale-thresh {t.cull_scale} "
        f"  --pipeline.model.reset-alpha-every {t.reset_alpha} "
        f"  --pipeline.model.random-init False "
        f"  {dataparser_arg} && "
        f"CONFIG_PATH=$(find /tmp/outputs/ -path '*/splatfacto/*/config.yml' | head -n 1) && "
        f"echo \"Config found at: $CONFIG_PATH\" && "
        f"[ -n \"$CONFIG_PATH\" ] || {{ echo 'ERROR: config.yml not found under /tmp/outputs/'; exit 1; }} && "
        f"ns-export gaussian-splat "
        f"  --load-config $CONFIG_PATH "
        f"  --output-dir /tmp/export && "
        f"gcloud storage cp -r /tmp/outputs/* gs://{BUCKET}/{gcs_prefix}/trained/ && "
        f"gcloud storage cp -r /tmp/export/* gs://{BUCKET}/{gcs_prefix}/exported/"
    )

    aiplatform.init(project=PROJECT_ID, location=REGION)
    job = aiplatform.CustomJob(
        display_name=job_name,
        worker_pool_specs=[{
            "machine_spec": {
                "machine_type": "g2-standard-12",
                "accelerator_type": "NVIDIA_L4",
                "accelerator_count": 1,
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": IMAGE_TRAIN,
                "command": ["bash", "-c", command],
                "env": [
                    {"name": "QT_QPA_PLATFORM",       "value": "offscreen"},
                    {"name": "TORCH_COMPILE_DISABLE",  "value": "1"},
                    *(
                        [{"name": "WANDB_API_KEY", "value": WANDB_API_KEY}]
                        if t.use_wandb and WANDB_API_KEY else []
                    ),
                ],
            },
        }],
        staging_bucket=f"gs://{BUCKET}",
    )

    job.submit(service_account=SERVICE_ACCOUNT)
    resource_name = job.resource_name
    job_id = resource_name.split("/")[-1]

    _log.info(f"Train job submiteado: {job_name} (job_id={job_id})")
    print(f"Train job submiteado: {job_name}")
    print(f"Monitor → {VERTEX_CONSOLE} | Job ID: {job_id}")
    return resource_name
