"""
flows/tasks/vertex.py
=====================
Prefect tasks para submit y polling de Vertex AI Custom Jobs.
"""

from __future__ import annotations

import time
from datetime import datetime

from prefect import task
from google.cloud import aiplatform
from google.cloud.aiplatform_v1 import JobServiceClient
from google.cloud.aiplatform_v1.types import JobState
from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import ServiceUnavailable, DeadlineExceeded, InternalServerError

from flows.config import (
    PROJECT_ID, REGION, BUCKET, SERVICE_ACCOUNT,
    IMAGE_PREPROCESS, IMAGE_TRAIN,
    API_ENDPOINT, VERTEX_CONSOLE,
)
from flows.config import PreprocessParams, TrainParams


def _job_service_client() -> JobServiceClient:
    return JobServiceClient(
        client_options=ClientOptions(api_endpoint=API_ENDPOINT)
    )


_TRANSIENT_ERRORS = (ServiceUnavailable, DeadlineExceeded, InternalServerError)


@task(name="poll-vertex-job", log_prints=True)
def poll_vertex_job(job_resource_name: str, poll_interval: int = 30) -> None:
    """Bloquea hasta que el Vertex AI Custom Job llegue a un estado terminal.

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

    while True:
        try:
            client = _job_service_client()
            job = client.get_custom_job(name=job_resource_name)
            state = job.state
        except _TRANSIENT_ERRORS as exc:
            print(f"Error transitorio ({exc.__class__.__name__}), reintentando en {poll_interval}s...")
            time.sleep(poll_interval)
            continue

        if state == JobState.JOB_STATE_RUNNING:
            print(f"Job RUNNING — próximo check en {poll_interval}s...")
        elif state in (JobState.JOB_STATE_QUEUED, JobState.JOB_STATE_PENDING):
            print(f"Job PENDING — próximo check en {poll_interval}s...")

        if state in terminal:
            break

        time.sleep(poll_interval)

    if state != JobState.JOB_STATE_SUCCEEDED:
        raise RuntimeError(
            f"Vertex AI job terminó con estado: {state.name}\n"
            f"Job: {job_resource_name}\n"
            f"Ver logs: {VERTEX_CONSOLE}"
        )

    print(f"Job SUCCEEDED: {job_resource_name}")


@task(name="submit-preprocess-job", log_prints=True)
def submit_preprocess_job(dataset: str, p: PreprocessParams) -> str:
    """Construye comando COLMAP, submite Vertex AI Custom Job, retorna resource name."""
    job_name = f"preprocess_{dataset}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    skip_flag = "--skip-colmap" if p.skip_colmap else ""

    command = (
        f"mkdir -p /tmp/processed && "
        f"ns-process-data images "
        f"  --data /gcs/{BUCKET}/{dataset}/raw "
        f"  --output-dir /tmp/processed "
        f"  --no-gpu "
        f"  --matching-method {p.matching_method} "
        f"  --sfm-tool {p.sfm_tool} "
        f"  --feature-type {p.feature_type} "
        f"  --matcher-type {p.matcher_type} "
        f"  --num-downscales {p.num_downscales} "
        f"  {skip_flag} && "
        f"gcloud storage cp -r /tmp/processed/* gs://{BUCKET}/{dataset}/processed/"
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

    resolved_path = data_path or f"/gcs/{BUCKET}/{dataset}/processed"
    dataparser_arg = f"{t.dataparser} " if t.dataparser else ""

    command = (
        f"export TORCH_COMPILE_DISABLE=1 && "
        f"export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/nvidia/lib64 && "
        f"ns-train splatfacto "
        f"  --data {resolved_path} "
        f"  --output-dir /tmp/outputs "
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
        f"gcloud storage cp -r /tmp/outputs/* gs://{BUCKET}/{dataset}/trained/ && "
        f"gcloud storage cp -r /tmp/export/* gs://{BUCKET}/{dataset}/exported/"
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
                ],
            },
        }],
        staging_bucket=f"gs://{BUCKET}",
    )

    job.submit(service_account=SERVICE_ACCOUNT)
    resource_name = job.resource_name
    job_id = resource_name.split("/")[-1]

    print(f"Train job submiteado: {job_name}")
    print(f"Monitor → {VERTEX_CONSOLE} | Job ID: {job_id}")
    return resource_name
