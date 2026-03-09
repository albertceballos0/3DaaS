"""
flows/tasks/vertex.py
=====================
Prefect tasks for submitting and polling Vertex AI Custom Jobs.
"""

from __future__ import annotations

import time
from datetime import datetime

from google.cloud import aiplatform
from google.cloud.aiplatform_v1 import JobServiceClient
from google.cloud.aiplatform_v1.types import JobState
from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import ServiceUnavailable, DeadlineExceeded, InternalServerError
from prefect import task, get_run_logger

from flows.config import (
    PROJECT_ID, REGION, BUCKET, SERVICE_ACCOUNT,
    IMAGE_PREPROCESS, IMAGE_TRAIN,
    API_ENDPOINT, VERTEX_CONSOLE,
)
from flows.config import PreprocessParams, TrainParams
from flows.notify import notify, notify_job_url


def _job_service_client() -> JobServiceClient:
    return JobServiceClient(
        client_options=ClientOptions(api_endpoint=API_ENDPOINT)
    )


_TRANSIENT_ERRORS = (ServiceUnavailable, DeadlineExceeded, InternalServerError)


def _poll_job(job_resource_name: str, poll_interval: int = 30) -> None:
    """Block until the Vertex AI Custom Job reaches a terminal state.

    A fresh JobServiceClient is created on every poll to avoid stale gRPC
    connections being reset after long idle periods (~40 min timeout).
    Transient network errors are caught and retried transparently.
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
            # Re-create client each iteration — prevents connection-reset errors
            # on long-running jobs (gRPC keepalive ~40 min).
            client = _job_service_client()
            job = client.get_custom_job(name=job_resource_name)
            state = job.state
        except _TRANSIENT_ERRORS as exc:
            notify(f"Transient API error ({exc.__class__.__name__}), retrying in {poll_interval}s …", "WAIT")
            time.sleep(poll_interval)
            continue

        if state == JobState.JOB_STATE_RUNNING:
            notify(f"Job RUNNING — next check in {poll_interval}s …", "WAIT")
        elif state in (JobState.JOB_STATE_QUEUED, JobState.JOB_STATE_PENDING):
            notify(f"Job PENDING — next check in {poll_interval}s …", "WAIT")

        if state in terminal:
            break

        time.sleep(poll_interval)

    if state != JobState.JOB_STATE_SUCCEEDED:
        raise RuntimeError(
            f"Vertex AI job ended with state: {state.name}\n"
            f"Job: {job_resource_name}\n"
            f"Check logs at {VERTEX_CONSOLE}"
        )


@task(name="submit-preprocess-job", retries=2, retry_delay_seconds=30)
def submit_preprocess_job(dataset: str, p: PreprocessParams) -> str:
    """Build COLMAP command, submit Vertex AI Custom Job, return resource name."""
    logger = get_run_logger()
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

    notify(f"Preprocess job submitted: {job_name}", "START")
    notify_job_url(job_id, VERTEX_CONSOLE)
    logger.info("Preprocess job resource: %s", resource_name)
    return resource_name


@task(name="submit-train-job", retries=2, retry_delay_seconds=30)
def submit_train_job(dataset: str, t: TrainParams) -> str:
    """Build splatfacto command, submit Vertex AI Custom Job, return resource name."""
    logger = get_run_logger()
    job_name = f"train_{dataset}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    command = (
        f"export TORCH_COMPILE_DISABLE=1 && "
        f"export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/nvidia/lib64 && "
        f"ns-train splatfacto "
        f"  --data /gcs/{BUCKET}/{dataset}/processed "
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
        f"  --pipeline.model.random-init False && "
        f"CONFIG_PATH=$(find /tmp/outputs/{dataset}/splatfacto/ -name 'config.yml' | head -n 1) && "
        f"echo \"Config found at: $CONFIG_PATH\" && "
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

    notify(f"Train job submitted: {job_name}", "START")
    notify_job_url(job_id, VERTEX_CONSOLE)
    logger.info("Train job resource: %s", resource_name)
    return resource_name


@task(name="wait-for-vertex-job", retries=1, retry_delay_seconds=60)
def wait_for_vertex_job(resource_name: str, stage: str, poll_interval: int = 30) -> None:
    """Poll a Vertex AI job until it reaches a terminal state."""
    notify(f"Waiting for {stage} job to complete …", "WAIT")
    _poll_job(resource_name, poll_interval=poll_interval)
    notify(f"{stage} job SUCCEEDED", "OK")
