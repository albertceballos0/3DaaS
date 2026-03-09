"""
flows/tasks/gcs.py
==================
Prefect tasks for validating GCS inputs and outputs.
"""

from __future__ import annotations

from google.cloud import storage
from prefect import task, get_run_logger

from flows.config import BUCKET
from flows.notify import notify


@task(name="validate-raw-input", retries=2, retry_delay_seconds=10)
def validate_raw_input(dataset: str) -> int:
    """Assert that gs://<BUCKET>/<dataset>/raw/ contains at least one image file."""
    logger = get_run_logger()
    notify(f"Validating raw input: gs://{BUCKET}/{dataset}/raw/")

    client = storage.Client()
    blobs = list(client.list_blobs(BUCKET, prefix=f"{dataset}/raw/", max_results=500))

    image_exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    images = [b for b in blobs if any(b.name.endswith(ext) for ext in image_exts)]

    if not images:
        raise ValueError(
            f"No images found at gs://{BUCKET}/{dataset}/raw/ "
            f"(found {len(blobs)} total objects). "
            f"Upload images before running the pipeline."
        )

    notify(f"Raw input OK — {len(images)} image(s) found", "OK")
    logger.info("Raw images: %d", len(images))
    return len(images)


@task(name="check-processed-exists", retries=2, retry_delay_seconds=10)
def check_processed_exists(dataset: str) -> None:
    """Check that processed data exists before skipping preprocess stage."""
    notify(f"Checking processed data: gs://{BUCKET}/{dataset}/processed/")

    client = storage.Client()
    blobs = {b.name for b in client.list_blobs(BUCKET, prefix=f"{dataset}/processed/")}

    transforms = f"{dataset}/processed/transforms.json"
    if transforms not in blobs:
        raise FileNotFoundError(
            f"skip_preprocess=True but no processed data found at gs://{BUCKET}/{transforms}. "
            f"Run the preprocessing stage first, or set skip_preprocess=False."
        )

    notify("Processed data found — skipping preprocess stage", "OK")


@task(name="validate-processed-output", retries=3, retry_delay_seconds=15)
def validate_processed_output(dataset: str) -> None:
    """Assert that transforms.json exists in <dataset>/processed/ after COLMAP."""
    notify(f"Validating COLMAP output: gs://{BUCKET}/{dataset}/processed/")

    client = storage.Client()
    blobs = {b.name for b in client.list_blobs(BUCKET, prefix=f"{dataset}/processed/")}

    transforms = f"{dataset}/processed/transforms.json"
    if transforms not in blobs:
        raise FileNotFoundError(
            f"transforms.json not found at gs://{BUCKET}/{transforms}. "
            f"COLMAP may have failed silently."
        )

    notify("Processed output OK — transforms.json found", "OK")


@task(name="validate-exported-output", retries=3, retry_delay_seconds=15)
def validate_exported_output(dataset: str) -> str:
    """Assert that a .ply file exists in <dataset>/exported/ and return its GCS URI."""
    notify(f"Validating export output: gs://{BUCKET}/{dataset}/exported/")

    client = storage.Client()
    blobs = list(client.list_blobs(BUCKET, prefix=f"{dataset}/exported/"))

    ply_blobs = [b for b in blobs if b.name.endswith(".ply")]
    if not ply_blobs:
        raise FileNotFoundError(
            f"No .ply file found at gs://{BUCKET}/{dataset}/exported/. "
            f"Export may have failed."
        )

    ply_uri = f"gs://{BUCKET}/{ply_blobs[0].name}"
    size_mb = ply_blobs[0].size / (1024 ** 2)
    notify(f"Export OK — {ply_uri} ({size_mb:.1f} MB)", "OK")
    return ply_uri
