"""
tests/conftest.py
=================
Shared fixtures and environment setup for all tests.

Sets required GCP env vars BEFORE any project module is imported so that
flows/config.py does not raise EnvironmentError during collection.
"""
import os

# ── Env vars must be set before importing any flows.* module ──────────────────
os.environ.setdefault("GCP_PROJECT_ID",      "test-project")
os.environ.setdefault("GCP_REGION",          "us-central1")
os.environ.setdefault("GCS_BUCKET",          "test-bucket")
os.environ.setdefault("GCP_SERVICE_ACCOUNT", "sa@test.iam.gserviceaccount.com")
os.environ.setdefault("IMAGE_PREPROCESS",    "gcr.io/test/preprocess:v1")
os.environ.setdefault("IMAGE_TRAIN",         "gcr.io/test/train:v1")

import pytest
from unittest.mock import MagicMock


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_blob(name: str, size: int = 1024 * 512) -> MagicMock:
    """Return a mock GCS Blob with .name and .size set."""
    blob = MagicMock()
    blob.name = name
    blob.size = size
    return blob


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def silence_prefect_logger(monkeypatch):
    """Replace get_run_logger with a no-op so tasks can run outside a flow."""
    fake_logger = MagicMock()
    monkeypatch.setattr("flows.tasks.gcs.get_run_logger",    lambda: fake_logger, raising=False)
    monkeypatch.setattr("flows.tasks.vertex.get_run_logger", lambda: fake_logger, raising=False)
    return fake_logger


@pytest.fixture()
def mock_gcs_client(monkeypatch):
    """Patch google.cloud.storage.Client in gcs tasks and return the mock instance."""
    mock_client = MagicMock()
    monkeypatch.setattr("flows.tasks.gcs.storage.Client", lambda: mock_client)
    return mock_client


@pytest.fixture()
def mock_aiplatform(monkeypatch):
    """Patch google.cloud.aiplatform used in vertex tasks."""
    mock = MagicMock()
    monkeypatch.setattr("flows.tasks.vertex.aiplatform", mock)
    return mock


@pytest.fixture()
def mock_job_service_client(monkeypatch):
    """Patch _job_service_client() and return the mock JobServiceClient instance."""
    mock_client = MagicMock()
    monkeypatch.setattr("flows.tasks.vertex._job_service_client", lambda: mock_client)
    return mock_client
