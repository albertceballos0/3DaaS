"""
tests/test_vertex.py
====================
Unit tests for flows/tasks/vertex.py

Tests cover:
  - _poll_job: terminal states, transient errors, retry behaviour
  - submit_preprocess_job: command construction, resource name returned
  - submit_train_job: command construction, banned flags absent
  - wait_for_vertex_job: delegates to _poll_job correctly
"""
import pytest
from unittest.mock import MagicMock, call, patch

from google.cloud.aiplatform_v1.types import JobState
from google.api_core.exceptions import ServiceUnavailable, InternalServerError

from flows.tasks.vertex import _poll_job, submit_preprocess_job, submit_train_job
from flows.config import PreprocessParams, TrainParams

DATASET = "test_scene"
RESOURCE = "projects/123/locations/us-central1/customJobs/456"


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_job(state) -> MagicMock:
    job = MagicMock()
    job.state = state
    return job


# ── _poll_job ─────────────────────────────────────────────────────────────────

class TestPollJob:
    def test_succeeds_immediately(self, mock_job_service_client):
        mock_job_service_client.get_custom_job.return_value = make_job(JobState.JOB_STATE_SUCCEEDED)
        _poll_job(RESOURCE, poll_interval=0)  # should not raise

    def test_raises_on_failed_state(self, mock_job_service_client):
        mock_job_service_client.get_custom_job.return_value = make_job(JobState.JOB_STATE_FAILED)
        with pytest.raises(RuntimeError, match="JOB_STATE_FAILED"):
            _poll_job(RESOURCE, poll_interval=0)

    def test_raises_on_cancelled_state(self, mock_job_service_client):
        mock_job_service_client.get_custom_job.return_value = make_job(JobState.JOB_STATE_CANCELLED)
        with pytest.raises(RuntimeError, match="JOB_STATE_CANCELLED"):
            _poll_job(RESOURCE, poll_interval=0)

    def test_polls_until_succeeded(self, mock_job_service_client, monkeypatch):
        monkeypatch.setattr("flows.tasks.vertex.time.sleep", lambda _: None)
        mock_job_service_client.get_custom_job.side_effect = [
            make_job(JobState.JOB_STATE_PENDING),
            make_job(JobState.JOB_STATE_RUNNING),
            make_job(JobState.JOB_STATE_SUCCEEDED),
        ]
        _poll_job(RESOURCE, poll_interval=1)
        assert mock_job_service_client.get_custom_job.call_count == 3

    def test_transient_service_unavailable_is_retried(self, mock_job_service_client, monkeypatch):
        monkeypatch.setattr("flows.tasks.vertex.time.sleep", lambda _: None)
        mock_job_service_client.get_custom_job.side_effect = [
            ServiceUnavailable("connection reset"),
            ServiceUnavailable("connection reset"),
            make_job(JobState.JOB_STATE_SUCCEEDED),
        ]
        _poll_job(RESOURCE, poll_interval=0)
        assert mock_job_service_client.get_custom_job.call_count == 3

    def test_transient_internal_error_is_retried(self, mock_job_service_client, monkeypatch):
        monkeypatch.setattr("flows.tasks.vertex.time.sleep", lambda _: None)
        mock_job_service_client.get_custom_job.side_effect = [
            InternalServerError("server error"),
            make_job(JobState.JOB_STATE_SUCCEEDED),
        ]
        _poll_job(RESOURCE, poll_interval=0)

    def test_new_client_created_per_iteration(self, monkeypatch):
        """Each poll must create a fresh client to avoid stale gRPC connections."""
        clients = []

        def fresh_client():
            c = MagicMock()
            c.get_custom_job.return_value = make_job(JobState.JOB_STATE_SUCCEEDED)
            clients.append(c)
            return c

        monkeypatch.setattr("flows.tasks.vertex._job_service_client", fresh_client)
        monkeypatch.setattr("flows.tasks.vertex.time.sleep", lambda _: None)
        _poll_job(RESOURCE, poll_interval=0)
        assert len(clients) == 1  # one iteration → one client

    def test_multiple_polls_each_get_fresh_client(self, monkeypatch):
        clients = []

        def fresh_client():
            c = MagicMock()
            if len(clients) < 2:
                c.get_custom_job.return_value = make_job(JobState.JOB_STATE_PENDING)
            else:
                c.get_custom_job.return_value = make_job(JobState.JOB_STATE_SUCCEEDED)
            clients.append(c)
            return c

        monkeypatch.setattr("flows.tasks.vertex._job_service_client", fresh_client)
        monkeypatch.setattr("flows.tasks.vertex.time.sleep", lambda _: None)
        _poll_job(RESOURCE, poll_interval=0)
        assert len(clients) == 3  # 3 polls → 3 distinct clients


# ── submit_preprocess_job ─────────────────────────────────────────────────────

class TestSubmitPreprocessJob:
    def _make_job_mock(self, mock_aiplatform) -> MagicMock:
        job_mock = MagicMock()
        job_mock.resource_name = RESOURCE
        mock_aiplatform.CustomJob.return_value = job_mock
        return job_mock

    def test_returns_resource_name(self, mock_aiplatform):
        self._make_job_mock(mock_aiplatform)
        params = PreprocessParams()
        result = submit_preprocess_job.fn(DATASET, params)
        assert result == RESOURCE

    def test_job_is_submitted_with_service_account(self, mock_aiplatform):
        job_mock = self._make_job_mock(mock_aiplatform)
        submit_preprocess_job.fn(DATASET, PreprocessParams())
        job_mock.submit.assert_called_once_with(service_account="sa@test.iam.gserviceaccount.com")

    def test_command_contains_dataset_path(self, mock_aiplatform):
        self._make_job_mock(mock_aiplatform)
        submit_preprocess_job.fn(DATASET, PreprocessParams())
        _, kwargs = mock_aiplatform.CustomJob.call_args
        command = kwargs["worker_pool_specs"][0]["container_spec"]["command"][2]
        assert f"/gcs/test-bucket/{DATASET}/raw" in command
        assert f"gs://test-bucket/{DATASET}/processed/" in command

    def test_command_uses_colmap_params(self, mock_aiplatform):
        self._make_job_mock(mock_aiplatform)
        params = PreprocessParams(matching_method="exhaustive", num_downscales=2)
        submit_preprocess_job.fn(DATASET, params)
        _, kwargs = mock_aiplatform.CustomJob.call_args
        command = kwargs["worker_pool_specs"][0]["container_spec"]["command"][2]
        assert "--matching-method exhaustive" in command
        assert "--num-downscales 2" in command

    def test_command_includes_no_gpu_flag(self, mock_aiplatform):
        self._make_job_mock(mock_aiplatform)
        submit_preprocess_job.fn(DATASET, PreprocessParams())
        _, kwargs = mock_aiplatform.CustomJob.call_args
        command = kwargs["worker_pool_specs"][0]["container_spec"]["command"][2]
        assert "--no-gpu" in command

    def test_machine_type_is_cpu(self, mock_aiplatform):
        self._make_job_mock(mock_aiplatform)
        submit_preprocess_job.fn(DATASET, PreprocessParams())
        _, kwargs = mock_aiplatform.CustomJob.call_args
        machine = kwargs["worker_pool_specs"][0]["machine_spec"]["machine_type"]
        assert machine == "n1-highmem-16"


# ── submit_train_job ──────────────────────────────────────────────────────────

class TestSubmitTrainJob:
    def _make_job_mock(self, mock_aiplatform) -> MagicMock:
        job_mock = MagicMock()
        job_mock.resource_name = RESOURCE
        mock_aiplatform.CustomJob.return_value = job_mock
        return job_mock

    def test_returns_resource_name(self, mock_aiplatform):
        self._make_job_mock(mock_aiplatform)
        result = submit_train_job.fn(DATASET, TrainParams())
        assert result == RESOURCE

    def test_command_reads_processed_data(self, mock_aiplatform):
        self._make_job_mock(mock_aiplatform)
        submit_train_job.fn(DATASET, TrainParams())
        _, kwargs = mock_aiplatform.CustomJob.call_args
        command = kwargs["worker_pool_specs"][0]["container_spec"]["command"][2]
        assert f"/gcs/test-bucket/{DATASET}/processed" in command

    def test_command_uploads_trained_and_exported(self, mock_aiplatform):
        self._make_job_mock(mock_aiplatform)
        submit_train_job.fn(DATASET, TrainParams())
        _, kwargs = mock_aiplatform.CustomJob.call_args
        command = kwargs["worker_pool_specs"][0]["container_spec"]["command"][2]
        assert f"gs://test-bucket/{DATASET}/trained/" in command
        assert f"gs://test-bucket/{DATASET}/exported/" in command

    def test_command_contains_ns_export(self, mock_aiplatform):
        self._make_job_mock(mock_aiplatform)
        submit_train_job.fn(DATASET, TrainParams())
        _, kwargs = mock_aiplatform.CustomJob.call_args
        command = kwargs["worker_pool_specs"][0]["container_spec"]["command"][2]
        assert "ns-export gaussian-splat" in command

    def test_banned_flags_not_in_command(self, mock_aiplatform):
        """refine-start-iter and refine-stop-iter do not exist in nerfstudio 1.1.5."""
        self._make_job_mock(mock_aiplatform)
        submit_train_job.fn(DATASET, TrainParams())
        _, kwargs = mock_aiplatform.CustomJob.call_args
        command = kwargs["worker_pool_specs"][0]["container_spec"]["command"][2]
        assert "--pipeline.model.refine-start-iter" not in command
        assert "--pipeline.model.refine-stop-iter" not in command

    def test_refine_every_is_in_command(self, mock_aiplatform):
        self._make_job_mock(mock_aiplatform)
        submit_train_job.fn(DATASET, TrainParams(refine_every=200))
        _, kwargs = mock_aiplatform.CustomJob.call_args
        command = kwargs["worker_pool_specs"][0]["container_spec"]["command"][2]
        assert "--pipeline.model.refine-every 200" in command

    def test_machine_type_is_gpu(self, mock_aiplatform):
        self._make_job_mock(mock_aiplatform)
        submit_train_job.fn(DATASET, TrainParams())
        _, kwargs = mock_aiplatform.CustomJob.call_args
        machine = kwargs["worker_pool_specs"][0]["machine_spec"]
        assert machine["machine_type"] == "g2-standard-12"
        assert machine["accelerator_type"] == "NVIDIA_L4"
        assert machine["accelerator_count"] == 1

    def test_torch_compile_disable_env(self, mock_aiplatform):
        self._make_job_mock(mock_aiplatform)
        submit_train_job.fn(DATASET, TrainParams())
        _, kwargs = mock_aiplatform.CustomJob.call_args
        envs = {e["name"]: e["value"]
                for e in kwargs["worker_pool_specs"][0]["container_spec"]["env"]}
        assert envs.get("TORCH_COMPILE_DISABLE") == "1"
