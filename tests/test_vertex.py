"""
tests/test_vertex.py
====================
Unit tests para flows/tasks/vertex.py.

Las tasks de Prefect se invocan via .fn() para evitar la maquinaria de Prefect.
"""
import pytest
from unittest.mock import MagicMock

from google.cloud.aiplatform_v1.types import JobState
from google.api_core.exceptions import ServiceUnavailable, InternalServerError

from flows.tasks.vertex import (
    poll_vertex_job, submit_preprocess_job, submit_train_job,
    _parse_train_progress,
)
from flows.config import PreprocessParams, TrainParams, GCS_DATASET_PREFIX

DATASET  = "test_scene"
RESOURCE = "projects/123/locations/us-central1/customJobs/456"


def make_job(state) -> MagicMock:
    job = MagicMock()
    job.state = state
    return job


class TestPollJob:
    def test_succeeds_immediately(self, mock_job_service_client):
        mock_job_service_client.get_custom_job.return_value = make_job(JobState.JOB_STATE_SUCCEEDED)
        poll_vertex_job.fn(RESOURCE, poll_interval=0)

    def test_raises_on_failed_state(self, mock_job_service_client):
        mock_job_service_client.get_custom_job.return_value = make_job(JobState.JOB_STATE_FAILED)
        with pytest.raises(RuntimeError, match="JOB_STATE_FAILED"):
            poll_vertex_job.fn(RESOURCE, poll_interval=0)

    def test_raises_on_cancelled_state(self, mock_job_service_client):
        mock_job_service_client.get_custom_job.return_value = make_job(JobState.JOB_STATE_CANCELLED)
        with pytest.raises(RuntimeError, match="JOB_STATE_CANCELLED"):
            poll_vertex_job.fn(RESOURCE, poll_interval=0)

    def test_polls_until_succeeded(self, mock_job_service_client, monkeypatch):
        monkeypatch.setattr("flows.tasks.vertex.time.sleep", lambda _: None)
        mock_job_service_client.get_custom_job.side_effect = [
            make_job(JobState.JOB_STATE_PENDING),
            make_job(JobState.JOB_STATE_RUNNING),
            make_job(JobState.JOB_STATE_SUCCEEDED),
        ]
        poll_vertex_job.fn(RESOURCE, poll_interval=1)
        assert mock_job_service_client.get_custom_job.call_count == 3

    def test_transient_service_unavailable_is_retried(self, mock_job_service_client, monkeypatch):
        monkeypatch.setattr("flows.tasks.vertex.time.sleep", lambda _: None)
        mock_job_service_client.get_custom_job.side_effect = [
            ServiceUnavailable("connection reset"),
            ServiceUnavailable("connection reset"),
            make_job(JobState.JOB_STATE_SUCCEEDED),
        ]
        poll_vertex_job.fn(RESOURCE, poll_interval=0)
        assert mock_job_service_client.get_custom_job.call_count == 3

    def test_transient_internal_error_is_retried(self, mock_job_service_client, monkeypatch):
        monkeypatch.setattr("flows.tasks.vertex.time.sleep", lambda _: None)
        mock_job_service_client.get_custom_job.side_effect = [
            InternalServerError("server error"),
            make_job(JobState.JOB_STATE_SUCCEEDED),
        ]
        poll_vertex_job.fn(RESOURCE, poll_interval=0)

    def test_new_client_created_per_iteration(self, monkeypatch):
        clients = []
        def fresh_client():
            c = MagicMock()
            c.get_custom_job.return_value = make_job(JobState.JOB_STATE_SUCCEEDED)
            clients.append(c)
            return c
        monkeypatch.setattr("flows.tasks.vertex._job_service_client", fresh_client)
        monkeypatch.setattr("flows.tasks.vertex.time.sleep", lambda _: None)
        poll_vertex_job.fn(RESOURCE, poll_interval=0)
        assert len(clients) == 1

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
        poll_vertex_job.fn(RESOURCE, poll_interval=0)
        assert len(clients) == 3


class TestParseTrainProgress:
    def test_parses_step_and_total(self):
        logs = ["[step 3000/30000] training..."]
        p = _parse_train_progress(logs)
        assert p is not None
        assert p["step"] == 3000
        assert p["total_steps"] == 30000
        assert p["pct"] == 10.0

    def test_parses_loss(self):
        logs = ["step 1500/30000 Loss: 0.0234"]
        p = _parse_train_progress(logs)
        assert p["train_loss"] == pytest.approx(0.0234)

    def test_parses_train_loss_key(self):
        logs = ["[step: 001000/030000] train_loss=0.0567"]
        p = _parse_train_progress(logs)
        assert p["step"] == 1000
        assert p["train_loss"] == pytest.approx(0.0567)

    def test_returns_most_recent_step(self):
        logs = [
            "step 1000/30000 Loss: 0.05",
            "step 2000/30000 Loss: 0.04",
        ]
        p = _parse_train_progress(logs)
        assert p["step"] == 2000

    def test_returns_none_on_no_match(self):
        assert _parse_train_progress([]) is None
        assert _parse_train_progress(["COLMAP done", "Writing transforms.json"]) is None

    def test_pct_rounded_to_one_decimal(self):
        logs = ["step 1/3 loss=0.01"]
        p = _parse_train_progress(logs)
        assert p["pct"] == round(1 / 3 * 100, 1)

    # ── Formato nerfstudio rich output (Cloud Logging real) ──────────────────

    def test_parses_nerfstudio_rich_format(self):
        logs = ["4350 (14.50%)       68.353 ms            29 m, 13 s           9.36 M"]
        p = _parse_train_progress(logs)
        assert p is not None
        assert p["step"] == 4350
        assert p["pct"] == 14.5
        assert p["total_steps"] == 30000

    def test_parses_nerfstudio_rich_with_ansi(self):
        logs = ["4350 (14.50%)       68.353 ms            29 m, 13 s           9.36 M\x1b[0m"]
        p = _parse_train_progress(logs)
        assert p is not None
        assert p["step"] == 4350
        assert p["pct"] == 14.5

    def test_nerfstudio_rich_infers_total_steps(self):
        # step=15000, pct=50.0 → total=30000
        logs = ["15000 (50.00%)   45.2 ms   15 m, 0 s   10.1 M"]
        p = _parse_train_progress(logs)
        assert p["total_steps"] == 30000

    def test_nerfstudio_rich_returns_most_recent(self):
        logs = [
            "1000 (3.33%)    60 ms  ...",
            "4350 (14.50%)   68 ms  ...",
        ]
        p = _parse_train_progress(logs)
        assert p["step"] == 4350


class TestPollJobWithRunId:
    def test_updates_firestore_with_vertex_job_id(self, mock_job_service_client, monkeypatch):
        mock_db = MagicMock()
        monkeypatch.setattr("flows.tasks.vertex.db", mock_db)
        mock_job_service_client.get_custom_job.return_value = make_job(JobState.JOB_STATE_SUCCEEDED)

        poll_vertex_job.fn(RESOURCE, run_id="run-1", poll_interval=0)

        first_call_data = mock_db.update_run.call_args_list[0][0][1]
        assert first_call_data.get("vertex_job_id") == "456"

    def test_sends_job_submitted_webhook(self, mock_job_service_client, monkeypatch):
        mock_webhook = MagicMock()
        monkeypatch.setattr("flows.tasks.vertex.send_webhook", mock_webhook)
        mock_job_service_client.get_custom_job.return_value = make_job(JobState.JOB_STATE_SUCCEEDED)

        poll_vertex_job.fn(RESOURCE, run_id="run-1", job_type="train", poll_interval=0)

        events = [c[0][0] for c in mock_webhook.call_args_list]
        assert "job_submitted" in events
        assert "job_completed" in events

    def test_sends_job_failed_webhook_on_failure(self, mock_job_service_client, monkeypatch):
        mock_webhook = MagicMock()
        monkeypatch.setattr("flows.tasks.vertex.send_webhook", mock_webhook)
        mock_job_service_client.get_custom_job.return_value = make_job(JobState.JOB_STATE_FAILED)

        with pytest.raises(RuntimeError):
            poll_vertex_job.fn(RESOURCE, run_id="run-1", poll_interval=0)

        events = [c[0][0] for c in mock_webhook.call_args_list]
        assert "job_failed" in events

    def test_no_firestore_calls_without_run_id(self, mock_job_service_client, monkeypatch):
        mock_db = MagicMock()
        monkeypatch.setattr("flows.tasks.vertex.db", mock_db)
        mock_job_service_client.get_custom_job.return_value = make_job(JobState.JOB_STATE_SUCCEEDED)

        poll_vertex_job.fn(RESOURCE, poll_interval=0)

        mock_db.update_run.assert_not_called()

    def test_heartbeat_webhook_sent_while_running(self, mock_job_service_client, monkeypatch):
        mock_webhook = MagicMock()
        monkeypatch.setattr("flows.tasks.vertex.send_webhook", mock_webhook)
        monkeypatch.setattr("flows.tasks.vertex.time.sleep", lambda _: None)
        monkeypatch.setattr(
            "flows.tasks.vertex._fetch_vertex_logs",
            lambda job_id, since=None: (["step 3000/30000 Loss: 0.02"], None),
        )
        mock_job_service_client.get_custom_job.side_effect = [
            make_job(JobState.JOB_STATE_RUNNING),
            make_job(JobState.JOB_STATE_SUCCEEDED),
        ]

        poll_vertex_job.fn(RESOURCE, run_id="run-1", job_type="train", poll_interval=0)

        events = [c[0][0] for c in mock_webhook.call_args_list]
        assert "job_heartbeat" in events
        hb = next(c for c in mock_webhook.call_args_list if c[0][0] == "job_heartbeat")
        assert hb[0][1]["progress"]["step"] == 3000
        assert hb[0][1]["progress"]["pct"] == 10.0
        assert hb[0][1]["state"] == "JOB_STATE_RUNNING"

    def test_heartbeat_sent_every_running_poll(self, mock_job_service_client, monkeypatch):
        mock_webhook = MagicMock()
        monkeypatch.setattr("flows.tasks.vertex.send_webhook", mock_webhook)
        monkeypatch.setattr("flows.tasks.vertex.time.sleep", lambda _: None)
        mock_job_service_client.get_custom_job.side_effect = [
            make_job(JobState.JOB_STATE_RUNNING),
            make_job(JobState.JOB_STATE_RUNNING),
            make_job(JobState.JOB_STATE_RUNNING),
            make_job(JobState.JOB_STATE_SUCCEEDED),
        ]

        poll_vertex_job.fn(RESOURCE, run_id="run-1", poll_interval=0)

        heartbeats = [c for c in mock_webhook.call_args_list if c[0][0] == "job_heartbeat"]
        assert len(heartbeats) == 3


class TestSubmitPreprocessJob:
    def _make_job_mock(self, mock_aiplatform) -> MagicMock:
        job_mock = MagicMock()
        job_mock.resource_name = RESOURCE
        mock_aiplatform.CustomJob.return_value = job_mock
        return job_mock

    def test_returns_resource_name(self, mock_aiplatform):
        self._make_job_mock(mock_aiplatform)
        result = submit_preprocess_job.fn(DATASET, PreprocessParams())
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
        assert f"/gcs/test-bucket/{GCS_DATASET_PREFIX}/{DATASET}/raw" in command
        assert f"gs://test-bucket/{GCS_DATASET_PREFIX}/{DATASET}/processed/" in command

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
        assert f"/gcs/test-bucket/{GCS_DATASET_PREFIX}/{DATASET}/processed" in command

    def test_command_uploads_trained_and_exported(self, mock_aiplatform):
        self._make_job_mock(mock_aiplatform)
        submit_train_job.fn(DATASET, TrainParams())
        _, kwargs = mock_aiplatform.CustomJob.call_args
        command = kwargs["worker_pool_specs"][0]["container_spec"]["command"][2]
        assert f"gs://test-bucket/{GCS_DATASET_PREFIX}/{DATASET}/trained/" in command
        assert f"gs://test-bucket/{GCS_DATASET_PREFIX}/{DATASET}/exported/" in command

    def test_command_contains_ns_export(self, mock_aiplatform):
        self._make_job_mock(mock_aiplatform)
        submit_train_job.fn(DATASET, TrainParams())
        _, kwargs = mock_aiplatform.CustomJob.call_args
        command = kwargs["worker_pool_specs"][0]["container_spec"]["command"][2]
        assert "ns-export gaussian-splat" in command

    def test_banned_flags_not_in_command(self, mock_aiplatform):
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

    def test_custom_data_path_overrides_default(self, mock_aiplatform):
        self._make_job_mock(mock_aiplatform)
        custom_path = f"/gcs/test-bucket/{DATASET}/data"
        submit_train_job.fn(DATASET, TrainParams(), data_path=custom_path)
        _, kwargs = mock_aiplatform.CustomJob.call_args
        command = kwargs["worker_pool_specs"][0]["container_spec"]["command"][2]
        assert custom_path in command
        assert f"{GCS_DATASET_PREFIX}/{DATASET}/processed" not in command

    def test_dataparser_inserted_after_splatfacto(self, mock_aiplatform):
        self._make_job_mock(mock_aiplatform)
        submit_train_job.fn(DATASET, TrainParams(dataparser="dnerf-data"))
        _, kwargs = mock_aiplatform.CustomJob.call_args
        command = kwargs["worker_pool_specs"][0]["container_spec"]["command"][2]
        assert "dnerf-data" in command

    def test_empty_dataparser_not_in_command(self, mock_aiplatform):
        self._make_job_mock(mock_aiplatform)
        submit_train_job.fn(DATASET, TrainParams(dataparser=""))
        _, kwargs = mock_aiplatform.CustomJob.call_args
        command = kwargs["worker_pool_specs"][0]["container_spec"]["command"][2]
        # El comando debe ser "splatfacto  --data" sin argumento extra entre ellos
        assert "splatfacto  --data" in command or "splatfacto\n  --data" in command or "splatfacto   --data" in command
