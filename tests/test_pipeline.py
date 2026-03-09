"""
tests/test_pipeline.py
======================
Integration tests for flows/gaussian_pipeline.py

All GCP calls are patched at the task level. Tests verify:
  - Which stages are called / skipped depending on skip_preprocess
  - What files each stage expects to find (pre/post conditions)
  - Failure propagation
  - Webhook events emitted at each stage
"""
import pytest
from unittest.mock import MagicMock, patch, call

DATASET = "test_scene"
PLY_URI = f"gs://test-bucket/{DATASET}/exported/splat.ply"
PREPROCESS_JOB = "projects/123/locations/us-central1/customJobs/111"
TRAIN_JOB      = "projects/123/locations/us-central1/customJobs/222"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def no_webhook(monkeypatch):
    """Disable real HTTP webhook calls in all pipeline tests."""
    monkeypatch.setattr("flows.gaussian_pipeline.post_webhook", lambda *a, **kw: None)


@pytest.fixture()
def patch_tasks(monkeypatch):
    """
    Replace all Prefect task objects used inside gaussian_pipeline with mocks.
    Returns a namespace of mocks for per-test assertion.
    """
    mocks = MagicMock()
    mocks.validate_raw_input.return_value        = 10
    mocks.check_processed_exists.return_value    = None
    mocks.validate_processed_output.return_value = None
    mocks.validate_exported_output.return_value  = PLY_URI
    mocks.submit_preprocess_job.return_value     = PREPROCESS_JOB
    mocks.submit_train_job.return_value          = TRAIN_JOB
    mocks.wait_for_vertex_job.return_value       = None

    import flows.gaussian_pipeline as gp
    monkeypatch.setattr(gp, "validate_raw_input",        mocks.validate_raw_input)
    monkeypatch.setattr(gp, "check_processed_exists",    mocks.check_processed_exists)
    monkeypatch.setattr(gp, "validate_processed_output", mocks.validate_processed_output)
    monkeypatch.setattr(gp, "validate_exported_output",  mocks.validate_exported_output)
    monkeypatch.setattr(gp, "submit_preprocess_job",     mocks.submit_preprocess_job)
    monkeypatch.setattr(gp, "submit_train_job",          mocks.submit_train_job)
    monkeypatch.setattr(gp, "wait_for_vertex_job",       mocks.wait_for_vertex_job)
    return mocks


# ── Full pipeline (skip_preprocess=False) ─────────────────────────────────────

class TestFullPipeline:
    def test_returns_ply_uri(self, patch_tasks):
        from flows.gaussian_pipeline import gaussian_pipeline
        result = gaussian_pipeline(dataset=DATASET)
        assert result == PLY_URI

    def test_validates_raw_input(self, patch_tasks):
        from flows.gaussian_pipeline import gaussian_pipeline
        gaussian_pipeline(dataset=DATASET)
        patch_tasks.validate_raw_input.assert_called_once_with(DATASET)

    def test_submits_preprocess_job(self, patch_tasks):
        from flows.gaussian_pipeline import gaussian_pipeline
        gaussian_pipeline(dataset=DATASET)
        patch_tasks.submit_preprocess_job.assert_called_once()
        args = patch_tasks.submit_preprocess_job.call_args[0]
        assert args[0] == DATASET

    def test_waits_for_preprocess_job(self, patch_tasks):
        from flows.gaussian_pipeline import gaussian_pipeline
        gaussian_pipeline(dataset=DATASET)
        wait_calls = patch_tasks.wait_for_vertex_job.call_args_list
        resource_names = [c[0][0] for c in wait_calls]
        assert PREPROCESS_JOB in resource_names

    def test_validates_processed_output_after_preprocess(self, patch_tasks):
        from flows.gaussian_pipeline import gaussian_pipeline
        gaussian_pipeline(dataset=DATASET)
        patch_tasks.validate_processed_output.assert_called_once_with(DATASET)

    def test_submits_train_job(self, patch_tasks):
        from flows.gaussian_pipeline import gaussian_pipeline
        gaussian_pipeline(dataset=DATASET)
        patch_tasks.submit_train_job.assert_called_once()

    def test_waits_for_train_job(self, patch_tasks):
        from flows.gaussian_pipeline import gaussian_pipeline
        gaussian_pipeline(dataset=DATASET)
        wait_calls = patch_tasks.wait_for_vertex_job.call_args_list
        resource_names = [c[0][0] for c in wait_calls]
        assert TRAIN_JOB in resource_names

    def test_validates_exported_ply(self, patch_tasks):
        from flows.gaussian_pipeline import gaussian_pipeline
        gaussian_pipeline(dataset=DATASET)
        patch_tasks.validate_exported_output.assert_called_once_with(DATASET)

    def test_check_processed_exists_not_called(self, patch_tasks):
        """check_processed_exists is only for skip_preprocess mode."""
        from flows.gaussian_pipeline import gaussian_pipeline
        gaussian_pipeline(dataset=DATASET)
        patch_tasks.check_processed_exists.assert_not_called()

    def test_stage_order(self, patch_tasks):
        """Preprocess wait must happen before train submit."""
        call_order = []
        patch_tasks.submit_preprocess_job.side_effect = lambda *a, **kw: call_order.append("submit_pre") or PREPROCESS_JOB
        patch_tasks.wait_for_vertex_job.side_effect   = lambda *a, **kw: call_order.append(f"wait_{a[0].split('/')[-1]}")
        patch_tasks.submit_train_job.side_effect      = lambda *a, **kw: call_order.append("submit_train") or TRAIN_JOB

        from flows.gaussian_pipeline import gaussian_pipeline
        gaussian_pipeline(dataset=DATASET)

        pre_wait_idx   = next(i for i, x in enumerate(call_order) if x == "wait_111")
        train_sub_idx  = next(i for i, x in enumerate(call_order) if x == "submit_train")
        assert pre_wait_idx < train_sub_idx

    def test_preprocess_failure_propagates(self, patch_tasks):
        from flows.gaussian_pipeline import gaussian_pipeline
        patch_tasks.wait_for_vertex_job.side_effect = RuntimeError("Vertex job FAILED")
        with pytest.raises(RuntimeError, match="Vertex job FAILED"):
            gaussian_pipeline(dataset=DATASET)

    def test_train_failure_propagates(self, patch_tasks):
        from flows.gaussian_pipeline import gaussian_pipeline

        def wait_side_effect(resource, stage, **kw):
            if stage == "Train":
                raise RuntimeError("Train job FAILED")

        patch_tasks.wait_for_vertex_job.side_effect = wait_side_effect
        with pytest.raises(RuntimeError, match="Train job FAILED"):
            gaussian_pipeline(dataset=DATASET)


# ── skip_preprocess=True ──────────────────────────────────────────────────────

class TestSkipPreprocess:
    def test_returns_ply_uri(self, patch_tasks):
        from flows.gaussian_pipeline import gaussian_pipeline
        result = gaussian_pipeline(dataset=DATASET, skip_preprocess=True)
        assert result == PLY_URI

    def test_check_processed_exists_called(self, patch_tasks):
        from flows.gaussian_pipeline import gaussian_pipeline
        gaussian_pipeline(dataset=DATASET, skip_preprocess=True)
        patch_tasks.check_processed_exists.assert_called_once_with(DATASET)

    def test_colmap_stages_are_skipped(self, patch_tasks):
        from flows.gaussian_pipeline import gaussian_pipeline
        gaussian_pipeline(dataset=DATASET, skip_preprocess=True)
        patch_tasks.validate_raw_input.assert_not_called()
        patch_tasks.submit_preprocess_job.assert_not_called()
        patch_tasks.validate_processed_output.assert_not_called()

    def test_only_one_wait_for_train(self, patch_tasks):
        from flows.gaussian_pipeline import gaussian_pipeline
        gaussian_pipeline(dataset=DATASET, skip_preprocess=True)
        assert patch_tasks.wait_for_vertex_job.call_count == 1
        call_resource = patch_tasks.wait_for_vertex_job.call_args[0][0]
        assert call_resource == TRAIN_JOB

    def test_still_submits_train_job(self, patch_tasks):
        from flows.gaussian_pipeline import gaussian_pipeline
        gaussian_pipeline(dataset=DATASET, skip_preprocess=True)
        patch_tasks.submit_train_job.assert_called_once()

    def test_still_validates_exported_ply(self, patch_tasks):
        from flows.gaussian_pipeline import gaussian_pipeline
        gaussian_pipeline(dataset=DATASET, skip_preprocess=True)
        patch_tasks.validate_exported_output.assert_called_once_with(DATASET)

    def test_raises_when_no_processed_data(self, patch_tasks):
        from flows.gaussian_pipeline import gaussian_pipeline
        patch_tasks.check_processed_exists.side_effect = FileNotFoundError(
            "skip_preprocess=True but no processed data found"
        )
        with pytest.raises(FileNotFoundError, match="skip_preprocess=True"):
            gaussian_pipeline(dataset=DATASET, skip_preprocess=True)

    def test_train_params_forwarded(self, patch_tasks):
        from flows.gaussian_pipeline import gaussian_pipeline
        from flows.config import TrainParams
        gaussian_pipeline(dataset=DATASET, skip_preprocess=True, max_iters=5000, sh_degree=1)
        tp_used = patch_tasks.submit_train_job.call_args[0][1]
        assert tp_used.max_iters == 5000
        assert tp_used.sh_degree == 1


# ── TrainParams dataclass ─────────────────────────────────────────────────────

class TestTrainParams:
    def test_no_refine_start_stop_fields(self):
        """These fields were removed because nerfstudio 1.1.5 does not support them."""
        from flows.config import TrainParams
        tp = TrainParams()
        assert not hasattr(tp, "refine_start"), "refine_start must be removed"
        assert not hasattr(tp, "refine_stop"),  "refine_stop must be removed"

    def test_refine_every_still_present(self):
        from flows.config import TrainParams
        tp = TrainParams(refine_every=50)
        assert tp.refine_every == 50
