"""
tests/test_pipeline.py
======================
Tests para la configuracion del pipeline y forwarding de params al Prefect flow.
"""
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

DATASET  = "test_scene"
RESOURCE = "projects/123/locations/us-central1/customJobs/456"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def mock_db(monkeypatch):
    store: dict = {}
    import api.db as db
    monkeypatch.setattr(db, "create_run",  lambda rid, d: store.update({rid: dict(d)}))
    monkeypatch.setattr(db, "update_run",  lambda rid, u: store[rid].update(u) if rid in store else None)
    monkeypatch.setattr(db, "get_run",     lambda rid: dict(store[rid]) if rid in store else None)
    monkeypatch.setattr(db, "list_runs",   lambda limit=100: list(store.values())[:limit])
    return store


@pytest.fixture(autouse=True)
def mock_prefect_trigger(monkeypatch):
    import api.main as m
    monkeypatch.setattr(m, "_trigger_prefect_flow", MagicMock(return_value="flow-run-id"))


@pytest.fixture()
def client():
    from api.main import app
    return TestClient(app)


# ── TrainParams dataclass ─────────────────────────────────────────────────────

class TestTrainParams:
    def test_no_refine_start_stop_fields(self):
        from flows.config import TrainParams
        tp = TrainParams()
        assert not hasattr(tp, "refine_start"), "refine_start debe estar eliminado"
        assert not hasattr(tp, "refine_stop"),  "refine_stop debe estar eliminado"

    def test_refine_every_still_present(self):
        from flows.config import TrainParams
        tp = TrainParams(refine_every=50)
        assert tp.refine_every == 50


# ── Prefect flow params forwarding ────────────────────────────────────────────

class TestPrefectFlowParamsForwarding:
    def test_trigger_called_with_correct_dataset(self, client, monkeypatch):
        import api.main as m
        mock_trigger = MagicMock(return_value="flow-run-id")
        monkeypatch.setattr(m, "_trigger_prefect_flow", mock_trigger)

        client.post("/pipeline/start", json={"dataset": DATASET})
        assert mock_trigger.call_count == 1
        _, req = mock_trigger.call_args[0]
        assert req.dataset == DATASET

    def test_skip_preprocess_forwarded(self, client, monkeypatch):
        import api.main as m
        mock_trigger = MagicMock(return_value="flow-run-id")
        monkeypatch.setattr(m, "_trigger_prefect_flow", mock_trigger)

        client.post("/pipeline/start", json={"dataset": DATASET, "skip_preprocess": True})
        _, req = mock_trigger.call_args[0]
        assert req.skip_preprocess is True

    def test_train_params_forwarded(self, client, monkeypatch):
        import api.main as m
        mock_trigger = MagicMock(return_value="flow-run-id")
        monkeypatch.setattr(m, "_trigger_prefect_flow", mock_trigger)

        client.post("/pipeline/start", json={
            "dataset": DATASET,
            "max_iters": 5000,
            "sh_degree": 1,
        })
        _, req = mock_trigger.call_args[0]
        assert req.max_iters == 5000
        assert req.sh_degree == 1

    def test_dataset_type_forwarded(self, client, monkeypatch):
        import api.main as m
        mock_trigger = MagicMock(return_value="flow-run-id")
        monkeypatch.setattr(m, "_trigger_prefect_flow", mock_trigger)

        client.post("/pipeline/start", json={"dataset": DATASET, "dataset_type": "dnerf"})
        _, req = mock_trigger.call_args[0]
        assert req.dataset_type == "dnerf"

    def test_dataset_type_defaults_to_auto(self, client, monkeypatch):
        import api.main as m
        mock_trigger = MagicMock(return_value="flow-run-id")
        monkeypatch.setattr(m, "_trigger_prefect_flow", mock_trigger)

        client.post("/pipeline/start", json={"dataset": DATASET})
        _, req = mock_trigger.call_args[0]
        assert req.dataset_type == "auto"

    def test_dataparser_override_forwarded(self, client, monkeypatch):
        import api.main as m
        mock_trigger = MagicMock(return_value="flow-run-id")
        monkeypatch.setattr(m, "_trigger_prefect_flow", mock_trigger)

        client.post("/pipeline/start", json={"dataset": DATASET, "dataparser": "blender-data"})
        _, req = mock_trigger.call_args[0]
        assert req.dataparser == "blender-data"
