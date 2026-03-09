"""
tests/test_api.py
=================
Tests for the FastAPI gateway (api/main.py).

Firestore (api.db) is mocked so tests run without a real GCP connection.
gaussian_pipeline is also mocked to run synchronously in most tests.

Convention:
  - Fixtures used only for side effects are listed via @pytest.mark.usefixtures()
    so static analysis tools don't flag them as unused parameters.
  - Fixtures whose return value is inspected are kept as method parameters.
"""
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

PLY_URI = "gs://test-bucket/test_scene/exported/splat.ply"
DATASET = "test_scene"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def mock_db(monkeypatch):
    """
    Replace all api.db functions with in-memory stubs so no Firestore
    connection is needed. Returns a dict acting as the fake store.
    """
    store: dict = {}

    def _create(run_id, data):
        store[run_id] = dict(data)

    def _update(run_id, updates):
        if run_id in store:
            store[run_id].update(updates)

    def _get(run_id):
        return dict(store[run_id]) if run_id in store else None

    def _list(limit=100):
        return list(store.values())[:limit]

    import api.db as db
    monkeypatch.setattr(db, "create_run",  _create)
    monkeypatch.setattr(db, "update_run",  _update)
    monkeypatch.setattr(db, "get_run",     _get)
    monkeypatch.setattr(db, "list_runs",   _list)

    return store


@pytest.fixture()
def sync_thread(monkeypatch):
    """Run the background thread synchronously so pipeline completes before assertions."""
    import api.main as m

    class _SyncThread:
        def __init__(self, target, args, daemon=False):
            self._target = target
            self._args = args
        def start(self):
            self._target(*self._args)

    monkeypatch.setattr(m.threading, "Thread", _SyncThread)


@pytest.fixture()
def mock_pipeline(monkeypatch):
    """Patch gaussian_pipeline to return a PLY URI immediately."""
    import api.main as m
    mock = MagicMock(return_value=PLY_URI)
    monkeypatch.setattr(m, "gaussian_pipeline", mock)
    return mock


@pytest.fixture()
def client():
    from api.main import app
    return TestClient(app)


# ── /health ───────────────────────────────────────────────────────────────────

def test_health_check(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ── POST /pipeline/start ──────────────────────────────────────────────────────

@pytest.mark.usefixtures("sync_thread", "mock_pipeline")
class TestStartPipeline:
    def test_returns_202_with_run_id(self, client):
        resp = client.post("/pipeline/start", json={"dataset": DATASET})
        assert resp.status_code == 202
        data = resp.json()
        assert "run_id" in data
        assert data["dataset"] == DATASET

    def test_run_persisted_to_firestore(self, client, mock_db):
        resp = client.post("/pipeline/start", json={"dataset": DATASET})
        run_id = resp.json()["run_id"]
        assert run_id in mock_db

    def test_run_completes_with_done_status(self, client):
        resp = client.post("/pipeline/start", json={"dataset": DATASET})
        data = resp.json()
        assert data["status"] == "done"
        assert data["ply_uri"] == PLY_URI
        assert data["completed_at"] is not None

    def test_run_fails_when_pipeline_raises(self, client, monkeypatch):
        import api.main as m
        monkeypatch.setattr(m, "gaussian_pipeline",
                            MagicMock(side_effect=RuntimeError("Vertex job FAILED")))
        resp = client.post("/pipeline/start", json={"dataset": DATASET})
        run_id = resp.json()["run_id"]
        run = client.get(f"/pipeline/{run_id}").json()
        assert run["status"] == "failed"
        assert "Vertex job FAILED" in run["error"]

    def test_initial_status_is_queued(self, client, monkeypatch):
        import api.main as m

        class _BlockingThread:
            def __init__(self, target, args, daemon=False): pass
            def start(self): pass

        monkeypatch.setattr(m.threading, "Thread", _BlockingThread)
        monkeypatch.setattr(m, "gaussian_pipeline", MagicMock())

        resp = client.post("/pipeline/start", json={"dataset": DATASET})
        assert resp.json()["status"] == "queued"

    def test_params_stored_in_firestore(self, client, mock_db):
        client.post("/pipeline/start", json={
            "dataset": DATASET,
            "max_iters": 5000,
            "skip_preprocess": True,
        })
        run = next(iter(mock_db.values()))
        assert run["params"]["max_iters"] == 5000
        assert run["params"]["skip_preprocess"] is True

    def test_skip_preprocess_passed_to_pipeline(self, client, mock_pipeline):
        client.post("/pipeline/start", json={"dataset": DATASET, "skip_preprocess": True})
        _, kwargs = mock_pipeline.call_args
        assert kwargs.get("skip_preprocess") is True

    def test_skip_preprocess_defaults_to_false(self, client, mock_pipeline):
        client.post("/pipeline/start", json={"dataset": DATASET})
        _, kwargs = mock_pipeline.call_args
        assert kwargs.get("skip_preprocess") is False

    def test_train_params_forwarded_to_pipeline(self, client, mock_pipeline):
        client.post("/pipeline/start", json={
            "dataset": DATASET,
            "max_iters": 5000,
            "sh_degree": 1,
        })
        _, kwargs = mock_pipeline.call_args
        assert kwargs["max_iters"] == 5000
        assert kwargs["sh_degree"] == 1

    def test_webhook_url_forwarded(self, client, mock_pipeline):
        client.post("/pipeline/start", json={
            "dataset": DATASET,
            "webhook_url": "https://example.com/hook",
        })
        _, kwargs = mock_pipeline.call_args
        assert kwargs["webhook_url"] == "https://example.com/hook"

    def test_multiple_runs_get_unique_ids(self, client):
        r1 = client.post("/pipeline/start", json={"dataset": DATASET}).json()
        r2 = client.post("/pipeline/start", json={"dataset": DATASET}).json()
        assert r1["run_id"] != r2["run_id"]

    def test_all_runs_persisted(self, client, mock_db):
        client.post("/pipeline/start", json={"dataset": "scene_a"})
        client.post("/pipeline/start", json={"dataset": "scene_b"})
        assert len(mock_db) == 2


# ── GET /pipeline/{run_id} ────────────────────────────────────────────────────

@pytest.mark.usefixtures("sync_thread", "mock_pipeline")
class TestGetRun:
    def test_returns_run_data(self, client):
        run_id = client.post("/pipeline/start", json={"dataset": DATASET}).json()["run_id"]
        resp = client.get(f"/pipeline/{run_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["run_id"] == run_id
        assert data["dataset"] == DATASET
        assert data["status"] == "done"
        assert data["ply_uri"] == PLY_URI

    def test_404_for_unknown_run(self, client):
        resp = client.get("/pipeline/nonexistent-id")
        assert resp.status_code == 404

    def test_response_has_all_expected_fields(self, client):
        run_id = client.post("/pipeline/start", json={"dataset": DATASET}).json()["run_id"]
        data = client.get(f"/pipeline/{run_id}").json()
        for field in ("run_id", "dataset", "status", "stage",
                      "started_at", "completed_at", "ply_uri", "error", "params"):
            assert field in data, f"Missing field: {field}"

    def test_params_field_contains_all_pipeline_params(self, client):
        run_id = client.post("/pipeline/start", json={"dataset": DATASET}).json()["run_id"]
        params = client.get(f"/pipeline/{run_id}").json()["params"]
        for key in ("max_iters", "sh_degree", "skip_preprocess", "matching_method",
                    "refine_every", "cull_scale", "reset_alpha"):
            assert key in params, f"Missing param: {key}"


# ── GET /pipeline ─────────────────────────────────────────────────────────────

class TestListRuns:
    def test_empty_list_initially(self, client):
        resp = client.get("/pipeline")
        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.usefixtures("sync_thread", "mock_pipeline")
    def test_lists_all_runs(self, client):
        client.post("/pipeline/start", json={"dataset": "scene_a"})
        client.post("/pipeline/start", json={"dataset": "scene_b"})
        resp = client.get("/pipeline")
        assert resp.status_code == 200
        datasets = {r["dataset"] for r in resp.json()}
        assert datasets == {"scene_a", "scene_b"}

    @pytest.mark.usefixtures("sync_thread", "mock_pipeline")
    def test_limit_query_param(self, client):
        for i in range(5):
            client.post("/pipeline/start", json={"dataset": f"scene_{i}"})
        resp = client.get("/pipeline?limit=3")
        assert resp.status_code == 200
        assert len(resp.json()) <= 3

    def test_limit_validation_rejects_zero(self, client):
        resp = client.get("/pipeline?limit=0")
        assert resp.status_code == 422

    def test_limit_validation_rejects_over_500(self, client):
        resp = client.get("/pipeline?limit=501")
        assert resp.status_code == 422
