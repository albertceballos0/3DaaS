"""
tests/test_api.py
=================
Tests para el FastAPI gateway (api/main.py).

api.db se mockea con un store en memoria.
_trigger_prefect_flow se mockea para evitar llamadas reales a Prefect Cloud.

Todas las respuestas siguen el envelope ApiResponse[T]:
  { "success": true,  "data": <T>,   "error": null }
  { "success": false, "data": null,  "error": { "code": "...", "message": "..." } }
"""
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

DATASET             = "test_scene"
PREFECT_FLOW_RUN_ID = "prefect-run-uuid-1234"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def mock_db(monkeypatch):
    """Reemplaza todas las funciones de api.db con stubs en memoria."""
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
    monkeypatch.setattr(db, "create_run", _create)
    monkeypatch.setattr(db, "update_run", _update)
    monkeypatch.setattr(db, "get_run",    _get)
    monkeypatch.setattr(db, "list_runs",  _list)
    return store


@pytest.fixture(autouse=True)
def mock_prefect(monkeypatch):
    """Mockea _trigger_prefect_flow para evitar llamadas reales a Prefect Cloud."""
    import api.main as m
    mock = MagicMock(return_value=PREFECT_FLOW_RUN_ID)
    monkeypatch.setattr(m, "_trigger_prefect_flow", mock)
    return mock


@pytest.fixture()
def client():
    from api.main import app
    return TestClient(app)


# ── /health ───────────────────────────────────────────────────────────────────

def test_health_check(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["status"] == "ok"
    assert "version" in body["data"]


# ── POST /pipeline/start ──────────────────────────────────────────────────────

class TestStartPipeline:
    def test_returns_201_with_run_id(self, client):
        resp = client.post("/pipeline/start", json={"dataset": DATASET})
        assert resp.status_code == 201
        data = resp.json()["data"]
        assert "run_id" in data
        assert data["dataset"] == DATASET

    def test_envelope_success_true(self, client):
        resp = client.post("/pipeline/start", json={"dataset": DATASET})
        assert resp.json()["success"] is True
        assert resp.json()["error"] is None

    def test_initial_status_is_queued(self, client):
        resp = client.post("/pipeline/start", json={"dataset": DATASET})
        assert resp.json()["data"]["status"] == "queued"

    def test_run_persisted_to_firestore(self, client, mock_db):
        resp = client.post("/pipeline/start", json={"dataset": DATASET})
        run_id = resp.json()["data"]["run_id"]
        assert run_id in mock_db

    def test_prefect_flow_triggered(self, client, mock_prefect):
        client.post("/pipeline/start", json={"dataset": DATASET})
        mock_prefect.assert_called_once()

    def test_prefect_flow_run_id_stored(self, client, mock_db):
        resp = client.post("/pipeline/start", json={"dataset": DATASET})
        run_id = resp.json()["data"]["run_id"]
        assert mock_db[run_id].get("prefect_flow_run_id") == PREFECT_FLOW_RUN_ID

    def test_params_stored_in_firestore(self, client, mock_db):
        client.post("/pipeline/start", json={
            "dataset": DATASET,
            "max_iters": 5000,
            "skip_preprocess": True,
        })
        run = next(iter(mock_db.values()))
        assert run["params"]["max_iters"] == 5000
        assert run["params"]["skip_preprocess"] is True

    def test_multiple_runs_get_unique_ids(self, client):
        r1 = client.post("/pipeline/start", json={"dataset": DATASET}).json()["data"]
        r2 = client.post("/pipeline/start", json={"dataset": DATASET}).json()["data"]
        assert r1["run_id"] != r2["run_id"]

    def test_all_runs_persisted(self, client, mock_db):
        client.post("/pipeline/start", json={"dataset": "scene_a"})
        client.post("/pipeline/start", json={"dataset": "scene_b"})
        assert len(mock_db) == 2

    def test_response_has_all_expected_fields(self, client):
        data = client.post("/pipeline/start", json={"dataset": DATASET}).json()["data"]
        for field in ("run_id", "dataset", "status", "stage",
                      "started_at", "completed_at", "ply_uri", "error", "params"):
            assert field in data, f"Campo faltante: {field}"

    def test_params_field_contains_pipeline_params(self, client):
        data = client.post("/pipeline/start", json={"dataset": DATASET}).json()["data"]
        params = data["params"]
        for key in ("max_iters", "sh_degree", "skip_preprocess", "matching_method",
                    "refine_every", "cull_scale", "reset_alpha"):
            assert key in params, f"Param faltante: {key}"


# ── GET /pipeline/{run_id} ────────────────────────────────────────────────────

class TestGetRun:
    def test_returns_run_data(self, client):
        run_id = client.post("/pipeline/start", json={"dataset": DATASET}).json()["data"]["run_id"]
        resp = client.get(f"/pipeline/{run_id}")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["run_id"] == run_id
        assert data["dataset"] == DATASET

    def test_envelope_success_true(self, client):
        run_id = client.post("/pipeline/start", json={"dataset": DATASET}).json()["data"]["run_id"]
        resp = client.get(f"/pipeline/{run_id}")
        assert resp.json()["success"] is True

    def test_404_for_unknown_run(self, client):
        resp = client.get("/pipeline/nonexistent-id")
        assert resp.status_code == 404
        body = resp.json()
        assert body["success"] is False
        assert body["error"]["code"] == "HTTP_404"


# ── GET /pipeline ─────────────────────────────────────────────────────────────

class TestListRuns:
    def test_empty_list_initially(self, client):
        resp = client.get("/pipeline")
        assert resp.status_code == 200
        assert resp.json()["data"] == []

    def test_lists_all_runs(self, client):
        client.post("/pipeline/start", json={"dataset": "scene_a"})
        client.post("/pipeline/start", json={"dataset": "scene_b"})
        resp = client.get("/pipeline")
        assert resp.status_code == 200
        datasets = {r["dataset"] for r in resp.json()["data"]}
        assert datasets == {"scene_a", "scene_b"}

    def test_limit_query_param(self, client):
        for i in range(5):
            client.post("/pipeline/start", json={"dataset": f"scene_{i}"})
        resp = client.get("/pipeline?limit=3")
        assert resp.status_code == 200
        assert len(resp.json()["data"]) <= 3

    def test_limit_validation_rejects_zero(self, client):
        resp = client.get("/pipeline?limit=0")
        assert resp.status_code == 422

    def test_limit_validation_rejects_over_500(self, client):
        resp = client.get("/pipeline?limit=501")
        assert resp.status_code == 422


# ── DELETE /pipeline/{run_id} ─────────────────────────────────────────────────

class TestCancelRun:
    def test_cancel_returns_200(self, client):
        run_id = client.post("/pipeline/start", json={"dataset": DATASET}).json()["data"]["run_id"]
        resp = client.delete(f"/pipeline/{run_id}")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["run_id"] == run_id
        assert "cancelled_in_prefect" in data

    def test_cancel_updates_status(self, client, mock_db):
        run_id = client.post("/pipeline/start", json={"dataset": DATASET}).json()["data"]["run_id"]
        client.delete(f"/pipeline/{run_id}")
        assert mock_db[run_id]["status"] == "cancelled"

    def test_cancel_404_for_unknown_run(self, client):
        resp = client.delete("/pipeline/nonexistent-id")
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "HTTP_404"
