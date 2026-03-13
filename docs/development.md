# Development Guide — 3DaaS

## Prerequisites

- **Docker Desktop** (or Docker Engine + compose plugin) — tested with Docker 24+
- **Python 3.12** — for running tests locally
- **gcloud CLI** — for interacting with GCP
- **GCP access** — service account key JSON for project `skillful-air-480018-f2`

---

## Initial Setup

### 1. Clone and configure

```bash
git clone <repo-url>
cd 3DaaS

cp .env.example .env
# Edit .env and fill in your values
```

Minimum required in `.env` for local development:

```bash
APP_ENV=local                          # uses pipeline_runs_dev in Firestore
GCP_PROJECT_ID=skillful-air-480018-f2
GCP_REGION=us-central1
GCS_BUCKET=bucket-saas-project
GCP_SERVICE_ACCOUNT=custom-models@skillful-air-480018-f2.iam.gserviceaccount.com
IMAGE_PREPROCESS=us-central1-docker.pkg.dev/skillful-air-480018-f2/nerfstudio-repo/image-preprocess:v2
IMAGE_TRAIN=us-central1-docker.pkg.dev/skillful-air-480018-f2/nerfstudio-repo/image-train-l4:v1
GCP_SA_KEY_PATH=./skillful-air-480018-f2-bcb000b8c0e2.json
PREFECT_API_URL=http://localhost:4200/api
PREFECT_API_KEY=
PREFECT_DEPLOYMENT=gaussian-pipeline/prod
```

### 2. Start the full stack

```bash
docker compose up --build
```

Services that start:

| Service | URL | Description |
|---|---|---|
| `prefect-server` | http://localhost:4200 | Prefect UI + API |
| `prefect-worker` | — | Executes flows from the `gcp-worker` pool |
| `api` | http://localhost:8080 | FastAPI gateway |

Wait ~60 seconds for the worker to register the deployment. Check:

```bash
docker compose logs -f prefect-worker
# Look for: "==> Prefect worker started."
```

### 3. Verify everything is working

```bash
curl http://localhost:8080/health
# {"status": "ok"}

curl http://localhost:4200/api/health
# {"status": "healthy"}
```

---

## Everyday Development Workflow

### Changing flow code (`flows/`)

The worker mounts `utils/` as a volume — changes to `utils/logger.py` are reflected immediately without rebuild.

For changes to `flows/` or `api/`, rebuild:

```bash
docker compose up --build prefect-worker api
# Or rebuild everything:
docker compose up --build
```

> **Tip:** Run `prefect-server` detached and rebuild only the worker:
> ```bash
> docker compose up -d prefect-server
> docker compose up --build prefect-worker
> ```

### Triggering a pipeline run manually

```bash
# Start a run
curl -X POST http://localhost:8080/pipeline/start \
  -H "Content-Type: application/json" \
  -d '{"dataset": "test_scene"}'

# Monitor status
curl http://localhost:8080/pipeline/<run_id>

# List all runs
curl http://localhost:8080/pipeline
```

From the **Prefect UI** (http://localhost:4200) you can:
- See all flow runs and their states
- Inspect task logs step by step
- Manually cancel a run
- Re-trigger a deployment

### Uploading test data to GCS

```bash
# Upload raw images (for full pipeline with COLMAP)
./scripts/sync_dataset.sh <dataset_name> raw ./path/to/images/

# Upload pre-processed data (skip COLMAP)
./scripts/sync_dataset.sh <dataset_name> data ./path/to/nerfstudio_data/
```

---

## Running Tests

Tests use mocked GCP services — no real GCP credentials needed.

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_vertex.py -v

# Run with coverage
pytest tests/ --cov=flows --cov=api --cov-report=term-missing
```

### Test structure

```
tests/
├── conftest.py       Shared fixtures — mocked GCS, Vertex AI, Firestore, webhooks
├── test_api.py       FastAPI endpoint tests
├── test_gcs.py       GCS task tests (detect, validate)
├── test_pipeline.py  Full pipeline flow tests (resume, skip_preprocess, etc.)
└── test_vertex.py    Vertex AI job submission and polling tests
```

### Key testing patterns

Prefect `@task` functions are called with `.fn()` to bypass the Prefect context:

```python
# In tests, always call task functions with .fn()
result = submit_preprocess_job.fn(run_id, dataset, params)
poll_vertex_job.fn(run_id, job_id)
```

GCP clients are mocked in `conftest.py` via `unittest.mock.patch`.

---

## Logging

### Viewing logs

```bash
# Stream all services
docker compose logs -f

# Worker only (where flows run)
docker compose logs -f prefect-worker

# API only
docker compose logs -f api

# Last 100 lines
docker compose logs --tail=100 prefect-worker
```

### Log files on disk

Logs are written to `./logs/` (mounted from containers):

```
logs/
├── worker/
│   ├── app.log       # INFO+ — all worker activity
│   ├── error.log     # ERROR+ — only errors (60 day retention)
│   └── debug.log     # DEBUG+ — verbose (7 day retention, local only)
└── api/
    ├── app.log
    └── error.log
```

```bash
# View worker errors
tail -f logs/worker/error.log

# Search for a specific run_id in all logs
grep "550e8400" logs/worker/app.log

# Follow all worker debug output
tail -f logs/worker/debug.log
```

### Log levels by environment

| Environment | Console level | Files |
|---|---|---|
| `APP_ENV=local` | `DEBUG` | app.log + error.log + debug.log |
| `APP_ENV=production` | `INFO` | app.log + error.log |

---

## Hot Reload for `utils/`

`utils/` is mounted as a Docker volume in both containers:

```yaml
volumes:
  - ./utils:/app/utils
```

Changes to `utils/logger.py` are picked up on the next Python import — no rebuild needed. For changes to take effect on already-running code, restart the container:

```bash
docker compose restart prefect-worker
```

---

## Environment Variables Reference

All variables and their defaults:

| Variable | Default | Description |
|---|---|---|
| `APP_ENV` | `production` | `local` → `pipeline_runs_dev`; `production` → `pipeline_runs` |
| `GCP_PROJECT_ID` | _(required)_ | GCP project |
| `GCP_REGION` | _(required)_ | GCP region |
| `GCS_BUCKET` | _(required)_ | GCS bucket name |
| `GCP_SERVICE_ACCOUNT` | _(required)_ | SA email |
| `IMAGE_PREPROCESS` | _(required)_ | Preprocess Docker image URI |
| `IMAGE_TRAIN` | _(required)_ | Training Docker image URI |
| `GCP_SA_KEY_PATH` | `""` | Path to SA JSON key (local only; empty on VM) |
| `GOOGLE_APPLICATION_CREDENTIALS` | `/secrets/sa-key.json` | Set by docker-compose from SA key |
| `PREFECT_API_URL` | `http://localhost:4200/api` | Prefect server URL |
| `PREFECT_API_KEY` | `""` | Prefect API key (empty for self-hosted) |
| `PREFECT_DEPLOYMENT` | `gaussian-pipeline/prod` | Deployment name |
| `PREFECT_WORK_POOL` | `gcp-worker` | Work pool name |
| `PREFECT_UI_API_URL` | `http://localhost:4200/api` | Browser-facing Prefect URL |
| `WEBHOOK_URL` | `""` | Webhook endpoint for pipeline events |
| `WANDB_API_KEY` | `""` | Weights & Biases API key (optional) |
| `GCE_VM_NAME` | — | VM name (for deploy_vm.sh) |
| `GCE_VM_ZONE` | — | VM zone (for deploy_vm.sh) |
| `GCE_VM_USER` | — | SSH user on VM (for deploy_vm.sh) |

---

## Adding a New Task

1. Create or edit a file in `flows/tasks/`
2. Decorate with `@task` from Prefect:
   ```python
   from prefect import task
   from utils.logger import get_logger

   _log = get_logger("worker.my_task")

   @task(name="my-task")
   def my_task(run_id: str, ...) -> ...:
       _log.info(f"[{run_id}] Starting my task")
       ...
   ```
3. Import and call it in `flows/pipeline.py`
4. Export from `flows/tasks/__init__.py`
5. Add tests in `tests/test_*.py`, calling the function with `.fn()`

---

## Debugging a Stuck Flow Run

### 1. Check Prefect UI

http://localhost:4200 → Flow Runs → click the stuck run → inspect task logs

### 2. Check worker logs

```bash
docker compose logs --tail=200 prefect-worker | grep -i error
```

### 3. Check Firestore state

```python
# In a Python shell
from api.db import get_run
print(get_run("<run_id>"))
```

### 4. Check Vertex AI job state

```bash
gcloud ai custom-jobs describe <job_id> --region=us-central1
gcloud ai custom-jobs stream-logs <job_id> --region=us-central1
```

### 5. Manually cancel and restart

```bash
# Via API
curl -X DELETE http://localhost:8080/pipeline/<run_id>

# Then re-trigger
curl -X POST http://localhost:8080/pipeline/start \
  -H "Content-Type: application/json" \
  -d '{"dataset": "<dataset>"}'
```

### 6. Run stale recovery manually

```bash
docker compose exec prefect-worker python /app/scripts/recover_stale_runs.py
```
