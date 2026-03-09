# AGENTS.md — AI Agent Guide for 3DaaS · Gaussian Splatting Pipeline

This document defines how AI agents (Claude Code, Copilot, etc.) must understand, navigate, and contribute to this repository. Read this before making any changes.

---

## Repository Scope

This repo implements **Phase 1** and **Phase 2** of the 3DaaS Gaussian Splatting pipeline:

| Phase       | Name          | What lives here                                                                                                                |
| ----------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **Phase 1** | Core Compute  | Vertex AI job specs, Docker images, shell scripts that run COLMAP preprocessing and Gaussian Splatting training on GCP         |
| **Phase 2** | Orchestration | Prefect flows that automate the end-to-end pipeline on GCP, including job submission, status polling, and webhook notification |

> This repo does **not** contain: the Laravel backend, the Next.js frontend, the CLIP/ChromaDB search layer, or Terraform infrastructure code. Those live in separate repositories.

---

## System Context (Where This Repo Fits)

This repo is the **MLOps engine** of the broader 3DaaS platform:

```
[Other repo] Laravel → triggers Prefect flow via HTTP  ──►
                                                          [THIS REPO] Prefect
                                                            ├─ Phase 1: COLMAP on Vertex AI (CPU)
                                                            └─ Phase 2: splatfacto on Vertex AI (GPU)
                                                                        ↓
                                                          GCS: <dataset>_exported/*.ply
                                                                        ↓
                                                          Prefect → webhook back to Laravel
```

---

## Repository Structure

```
3DaaS/
├── specs/
│   ├── preprocess_spec.json      # Vertex AI spec: COLMAP job (n1-highmem-16, CPU)
│   └── train-export-spec.json    # Vertex AI spec: training job (g2-standard-12, L4 GPU)
├── scripts/
│   ├── run_preprocess_job.sh     # Submit Phase 1 job to Vertex AI
│   └── run_train_job.sh          # Submit Phase 2 job to Vertex AI
├── docker-images/
│   └── Dockerimage-process       # Dockerfile for COLMAP preprocessing image
├── flows/                        # Prefect flows (Phase 2 — in development)
│   └── gaussian_pipeline.py      # Main Prefect flow: preprocess → train → notify
├── sync_official_image.sh        # Syncs dromni/nerfstudio:1.1.5 to Artifact Registry
├── AGENTS.md                     # This file
└── specs.pdf                     # Full technical specification
```

---

## Phase 1 — Core Compute (Implemented)

### COLMAP Preprocessing

- **Script:** `scripts/run_preprocess_job.sh <dataset_name> [OPTIONS]`
- **Spec:** `specs/preprocess_spec.json`
- **Machine:** `n1-highmem-16` — CPU only, **no GPU**
- **Image:** `image-preprocess:v2`
- Reads raw images from `gs://bucket-saas-project/<dataset>/raw/`
- Writes COLMAP output to `gs://bucket-saas-project/<dataset>/processed/`
- Optional flags: `--matching-method`, `--sfm-tool`, `--feature-type`, `--matcher-type`, `--num-downscales`, `--skip-colmap`

### Gaussian Splatting Training + Export

- **Script:** `scripts/run_train_job.sh <dataset_name> [OPTIONS]`
- **Spec:** `specs/train-export-spec.json`
- **Machine:** `g2-standard-12` + NVIDIA L4 GPU ×1
- **Image:** `image-train-l4:v1` (official `dromni/nerfstudio:1.1.5`)
- Reads from `gs://bucket-saas-project/<dataset>/processed/`
- Trains `splatfacto` (default: 30,000 iterations)
- Exports `.ply` to `gs://bucket-saas-project/<dataset>/exported/`
- Saves checkpoints to `gs://bucket-saas-project/<dataset>/trained/`
- Optional flags: `--max-iters`, `--sh-degree`, `--num-random`, `--densify-grad`, `--refine-start`, `--refine-stop`, `--cull-scale`, and more (see `--help`)

### GCS Folder Structure (must be respected everywhere)

```
gs://bucket-saas-project/
  <dataset>/
    raw/         # Raw input images (user uploads here)
    processed/   # Stage 1 output: transforms.json + undistorted images
    trained/     # Stage 2 output: Nerfstudio checkpoints
    exported/    # Stage 2 output: final .ply (~200 MB)
```

All dataset data is nested under a single `<dataset>/` prefix — this allows clean deletion,
listing, and per-dataset IAM scoping. Never write directly to the bucket root.

---

## Phase 2 — Prefect Orchestration (In Development)

### Goal

Replace the manual two-step shell script workflow with a fully automated Prefect flow that:

1. Receives a trigger (dataset name + GCS path)
2. Submits the COLMAP job and waits for completion
3. On success, submits the training job and waits for completion
4. On success, sends a webhook back to the Laravel backend
5. On any failure, retries with backoff and notifies of failure

### Prefect Flow Design

```
@flow: gaussian_pipeline(dataset_name)
  ├─ @task: submit_preprocess_job(dataset_name)
  │    └─ polls Vertex AI job status until SUCCEEDED or FAILED
  ├─ @task: submit_train_job(dataset_name)          # only if preprocess succeeded
  │    └─ polls Vertex AI job status until SUCCEEDED or FAILED
  └─ @task: notify_completion(dataset_name, status)
       └─ POST webhook to Laravel backend
```

### Key Prefect Constraints

- Training must **only** run after preprocessing **succeeds** — never bypass this dependency.
- Use Prefect's `@task` retries with exponential backoff for job submission failures.
- Do not implement busy-wait loops — use Vertex AI job state polling with `time.sleep` intervals or Prefect's built-in wait primitives.
- Prefect flows run locally or as a Prefect Worker on a lightweight VM/Cloud Run — not on GKE in this repo.

### Vertex AI Job Polling Pattern

```python
import time
from google.cloud import aiplatform

def wait_for_job(job_name: str, poll_interval: int = 30):
    client = aiplatform.gapic.JobServiceClient(...)
    while True:
        job = client.get_custom_job(name=job_name)
        state = job.state.name
        if state == "JOB_STATE_SUCCEEDED":
            return True
        if state in ("JOB_STATE_FAILED", "JOB_STATE_CANCELLED"):
            raise RuntimeError(f"Job {job_name} ended with state {state}")
        time.sleep(poll_interval)
```

---

## GCP Resources

| Resource          | Value                                                                |
| ----------------- | -------------------------------------------------------------------- |
| Project ID        | `skillful-air-480018-f2`                                             |
| Region            | `us-central1`                                                        |
| GCS Bucket        | `bucket-saas-project`                                                |
| Artifact Registry | `us-central1-docker.pkg.dev/skillful-air-480018-f2/nerfstudio-repo/` |
| Service Account   | `custom-models@skillful-air-480018-f2.iam.gserviceaccount.com`       |

---

## Agent Rules & Constraints

### Security

- **Never commit secrets.** Service account key files (`*.json`) must be in `.gitignore`.
- Never hardcode GCP project IDs, bucket names, or service account emails — use environment variables.
- Use GCP Secret Manager or environment injection for credentials in Prefect flows.

### Vertex AI Specs (`specs/*.json`)

- The preprocessing spec must remain **CPU-only** (`n1-highmem-16`) — do NOT add an accelerator.
- The training spec requires **NVIDIA L4** on `g2-standard-12` — do not downgrade the machine type.
- `TORCH_COMPILE_DISABLE=1` must remain set in the training container — do not remove it.
- `QT_QPA_PLATFORM=offscreen` must remain set in all specs — containers are headless.
- Do not modify machine types or GPU config without explicit user review.

### Docker Images

- Tag images with explicit versions (`:v2`, `:v1`) — **never use `:latest` in production specs**.
- The training image is synced from `dromni/nerfstudio:1.1.5` — do not change the base version without updating `sync_official_image.sh`.
- The preprocessing image is built from `docker-images/Dockerimage-process`.
- Do not add Python packages to training containers without rebuilding the image.

### Pipeline & Prefect

- **Training must only run after preprocessing succeeds.** This is a hard dependency — never bypass it.
- Follow the GCS folder structure strictly: `<dataset>/processed/`, `<dataset>/trained/`, `<dataset>/exported/`. Never use the old flat `_processed`/`_trained`/`_exported` suffix convention.
- Prefect flows must be idempotent where possible — re-running a flow for the same dataset should not corrupt existing outputs without explicit override.
- Do not store Prefect state in local filesystem when running in cloud — use Prefect Cloud or a remote backend.

### What Agents Must NOT Do

- Do not run `gcloud` commands that create or delete GCP resources without explicit user confirmation.
- Do not modify `specs/*.json` machine types or GPU config without user review.
- Do not add GPU accelerators to the preprocessing spec.
- Do not implement synchronous busy-wait loops — use polling with sleep intervals.
- Do not add features, refactoring, or abstractions beyond what is directly requested.
- Do not introduce code for layers outside this repo's scope (no Laravel, Next.js, ChromaDB, CLIP code here).

---

## Key Commands

```bash
# Run Phase 1 manually (defaults)
./scripts/run_preprocess_job.sh <dataset_name>

# Run Phase 1 with custom COLMAP parameters
./scripts/run_preprocess_job.sh <dataset_name> --matching-method exhaustive --num-downscales 0

# Run Phase 2 manually (defaults)
./scripts/run_train_job.sh <dataset_name>

# Run Phase 2 with custom training parameters
./scripts/run_train_job.sh <dataset_name> --max-iters 50000 --sh-degree 2 --densify-grad 0.0001

# Monitor a running Vertex AI job
gcloud ai custom-jobs stream-logs <JOB_ID> --region=us-central1

# Build and push preprocessing image
docker build -t image-preprocess:v2 -f docker-images/Dockerimage-process .
docker tag image-preprocess:v2 us-central1-docker.pkg.dev/skillful-air-480018-f2/nerfstudio-repo/image-preprocess:v2
docker push us-central1-docker.pkg.dev/skillful-air-480018-f2/nerfstudio-repo/image-preprocess:v2

# Sync official nerfstudio training image
./sync_official_image.sh
```

# ── API ──────────────────────────────────────────────────────────────────────

## Endpoints

```
POST   /pipeline/start          → start new pipeline run
GET    /pipeline                → list all runs (paginated)
GET    /pipeline/{run_id}       → get run details
GET    /health                  → health check
```

## Request / Response

### Start Pipeline

**Request:**

```json
{
  "dataset": "test_scene",
  "webhook_url": "https://example.com/webhook",
  "poll_interval": 30,
  "skip_preprocess": false,
  "matching_method": "exhaustive",
  "sfm_tool": "colmap",
  "feature_type": "sift",
  "matcher_type": "exhaustive",
  "num_downscales_preprocess": 0,
  "skip_colmap": false,
  "max_iters": 30000,
  "sh_degree": 2,
  "num_random": 8,
  "num_downscales_train": 1,
  "res_schedule": "log2",
  "densify_grad": 0.0005,
  "densify_size": 100000,
  "n_split": 1,
  "refine_every": 1,
  "cull_scale": 0.5,
  "reset_alpha": 30
}
```

**Response (202 Accepted):**

```json
{
  "run_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
  "status": "pending",
  "created_at": "2026-03-10T03:30:00.000000Z",
  "completed_at": null,
  "ply_uri": null,
  "error": null,
  "params": { ... }
}
```

### Get Run

**Response:**

```json
{
  "run_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
  "status": "running",
  "created_at": "2026-03-10T03:30:00.000000Z",
  "completed_at": null,
  "ply_uri": null,
  "error": null,
  "params": { ... }
}
```

### List Runs

**Response:**

```json
[
  {
    "run_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
    "status": "running",
    "created_at": "2026-03-10T03:30:00.000000Z",
    "completed_at": null,
    "ply_uri": null,
    "error": null,
    "params": { ... }
  }
]
```

## Firestore Schema

Collection: `pipeline_runs`

```
{
  "run_id":       "string",
  "status":       "string",
  "created_at":   "timestamp",
  "completed_at": "timestamp | null",
  "ply_uri":      "string | null",
  "error":        "string | null",
  "params": {
    "dataset":                "string",
    "webhook_url":            "string",
    "poll_interval":          "integer",
    "skip_preprocess":        "boolean",
    "matching_method":        "string",
    "sfm_tool":               "string",
    "feature_type":           "string",
    "matcher_type":           "string",
    "num_downscales_preprocess":"integer",
    "skip_colmap":            "boolean",
    "max_iters":              "integer",
    "sh_degree":              "integer",
    "num_random":             "integer",
    "num_downscales_train":   "integer",
    "res_schedule":           "string",
    "densify_grad":           "float",
    "densify_size":           "integer",
    "n_split":                "integer",
    "refine_every":           "integer",
    "cull_scale":             "float",
    "reset_alpha":            "integer"
  }
}
```

## Webhook Payload (to Laravel)

**On success:**

```json
{
  "run_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
  "status": "done",
  "dataset": "test_scene",
  "ply_uri": "gs://bucket-saas-project/test_scene/exported/splat.ply",
  "completed_at": "2026-03-10T03:35:00.000000Z"
}
```

**On failure:**

```json
{
  "run_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
  "status": "failed",
  "dataset": "test_scene",
  "error": "Vertex AI job failed: ...",
  "completed_at": "2026-03-10T03:35:00.000000Z"
}
```

# ── Development ──────────────────────────────────────────────────────────────

## Running Locally

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Set environment variables
export GOOGLE_CLOUD_PROJECT=skillful-air-480018-f2
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
export WEBHOOK_URL=https://example.com/webhook

# Start API
uvicorn api.main:app --host [IP_ADDRESS] --port 8000
```

## Testing

```bash
# Run unit tests
pytest tests/

# Run integration tests
pytest tests/test_integration.py
```

## Firestore Setup

```bash
# Create Firestore database
gcloud firestore databases create --location=us-central1

# Create collection (auto-created on first write)
```

## Local Testing Without Firestore

```bash
# Use in-memory DB
export USE_FIRESTORE=false

uvicorn api.main:app --host [IP_ADDRESS] --port 8000
```
