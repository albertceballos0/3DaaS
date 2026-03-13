# Pipeline Reference — 3DaaS

## Overview

`gaussian_pipeline()` is a Prefect `@flow` defined in [flows/pipeline.py](../flows/pipeline.py). It orchestrates the full lifecycle of converting raw images into a 3D Gaussian Splat `.ply` file.

---

## Pipeline Stages

```
queued
  └─► detecting_dataset       Infer dataset format from GCS structure
        └─► validating_raw    Check raw/ images exist
              └─► preprocessing      Vertex AI: COLMAP (CPU)
                    └─► validating_processed  Check transforms.json exists
                          └─► training        Vertex AI: splatfacto (GPU L4)
                                └─► validating_exported  Check .ply exists
                                      └─► done
```

If `skip_preprocess=true`, the pipeline jumps from `validating_raw` directly to `validating_processed` (assumes `processed/` already has valid COLMAP output).

### Stage Details

| Stage | Module | Description |
|---|---|---|
| `detecting_dataset` | `flows/tasks/gcs.py` | Checks GCS structure to infer dataset type: `raw`, `nerfstudio`, `blender`, `dnerf` |
| `validating_raw` | `flows/tasks/gcs.py` | Verifies that `gs://<bucket>/<dataset>/raw/` contains image files |
| `preprocessing` | `flows/tasks/vertex.py` | Submits COLMAP Custom Job; writes `processed/` to GCS |
| `validating_processed` | `flows/tasks/gcs.py` | Verifies `processed/transforms.json` exists |
| `training` | `flows/tasks/vertex.py` | Submits splatfacto Custom Job; writes `exported/<dataset>.ply` |
| `validating_exported` | `flows/tasks/gcs.py` | Verifies that a `.ply` file exists in `exported/` |

---

## Dataset Types

The pipeline auto-detects the dataset type by inspecting GCS:

| Type | Detection | COLMAP needed? | Dataparser |
|---|---|---|---|
| `raw` | `raw/` folder has images, no `data/` | Yes | auto-selected after COLMAP |
| `nerfstudio` | `data/transforms.json` exists | No | `nerfstudio` |
| `blender` | `data/transforms_train.json` exists | No | `blender` |
| `dnerf` | `data/transforms_train.json` + time field | No | `dnerf` |

---

## Flow Parameters

```python
gaussian_pipeline(
    run_id: str,                   # UUID — links to Firestore document
    dataset: str,                  # GCS dataset name (folder in bucket)
    dataset_type: str = "auto",    # "auto" | "raw" | "nerfstudio" | "blender" | "dnerf"
    skip_preprocess: bool = False, # Skip COLMAP (dataset already has processed/)
    preprocess_params: dict = {},  # See PreprocessParams below
    train_params: dict = {},       # See TrainParams below
    resume_from_stage: str | None = None,  # For crash recovery
)
```

### PreprocessParams

Configured via `flows/config.py:PreprocessParams`:

| Parameter | Default | Description |
|---|---|---|
| `matching_method` | `vocab_tree` | Feature matching: `vocab_tree`, `exhaustive`, `sequential` |
| `sfm_tool` | `colmap` | Structure-from-motion tool |
| `feature_type` | `sift` | Feature detector: `sift`, `superpoint` |
| `matcher_type` | `NN` | Feature matcher: `NN`, `superglue` |
| `num_downscales` | `3` | Image downscale levels for COLMAP |
| `skip_colmap` | `false` | Skip COLMAP if transforms.json already exists in raw/ |

### TrainParams

Configured via `flows/config.py:TrainParams`:

| Parameter | Default | Description |
|---|---|---|
| `dataparser` | `""` (auto) | `nerfstudio`, `blender`, `dnerf`, or empty for auto |
| `max_iters` | `30000` | Training iterations |
| `sh_degree` | `3` | Spherical harmonics degree (0–3; higher = better color) |
| `num_random` | `100000` | Initial random Gaussians |
| `num_downscales` | `0` | Training image downscale |
| `res_schedule` | `250` | Resolution schedule steps |
| `densify_grad` | `0.0002` | Densification gradient threshold |
| `densify_size` | `0.01` | Densification size threshold |
| `n_split` | `2` | Number of Gaussian splits |
| `refine_every` | `100` | Refinement interval (steps) |
| `cull_scale` | `0.5` | Scale culling threshold |
| `reset_alpha` | `30` | Alpha reset interval (steps) |
| `use_wandb` | `false` | Enable Weights & Biases logging |

---

## Vertex AI Jobs

### Stage 1 — Preprocessing (COLMAP)

- **Machine:** `n1-highmem-16` (CPU only, 16 vCPUs, 104 GB RAM)
- **Image:** `image-preprocess:v2`
- **Spec:** [specs/preprocess_spec.json](../specs/preprocess_spec.json)
- **Command inside container:**
  ```bash
  gcloud storage cp -r gs://<bucket>/<dataset>/raw/ /data/raw/
  ns-process-data images \
    --data /data/raw \
    --output-dir /data/processed \
    --matching-method vocab_tree \
    ...
  gcloud storage cp -r /data/processed/ gs://<bucket>/<dataset>/processed/
  ```
- **Output:** `gs://<bucket>/<dataset>/processed/transforms.json` + resized images

### Stage 2 — Training + Export (splatfacto)

- **Machine:** `g2-standard-12` (12 vCPUs, 48 GB RAM) + 1× NVIDIA L4
- **Image:** `image-train-l4:v1`
- **Spec:** [specs/train-export-spec.json](../specs/train-export-spec.json)
- **Required env vars in spec:**
  - `TORCH_COMPILE_DISABLE=1` — prevents CUDA compilation issues on L4
  - `QT_QPA_PLATFORM=offscreen` — headless rendering
- **Command inside container:**
  ```bash
  gcloud storage cp -r gs://<bucket>/<dataset>/processed/ /data/processed/
  ns-train splatfacto \
    --data /data/processed \
    --output-dir /data/output \
    --max-num-iterations 30000 \
    ...
  ns-export gaussian-splat \
    --load-config /data/output/splatfacto/.../config.yml \
    --output-dir /data/exported/
  gcloud storage cp -r /data/output/ gs://<bucket>/<dataset>/trained/
  gcloud storage cp -r /data/exported/ gs://<bucket>/<dataset>/exported/
  ```
- **Output:** `gs://<bucket>/<dataset>/exported/<dataset>.ply`

---

## Progress Tracking

During training, the worker polls the Vertex AI job logs and parses training metrics:

```
Firestore → progress field:
{
  "step":      15000,       # current training step
  "loss":      0.042,       # current loss
  "pct":       50,          # percentage complete (step / max_iters * 100)
  "wandb_url": "https://wandb.ai/..."  # if use_wandb=true
}
```

The client can poll `GET /pipeline/{run_id}` to get real-time progress.

---

## Webhooks

If `WEBHOOK_URL` is set, the worker sends POST requests at key pipeline events:

| Event | When sent | Payload |
|---|---|---|
| `job_submitted` | After Vertex AI job is created | `{run_id, dataset, stage, job_id}` |
| `job_polling` | Every N poll intervals (not every tick) | `{run_id, dataset, stage, progress}` |
| `pipeline_completed` | When `done` | `{run_id, dataset, ply_uri}` |
| `pipeline_failed` | When `failed` | `{run_id, dataset, error}` |

Webhook failures are logged as warnings but do **not** fail the pipeline.

---

## Crash Recovery

When the Prefect worker restarts, `scripts/recover_stale_runs.py` automatically:

1. Queries Firestore for all runs with `status == "running"`
2. For each stale run:
   - Cancels the old Prefect flow run (if `prefect_flow_run_id` is stored)
   - Determines the resume stage:
     - **Vertex AI stages** (`preprocessing`, `training`): checks GCP job state
       - `JOB_STATE_SUCCEEDED` → resume from next validation stage
       - `RUNNING/PENDING/QUEUED` → resume polling the same job
       - Any other state → re-submit the job from the same stage
     - **Validation stages** → re-run from the same stage (fast, idempotent)
   - Creates a new Prefect flow run with `resume_from_stage` parameter
   - Updates Firestore to `status=queued` with the new `prefect_flow_run_id`

### Stage order for resume

```python
_STAGE_ORDER = [
    "detecting_dataset",
    "validating_raw",
    "preprocessing",
    "validating_processed",
    "training",
    "validating_exported",
]
```

A run with `resume_from_stage="validating_processed"` will skip `detecting_dataset`, `validating_raw`, and `preprocessing`, and start directly at `validating_processed`.

### `vertex_job_id` persistence

The `vertex_job_id` is written to Firestore **immediately after** job submission (before polling starts). This ensures that even if the worker crashes between submit and poll, the recovery script can check the GCP job state on restart.

---

## Firestore Schema

Collection: `pipeline_runs` (production) / `pipeline_runs_dev` (local — set by `APP_ENV`)

```
Field               Type        Description
─────────────────────────────────────────────────────────
run_id              string      UUID — primary key
dataset             string      GCS folder name
status              string      queued | running | done | failed
stage               string|null Current stage (null when done or queued)
started_at          string      ISO-8601 UTC timestamp
completed_at        string|null ISO-8601 UTC timestamp (null until done/failed)
ply_uri             string|null gs:// URI of exported .ply file
error               string|null Error message (null unless failed)
prefect_flow_run_id string|null Prefect flow run ID
vertex_job_id       string|null Full GCP resource name of active Vertex AI job
dataset_type        string|null Detected dataset type (raw/nerfstudio/blender/dnerf)
image_count         int|null    Number of images detected in raw/
params              object      Full copy of all pipeline parameters (see below)
progress            object|null Training progress {step, loss, pct, wandb_url}
recent_logs         string[]    Last N log lines from Vertex AI job
```

`params` contains all fields from `PreprocessParams` and `TrainParams` as flat keys.

---

## API Endpoints

### POST `/pipeline/start`

Triggers a new pipeline run.

**Request body:**
```json
{
  "dataset": "mi_escena",
  "skip_preprocess": false,
  "matching_method": "vocab_tree",
  "sfm_tool": "colmap",
  "feature_type": "sift",
  "matcher_type": "NN",
  "num_downscales_preprocess": 3,
  "skip_colmap": false,
  "dataparser": "",
  "max_iters": 30000,
  "sh_degree": 3,
  "num_random": 100000,
  "num_downscales_train": 0,
  "res_schedule": 250,
  "densify_grad": 0.0002,
  "densify_size": 0.01,
  "n_split": 2,
  "refine_every": 100,
  "cull_scale": 0.5,
  "reset_alpha": 30,
  "use_wandb": false
}
```

Only `dataset` is required. All others have defaults.

**Response: 202**
```json
{
  "run_id": "550e8400-...",
  "dataset": "mi_escena",
  "status": "queued",
  "stage": null,
  "started_at": "2026-03-10T12:00:00Z",
  "completed_at": null,
  "ply_uri": null,
  "error": null,
  "prefect_flow_run_id": "abc123",
  "params": {...}
}
```

### GET `/pipeline/{run_id}`

Returns the full Firestore document for a run.

**Response: 200** — same schema as above, with current values.
**Response: 404** — run not found.

### GET `/pipeline`

Lists runs ordered by `started_at` descending.

**Query params:** `?limit=N` (default 100, max 500)

### DELETE `/pipeline/{run_id}`

Cancels a run: cancels the Prefect flow run and marks Firestore as `failed`.

**Response: 200**
```json
{"message": "Run <run_id> cancelled"}
```

### GET `/health`

```json
{"status": "ok"}
```
