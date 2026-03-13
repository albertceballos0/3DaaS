# Architecture — 3DaaS

## System Overview

3DaaS is a cloud pipeline that converts raw images into 3D Gaussian Splat files (`.ply`) using COLMAP + Nerfstudio splatfacto on Google Cloud, orchestrated with Prefect.

---

## High-Level Architecture

```mermaid
graph TD
    Client["Client\n(Laravel / curl)"]
    API["FastAPI Gateway\napi/main.py\n:8080"]
    Prefect["Prefect Server\n:4200"]
    Worker["Prefect Worker\nflows/pipeline.py"]
    Firestore["Firestore\npipeline_runs(_dev)"]
    GCS["GCS Bucket\nbucket-saas-project"]
    Vertex1["Vertex AI Job\nCOLMAP (CPU)\nn1-highmem-16"]
    Vertex2["Vertex AI Job\nsplatfacto (GPU L4)\ng2-standard-12"]
    Webhook["Webhook\n(Laravel callback)"]

    Client -->|POST /pipeline/start| API
    API -->|create flow run| Prefect
    API -->|write run doc| Firestore
    Prefect -->|dispatch job| Worker
    Worker -->|poll + update status| Firestore
    Worker -->|submit job| Vertex1
    Worker -->|submit job| Vertex2
    Vertex1 -->|write processed/| GCS
    Vertex2 -->|write exported/| GCS
    Worker -->|send events| Webhook
    Client -->|GET /pipeline/{id}| API
    API -->|read doc| Firestore
```

---

## Component Architecture

### GCE VM (vm-3daas)

All three services run on a single `e2-standard-2` VM via docker-compose:

```
┌─────────────────────────────────────────────────────────┐
│  GCE VM  vm-3daas  (us-central1-a, e2-standard-2)       │
│                                                         │
│  ┌──────────────────────┐  ┌──────────────────────────┐ │
│  │   prefect-server     │  │   api                    │ │
│  │   :4200              │  │   :8080                  │ │
│  │   prefecthq/prefect  │  │   Dockerfile.api         │ │
│  └─────────┬────────────┘  └────────────┬─────────────┘ │
│            │ healthcheck OK             │               │
│  ┌─────────▼────────────────────────────▼─────────────┐ │
│  │   prefect-worker                                    │ │
│  │   Dockerfile.worker                                 │ │
│  │   work pool: gcp-worker                             │ │
│  │   flows/ + api/ + utils/                            │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                         │
│  Volumes:                                               │
│    ./logs/     → /app/logs  (shared across containers)  │
│    ./utils/    → /app/utils (hot reload utils module)   │
│    prefect-data → /root/.prefect                        │
└─────────────────────────────────────────────────────────┘
```

---

## Data Flow

```mermaid
sequenceDiagram
    participant C as Client
    participant A as API (FastAPI)
    participant P as Prefect Server
    participant W as Worker
    participant V1 as Vertex AI (COLMAP)
    participant V2 as Vertex AI (splatfacto)
    participant FS as Firestore
    participant GCS as GCS

    C->>A: POST /pipeline/start {dataset, params}
    A->>FS: create_run(run_id, status=queued)
    A->>P: create_flow_run(deployment_id, params)
    A-->>C: 202 {run_id, status=queued}

    P->>W: dispatch flow run

    W->>FS: update status=running, stage=validating_raw
    W->>GCS: list gs://<bucket>/<dataset>/raw/
    W->>FS: update stage=preprocessing

    W->>V1: submit_custom_job (COLMAP spec)
    W->>FS: update vertex_job_id
    loop poll every 30s
        W->>V1: get_custom_job(job_id)
        W->>FS: update stage_detail, logs
    end

    V1->>GCS: write processed/ (transforms.json + images)

    W->>FS: update stage=validating_processed
    W->>GCS: check transforms.json exists

    W->>FS: update stage=training
    W->>V2: submit_custom_job (splatfacto spec)
    W->>FS: update vertex_job_id
    loop poll every 30s
        W->>V2: get_custom_job(job_id)
        W->>FS: update progress (step, loss, %)
    end

    V2->>GCS: write exported/<dataset>.ply

    W->>FS: update status=done, ply_uri=gs://...
    W->>C: webhook pipeline_completed {ply_uri}

    C->>A: GET /pipeline/{run_id}
    A->>FS: get_run(run_id)
    A-->>C: {status=done, ply_uri=...}
```

---

## Module Map

```
flows/
├── pipeline.py        Main @flow — gaussian_pipeline()
│                      Resume logic, stage routing, Firestore updates
├── config.py          Environment config + PreprocessParams + TrainParams
└── tasks/
    ├── gcs.py         GCS validation tasks
    │   ├── detect_dataset_type()       → "raw" | "nerfstudio" | "blender" | "dnerf"
    │   ├── validate_raw_input()        → checks raw/ has images
    │   ├── validate_processed_output() → checks processed/transforms.json
    │   └── validate_exported_output()  → checks exported/*.ply
    ├── vertex.py      Vertex AI tasks
    │   ├── submit_preprocess_job()     → creates COLMAP Custom Job
    │   ├── submit_train_job()          → creates splatfacto Custom Job
    │   ├── poll_vertex_job()           → polls until done, parses logs
    │   ├── check_vertex_job_state()    → single state check (for recovery)
    │   └── build_job_resource_name()   → constructs full resource name
    └── notify.py      Webhook tasks
        └── send_webhook()              → POST to WEBHOOK_URL

api/
├── main.py            FastAPI endpoints
│   ├── POST /pipeline/start
│   ├── GET  /pipeline/{run_id}
│   ├── GET  /pipeline
│   ├── DELETE /pipeline/{run_id}
│   └── GET  /health
└── db.py              Firestore CRUD
    ├── create_run()
    ├── update_run()
    ├── get_run()
    ├── list_runs()
    └── delete_run()

utils/
└── logger.py          Centralized logging
    ├── setup_logging(component)   → configures root logger
    └── get_logger(name)           → returns named logger

scripts/
├── start_worker.sh          Worker container entrypoint
├── deploy_vm.sh             Deploy to GCE VM
├── sync_dataset.sh          Upload dataset to GCS
└── recover_stale_runs.py    Resume interrupted runs on startup
```

---

## State Machine

Pipeline runs follow a strict stage progression stored in Firestore:

```mermaid
stateDiagram-v2
    [*] --> queued: POST /pipeline/start
    queued --> detecting_dataset: worker picks up job
    detecting_dataset --> validating_raw: dataset type detected
    validating_raw --> preprocessing: raw images exist
    validating_raw --> validating_processed: skip_preprocess=true
    preprocessing --> validating_processed: COLMAP job succeeded
    validating_processed --> training: transforms.json exists
    training --> validating_exported: splatfacto job succeeded
    validating_exported --> done: .ply file exists
    detecting_dataset --> failed: error
    validating_raw --> failed: error
    preprocessing --> failed: error
    validating_processed --> failed: error
    training --> failed: error
    validating_exported --> failed: error
```

---

## GCS Layout

```
gs://bucket-saas-project/
├── <dataset>/
│   ├── raw/                ← User uploads images here
│   │   ├── image001.jpg
│   │   ├── image002.jpg
│   │   └── ...
│   ├── processed/          ← COLMAP output (written by Vertex AI Stage 1)
│   │   ├── transforms.json
│   │   └── images/
│   │       └── *.jpg
│   ├── trained/            ← Nerfstudio checkpoints (written by Vertex AI Stage 2)
│   │   └── splatfacto/
│   │       └── ...
│   ├── exported/           ← Final .ply file (written by Vertex AI Stage 2)
│   │   └── <dataset>.ply
│   └── data/               ← Pre-processed datasets (dnerf, blender, nerfstudio)
│       └── ...
```

---

## Vertex AI Job Specs

### Stage 1 — COLMAP (preprocessing)

Spec file: [specs/preprocess_spec.json](../specs/preprocess_spec.json)

| Parameter | Value |
|---|---|
| Machine type | `n1-highmem-16` |
| Accelerator | None (CPU only) |
| Image | `image-preprocess:v2` |
| Command | `ns-process-data images` + `gcloud storage cp` |
| Estimated runtime | 5–30 min depending on image count |

### Stage 2 — splatfacto (training + export)

Spec file: [specs/train-export-spec.json](../specs/train-export-spec.json)

| Parameter | Value |
|---|---|
| Machine type | `g2-standard-12` |
| Accelerator | 1× NVIDIA L4 |
| Image | `image-train-l4:v1` |
| Command | `ns-train splatfacto` + `ns-export gaussian-splat` + upload |
| Estimated runtime | 20–60 min depending on iterations |

---

## Firestore Schema

Collection: `pipeline_runs` (production) / `pipeline_runs_dev` (local)

```json
{
  "run_id":              "550e8400-e29b-41d4-a716-446655440000",
  "dataset":             "mi_escena",
  "status":              "running",
  "stage":               "training",
  "started_at":          "2026-03-10T12:00:00Z",
  "completed_at":        null,
  "ply_uri":             null,
  "error":               null,
  "prefect_flow_run_id": "abc123",
  "vertex_job_id":       "projects/.../locations/.../customJobs/456",
  "dataset_type":        "raw",
  "image_count":         150,
  "params": {
    "matching_method":         "vocab_tree",
    "sfm_tool":                "colmap",
    "feature_type":            "sift",
    "matcher_type":            "NN",
    "num_downscales_preprocess": 3,
    "skip_colmap":             false,
    "skip_preprocess":         false,
    "dataparser":              "nerfstudio",
    "max_iters":               30000,
    "sh_degree":               3,
    "num_random":              100000
  },
  "progress": {
    "step":    15000,
    "loss":    0.042,
    "pct":     50,
    "wandb_url": "https://wandb.ai/..."
  },
  "recent_logs": ["line 1", "line 2", "..."]
}
```

---

## Environment Isolation

| Variable | `local` | `production` |
|---|---|---|
| `APP_ENV` | `local` | `production` |
| Firestore collection | `pipeline_runs_dev` | `pipeline_runs` |
| Log level (console) | `DEBUG` | `INFO` |
| Log files | app.log + error.log + debug.log | app.log + error.log |

Switching between environments only requires changing `APP_ENV` in `.env`.

---

## Crash Recovery

When the Prefect worker restarts (crash, deploy, OOM), `recover_stale_runs.py` runs automatically before the worker starts accepting new jobs:

```mermaid
flowchart TD
    Start["Worker starts\nstart_worker.sh"]
    Register["prefect deploy --all\n(register deployment)"]
    Recovery["recover_stale_runs.py"]
    Query["Query Firestore\nstatus == running"]
    Empty{Any stale runs?}
    Done["Start worker\nprefect worker start"]

    Vertex{Stage is\nVertex AI?}
    Check["check_vertex_job_state(job_id)"]
    Succeeded{JOB_STATE_SUCCEEDED?}
    NextStage["resume_stage = next\nvalidation stage"]
    Running{RUNNING/PENDING\n/QUEUED?}
    SameStage["resume_stage = same\n(resume polling)"]
    Resubmit["resume_stage = same\n(will re-submit job)"]

    Cancel["Cancel old Prefect flow run"]
    NewRun["Create new Prefect flow run\nwith resume_from_stage"]
    UpdateFS["Firestore: status=queued"]

    Start --> Register --> Recovery --> Query --> Empty
    Empty -->|No| Done
    Empty -->|Yes| Vertex
    Vertex -->|Yes| Check --> Succeeded
    Succeeded -->|Yes| NextStage --> Cancel
    Succeeded -->|No| Running
    Running -->|Yes| SameStage --> Cancel
    Running -->|No| Resubmit --> Cancel
    Vertex -->|No| Cancel
    Cancel --> NewRun --> UpdateFS --> Done
```
