# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

Cloud pipeline for training and rendering 3D Gaussian Splats using **Nerfstudio (splatfacto)** on **Google Cloud Vertex AI**, orchestrated with **Prefect** (self-hosted on GCE VM).

- **GCP Project:** `skillful-air-480018-f2`
- **Region:** `us-central1`
- **GCS Bucket:** `bucket-saas-project`
- **Artifact Registry:** `us-central1-docker.pkg.dev/skillful-air-480018-f2/nerfstudio-repo/`
- **Service Account:** `custom-models@skillful-air-480018-f2.iam.gserviceaccount.com`
- **GCE VM:** `vm-3daas` (zone `us-central1-a`, type `e2-standard-2`)

## Architecture

```
Client (Laravel)
    ↓ POST /pipeline/start
FastAPI (GCE VM :8080)        ← api/main.py
    ↓ trigger Prefect flow run via HTTP
Prefect Worker (GCE VM)       ← flows/pipeline.py @flow
    ├─ validate_raw_input      ← flows/tasks/gcs.py @task
    ├─ submit_preprocess_job   ← flows/tasks/vertex.py @task
    ├─ poll_vertex_job         ← flows/tasks/vertex.py @task
    ├─ validate_processed_output
    ├─ submit_train_job
    ├─ poll_vertex_job
    └─ validate_exported_output
         ↓ updates Firestore directly
Firestore                     ← api/db.py (collection: pipeline_runs)
    ↓ read by
GET /pipeline/{run_id}
```

## Repository Structure

```
api/
  main.py            FastAPI — POST /pipeline/start, GET /pipeline/{id}
  db.py              Firestore CRUD
flows/
  pipeline.py        Prefect @flow — gaussian_pipeline()
  config.py          Env vars + dataclasses PreprocessParams / TrainParams
  tasks/
    gcs.py           @task: validate_raw_input, validate_processed_output, validate_exported_output
    vertex.py        @task: submit_preprocess_job, submit_train_job, poll_vertex_job
docker-images/
  Dockerfile.api       FastAPI container
  Dockerfile.worker    Prefect worker container
  Dockerimage-process  COLMAP preprocessing image (Vertex AI)
scripts/
  start_worker.sh           Worker entrypoint: pool → deployment → start
  deploy_vm.sh              Deploy to GCE VM via gcloud scp + ssh
  run_preprocess_job.sh     Manual Stage 1 submit
  run_train_job.sh          Manual Stage 2 submit
specs/
  preprocess_spec.json      Vertex AI: n1-highmem-16 (CPU)
  train-export-spec.json    Vertex AI: g2-standard-12 + L4 GPU
docker-compose.yml   Stack: prefect-server + worker + api
prefect.yaml         Prefect deployment config
.github/workflows/ci-cd.yml  CI/CD: test → deploy to GCE VM
```

## GCS Layout

```
gs://bucket-saas-project/
  <dataset>/
    raw/         <- original images (user uploads here)
    processed/   <- COLMAP output (transforms.json + images)
    trained/     <- Nerfstudio checkpoints
    exported/    <- final .ply file
```

Always `<dataset>/processed/`, NEVER `<dataset>_processed/`.

## Local Development

```bash
cp .env.example .env   # fill in values (GCP keys, SA key path, etc.)
docker compose up --build
# Prefect UI: http://localhost:4200
# API:        http://localhost:8080
```

## Deploy to GCE VM

```bash
# One-time VM setup (already done if vm-3daas exists):
# gcloud compute instances create vm-3daas --zone=us-central1-a --machine-type=e2-standard-2 ...

# Deploy:
./scripts/deploy_vm.sh

# Requires in .env: GCE_VM_NAME=vm-3daas, GCE_VM_ZONE=us-central1-a, GCE_VM_USER=<user>
```

## CI/CD

Push to `master` → GitHub Actions runs tests → deploys to GCE VM via gcloud.
Required GitHub secrets: `GCP_SA_KEY`, `GCS_BUCKET`, `GCP_SERVICE_ACCOUNT`, `IMAGE_PREPROCESS`, `IMAGE_TRAIN`, `WEBHOOK_URL`.
Required GitHub variables: `GCE_VM_NAME`, `GCE_VM_ZONE`, `GCE_VM_USER`.

## Required Environment Variables

```
GCP_PROJECT_ID, GCP_REGION, GCS_BUCKET, GCP_SERVICE_ACCOUNT,
IMAGE_PREPROCESS, IMAGE_TRAIN, GCP_SA_KEY_PATH,
PREFECT_API_URL (http://localhost:4200/api), PREFECT_API_KEY (empty for self-hosted),
PREFECT_DEPLOYMENT (gaussian-pipeline/prod),
GCE_VM_NAME, GCE_VM_ZONE, GCE_VM_USER
```

## Key Notes

- `TORCH_COMPILE_DISABLE=1` and `QT_QPA_PLATFORM=offscreen` must remain in Vertex AI specs.
- The worker updates Firestore directly — no `/internal/*` API callbacks.
- Prefect `@task` functions are called with `.fn()` in tests to bypass Prefect context.
- `PREFECT_API_KEY` is empty for self-hosted Prefect — only `PREFECT_API_URL` is required.
- GCE VM uses `--scopes=cloud-platform` so no SA key file needed in production.
