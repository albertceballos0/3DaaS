# AGENTS.md — AI Agent Guide for 3DaaS Gaussian Splatting Pipeline

Léelo antes de hacer cualquier cambio.

---

## Scope del Repositorio

Este repo es el **motor MLOps** del sistema 3DaaS. Contiene:

- Pipeline de Gaussian Splatting (COLMAP + splatfacto) en Vertex AI
- Orquestación con Prefect (self-hosted en GCE VM)
- API Gateway (FastAPI)
- Estado persistido en Firestore

**No contiene:** backend Laravel, frontend Next.js, capa CLIP/ChromaDB, ni infraestructura Terraform.

---

## Contexto del Sistema

```
[Laravel] → POST /pipeline/start → [FastAPI (GCE VM :8080)]
                                          ↓ dispara Prefect flow run
                                  [Prefect Worker (GCE VM)]
                                    ├─ Stage 1: COLMAP (Vertex AI CPU)
                                    └─ Stage 2: splatfacto (Vertex AI GPU L4)
                                          ↓ escribe directamente en Firestore
                              [Laravel] → GET /pipeline/{run_id}
```

---

## Estructura del Repositorio

```
api/
  main.py              FastAPI — endpoints públicos únicamente
  db.py                Firestore CRUD

flows/
  pipeline.py          @flow: gaussian_pipeline() — orquesta los 7 steps
  config.py            PreprocessParams, TrainParams, env vars
  tasks/
    gcs.py             @task: validate_raw_input, validate_processed_output, validate_exported_output
    vertex.py          @task: submit_preprocess_job, submit_train_job, poll_vertex_job

docker-images/
  Dockerfile.api       FastAPI para GCE VM (puerto 8080)
  Dockerfile.worker    Prefect worker para GCE VM
  Dockerimage-process  Imagen COLMAP (CPU, subida a Vertex AI)

scripts/
  start_worker.sh        Entrypoint del worker (pool + deployment + start)
  deploy_vm.sh           Deploy completo a GCE VM via gcloud
  run_preprocess_job.sh  Submit manual Stage 1
  run_train_job.sh       Submit manual Stage 2

specs/
  preprocess_spec.json     n1-highmem-16, CPU, image-preprocess:v2
  train-export-spec.json   g2-standard-12 + L4, image-train-l4:v1

docker-compose.yml   Stack local y producción: prefect-server + worker + api
prefect.yaml         Deployment Prefect self-hosted
.github/workflows/ci-cd.yml  CI/CD: test → deploy a GCE VM
```

---

## GCS Layout (respetar estrictamente)

```
gs://bucket-saas-project/
  <dataset>/
    raw/         ← imágenes originales
    processed/   ← COLMAP output (transforms.json + imágenes)
    trained/     ← checkpoints Nerfstudio
    exported/    ← .ply final
```

Siempre `<dataset>/processed/`, NUNCA `<dataset>_processed/`.

---

## API Endpoints

```
POST /pipeline/start    Inicia run → dispara Prefect flow
GET  /pipeline          Lista runs (paginado)
GET  /pipeline/{run_id} Estado del run (leído de Firestore)
GET  /health            Health check
```

No existen endpoints `/internal/*` — el flow actualiza Firestore directamente.

---

## Firestore Schema

Colección: `pipeline_runs`

```
run_id               str   — UUID del run
dataset              str   — nombre del dataset GCS
status               str   — queued | running | done | failed
stage                str   — validating_raw | preprocessing | validating_processed | training | null
started_at           str   — ISO-8601 UTC
completed_at         str   — ISO-8601 UTC (null mientras corre)
ply_uri              str   — gs:// URI del .ply (null hasta done)
error                str   — mensaje de error (null si ok)
params               dict  — parámetros del run
prefect_flow_run_id  str   — ID del flow run en Prefect
```

---

## Deploy

### Local

```bash
cp .env.example .env   # rellenar valores
docker compose up --build
# Prefect UI: http://localhost:4200
# API:        http://localhost:8080
```

### GCE VM (producción)

```bash
./scripts/deploy_vm.sh          # sync + build + up
./scripts/deploy_vm.sh --no-build   # solo sync, sin rebuild
./scripts/deploy_vm.sh --sync-only  # solo copiar archivos
```

El script requiere `GCE_VM_NAME`, `GCE_VM_ZONE` en `.env` y `gcloud` autenticado.

### CI/CD continuo

Push a `master` → GitHub Actions → test → `deploy_vm.sh` equivalente en la VM.
Ver `.github/workflows/ci-cd.yml`.

Secrets requeridos en GitHub Actions (Settings → Secrets and variables → Actions):
- `GCP_SA_KEY` — JSON completo de la SA key
- `GCS_BUCKET`, `GCP_SERVICE_ACCOUNT`, `IMAGE_PREPROCESS`, `IMAGE_TRAIN`, `WEBHOOK_URL`

Variables (no secretas):
- `GCE_VM_NAME`, `GCE_VM_ZONE`, `GCE_VM_USER`

---

## Reglas para Agentes

### Lo que NO hacer

- **No crear endpoints `/internal/*`** — el flujo de datos va: flow → Firestore → API GET.
- **No modificar machine types** en `specs/*.json` sin confirmación del usuario.
- **No agregar GPU** al spec de preprocessing — es CPU-only.
- **No eliminar** `TORCH_COMPILE_DISABLE=1` ni `QT_QPA_PLATFORM=offscreen`.
- **No usar** `<dataset>_processed/` — siempre `<dataset>/processed/`.
- **No implementar** busy-wait loops — usar polling con `time.sleep`.
- **No tocar** código de Laravel, Next.js ni ChromaDB (están en otros repos).

### Testing

- Los `@task` de Prefect se llaman con `.fn()` en tests (evita contexto Prefect).
- Mockear `api.db` con store en memoria, y `_trigger_prefect_flow` en tests de la API.
- `PREFECT_API_URL` y `PREFECT_API_KEY` no necesitan valor real en tests.

### Imágenes Docker

- Tags explícitas siempre (`:v2`, `:v1`) — nunca `:latest` en specs de Vertex AI.
- La imagen de training viene de `dromni/nerfstudio:1.1.5` via `sync_official_image.sh`.

---

## Comandos Clave

```bash
# Levantar en local
docker compose up --build

# Correr tests
pytest tests/ -v

# Deploy a VM
./scripts/deploy_vm.sh

# Submit manual Stage 1 (COLMAP)
./scripts/run_preprocess_job.sh <dataset>

# Submit manual Stage 2 (training)
./scripts/run_train_job.sh <dataset>

# Build y push imagen preprocessing
docker build -f docker-images/Dockerimage-process \
  -t us-central1-docker.pkg.dev/skillful-air-480018-f2/nerfstudio-repo/image-preprocess:v2 .
docker push us-central1-docker.pkg.dev/skillful-air-480018-f2/nerfstudio-repo/image-preprocess:v2

# Sincronizar imagen training oficial
./sync_official_image.sh

# Monitorear job Vertex AI
gcloud ai custom-jobs stream-logs <JOB_ID> --region=us-central1

# Logs en producción
gcloud compute ssh vm-3daas --zone=us-central1-a \
  --command="cd ~/3daas && docker compose logs -f prefect-worker"
```

---

## GCP Resources

| Recurso | Valor |
|---|---|
| Project ID | `skillful-air-480018-f2` |
| Region | `us-central1` |
| GCS Bucket | `bucket-saas-project` |
| Artifact Registry | `us-central1-docker.pkg.dev/skillful-air-480018-f2/nerfstudio-repo/` |
| Service Account | `custom-models@skillful-air-480018-f2.iam.gserviceaccount.com` |
| GCE VM | `vm-3daas` (zona `us-central1-a`) |
