# 3DaaS — Gaussian Splatting Pipeline on Google Cloud

Pipeline automatizado para procesar imágenes y generar archivos `.ply` de Gaussian Splats 3D usando **Nerfstudio (splatfacto)** en **Google Cloud Vertex AI**, orquestado con **Prefect**.

## Documentación detallada

| Documento | Contenido |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Arquitectura completa, diagramas Mermaid, flujo de datos, módulos |
| [docs/pipeline.md](docs/pipeline.md) | Stages, parámetros, Vertex AI jobs, Firestore schema, webhooks, recovery |
| [docs/development.md](docs/development.md) | Setup local, tests, logs, hot reload, debugging |
| [docs/deployment.md](docs/deployment.md) | Deploy GCE VM, CI/CD, IAM, gestión de la VM |
| [docs/commands.md](docs/commands.md) | Referencia completa de todos los comandos CLI |

---

## Índice

1. [Arquitectura](#arquitectura)
2. [Stack tecnológico](#stack-tecnológico)
3. [Estructura del repositorio](#estructura-del-repositorio)
4. [Desarrollo local](#desarrollo-local)
5. [Deploy en GCE VM (producción)](#deploy-en-gce-vm-producción)
6. [CI/CD — Deploy continuo](#cicd--deploy-continuo)
7. [API Reference](#api-reference)
8. [GCS — Layout de datos](#gcs--layout-de-datos)
9. [Imágenes Docker](#imágenes-docker)
10. [Monitoreo y debugging](#monitoreo-y-debugging)
11. [Logs del sistema](#logs-del-sistema)
12. [Entornos — local vs producción](#entornos--local-vs-producción)
13. [Recuperación de runs interrumpidos](#recuperación-de-runs-interrumpidos)
14. [Service Account & IAM](#service-account--iam)
15. [Seguridad](#seguridad)
16. [GCP Resources](#gcp-resources)

---

## Arquitectura

```
Cliente (Laravel / curl)
        │
        │  POST /pipeline/start
        ▼
┌───────────────────┐
│  FastAPI Gateway  │  ← api/main.py  (puerto 8080)
│  (GCE VM / local) │
└────────┬──────────┘
         │ crea run en Firestore
         │ dispara Prefect flow run via HTTP
         ▼
┌───────────────────────────────────────────────────┐
│  Prefect Worker  (GCE VM)   flows/pipeline.py     │
│                                                   │
│  1. validate_raw_input        gcs.py @task        │
│  2. submit_preprocess_job     vertex.py @task ──┐ │
│  3. poll_vertex_job           vertex.py @task   │ │
│  4. validate_processed_output gcs.py @task      │ │
│  5. submit_train_job          vertex.py @task ──┐ │
│  6. poll_vertex_job           vertex.py @task   │ │
│  7. validate_exported_output  gcs.py @task      │ │
│                                                 │ │
│  ↓ actualiza Firestore en cada step             │ │
└─────────────────────────────────────────────────┘ │
                                                     │
         ┌───────────────┐   ┌─────────────────────┐│
         │ Vertex AI Job │   │  Vertex AI Job      ││
         │ COLMAP (CPU)  │◄──┘  splatfacto (GPU L4)│◄┘
         │ n1-highmem-16 │   │  g2-standard-12     │
         └───────────────┘   └─────────────────────┘
                │                       │
                ▼                       ▼
     gs://<bucket>/<dataset>/processed/  →  exported/<dataset>.ply

        │
        │  GET /pipeline/{run_id}
        ▼
┌───────────────────┐
│    Firestore      │  {status, stage, ply_uri, ...}
└───────────────────┘
```

---

## Stack tecnológico

| Capa | Tecnología | Dónde corre |
|---|---|---|
| Orquestación | Prefect 3 (self-hosted) | GCE VM (`vm-3daas`) |
| API Gateway | FastAPI + uvicorn | GCE VM (mismo docker-compose) |
| Estado de runs | Google Cloud Firestore | GCP managed |
| Compute Stage 1 | Vertex AI Custom Job (CPU) | `n1-highmem-16` |
| Compute Stage 2 | Vertex AI Custom Job (GPU) | `g2-standard-12` + L4 |
| Almacenamiento | Google Cloud Storage | `bucket-saas-project` |
| Container Registry | Artifact Registry | `us-central1-docker.pkg.dev/...` |
| CI/CD | GitHub Actions | Push a `master` |

---

## Estructura del repositorio

```
3DaaS/
├── api/
│   ├── main.py              # FastAPI: POST /pipeline/start, GET /pipeline/{id}
│   └── db.py                # Firestore CRUD (colección: pipeline_runs)
│
├── flows/
│   ├── pipeline.py          # @flow gaussian_pipeline() — orquesta los 7 steps
│   ├── config.py            # PreprocessParams, TrainParams, variables de entorno
│   ├── requirements.txt     # Dependencias del worker (prefect, google-cloud-*)
│   └── tasks/
│       ├── __init__.py
│       ├── gcs.py           # @task: validaciones en GCS
│       └── vertex.py        # @task: submit/poll Vertex AI jobs
│
├── docker-images/
│   ├── Dockerfile.api       # Imagen FastAPI (API gateway)
│   ├── Dockerfile.worker    # Imagen Prefect worker (flows + dependencias GCP)
│   └── Dockerimage-process  # Imagen COLMAP preprocessing (subida a Vertex AI)
│
├── scripts/
│   ├── start_worker.sh           # Entrypoint del worker: crea pool → registra deployment → arranca
│   ├── deploy_vm.sh              # Deploy completo a GCE VM via gcloud
│   ├── sync_dataset.sh           # Sube imágenes/datasets a GCS
│   ├── recover_stale_runs.py     # Reanuda runs interrumpidos al arrancar el worker
│   └── tasks/
│       ├── run_preprocess_job.sh  # Submit manual Stage 1 (COLMAP)
│       └── run_train_job.sh       # Submit manual Stage 2 (training)
│
├── specs/
│   ├── preprocess_spec.json    # Vertex AI spec: n1-highmem-16, CPU
│   └── train-export-spec.json  # Vertex AI spec: g2-standard-12 + L4 GPU
│
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_gcs.py
│   ├── test_pipeline.py
│   └── test_vertex.py
│
├── .github/workflows/ci-cd.yml  # GitHub Actions: test → deploy a GCE VM
├── docker-compose.yml           # Stack completo: prefect-server + worker + api
├── prefect.yaml                 # Deployment Prefect self-hosted
├── requirements-dev.txt         # Dependencias para tests locales
└── .env.example                 # Plantilla de variables de entorno
```

---

## Desarrollo local

### Requisitos previos

- Docker Desktop (o Docker Engine + docker compose plugin)
- Cuenta GCP con acceso al proyecto `skillful-air-480018-f2`
- Service account key JSON descargada localmente

### 1. Configurar variables de entorno

```bash
cp .env.example .env
```

Edita `.env` y rellena los valores:

```bash
GCP_PROJECT_ID=skillful-air-480018-f2
GCP_REGION=us-central1
GCS_BUCKET=bucket-saas-project
GCP_SERVICE_ACCOUNT=custom-models@skillful-air-480018-f2.iam.gserviceaccount.com
IMAGE_PREPROCESS=us-central1-docker.pkg.dev/.../image-preprocess:v2
IMAGE_TRAIN=us-central1-docker.pkg.dev/.../image-train-l4:v1
GCP_SA_KEY_PATH=./nombre-del-fichero.json   # ruta al JSON de la SA key
PREFECT_API_URL=http://localhost:4200/api
PREFECT_API_KEY=                             # vacío en self-hosted
```

### 2. Levantar el stack

```bash
docker compose up --build
```

Esto arranca 3 servicios:

| Servicio | Puerto | Descripción |
|---|---|---|
| `prefect-server` | 4200 | UI + API de Prefect |
| `prefect-worker` | — | Ejecuta los flows (escucha el work pool `gcp-worker`) |
| `api` | 8080 | FastAPI gateway |

- **Prefect UI:** http://localhost:4200
- **API:** http://localhost:8080
- **Health:** http://localhost:8080/health

### 3. Probar el pipeline

```bash
# Iniciar un run
curl -X POST http://localhost:8080/pipeline/start \
  -H "Content-Type: application/json" \
  -d '{"dataset": "mi_escena"}'

# Ver estado
curl http://localhost:8080/pipeline/<run_id>
```

### 4. Ejecutar tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Los tests usan mocks para GCP — no necesitan credenciales reales.

### Flujo interno del docker-compose

```
prefect-server arranca (puerto 4200)
       ↓ healthcheck OK
prefect-worker arranca (start_worker.sh)
  1. Espera a que prefect-server responda en /health
  2. Crea work pool "gcp-worker" (type: process)
  3. Registra el deployment "prod" desde prefect.yaml
  4. Arranca el worker → escucha trabajos pendientes
       ↓
api arranca (puerto 8080)
  → PREFECT_API_URL apunta a http://prefect-server:4200/api
```

---

## Deploy en GCE VM (producción)

El stack completo (Prefect server + worker + API) corre en una única GCE VM con docker-compose.

### 1. Crear la VM (solo una vez)

```bash
gcloud compute instances create vm-3daas \
  --project=skillful-air-480018-f2 \
  --zone=us-central1-a \
  --machine-type=e2-standard-2 \
  --image-family=debian-12 \
  --image-project=debian-cloud \
  --boot-disk-size=30GB \
  --service-account=custom-models@skillful-air-480018-f2.iam.gserviceaccount.com \
  --scopes=cloud-platform \
  --tags=vm-3daas
```

> `--scopes=cloud-platform` da acceso a GCS, Vertex AI y Artifact Registry sin necesitar copiar la SA key. `GCP_SA_KEY_PATH` puede quedar vacío.

### 2. Abrir puertos en el firewall

```bash
gcloud compute firewall-rules create allow-3daas \
  --project=skillful-air-480018-f2 \
  --allow=tcp:4200,tcp:8080 \
  --target-tags=vm-3daas \
  --description="Prefect UI y API 3DaaS"
```

### 3. Configurar `.env` para la VM

Añade en tu `.env` local:

```bash
GCE_VM_NAME=vm-3daas
GCE_VM_ZONE=us-central1-a
GCE_VM_USER=tu_usuario_en_la_vm   # derivado del email gcloud (puntos → guiones bajos)
```

El `PREFECT_API_URL` en el `.env` para producción debe apuntar a `http://localhost:4200/api` (dentro de la misma VM el worker habla con el server por localhost).

### 4. Desplegar

```bash
./scripts/deploy_vm.sh
```

El script hace automáticamente:

1. Verifica que la VM está `RUNNING` (la inicia si está parada)
2. Resuelve el directorio HOME real en la VM
3. Sincroniza código con `gcloud compute scp`
4. Copia `.env` y SA key (si existe localmente)
5. Instala Docker y docker compose plugin (si no están)
6. Ejecuta `docker compose up --build -d`
7. Imprime las URLs de acceso

```
Opciones:
  --no-build    Sync sin reconstruir imágenes Docker
  --sync-only   Solo copia archivos, no levanta el stack
```

### 5. Acceder a los servicios

```bash
# Obtener IP externa de la VM
gcloud compute instances describe vm-3daas \
  --zone=us-central1-a --format="value(networkInterfaces[0].accessConfigs[0].natIP)"
```

- **Prefect UI:** `http://<IP_EXTERNA>:4200`
- **API:** `http://<IP_EXTERNA>:8080`
- **Health:** `http://<IP_EXTERNA>:8080/health`

---

## CI/CD — Deploy continuo

Cada push a `master` dispara el workflow `.github/workflows/ci-cd.yml`:

```
push a master
      │
      ▼
  [test] pytest tests/ -v
      │ ✓ pasan todos
      ▼
  [deploy]
    1. Autentica en GCP con la SA key (secret GCP_SA_KEY)
    2. Verifica/inicia la VM
    3. Sincroniza código a la VM (gcloud compute scp)
    4. Escribe .env desde los secrets de GitHub
    5. Instala Docker si no está
    6. docker compose up --build -d
    7. Health check → imprime URLs
```

Si los tests fallan, el deploy **no se ejecuta**.

### Secrets y variables requeridos en GitHub

Ve a **Settings → Secrets and variables → Actions** en el repo.

**Variables** (no secretas — `vars.*`):

| Nombre | Valor |
|---|---|
| `GCE_VM_NAME` | `vm-3daas` |
| `GCE_VM_ZONE` | `us-central1-a` |
| `GCE_VM_USER` | usuario en la VM (ej. `albert_ceballos`) |

**Secrets** (`secrets.*`):

| Nombre | Valor |
|---|---|
| `GCP_SA_KEY` | Contenido JSON completo de la service account key |
| `GCS_BUCKET` | `bucket-saas-project` |
| `GCP_SERVICE_ACCOUNT` | `custom-models@skillful-air-480018-f2.iam.gserviceaccount.com` |
| `IMAGE_PREPROCESS` | URL completa de la imagen de preprocessing |
| `IMAGE_TRAIN` | URL completa de la imagen de training |
| `WEBHOOK_URL` | URL de notificación a Laravel (puede estar vacío) |

> `PREFECT_API_URL` se escribe directamente como `http://localhost:4200/api` en el workflow — no necesita secret.

---

## API Reference

### `POST /pipeline/start`

Inicia un pipeline run. Retorna `202 Accepted` con el estado inicial del run.

**Body (JSON):**

```json
{
  "dataset": "mi_escena",
  "skip_preprocess": false,

  // Parámetros COLMAP (opcionales)
  "matching_method": "vocab_tree",
  "sfm_tool": "colmap",
  "feature_type": "sift",
  "num_downscales_preprocess": 3,

  // Parámetros training (opcionales)
  "max_iters": 30000,
  "sh_degree": 3,
  "num_random": 100000
}
```

**Respuesta:**

```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "dataset": "mi_escena",
  "status": "queued",
  "stage": null,
  "started_at": "2026-03-10T12:00:00Z",
  "completed_at": null,
  "ply_uri": null,
  "error": null,
  "prefect_flow_run_id": "abc123",
  "params": { ... }
}
```

### `GET /pipeline/{run_id}`

Estado de un run. Cuando el pipeline termina, `status` es `"done"` y `ply_uri` contiene la URI GCS del `.ply`.

**Estados posibles:**

| `status` | `stage` | Descripción |
|---|---|---|
| `queued` | `null` | Esperando worker |
| `running` | `validating_raw` | Validando imágenes de entrada |
| `running` | `preprocessing` | COLMAP corriendo en Vertex AI |
| `running` | `validating_processed` | Verificando output COLMAP |
| `running` | `training` | splatfacto corriendo en Vertex AI |
| `done` | `null` | Completado — `ply_uri` disponible |
| `failed` | `*` | Error — ver campo `error` |

### `GET /pipeline`

Lista todos los runs. Parámetro opcional: `?limit=N` (default 100, max 500).

### `GET /health`

```json
{"status": "ok"}
```

---

## GCS — Layout de datos

```
gs://bucket-saas-project/
  <dataset>/
    raw/            ← imágenes originales (el usuario sube aquí antes de llamar a la API)
    processed/      ← output COLMAP: transforms.json + imágenes redimensionadas
    trained/        ← checkpoints Nerfstudio (outputs del training)
    exported/       ← archivo .ply final (~50–500 MB según escena)
```

> **Importante:** siempre usar `<dataset>/processed/`, nunca `<dataset>_processed/`.

El usuario sube las imágenes a `gs://bucket-saas-project/<dataset>/raw/` antes de llamar a `POST /pipeline/start`.

---

## Imágenes Docker

| Imagen | Uso | Cómo actualizar |
|---|---|---|
| `image-preprocess:v2` | COLMAP en Vertex AI (CPU) | `docker build -f docker-images/Dockerimage-process . && docker push ...` |
| `image-train-l4:v1` | splatfacto en Vertex AI (GPU L4) | `./sync_official_image.sh` |
| `Dockerfile.api` | FastAPI gateway | Automático via CI/CD |
| `Dockerfile.worker` | Prefect worker | Automático via CI/CD |

### Actualizar imagen de preprocessing

```bash
docker build -f docker-images/Dockerimage-process \
  -t us-central1-docker.pkg.dev/skillful-air-480018-f2/nerfstudio-repo/image-preprocess:v2 .
docker push us-central1-docker.pkg.dev/skillful-air-480018-f2/nerfstudio-repo/image-preprocess:v2
```

### Sincronizar imagen de training

```bash
./sync_official_image.sh
```

---

## Logs del sistema

Cada contenedor escribe logs estructurados en `./logs/` (montado como volumen):

```
logs/
├── worker/
│   ├── app.log       # INFO+ — actividad general del worker
│   ├── error.log     # ERROR+ — solo errores (retención 60 días)
│   └── debug.log     # DEBUG+ — verbose, solo en local (retención 7 días)
└── api/
    ├── app.log
    └── error.log
```

```bash
# En local
docker compose logs -f prefect-worker
tail -f logs/worker/error.log
grep "<run_id>" logs/worker/app.log

# En producción
gcloud compute ssh vm-3daas --zone=us-central1-a \
  --command="cd ~/3daas && docker compose logs -f prefect-worker --tail=100"
```

El nivel de log por consola depende de `APP_ENV`:
- `local` → `DEBUG` (verbose, ideal para desarrollo)
- `production` → `INFO`

---

## Entornos — local vs producción

El sistema usa `APP_ENV` para separar completamente los datos:

| Variable | `local` | `production` |
|---|---|---|
| `APP_ENV` | `local` | `production` |
| Colección Firestore | `pipeline_runs_dev` | `pipeline_runs` |
| Nivel de log consola | `DEBUG` | `INFO` |
| Fichero debug.log | Sí | No |

Cambiar de entorno: editar `APP_ENV` en `.env` y reiniciar los contenedores.

---

## Recuperación de runs interrumpidos

Si el worker cae (OOM, reinicio, deploy), al volver a arrancar ejecuta automáticamente `scripts/recover_stale_runs.py`:

1. Busca en Firestore todos los runs con `status == "running"`
2. Cancela el flow run anterior en Prefect (evita duplicados)
3. Para runs en una stage de Vertex AI, comprueba el estado del job en GCP:
   - `JOB_STATE_SUCCEEDED` → reanuda desde el siguiente stage de validación
   - `RUNNING/PENDING` → reanuda el polling del mismo job
   - Otro estado → re-submitirá el job
4. Crea un nuevo flow run en Prefect con `resume_from_stage`
5. Actualiza Firestore a `status=queued`

El pipeline retoma exactamente donde se quedó — no repite trabajo ya completado en Vertex AI.

Para ejecutarlo manualmente:

```bash
docker compose exec prefect-worker python /app/scripts/recover_stale_runs.py
```

---

## Monitoreo y debugging

### Ver logs en producción (VM)

```bash
# Todos los servicios
gcloud compute ssh vm-3daas --zone=us-central1-a \
  --command="cd ~/3daas && docker compose logs -f"

# Solo el worker (donde corren los flows)
gcloud compute ssh vm-3daas --zone=us-central1-a \
  --command="cd ~/3daas && docker compose logs -f prefect-worker"

# Estado de contenedores
gcloud compute ssh vm-3daas --zone=us-central1-a \
  --command="cd ~/3daas && docker compose ps"
```

### Ver logs en local

```bash
docker compose logs -f prefect-worker
docker compose logs -f api
```

### Monitorear jobs Vertex AI

```bash
gcloud ai custom-jobs stream-logs <JOB_ID> --region=us-central1
```

O desde la consola GCP:
https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=skillful-air-480018-f2

### Prefect UI

Muestra todos los flow runs, estado, logs y tareas en tiempo real.

- **Local:** http://localhost:4200
- **Producción:** `http://<IP_VM>:4200`

---

## Service Account & IAM

Todo el proyecto usa una única Service Account (SA): `custom-models@skillful-air-480018-f2.iam.gserviceaccount.com`.

Esta SA se usa en tres contextos:
- **GCE VM:** adjuntada a la VM al crearla (`--service-account`) — no necesita key JSON
- **Desarrollo local:** key JSON descargada y montada via `GCP_SA_KEY_PATH`
- **CI/CD:** key JSON almacenada como secret `GCP_SA_KEY` en GitHub Actions

### Roles IAM necesarios

```bash
PROJECT=skillful-air-480018-f2
SA=custom-models@skillful-air-480018-f2.iam.gserviceaccount.com

for ROLE in \
  roles/storage.objectAdmin \
  roles/aiplatform.user \
  roles/datastore.user \
  roles/artifactregistry.reader \
  roles/logging.logWriter \
  roles/iam.serviceAccountUser; do
    gcloud projects add-iam-policy-binding "$PROJECT" \
      --member="serviceAccount:$SA" \
      --role="$ROLE"
done
```

| Rol | Para qué se usa |
|---|---|
| `roles/storage.objectAdmin` | Leer/escribir en GCS: `raw/`, `processed/`, `trained/`, `exported/` |
| `roles/aiplatform.user` | Crear y monitorear Custom Jobs en Vertex AI |
| `roles/datastore.user` | Leer/escribir en Firestore (colección `pipeline_runs`) |
| `roles/artifactregistry.reader` | Pull de imágenes Docker desde Artifact Registry en los jobs de Vertex AI |
| `roles/logging.logWriter` | Escribir logs desde los containers de Vertex AI |
| `roles/iam.serviceAccountUser` | Necesario para que Vertex AI lance jobs usando esta misma SA |

### Crear la Service Account (si no existe)

```bash
gcloud iam service-accounts create custom-models \
  --project=skillful-air-480018-f2 \
  --display-name="3DaaS Custom Models SA"
```

Luego asigna los roles del bloque anterior.

### Descargar la key JSON

Necesaria para desarrollo local y para el secret de CI/CD. **No commitear al repositorio.**

```bash
gcloud iam service-accounts keys create skillful-air-480018-f2-bcb000b8c0e2.json \
  --iam-account=custom-models@skillful-air-480018-f2.iam.gserviceaccount.com \
  --project=skillful-air-480018-f2
```

El fichero resultante se usa:
- **Local:** `GCP_SA_KEY_PATH=./skillful-air-480018-f2-bcb000b8c0e2.json` en `.env`
- **CI/CD:** contenido JSON completo como secret `GCP_SA_KEY` en GitHub

### Autenticación en la GCE VM (sin key JSON)

La VM se crea con `--service-account` y `--scopes=cloud-platform`, lo que equivale a tener las credenciales de la SA disponibles automáticamente via el [metadata server de GCE](https://cloud.google.com/compute/docs/access/service-accounts). No es necesario copiar ningún fichero `.json` a la VM.

En el `.env` de producción, `GCP_SA_KEY_PATH` se deja vacío:

```bash
GCP_SA_KEY_PATH=
```

El SDK de Google Cloud detecta las credenciales automáticamente (Application Default Credentials).

---

## Seguridad

### Firewall

Solo se abren los puertos estrictamente necesarios, y solo para instancias con el network tag `vm-3daas`:

| Puerto | Servicio | Quién accede |
|---|---|---|
| `8080` | API FastAPI | Cliente externo (Laravel, curl) |
| `4200` | Prefect UI | Administrador / CI/CD health check |

```bash
# Ver reglas activas
gcloud compute firewall-rules list --filter="targetTags:vm-3daas" --project=skillful-air-480018-f2
```

Para restringir la Prefect UI a una IP concreta (recomendado en producción):

```bash
gcloud compute firewall-rules update allow-3daas \
  --source-ranges=<TU_IP>/32 \
  --project=skillful-air-480018-f2
```

### Service Account key

- El fichero `.json` de la SA key está en `.gitignore` — **nunca se commitea**
- En la GCE VM no se usa ningún fichero de key (usa el metadata server)
- En CI/CD la key se almacena cifrada como secret de GitHub Actions, nunca en texto plano
- Si la key se compromete: `gcloud iam service-accounts keys list --iam-account=<SA>` y revocar con `keys delete`

### Acceso SSH a la VM

El acceso SSH se gestiona via `gcloud compute ssh`, que usa OS Login o claves SSH efímeras de Google. No hay contraseñas ni claves estáticas.

```bash
# Acceso directo
gcloud compute ssh vm-3daas --zone=us-central1-a --project=skillful-air-480018-f2

# Ejecutar un comando remoto sin abrir terminal
gcloud compute ssh vm-3daas --zone=us-central1-a --project=skillful-air-480018-f2 \
  --command="docker compose ps"
```

### Principio de mínimo privilegio

La SA `custom-models` no tiene roles de propietario ni editor del proyecto. Solo los roles exactos que necesita cada servicio. Si en el futuro se añade una funcionalidad nueva que requiera permisos adicionales, añadir el rol específico con scope mínimo.

---

## GCP Resources

| Recurso | Valor |
|---|---|
| Project ID | `skillful-air-480018-f2` |
| Region | `us-central1` |
| GCS Bucket | `bucket-saas-project` |
| Artifact Registry | `us-central1-docker.pkg.dev/skillful-air-480018-f2/nerfstudio-repo/` |
| Service Account | `custom-models@skillful-air-480018-f2.iam.gserviceaccount.com` |
| GCE VM | `vm-3daas` (zona `us-central1-a`, tipo `e2-standard-2`) |
| Firestore DB | `(default)` colección `pipeline_runs` |
