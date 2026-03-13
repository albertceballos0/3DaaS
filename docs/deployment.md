# Deployment Guide — 3DaaS

## Overview

The production stack runs entirely on a single **GCE VM** (`vm-3daas`, `e2-standard-2`, `us-central1-a`) using `docker compose`. The VM is deployed via `./scripts/deploy_vm.sh` or automatically by GitHub Actions on every push to `master`.

---

## Production Architecture

```
Internet
   │
   ├─ :8080 ──► FastAPI (api container)
   └─ :4200 ──► Prefect UI (prefect-server container)

GCE VM vm-3daas
   └─ docker compose
       ├─ prefect-server  (prefecthq/prefect:3-latest)
       ├─ prefect-worker  (Dockerfile.worker)
       └─ api             (Dockerfile.api)
```

---

## First-Time VM Setup

Run these commands once. If `vm-3daas` already exists, skip to [Deploy](#deploy).

### 1. Create the VM

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

> `--scopes=cloud-platform` grants full GCP access via the VM metadata server — no SA key JSON needed on the VM.

### 2. Open firewall ports

```bash
gcloud compute firewall-rules create allow-3daas \
  --project=skillful-air-480018-f2 \
  --allow=tcp:4200,tcp:8080 \
  --target-tags=vm-3daas \
  --description="Prefect UI (4200) and FastAPI (8080)"
```

To restrict access to specific IPs (recommended):

```bash
gcloud compute firewall-rules update allow-3daas \
  --source-ranges=<YOUR_IP>/32
```

### 3. Configure your local `.env`

Add to your local `.env`:

```bash
GCE_VM_NAME=vm-3daas
GCE_VM_ZONE=us-central1-a
GCE_VM_USER=albert_ceballos    # your gcloud user (dots replaced with underscores)
```

For production, also set:

```bash
APP_ENV=production              # uses pipeline_runs Firestore collection
GCP_SA_KEY_PATH=               # empty — VM uses metadata server
PREFECT_UI_API_URL=http://<VM_EXTERNAL_IP>:4200/api
```

---

## Deploy

```bash
./scripts/deploy_vm.sh
```

What the script does:

1. Reads `GCE_VM_NAME`, `GCE_VM_ZONE`, `GCE_VM_USER` from `.env`
2. Checks VM status — starts it if stopped
3. Resolves the real `$HOME` directory on the VM
4. Syncs code to the VM via `gcloud compute scp --recurse`
5. Copies `.env` and SA key JSON (if `GCP_SA_KEY_PATH` is set)
6. SSHs into the VM and:
   - Installs Docker + compose plugin (if missing)
   - Runs `docker compose up --build -d`
   - Waits for health check
7. Prints the external IP and access URLs

### Deploy options

```bash
./scripts/deploy_vm.sh --no-build    # sync code only, skip docker build
./scripts/deploy_vm.sh --sync-only   # only copy files, don't start containers
```

### After deploy

```bash
# Get external IP
gcloud compute instances describe vm-3daas \
  --zone=us-central1-a \
  --format="value(networkInterfaces[0].accessConfigs[0].natIP)"

# Access URLs
# API:         http://<IP>:8080
# Prefect UI:  http://<IP>:4200
# Health:      http://<IP>:8080/health
```

---

## CI/CD — GitHub Actions

Every push to `master` runs `.github/workflows/ci-cd.yml`:

```
push to master
      │
      ▼
  ┌─────────────────────────────────────────┐
  │  job: test                              │
  │  - pip install -r requirements-dev.txt  │
  │  - pytest tests/ -v                     │
  └──────────────────┬──────────────────────┘
                     │ all tests pass
                     ▼
  ┌─────────────────────────────────────────┐
  │  job: deploy                            │
  │  1. Auth with GCP SA key (secret)       │
  │  2. Verify/start VM                     │
  │  3. gcloud compute scp (sync code)      │
  │  4. Write .env from GitHub secrets      │
  │  5. Install Docker (if missing)         │
  │  6. docker compose up --build -d        │
  │  7. Health check                        │
  └─────────────────────────────────────────┘
```

If tests fail → deploy does **not** run.

### Required GitHub Secrets

Go to **Settings → Secrets and variables → Actions**:

**Secrets** (`secrets.*`):

| Secret | Description |
|---|---|
| `GCP_SA_KEY` | Full JSON content of the service account key |
| `GCS_BUCKET` | `bucket-saas-project` |
| `GCP_SERVICE_ACCOUNT` | `custom-models@skillful-air-480018-f2.iam.gserviceaccount.com` |
| `IMAGE_PREPROCESS` | Full URI of preprocess Docker image |
| `IMAGE_TRAIN` | Full URI of training Docker image |
| `WEBHOOK_URL` | Webhook callback URL (can be empty) |

**Variables** (`vars.*`):

| Variable | Value |
|---|---|
| `GCE_VM_NAME` | `vm-3daas` |
| `GCE_VM_ZONE` | `us-central1-a` |
| `GCE_VM_USER` | SSH username on the VM |

---

## Managing the Production VM

### Start / Stop

```bash
# Start (if stopped to save costs)
gcloud compute instances start vm-3daas --zone=us-central1-a

# Stop
gcloud compute instances stop vm-3daas --zone=us-central1-a

# Current status
gcloud compute instances describe vm-3daas --zone=us-central1-a --format="value(status)"
```

### SSH access

```bash
# Open interactive shell
gcloud compute ssh vm-3daas --zone=us-central1-a

# Run a single command
gcloud compute ssh vm-3daas --zone=us-central1-a \
  --command="cd ~/3daas && docker compose ps"
```

### View production logs

```bash
# All services, follow
gcloud compute ssh vm-3daas --zone=us-central1-a \
  --command="cd ~/3daas && docker compose logs -f"

# Worker only (where flows run)
gcloud compute ssh vm-3daas --zone=us-central1-a \
  --command="cd ~/3daas && docker compose logs -f prefect-worker --tail=100"

# API only
gcloud compute ssh vm-3daas --zone=us-central1-a \
  --command="cd ~/3daas && docker compose logs -f api --tail=50"
```

### Restart services

```bash
# Restart all
gcloud compute ssh vm-3daas --zone=us-central1-a \
  --command="cd ~/3daas && docker compose restart"

# Restart only worker (recovers stale runs automatically on restart)
gcloud compute ssh vm-3daas --zone=us-central1-a \
  --command="cd ~/3daas && docker compose restart prefect-worker"
```

### Check disk usage

```bash
gcloud compute ssh vm-3daas --zone=us-central1-a \
  --command="df -h && docker system df"
```

Clean up Docker artifacts if disk is full:

```bash
gcloud compute ssh vm-3daas --zone=us-central1-a \
  --command="docker system prune -f"
```

---

## Docker Images

### Worker image (Prefect flows)

Built from [docker-images/Dockerfile.worker](../docker-images/Dockerfile.worker). Rebuilt automatically by CI/CD on every deploy.

To rebuild manually:

```bash
docker build -f docker-images/Dockerfile.worker -t 3daas-worker:latest .
```

### API image

Built from [docker-images/Dockerfile.api](../docker-images/Dockerfile.api). Multi-stage build — dependencies in a builder stage, clean runtime.

### Vertex AI — Preprocess image

Built from `docker-images/Dockerimage-process`. Must be pushed to Artifact Registry manually:

```bash
docker build -f docker-images/Dockerimage-process \
  -t us-central1-docker.pkg.dev/skillful-air-480018-f2/nerfstudio-repo/image-preprocess:v2 .

docker push us-central1-docker.pkg.dev/skillful-air-480018-f2/nerfstudio-repo/image-preprocess:v2
```

### Vertex AI — Training image

Synced from the official Nerfstudio image:

```bash
./sync_official_image.sh
```

---

## Service Account & IAM

### Required roles

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

| Role | Purpose |
|---|---|
| `storage.objectAdmin` | Read/write GCS: raw/, processed/, trained/, exported/ |
| `aiplatform.user` | Create and monitor Vertex AI Custom Jobs |
| `datastore.user` | Read/write Firestore collection `pipeline_runs` |
| `artifactregistry.reader` | Pull Docker images from Artifact Registry in Vertex AI jobs |
| `logging.logWriter` | Write logs from Vertex AI containers |
| `iam.serviceAccountUser` | Required for Vertex AI to launch jobs as this SA |

### Key management

```bash
# List existing keys
gcloud iam service-accounts keys list \
  --iam-account=custom-models@skillful-air-480018-f2.iam.gserviceaccount.com

# Create new key (for local dev / CI/CD)
gcloud iam service-accounts keys create sa-key.json \
  --iam-account=custom-models@skillful-air-480018-f2.iam.gserviceaccount.com

# Revoke a compromised key
gcloud iam service-accounts keys delete <KEY_ID> \
  --iam-account=custom-models@skillful-air-480018-f2.iam.gserviceaccount.com
```

---

## Rollback

If a deploy breaks production:

```bash
# On the VM, pull a specific Git commit and redeploy
gcloud compute ssh vm-3daas --zone=us-central1-a --command="
  cd ~/3daas && \
  git fetch origin && \
  git checkout <previous_commit_hash> && \
  docker compose up --build -d
"
```

Or revert via `git revert` locally and push to `master` to trigger CI/CD.
