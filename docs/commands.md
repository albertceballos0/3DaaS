# Commands Reference — 3DaaS

All commands used to develop, operate, and deploy the 3DaaS pipeline.

---

## Docker Compose

### Start / Stop

```bash
# Start full stack (build first)
docker compose up --build

# Start in background
docker compose up --build -d

# Start only specific services
docker compose up --build prefect-server
docker compose up --build prefect-worker api

# Stop all services
docker compose down

# Stop and remove volumes (DELETES prefect-data)
docker compose down -v

# Restart a service
docker compose restart prefect-worker
docker compose restart api
```

### Logs

```bash
# Follow all services
docker compose logs -f

# Follow a specific service
docker compose logs -f prefect-worker
docker compose logs -f api
docker compose logs -f prefect-server

# Last N lines (no follow)
docker compose logs --tail=100 prefect-worker

# Last N lines + follow
docker compose logs -f --tail=50 prefect-worker
```

### Status & Exec

```bash
# Container status
docker compose ps

# Execute a command inside the worker
docker compose exec prefect-worker bash
docker compose exec prefect-worker python scripts/recover_stale_runs.py

# Execute inside API
docker compose exec api bash
```

### Build

```bash
# Rebuild without cache
docker compose build --no-cache

# Build a specific service
docker compose build prefect-worker
```

---

## Prefect CLI

### Deployments

```bash
# Register all deployments from prefect.yaml
prefect deploy --all --prefect-file prefect.yaml

# Register a specific deployment
prefect deploy gaussian-pipeline/prod --prefect-file prefect.yaml

# List deployments
prefect deployment ls
```

### Flow Runs

```bash
# Trigger a run from CLI (must have PREFECT_API_URL set)
prefect deployment run 'gaussian-pipeline/prod' \
  -p run_id=<uuid> \
  -p dataset=mi_escena

# List recent flow runs
prefect flow-run ls

# Cancel a flow run
prefect flow-run cancel <flow_run_id>

# Inspect a flow run
prefect flow-run inspect <flow_run_id>
```

### Worker

```bash
# Start a worker manually (local dev, no Docker)
PREFECT_API_URL=http://localhost:4200/api \
  prefect worker start --pool gcp-worker

# Create a work pool
prefect work-pool create gcp-worker --type process

# List work pools
prefect work-pool ls
```

### Server (for local CLI usage without Docker)

```bash
# Start Prefect server locally (if not using docker compose)
prefect server start --host 0.0.0.0

# Check server health
curl http://localhost:4200/api/health
```

---

## gcloud — GCE VM

### VM Management

```bash
# Create VM (one-time setup)
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

# Start / Stop
gcloud compute instances start vm-3daas --zone=us-central1-a
gcloud compute instances stop vm-3daas --zone=us-central1-a

# Get external IP
gcloud compute instances describe vm-3daas \
  --zone=us-central1-a \
  --format="value(networkInterfaces[0].accessConfigs[0].natIP)"

# Check status
gcloud compute instances describe vm-3daas \
  --zone=us-central1-a \
  --format="value(status)"

# List all VMs
gcloud compute instances list --project=skillful-air-480018-f2
```

### SSH & Remote Commands

```bash
# Open interactive shell
gcloud compute ssh vm-3daas --zone=us-central1-a

# Run a single command on the VM
gcloud compute ssh vm-3daas --zone=us-central1-a \
  --command="cd ~/3daas && docker compose ps"

# Follow logs remotely
gcloud compute ssh vm-3daas --zone=us-central1-a \
  --command="cd ~/3daas && docker compose logs -f prefect-worker"
```

### File Transfer

```bash
# Copy files to VM
gcloud compute scp --recurse ./flows vm-3daas:~/3daas/ --zone=us-central1-a

# Copy a single file
gcloud compute scp .env vm-3daas:~/3daas/.env --zone=us-central1-a

# Copy from VM to local
gcloud compute scp vm-3daas:~/3daas/logs/worker/error.log . --zone=us-central1-a
```

### Firewall

```bash
# Create firewall rule for 3DaaS ports
gcloud compute firewall-rules create allow-3daas \
  --project=skillful-air-480018-f2 \
  --allow=tcp:4200,tcp:8080 \
  --target-tags=vm-3daas

# Restrict to specific IP
gcloud compute firewall-rules update allow-3daas \
  --source-ranges=<YOUR_IP>/32

# List rules for vm-3daas tag
gcloud compute firewall-rules list \
  --filter="targetTags:vm-3daas" \
  --project=skillful-air-480018-f2
```

---

## gcloud — GCS

### Datasets

```bash
# Upload raw images for a dataset (full pipeline with COLMAP)
./scripts/sync_dataset.sh <dataset> raw ./local/images/

# Upload pre-processed data (skip COLMAP)
./scripts/sync_dataset.sh <dataset> data ./local/nerfstudio_data/

# Manually copy to GCS
gcloud storage cp -r ./images/ gs://bucket-saas-project/<dataset>/raw/

# Sync (like rsync)
gcloud storage rsync --recursive ./images/ gs://bucket-saas-project/<dataset>/raw/

# List contents of a dataset
gcloud storage ls gs://bucket-saas-project/<dataset>/

# List all datasets
gcloud storage ls gs://bucket-saas-project/

# Download exported .ply
gcloud storage cp gs://bucket-saas-project/<dataset>/exported/<dataset>.ply .

# Delete a dataset (all stages)
gcloud storage rm -r gs://bucket-saas-project/<dataset>/
```

---

## gcloud — Vertex AI

### Custom Jobs

```bash
# List recent jobs
gcloud ai custom-jobs list \
  --region=us-central1 \
  --project=skillful-air-480018-f2

# Describe a job
gcloud ai custom-jobs describe <JOB_ID> \
  --region=us-central1

# Stream logs from a running job
gcloud ai custom-jobs stream-logs <JOB_ID> \
  --region=us-central1

# Cancel a running job
gcloud ai custom-jobs cancel <JOB_ID> \
  --region=us-central1

# Submit a job manually (preprocess)
./scripts/tasks/run_preprocess_job.sh

# Submit a job manually (training)
./scripts/tasks/run_train_job.sh
```

---

## gcloud — IAM & Service Accounts

```bash
# List service accounts
gcloud iam service-accounts list --project=skillful-air-480018-f2

# Create service account
gcloud iam service-accounts create custom-models \
  --project=skillful-air-480018-f2 \
  --display-name="3DaaS Custom Models SA"

# Grant a role
gcloud projects add-iam-policy-binding skillful-air-480018-f2 \
  --member="serviceAccount:custom-models@skillful-air-480018-f2.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"

# List SA keys
gcloud iam service-accounts keys list \
  --iam-account=custom-models@skillful-air-480018-f2.iam.gserviceaccount.com

# Create SA key (local dev / CI/CD)
gcloud iam service-accounts keys create sa-key.json \
  --iam-account=custom-models@skillful-air-480018-f2.iam.gserviceaccount.com

# Revoke a key
gcloud iam service-accounts keys delete <KEY_ID> \
  --iam-account=custom-models@skillful-air-480018-f2.iam.gserviceaccount.com
```

---

## gcloud — Artifact Registry

```bash
# Configure Docker auth for Artifact Registry
gcloud auth configure-docker us-central1-docker.pkg.dev

# List images in the repo
gcloud artifacts docker images list \
  us-central1-docker.pkg.dev/skillful-air-480018-f2/nerfstudio-repo

# Push an image
docker push us-central1-docker.pkg.dev/skillful-air-480018-f2/nerfstudio-repo/image-preprocess:v2
```

---

## Custom Scripts

### Deploy to GCE VM

```bash
./scripts/deploy_vm.sh              # full deploy (build + up)
./scripts/deploy_vm.sh --no-build   # deploy without rebuilding Docker images
./scripts/deploy_vm.sh --sync-only  # only sync files, don't restart services
```

### Sync dataset to GCS

```bash
# Upload raw images (will run COLMAP in the pipeline)
./scripts/sync_dataset.sh <dataset_name> raw ./path/to/images/

# Upload pre-processed Nerfstudio data (skip COLMAP)
./scripts/sync_dataset.sh <dataset_name> data ./path/to/nerfstudio_data/
```

### Sync official Nerfstudio training image

```bash
./sync_official_image.sh
# Pulls the official Nerfstudio image and re-tags it in our Artifact Registry
```

### Recover stale runs manually

```bash
# Inside the worker container
docker compose exec prefect-worker python /app/scripts/recover_stale_runs.py

# Or from local (with .env loaded)
python scripts/recover_stale_runs.py
```

---

## API (curl examples)

```bash
BASE=http://localhost:8080   # or http://<VM_IP>:8080 for production

# Health check
curl $BASE/health

# Start a pipeline run
curl -X POST $BASE/pipeline/start \
  -H "Content-Type: application/json" \
  -d '{"dataset": "mi_escena", "max_iters": 30000}'

# Get run status
curl $BASE/pipeline/<run_id>

# List all runs
curl "$BASE/pipeline?limit=20"

# Cancel a run
curl -X DELETE $BASE/pipeline/<run_id>
```

---

## Tests

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_vertex.py -v
pytest tests/test_pipeline.py -v

# Run with coverage
pytest tests/ --cov=flows --cov=api --cov=utils --cov-report=term-missing

# Run a specific test by name
pytest tests/ -k "test_poll_vertex_job_success"

# Run with output (no capture)
pytest tests/ -v -s
```

---

## Docker — Vertex AI Images

### Build preprocess image

```bash
docker build \
  -f docker-images/Dockerimage-process \
  -t us-central1-docker.pkg.dev/skillful-air-480018-f2/nerfstudio-repo/image-preprocess:v2 \
  .
docker push us-central1-docker.pkg.dev/skillful-air-480018-f2/nerfstudio-repo/image-preprocess:v2
```

### Build worker image manually

```bash
docker build \
  -f docker-images/Dockerfile.worker \
  -t 3daas-worker:latest \
  .
```

### Build API image manually

```bash
docker build \
  -f docker-images/Dockerfile.api \
  -t 3daas-api:latest \
  .
```

---

## Git & CI/CD

```bash
# Deploy to production (via CI/CD)
git push origin master

# Check GitHub Actions status
gh run list --branch master
gh run view <run_id>

# View CI/CD workflow
cat .github/workflows/ci-cd.yml
```
