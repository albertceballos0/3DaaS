```markdown
# Gaussian Splatting on Google Cloud (Vertex AI)

This project implements a containerized pipeline for training and rendering 3D Gaussian Splats using the `gcloud` CLI and Vertex AI infrastructure.

## Project Structure
```text
.
├── Dockerfile              # GPU-enabled container definition
├── Makefile                # Automation for build/deploy
├── configs/                # Hyperparameter configurations
├── data/                   # Local dataset mount point
├── scripts/                # Utility scripts for data processing
├── src/
│   ├── train.py            # Main training entrypoint
│   └── render.py           # Inference/Rendering script
└── requirements.txt        # Python dependencies
```

## Local Development

### Build Docker Image
```bash
docker build -t gaussian-splatting:latest .
```

### Run Locally (GPU required)
```bash
docker run --gpus all \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/output:/workspace/output \
    gaussian-splatting:latest python src/train.py --data_path /workspace/data/scene
```

## Google Cloud Infrastructure

### 1. Configure GCR/Artifact Registry
```bash
gcloud auth configure-docker us-central1-docker.pkg.dev
```

### 2. Tag and Push Image
```bash
export PROJECT_ID=$(gcloud config get-value project)
export REPO_URL=us-central1-docker.pkg.dev/$PROJECT_ID/gaussian-repo/gaussian-splatting:latest

docker tag gaussian-splatting:latest $REPO_URL
docker push $REPO_URL
```

### 3. Create Vertex AI Custom Training Job
```bash
gcloud ai custom-jobs create \
    --region=us-central1 \
    --display-name=gaussian-splatting-train \
    --worker-pool-spec=machine-type=n1-standard-8,replica-count=1,container-image-uri=$REPO_URL,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1 \
    --args="--data_path=gs://your-bucket/data/scene","--output_path=gs://your-bucket/models/exp1"
```

### 4. Monitor Job
```bash
gcloud ai custom-jobs stream-logs [JOB_ID]
```
```

## Google Cloud Artifact Registry

```bash
gcloud auth configure-docker us-central1-docker.pkg.dev
```

gcloud artifacts repositories create [NOMBRE_DEL_REPOSITORIO] \
    --repository-format=docker \
    --location=[REGION] \
    --description="Mi repositorio de Docker"


docker build -t [NOMBRE_IMAGEN_LOCAL]:[ETIQUETA_LOCAL] .

docker tag [NOMBRE_IMAGEN_LOCAL]:[ETIQUETA_LOCAL] [REGION]-docker.pkg.dev/[ID_PROYECTO]/[NOMBRE_REPOSITORIO]/[NOMBRE_IMAGEN_EN_AR]:[ETIQUETA_AR]

docker push [REGION]-docker.pkg.dev/[ID_PROYECTO]/[NOMBRE_REPOSITORIO]/[NOMBRE_IMAGEN_EN_AR]:[ETIQUETA_AR]

docker tag image-train-l4:latest us-central1-docker.pkg.dev/skillful-air-480018-f2/nerfstudio-repo/image-train-l4:v1

docker push us-central1-docker.pkg.dev/skillful-air-480018-f2/nerfstudio-repo/image-train-l4:v1