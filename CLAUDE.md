# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project implements a cloud pipeline for training and rendering 3D Gaussian Splats using **Nerfstudio (splatfacto)** on **Google Cloud Vertex AI**. The workflow consists of two sequential stages: preprocessing raw images with COLMAP, then training the Gaussian Splat model and exporting to `.ply` format.

- **GCP Project ID:** `skillful-air-480018-f2`
- **Region:** `us-central1`
- **GCS Bucket:** `bucket-saas-project`
- **Artifact Registry:** `us-central1-docker.pkg.dev/skillful-air-480018-f2/nerfstudio-repo/`
- **Service Account:** `custom-models@skillful-air-480018-f2.iam.gserviceaccount.com`

## Pipeline Architecture

### Stage 1 — Preprocessing
- **Script:** `scripts/run_preprocess_job.sh <dataset_name>`
- **Vertex AI spec:** `specs/preprocess_spec.json`
- **Machine:** `n1-highmem-16` (CPU only, no GPU)
- **Container:** `image-preprocess:v2`
- Runs `ns-process-data images` (COLMAP) on raw images from `gs://bucket-saas-project/<dataset_name>/`
- Outputs processed data to `gs://bucket-saas-project/<dataset_name>_processed/`

### Stage 2 — Training + Export
- **Script:** `scripts/run_train_job.sh <dataset_name>`
- **Vertex AI spec:** `specs/train-export-spec.json`
- **Machine:** `g2-standard-12` with 1x NVIDIA L4 GPU
- **Container:** `image-train-l4:v1`
- Reads from `gs://bucket-saas-project/<dataset_name>_processed/`
- Runs `ns-train splatfacto` for 30,000 iterations
- Runs `ns-export gaussian-splat` to produce `.ply`
- Uploads trained model to `gs://bucket-saas-project/<dataset_name>_trained/`
- Uploads exported `.ply` to `gs://bucket-saas-project/<dataset_name>_exported/`

### Docker Images
- `docker-images/Dockerimage-process` — CPU preprocessing image (Python 3.10 slim + COLMAP + Nerfstudio CPU)
- Training image (`image-train-l4`) is assumed to be the official `dromni/nerfstudio:1.1.5` image synced via `sync_official_image.sh`

## Key Commands

### Run the full pipeline
```bash
# Step 1: Preprocess images
./scripts/run_preprocess_job.sh <dataset_name>

# Step 2: Train and export
./scripts/run_train_job.sh <dataset_name>
```

### Sync official Nerfstudio image to GCP
```bash
./sync_official_image.sh
```

### Build and push the preprocessing image
```bash
docker build -t image-preprocess:v2 -f docker-images/Dockerimage-process .
docker tag image-preprocess:v2 us-central1-docker.pkg.dev/skillful-air-480018-f2/nerfstudio-repo/image-preprocess:v2
docker push us-central1-docker.pkg.dev/skillful-air-480018-f2/nerfstudio-repo/image-preprocess:v2
```

### Monitor Vertex AI jobs
```bash
gcloud ai custom-jobs stream-logs <JOB_ID> --region=us-central1
# Or view in console: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=skillful-air-480018-f2
```

## GCS Data Layout

```
gs://bucket-saas-project/
  <dataset_name>/           # Raw input images
  <dataset_name>_processed/ # COLMAP output (transforms.json + images)
  <dataset_name>_trained/   # Nerfstudio checkpoint outputs
  <dataset_name>_exported/  # Final .ply Gaussian Splat file
```

## Important Notes

- The preprocessing container mounts GCS via the `/gcs/` FUSE path automatically provided by Vertex AI.
- `TORCH_COMPILE_DISABLE=1` is set at training time to avoid compilation issues on Vertex AI.
- The preprocessing job uses `--no-gpu` flag since the machine has no accelerator.
- `QT_QPA_PLATFORM=offscreen` is set in both specs to prevent display errors in headless containers.
