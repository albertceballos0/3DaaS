#!/bin/bash
# =============================================================================
# deploy_api.sh — Build, push and deploy the FastAPI gateway to Cloud Run
#
# Usage:
#   ./scripts/deploy_api.sh [OPTIONS]
#
# Options:
#   --tag TAG     Docker image tag  (default: git short SHA or "latest")
#   --no-build    Skip docker build/push, redeploy the existing :latest image
#   -h, --help    Show this help message
#
# Prerequisites:
#   - .env file in the repo root (copy from .env.example)
#   - gcloud authenticated: gcloud auth login / gcloud auth configure-docker
#   - Docker daemon running
# =============================================================================

set -euo pipefail

# ── Load .env from repo root ──────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="${REPO_ROOT}/.env"
if [ -f "$ENV_FILE" ]; then
    set -a; source "$ENV_FILE"; set +a
else
    echo "Error: .env not found at ${ENV_FILE}. Copy .env.example → .env and fill in values." >&2
    exit 1
fi

# ── GCP config (from .env) ────────────────────────────────────────────────────
PROJECT_ID="${GCP_PROJECT_ID}"
REGION="${GCP_REGION}"
BUCKET="${GCS_BUCKET}"
SERVICE_ACCOUNT="${GCP_SERVICE_ACCOUNT}"
WEBHOOK_URL="${WEBHOOK_URL:-}"

REGISTRY="${REGION}-docker.pkg.dev/${PROJECT_ID}/nerfstudio-repo"
IMAGE_NAME="image-api"
SERVICE_NAME="api-3daas"
DOCKERFILE="${REPO_ROOT}/docker-images/Dockerfile.api"

IMAGE_PREPROCESS_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/nerfstudio-repo/image-preprocess:v2"
IMAGE_TRAIN_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/nerfstudio-repo/image-train-l4:v1"

# ── Defaults ──────────────────────────────────────────────────────────────────
TAG=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo "latest")
BUILD=true

# ── Parse arguments ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag)      TAG="$2";    shift 2 ;;
        --no-build) BUILD=false; shift 1 ;;
        -h|--help)
            sed -n '2,13p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${TAG}"
LATEST_IMAGE="${REGISTRY}/${IMAGE_NAME}:latest"

echo "======================================================="
echo "  Cloud Run — API Deploy"
echo "======================================================="
echo "  Service:    $SERVICE_NAME"
echo "  Region:     $REGION"
echo "  Image:      $FULL_IMAGE"
echo "  Build:      $BUILD"
echo "======================================================="

# ── Build & push ──────────────────────────────────────────────────────────────
if [ "$BUILD" = true ]; then
    echo ""
    echo "→ Configuring Docker for Artifact Registry..."
    gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

    LOCAL_IMAGE="${IMAGE_NAME}:${TAG}"

    echo "→ Building image (local tag: ${LOCAL_IMAGE})..."
    docker build \
        -f "$DOCKERFILE" \
        -t "$LOCAL_IMAGE" \
        "$REPO_ROOT"

    echo "→ Tagging for Artifact Registry..."
    docker tag "$LOCAL_IMAGE" "$FULL_IMAGE"
    docker tag "$LOCAL_IMAGE" "$LATEST_IMAGE"

    echo "→ Pushing to Artifact Registry..."
    docker push "$FULL_IMAGE"
    docker push "$LATEST_IMAGE"
fi

# ── Deploy to Cloud Run ───────────────────────────────────────────────────────
echo ""
echo "→ Deploying to Cloud Run..."

ENV_VARS="\
GCP_PROJECT_ID=${PROJECT_ID},\
GCP_REGION=${REGION},\
GCS_BUCKET=${BUCKET},\
GCP_SERVICE_ACCOUNT=${SERVICE_ACCOUNT},\
IMAGE_PREPROCESS=${IMAGE_PREPROCESS_URI},\
IMAGE_TRAIN=${IMAGE_TRAIN_URI}"

if [ -n "$WEBHOOK_URL" ]; then
    ENV_VARS="${ENV_VARS},WEBHOOK_URL=${WEBHOOK_URL}"
fi

gcloud run deploy "$SERVICE_NAME" \
    --image "$FULL_IMAGE" \
    --region "$REGION" \
    --platform managed \
    --allow-unauthenticated \
    --service-account "$SERVICE_ACCOUNT" \
    --port 8080 \
    --set-env-vars "$ENV_VARS"

echo ""
echo "======================================================="
echo "  Deploy complete!"
echo "  URL: $(gcloud run services describe "$SERVICE_NAME" \
        --region "$REGION" --format 'value(status.url)')"
echo "======================================================="
