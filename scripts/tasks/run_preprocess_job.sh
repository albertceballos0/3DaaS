#!/bin/bash
# =============================================================================
# run_preprocess_job.sh — Submit a COLMAP preprocessing job to Vertex AI
#
# Usage:
#   ./run_preprocess_job.sh <dataset_name> [OPTIONS]
#
# Positional:
#   dataset_name          Name of the dataset folder in GCS
#                         Input:  gs://bucket-saas-project/<dataset>/raw/
#                         Output: gs://bucket-saas-project/<dataset>/processed/
#
# Options:
#   --matching-method     vocab_tree | exhaustive | sequential  (default: vocab_tree)
#   --sfm-tool            colmap | hloc                         (default: colmap)
#   --feature-type        sift | superpoint | superpoint_aachen (default: sift)
#   --matcher-type        NN | superglue | supergluefast        (default: NN)
#   --num-downscales      N — number of image downscale levels  (default: 3)
#   --skip-colmap         Flag — skip COLMAP, use existing transforms.json
#   -h, --help            Show this help message
#
# Examples:
#   ./run_preprocess_job.sh scene_01
#   ./run_preprocess_job.sh scene_01 --matching-method exhaustive --num-downscales 0
#   ./run_preprocess_job.sh scene_01 --sfm-tool hloc --feature-type superpoint
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
SPEC_FILE="${REPO_ROOT}/specs/preprocess_spec.json"

# ── Defaults ─────────────────────────────────────────────────────────────────
MATCHING_METHOD="vocab_tree"
SFM_TOOL="colmap"
FEATURE_TYPE="sift"
MATCHER_TYPE="NN"
NUM_DOWNSCALES="3"
SKIP_COLMAP="false"

# ── Parse arguments ──────────────────────────────────────────────────────────
if [ $# -eq 0 ] || [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    sed -n '2,20p' "$0" | sed 's/^# \?//'
    exit 0
fi

DATASET_NAME="$1"
shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        --matching-method) MATCHING_METHOD="$2"; shift 2 ;;
        --sfm-tool)        SFM_TOOL="$2";        shift 2 ;;
        --feature-type)    FEATURE_TYPE="$2";    shift 2 ;;
        --matcher-type)    MATCHER_TYPE="$2";    shift 2 ;;
        --num-downscales)  NUM_DOWNSCALES="$2";  shift 2 ;;
        --skip-colmap)     SKIP_COLMAP="true";   shift 1 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── GCS paths ────────────────────────────────────────────────────────────────
GCS_INPUT="gs://${BUCKET}/${DATASET_NAME}/raw"
GCS_OUTPUT="gs://${BUCKET}/${DATASET_NAME}/processed"

# ── Validate ──────────────────────────────────────────────────────────────────
if [ -z "$DATASET_NAME" ]; then
    echo "Error: dataset_name is required." >&2
    exit 1
fi

if [[ ! "$MATCHING_METHOD" =~ ^(vocab_tree|exhaustive|sequential)$ ]]; then
    echo "Error: --matching-method must be vocab_tree, exhaustive, or sequential." >&2
    exit 1
fi

if [[ ! "$SFM_TOOL" =~ ^(colmap|hloc)$ ]]; then
    echo "Error: --sfm-tool must be colmap or hloc." >&2
    exit 1
fi

# ── Build skip-colmap flag ────────────────────────────────────────────────────
SKIP_FLAG=""
if [ "$SKIP_COLMAP" = "true" ]; then
    SKIP_FLAG="--skip-colmap"
fi

# ── Job name ─────────────────────────────────────────────────────────────────
JOB_NAME="preprocess_${DATASET_NAME}_$(date +%Y%m%d_%H%M%S)"

echo "======================================================="
echo "  Vertex AI — COLMAP Preprocessing"
echo "======================================================="
echo "  Dataset:          $DATASET_NAME"
echo "  Job name:         $JOB_NAME"
echo "  Input  (GCS):     $GCS_INPUT"
echo "  Output (GCS):     $GCS_OUTPUT"
echo "-------------------------------------------------------"
echo "  matching-method:  $MATCHING_METHOD"
echo "  sfm-tool:         $SFM_TOOL"
echo "  feature-type:     $FEATURE_TYPE"
echo "  matcher-type:     $MATCHER_TYPE"
echo "  num-downscales:   $NUM_DOWNSCALES"
echo "  skip-colmap:      $SKIP_COLMAP"
echo "======================================================="

# ── Command executed inside the container ────────────────────────────────────
COMMAND_STR="\
mkdir -p /tmp/processed && \
ns-process-data images \
  --data /gcs/${BUCKET}/${DATASET_NAME}/raw \
  --output-dir /tmp/processed \
  --no-gpu \
  --matching-method ${MATCHING_METHOD} \
  --sfm-tool ${SFM_TOOL} \
  --feature-type ${FEATURE_TYPE} \
  --matcher-type ${MATCHER_TYPE} \
  --num-downscales ${NUM_DOWNSCALES} \
  ${SKIP_FLAG} && \
gcloud storage cp -r /tmp/processed/* gs://${BUCKET}/${DATASET_NAME}/processed/"

# ── Submit job ────────────────────────────────────────────────────────────────
gcloud ai custom-jobs create \
    --region="$REGION" \
    --display-name="$JOB_NAME" \
    --config="$SPEC_FILE" \
    --service-account="$SERVICE_ACCOUNT" \
    --command="bash,-c,$COMMAND_STR"

echo ""
echo "  Job submitted to Vertex AI."
echo "  Monitor: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=${PROJECT_ID}"
echo "======================================================="
