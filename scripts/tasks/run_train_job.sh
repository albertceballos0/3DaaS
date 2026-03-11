#!/bin/bash
# =============================================================================
# run_train_job.sh — Submit a Gaussian Splatting training job to Vertex AI
#
# Usage:
#   ./run_train_job.sh <dataset_name> [OPTIONS]
#
# Positional:
#   dataset_name              Name of the dataset folder in GCS
#                             Input:  gs://bucket-saas-project/<dataset>/processed/
#                             Output: gs://bucket-saas-project/<dataset>/trained/
#                                     gs://bucket-saas-project/<dataset>/exported/
#
# Options (splatfacto model):
#   --max-iters       N       Max training iterations          (default: 30000)
#   --sh-degree       N       Spherical harmonics degree 0-3   (default: 3)
#   --num-random      N       Initial random Gaussians         (default: 100000)
#   --num-downscales  N       Image pyramid downscale levels   (default: 0)
#   --res-schedule    N       Resolution schedule iters        (default: 250)
#   --densify-grad    F       Densification gradient threshold (default: 0.0002)
#   --densify-size    F       Densification size threshold     (default: 0.01)
#   --n-split         N       Split samples per densification  (default: 2)
#   --refine-start    N       Start refine at iteration        (default: 500)
#   --refine-stop     N       Stop refine at iteration         (default: 25000)
#   --refine-every    N       Refine every N iterations        (default: 100)
#   --cull-scale      F       Cull scale threshold             (default: 0.5)
#   --reset-alpha     N       Reset alpha every N Gaussians    (default: 30)
#   -h, --help                Show this help message
#
# Examples:
#   ./run_train_job.sh scene_01
#   ./run_train_job.sh scene_01 --max-iters 50000 --sh-degree 2
#   ./run_train_job.sh scene_01 --densify-grad 0.0001 --refine-stop 30000
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
SPEC_FILE="${REPO_ROOT}/specs/train-export-spec.json"

# ── Defaults ──────────────────────────────────────────────────────────────────
MAX_ITERS=30000
SH_DEGREE=3
NUM_RANDOM=100000
NUM_DOWNSCALES=0
RES_SCHEDULE=250
DENSIFY_GRAD=0.0002
DENSIFY_SIZE=0.01
N_SPLIT=2
REFINE_START=500
REFINE_STOP=25000
REFINE_EVERY=100
CULL_SCALE=0.5
RESET_ALPHA=30

# ── Parse arguments ──────────────────────────────────────────────────────────
if [ $# -eq 0 ] || [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    sed -n '2,27p' "$0" | sed 's/^# \?//'
    exit 0
fi

DATASET_NAME="$1"
shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-iters)      MAX_ITERS="$2";      shift 2 ;;
        --sh-degree)      SH_DEGREE="$2";      shift 2 ;;
        --num-random)     NUM_RANDOM="$2";     shift 2 ;;
        --num-downscales) NUM_DOWNSCALES="$2"; shift 2 ;;
        --res-schedule)   RES_SCHEDULE="$2";   shift 2 ;;
        --densify-grad)   DENSIFY_GRAD="$2";   shift 2 ;;
        --densify-size)   DENSIFY_SIZE="$2";   shift 2 ;;
        --n-split)        N_SPLIT="$2";        shift 2 ;;
        --refine-start)   REFINE_START="$2";   shift 2 ;;
        --refine-stop)    REFINE_STOP="$2";    shift 2 ;;
        --refine-every)   REFINE_EVERY="$2";   shift 2 ;;
        --cull-scale)     CULL_SCALE="$2";     shift 2 ;;
        --reset-alpha)    RESET_ALPHA="$2";    shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Validate ──────────────────────────────────────────────────────────────────
if [ -z "$DATASET_NAME" ]; then
    echo "Error: dataset_name is required." >&2
    exit 1
fi

if [ "$SH_DEGREE" -lt 0 ] || [ "$SH_DEGREE" -gt 3 ]; then
    echo "Error: --sh-degree must be between 0 and 3." >&2
    exit 1
fi

if [ "$REFINE_START" -ge "$REFINE_STOP" ]; then
    echo "Error: --refine-start must be less than --refine-stop." >&2
    exit 1
fi

# ── GCS paths ─────────────────────────────────────────────────────────────────
GCS_INPUT="gs://${BUCKET}/${DATASET_NAME}/processed"
GCS_TRAINED="gs://${BUCKET}/${DATASET_NAME}/trained"
GCS_EXPORTED="gs://${BUCKET}/${DATASET_NAME}/exported"

# ── Job name ──────────────────────────────────────────────────────────────────
JOB_NAME="train_${DATASET_NAME}_$(date +%Y%m%d_%H%M%S)"

echo "======================================================="
echo "  Vertex AI — Gaussian Splatting Training"
echo "======================================================="
echo "  Dataset:          $DATASET_NAME"
echo "  Job name:         $JOB_NAME"
echo "  Input  (GCS):     $GCS_INPUT"
echo "  Trained (GCS):    $GCS_TRAINED"
echo "  Exported (GCS):   $GCS_EXPORTED"
echo "-------------------------------------------------------"
echo "  max-iters:        $MAX_ITERS"
echo "  sh-degree:        $SH_DEGREE"
echo "  num-random:       $NUM_RANDOM"
echo "  num-downscales:   $NUM_DOWNSCALES"
echo "  res-schedule:     $RES_SCHEDULE"
echo "  densify-grad:     $DENSIFY_GRAD"
echo "  densify-size:     $DENSIFY_SIZE"
echo "  n-split:          $N_SPLIT"
echo "  refine-start:     $REFINE_START"
echo "  refine-stop:      $REFINE_STOP"
echo "  refine-every:     $REFINE_EVERY"
echo "  cull-scale:       $CULL_SCALE"
echo "  reset-alpha:      $RESET_ALPHA"
echo "======================================================="

# ── Command executed inside the container ────────────────────────────────────
# Note: CONFIG_PATH is resolved at runtime inside the container, so the find
# command uses a single-escaped $ to defer expansion to the remote shell.
COMMAND_STR="\
export TORCH_COMPILE_DISABLE=1 && \
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/nvidia/lib64 && \
ns-train splatfacto \
  --data /gcs/${BUCKET}/${DATASET_NAME}/processed \
  --output-dir /tmp/outputs \
  --viewer.quit-on-train-completion True \
  --project-name ${DATASET_NAME} \
  --max-num-iterations ${MAX_ITERS} \
  --pipeline.model.sh-degree ${SH_DEGREE} \
  --pipeline.model.sh-degree-interval 1000 \
  --pipeline.model.num-random ${NUM_RANDOM} \
  --pipeline.model.num-downscales ${NUM_DOWNSCALES} \
  --pipeline.model.resolution-schedule ${RES_SCHEDULE} \
  --pipeline.model.densify-grad-thresh ${DENSIFY_GRAD} \
  --pipeline.model.densify-size-thresh ${DENSIFY_SIZE} \
  --pipeline.model.n-split-samples ${N_SPLIT} \
  --pipeline.model.refine-start-iter ${REFINE_START} \
  --pipeline.model.refine-stop-iter ${REFINE_STOP} \
  --pipeline.model.refine-every ${REFINE_EVERY} \
  --pipeline.model.cull-scale-thresh ${CULL_SCALE} \
  --pipeline.model.reset-alpha-every ${RESET_ALPHA} \
  --pipeline.model.random-init False && \
CONFIG_PATH=\$(find /tmp/outputs/${DATASET_NAME}/splatfacto/ -name 'config.yml' | head -n 1) && \
echo \"Config found at: \$CONFIG_PATH\" && \
ns-export gaussian-splat \
  --load-config \$CONFIG_PATH \
  --output-dir /tmp/export && \
gcloud storage cp -r /tmp/outputs/* gs://${BUCKET}/${DATASET_NAME}/trained/ && \
gcloud storage cp -r /tmp/export/* gs://${BUCKET}/${DATASET_NAME}/exported/"

# ── Submit job ────────────────────────────────────────────────────────────────
gcloud ai custom-jobs create \
    --region="$REGION" \
    --display-name="$JOB_NAME" \
    --config="$SPEC_FILE" \
    --service-account="$SERVICE_ACCOUNT" \
    --command="bash","-c","$COMMAND_STR"

echo ""
echo "  Job submitted to Vertex AI."
echo "  Monitor: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=${PROJECT_ID}"
echo "======================================================="
