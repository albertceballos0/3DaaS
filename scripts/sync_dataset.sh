#!/usr/bin/env bash
# sync_dataset.sh — Sube un dataset local al bucket GCS
#
# Uso:
#   ./scripts/sync_dataset.sh <local_folder> <dataset_name> [raw|data]
#
# Tipos:
#   raw   → gs://<bucket>/<dataset>/raw/   Imágenes para pipeline completo (COLMAP → train)
#   data  → gs://<bucket>/<dataset>/data/  Dataset pre-procesado (dnerf, blender, nerfstudio...)
#
# Ejemplo:
#   ./scripts/sync_dataset.sh ~/datasets/lego lego data
#   ./scripts/sync_dataset.sh ~/datasets/garden garden raw

set -euo pipefail

LOCAL_DIR="${1:-}"
DATASET_NAME="${2:-}"
FOLDER_TYPE="${3:-raw}"

# ── Load .env ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../.env"
if [ -f "$ENV_FILE" ]; then
    set -o allexport
    source "$ENV_FILE"
    set +o allexport
fi

GCS_BUCKET="${GCS_BUCKET:-}"

# ── Validation ─────────────────────────────────────────────────────────────────
if [ -z "$LOCAL_DIR" ] || [ -z "$DATASET_NAME" ]; then
    echo "Uso: $0 <local_folder> <dataset_name> [raw|data]"
    echo ""
    echo "  raw   → pipeline completo (imágenes sin procesar, se aplica COLMAP)"
    echo "  data  → solo training (dataset pre-procesado: dnerf, blender, nerfstudio...)"
    echo ""
    echo "Ejemplo:"
    echo "  $0 ~/datasets/lego   lego   data   # dataset dnerf/blender ya procesado"
    echo "  $0 ~/photos/garden   garden raw    # fotos para pipeline completo"
    exit 1
fi

if [ -z "$GCS_BUCKET" ]; then
    echo "ERROR: GCS_BUCKET no está configurado. Revisa tu .env"
    exit 1
fi

if [ ! -d "$LOCAL_DIR" ]; then
    echo "ERROR: La carpeta '$LOCAL_DIR' no existe"
    exit 1
fi

if [[ "$FOLDER_TYPE" != "raw" && "$FOLDER_TYPE" != "data" ]]; then
    echo "ERROR: El tipo debe ser 'raw' o 'data' (recibido: '$FOLDER_TYPE')"
    exit 1
fi

if [ "$APP_ENV" == "production" ]; then
    GCS_DEST="gs://${GCS_BUCKET}/pipeline_runs/${DATASET_NAME}/${FOLDER_TYPE}/"
else
    GCS_DEST="gs://${GCS_BUCKET}/pipeline_runs_dev/${DATASET_NAME}/${FOLDER_TYPE}/"
fi

# ── Info ───────────────────────────────────────────────────────────────────────
FILE_COUNT=$(find "$LOCAL_DIR" -type f | wc -l | tr -d ' ')
TOTAL_SIZE=$(du -sh "$LOCAL_DIR" 2>/dev/null | cut -f1)

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Dataset:  $DATASET_NAME"
echo "  Tipo:     $FOLDER_TYPE"
echo "  Origen:   $LOCAL_DIR"
echo "  Destino:  $GCS_DEST"
echo "  Archivos: $FILE_COUNT  ($TOTAL_SIZE)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── Confirm ────────────────────────────────────────────────────────────────────
read -r -p "¿Continuar? [y/N] " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Cancelado."
    exit 0
fi

echo ""
echo "Sincronizando con gcloud storage rsync..."
gcloud storage rsync --recursive "$LOCAL_DIR" "$GCS_DEST"

echo ""
echo "Sync completado: $GCS_DEST"
echo ""
echo "Para lanzar el pipeline (API):"
if [[ "$FOLDER_TYPE" == "data" ]]; then
    echo "  POST /pipeline/start"
    echo "  { \"dataset\": \"$DATASET_NAME\", \"dataset_type\": \"dnerf\" }"
    echo ""
    echo "  Tipos soportados: dnerf, blender, nerfstudio, instant-ngp"
else
    echo "  POST /pipeline/start"
    echo "  { \"dataset\": \"$DATASET_NAME\" }"
fi
