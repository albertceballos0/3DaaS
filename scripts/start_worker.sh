#!/usr/bin/env bash
# scripts/start_worker.sh
# ========================
# Entrypoint del Prefect worker container.
# 1. Espera a que el Prefect server esté listo
# 2. Crea el work pool (si no existe)
# 3. Registra el deployment prod
# 4. Arranca el worker

set -euo pipefail

PREFECT_API_URL="${PREFECT_API_URL:-http://localhost:4200/api}"
WORK_POOL="${PREFECT_WORK_POOL:-gcp-worker}"

echo "==> Waiting for Prefect server at ${PREFECT_API_URL} ..."
until curl -sf "${PREFECT_API_URL}/health" > /dev/null 2>&1; do
  sleep 3
done
echo "==> Prefect server is up."

echo "==> Creating work pool '${WORK_POOL}' (if not exists)..."
prefect work-pool create "${WORK_POOL}" --type process 2>/dev/null || true

echo "==> Registering deployment..."
prefect deploy --all --prefect-file /app/prefect.yaml

# Recovery DESPUÉS del deploy: necesita el deployment registrado para poder
# crear nuevos flow runs que reanuden los runs interrumpidos.
echo "==> Recovering stale runs from previous worker crash..."
python /app/scripts/recover_stale_runs.py || echo "WARNING: stale run recovery failed (non-fatal)"

echo "==> Starting worker for pool '${WORK_POOL}'..."
exec prefect worker start --pool "${WORK_POOL}"
