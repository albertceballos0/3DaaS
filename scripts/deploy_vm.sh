#!/bin/bash
# =============================================================================
# deploy_vm.sh — Sube el código a la GCE VM y levanta docker-compose
#
# Uso:
#   ./scripts/deploy_vm.sh            # sync + docker compose up --build -d
#   ./scripts/deploy_vm.sh --no-build # sync sin reconstruir imágenes
#   ./scripts/deploy_vm.sh --sync-only # solo copia archivos, no levanta stack
#
# Requisitos:
#   - .env con GCE_VM_NAME y GCE_VM_ZONE definidos
#   - gcloud autenticado: gcloud auth login
#   - SSH key configurada con la VM (lo hace gcloud automáticamente)
# =============================================================================

set -euo pipefail

# ── Colores ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${GREEN}→${NC} $*"; }
warn()    { echo -e "${YELLOW}⚠${NC}  $*"; }
error()   { echo -e "${RED}✗${NC}  $*" >&2; exit 1; }
success() { echo -e "${GREEN}✓${NC} $*"; }

# ── Cargar .env ───────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="${REPO_ROOT}/.env"
[ -f "$ENV_FILE" ] || error ".env no encontrado. Copia .env.example → .env y rellena los valores."
set -a; source "$ENV_FILE"; set +a

# ── Config desde .env ─────────────────────────────────────────────────────────
VM_NAME="${GCE_VM_NAME:?Falta GCE_VM_NAME en .env}"
VM_ZONE="${GCE_VM_ZONE:?Falta GCE_VM_ZONE en .env}"
PROJECT_ID="${GCP_PROJECT_ID:?Falta GCP_PROJECT_ID en .env}"
SA_KEY_PATH="${GCP_SA_KEY_PATH:-}"
REMOTE_DIR='$HOME/3daas'   # expandido en la VM, no en local

# ── Flags ─────────────────────────────────────────────────────────────────────
BUILD=true
DEPLOY=true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-build)   BUILD=false;  shift ;;
        --sync-only)  DEPLOY=false; shift ;;
        -h|--help)
            sed -n '2,10p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) error "Opción desconocida: $1" ;;
    esac
done

# ── Helper: ejecutar comando en la VM ─────────────────────────────────────────
vm_exec() {
    gcloud compute ssh "${VM_NAME}" \
        --zone="${VM_ZONE}" \
        --project="${PROJECT_ID}" \
        --command="$1"
}

echo "======================================================="
echo "  Deploy 3DaaS → GCE VM"
echo "======================================================="
echo "  VM:      ${VM_NAME} (${VM_ZONE})"
echo "  Destino: ${REMOTE_DIR}"
echo "  Build:   ${BUILD}"
echo "======================================================="

# ── 1. Verificar que la VM existe y está corriendo ───────────────────────────
info "Verificando estado de la VM..."
STATUS=$(gcloud compute instances describe "${VM_NAME}" \
    --zone="${VM_ZONE}" --project="${PROJECT_ID}" \
    --format="value(status)" 2>/dev/null || echo "NOT_FOUND")

if [ "$STATUS" = "NOT_FOUND" ]; then
    error "VM '${VM_NAME}' no encontrada en zona '${VM_ZONE}'. Créala primero en GCP Console."
elif [ "$STATUS" != "RUNNING" ]; then
    warn "VM está en estado '${STATUS}'. Intentando iniciar..."
    gcloud compute instances start "${VM_NAME}" --zone="${VM_ZONE}" --project="${PROJECT_ID}"
    info "Esperando a que la VM arranque..."
    sleep 15
fi
success "VM disponible"

# ── 2. Resolver HOME real en la VM y crear directorio ────────────────────────
info "Resolviendo directorio remoto..."
REMOTE_DIR_ABS=$(vm_exec 'echo $HOME/3daas' | tail -1)
info "Directorio remoto: ${REMOTE_DIR_ABS}"
vm_exec "mkdir -p '${REMOTE_DIR_ABS}'"

# ── 3. Sincronizar código ─────────────────────────────────────────────────────
info "Sincronizando código (rsync)..."
gcloud compute scp --recurse \
    --zone="${VM_ZONE}" \
    --project="${PROJECT_ID}" \
    --compress \
    "${REPO_ROOT}/api" \
    "${REPO_ROOT}/flows" \
    "${REPO_ROOT}/scripts" \
    "${REPO_ROOT}/docker-images" \
    "${REPO_ROOT}/docker-compose.yml" \
    "${REPO_ROOT}/prefect.yaml" \
    "${REPO_ROOT}/requirements.txt" \
    "${VM_NAME}:${REMOTE_DIR_ABS}/"
success "Código sincronizado"

# ── 4. Copiar .env ────────────────────────────────────────────────────────────
info "Copiando .env..."
gcloud compute scp \
    --zone="${VM_ZONE}" \
    --project="${PROJECT_ID}" \
    "${ENV_FILE}" \
    "${VM_NAME}:${REMOTE_DIR_ABS}/.env"
success ".env copiado"

# ── 5. Copiar service account key (si existe localmente) ─────────────────────
if [ -n "$SA_KEY_PATH" ] && [ -f "${REPO_ROOT}/${SA_KEY_PATH#./}" ]; then
    info "Copiando service account key..."
    gcloud compute scp \
        --zone="${VM_ZONE}" \
        --project="${PROJECT_ID}" \
        "${REPO_ROOT}/${SA_KEY_PATH#./}" \
        "${VM_NAME}:${REMOTE_DIR_ABS}/"
    success "SA key copiada"
else
    warn "SA key no encontrada localmente (${SA_KEY_PATH}). Si la VM usa Workload Identity, está bien."
fi

# ── 6. Instalar Docker en la VM si no está ───────────────────────────────────
info "Verificando Docker en la VM..."
if ! vm_exec "docker --version > /dev/null 2>&1"; then
    warn "Docker no encontrado. Instalando..."
    vm_exec "
        curl -fsSL https://get.docker.com | sudo sh
        sudo usermod -aG docker \$USER
        sudo systemctl enable docker
        sudo systemctl start docker
    "
    success "Docker instalado"
else
    success "Docker ya instalado"
fi

# ── 7. Verificar docker compose (plugin) ─────────────────────────────────────
if ! vm_exec "docker compose version > /dev/null 2>&1"; then
    warn "docker compose plugin no encontrado. Instalando..."
    vm_exec "
        sudo apt-get update -qq
        sudo apt-get install -y --no-install-recommends docker-compose-plugin
    "
fi

# ── 8. Levantar el stack ──────────────────────────────────────────────────────
if [ "$DEPLOY" = true ]; then
    COMPOSE_CMD="cd ${REMOTE_DIR_ABS} && docker compose up -d"
    [ "$BUILD" = true ] && COMPOSE_CMD="cd ${REMOTE_DIR_ABS} && docker compose up --build -d"

    info "Levantando docker-compose stack..."
    vm_exec "$COMPOSE_CMD"
    success "Stack levantado"

    # Mostrar estado
    echo ""
    info "Estado de los contenedores:"
    vm_exec "cd ${REMOTE_DIR_ABS} && docker compose ps"

    # Obtener IP externa
    EXTERNAL_IP=$(gcloud compute instances describe "${VM_NAME}" \
        --zone="${VM_ZONE}" --project="${PROJECT_ID}" \
        --format="value(networkInterfaces[0].accessConfigs[0].natIP)" 2>/dev/null || echo "N/A")

    echo ""
    echo "======================================================="
    success "Deploy completo"
    echo "  Prefect UI: http://${EXTERNAL_IP}:4200"
    echo "  API:        http://${EXTERNAL_IP}:8080"
    echo "  Health:     http://${EXTERNAL_IP}:8080/health"
    echo ""
    echo "  Logs del worker:"
    echo "    gcloud compute ssh ${VM_NAME} --zone=${VM_ZONE} --command='cd ${REMOTE_DIR_ABS} && docker compose logs -f prefect-worker'"
    echo "======================================================="
fi
