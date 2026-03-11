Plan de puesta en marcha
PARTE 1 — Local (desarrollo y pruebas)
Paso 1 — Prereqs

# Docker Desktop corriendo

# Cuenta GCP con acceso al proyecto skillful-air-480018-f2

# gcloud CLI instalado y autenticado

gcloud auth login
gcloud config set project skillful-air-480018-f2
Paso 2 — Configurar .env

cp .env.example .env
Editar .env:

GCP_SA_KEY_PATH → ruta al JSON de la service account (ya está en el repo: ./skillful-air-480018-f2-bcb000b8c0e2.json)
PREFECT_API_URL=http://localhost:4200/api ✓ (ya configurado en .env.example)
El resto ya tiene los valores correctos
Paso 3 — Levantar el stack

docker compose up --build
Esto arranca:

Prefect server → http://localhost:4200 (UI para ver flow runs)
Prefect worker → se auto-registra, crea el work pool gcp-worker y el deployment prod
API → http://localhost:8080
Paso 4 — Verificar

# API up?

curl http://localhost:8080/health

# → {"status": "ok"}

# Worker registrado? (en la UI de Prefect o CLI)

export PREFECT_API_URL=http://localhost:4200/api
prefect work-pool ls
prefect deployment ls
Paso 5 — Disparar un pipeline de prueba

curl -X POST http://localhost:8080/pipeline/start \
 -H "Content-Type: application/json" \
 -d '{"dataset": "mi_escena", "skip_preprocess": false}'

# Responde con run_id → monitorear estado:

curl http://localhost:8080/pipeline/<run_id>

# Ver el flow run en la UI: http://localhost:4200

PARTE 2 — Remoto (producción en GCP)
Paso 1 — Crear la GCE VM (una sola vez)

gcloud compute instances create prefect-worker \
 --project=skillful-air-480018-f2 \
 --zone=us-central1-a \
 --machine-type=e2-small \
 --image-family=debian-12 \
 --image-project=debian-cloud \
 --scopes=cloud-platform \
 --service-account=custom-models@skillful-air-480018-f2.iam.gserviceaccount.com \
 --tags=prefect-server

# Abrir puerto 4200 para la API de Cloud Run

gcloud compute firewall-rules create allow-prefect \
 --direction=INGRESS --action=ALLOW \
 --rules=tcp:4200 \
 --source-ranges=0.0.0.0/0 \
 --target-tags=prefect-server
Paso 2 — Instalar Docker en la VM

gcloud compute ssh prefect-worker --zone=us-central1-a

# En la VM:

curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker
Paso 3 — Copiar archivos a la VM

# Desde tu máquina local:

gcloud compute scp docker-compose.yml prefect-worker:~/ --zone=us-central1-a
gcloud compute scp .env prefect-worker:~/ --zone=us-central1-a
gcloud compute scp docker-images/Dockerfile.worker prefect-worker:~/ --zone=us-central1-a
O mejor, clonar el repo directamente en la VM:

# En la VM:

git clone https://github.com/<tu-org>/3DaaS.git
cd 3DaaS
cp .env.example .env

# Editar .env con los valores de producción

# (En GCE con Workload Identity no necesitas el JSON key — dejar GCP_SA_KEY_PATH vacío)

Paso 4 — Levantar en la VM

# En la VM (solo server + worker, sin API local):

docker compose up -d prefect-server prefect-worker

# Ver logs:

docker compose logs -f prefect-worker
Paso 5 — Configurar secretos en GitHub Actions
En https://github.com/<org>/3DaaS/settings/secrets/actions añadir:

Secret Valor
GCP_SA_KEY Contenido del JSON de la service account
PREFECT_API_URL http://<IP_INTERNA_VM>:4200/api
PREFECT_API_KEY (vacío — self-hosted no necesita key)
WEBHOOK_URL URL de tu webhook Laravel (opcional)

# Obtener IP interna de la VM:

gcloud compute instances describe prefect-worker \
 --zone=us-central1-a \
 --format="value(networkInterfaces[0].networkIP)"
Paso 6 — Deploy automático (push a master)

git push origin master

# GitHub Actions:

# 1. pytest → pasa

# 2. Build + push imagen API a Artifact Registry

# 3. Deploy a Cloud Run con PREFECT_API_URL apuntando a la VM

Paso 7 — Verificar producción

# URL del servicio Cloud Run:

gcloud run services describe api-3daas \
 --region=us-central1 \
 --format="value(status.url)"

curl https://<cloud-run-url>/health
Diagrama final

[Local dev] [GCP Production]
docker compose up  
 prefect-server ─────────────► GCE VM: prefect-server:4200
prefect-worker GCE VM: prefect-worker
api:8080 Cloud Run: api-3daas (auto via CI/CD)
↕ PREFECT_API_URL
Firestore, GCS, Vertex AI
