#!/bin/bash

# 1. Configuración
PROJECT_ID=$(gcloud config get-value project)
REGION="us-central1"
REPO_NAME="nerfstudio-repo"
OFFICIAL_IMAGE="dromni/nerfstudio:1.1.5" # Imagen oficial en Docker Hub

# URI de destino en tu GCP
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/nerfstudio-official:latest"

echo "-------------------------------------------------------"
echo "🔄 Sincronizando Imagen Oficial de Nerfstudio a GCP"
echo "📍 Origen:  $OFFICIAL_IMAGE"
echo "📍 Destino: $IMAGE_URI"
echo "-------------------------------------------------------"

# 2. Crear el repositorio (por si no existe)
gcloud artifacts repositories create $REPO_NAME \
    --repository-format=docker \
    --location=$REGION \
    --description="Repositorio para imágenes oficiales" \
    --quiet || echo "⚠️ El repositorio ya existe."

# 3. Configurar Docker para GCP
gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet

# 4. Descargar la imagen oficial de Docker Hub
echo "📥 Descargando imagen oficial..."
docker pull $OFFICIAL_IMAGE

# 5. Cambiar el nombre (Tag) para que apunte a tu registro de Google
echo "🏷️ Etiquetando imagen..."
docker tag $OFFICIAL_IMAGE $IMAGE_URI

# 6. Subir a Google Artifact Registry
echo "☁️ Subiendo a GCP..."
docker push $IMAGE_URI

echo "-------------------------------------------------------"
echo "✅ ¡Listo! Imagen oficial disponible en tu proyecto."
echo "Usa esta URI en tus archivos .json de Vertex AI."
echo "-------------------------------------------------------"