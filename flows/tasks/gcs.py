"""
flows/tasks/gcs.py
==================
Prefect tasks para validar inputs/outputs en GCS.
"""

from __future__ import annotations

from prefect import task
from google.cloud import storage

from flows.config import BUCKET

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


@task(name="detect-dataset-type", log_prints=True)
def detect_dataset_type(dataset: str) -> str:
    """Inspecciona la estructura GCS y devuelve el tipo de dataset.

    Returns:
        "raw"        → <dataset>/raw/ tiene imágenes  → pipeline completo (COLMAP + train)
        "nerfstudio" → <dataset>/data/transforms.json → solo training (formato nerfstudio)
        "<type>"     → <dataset>/data/transforms_train.json → solo training (dnerf, blender...)

    El tipo exacto para datasets con transforms_train.json se lee del archivo
    ``<dataset>/data/dataset_type.txt`` si existe, o se devuelve ``"dnerf"`` por defecto.

    Raises:
        ValueError si no se puede determinar el tipo.
    """
    print(f"Detectando tipo de dataset: gs://{BUCKET}/{dataset}/")

    client = storage.Client()

    # Comprobar raw/ con imágenes → pipeline completo
    raw_blobs = list(client.list_blobs(BUCKET, prefix=f"{dataset}/raw/", max_results=20))
    images = [b for b in raw_blobs if any(b.name.endswith(ext) for ext in _IMAGE_EXTS)]
    if images:
        print(f"Tipo detectado: 'raw' ({len(images)} imágenes en raw/)")
        return "raw"

    # Comprobar data/ → dataset pre-procesado
    data_blobs = {b.name for b in client.list_blobs(BUCKET, prefix=f"{dataset}/data/")}

    if f"{dataset}/data/transforms.json" in data_blobs:
        print("Tipo detectado: 'nerfstudio' (transforms.json en data/)")
        return "nerfstudio"

    if f"{dataset}/data/transforms_train.json" in data_blobs:
        # Intentar leer dataset_type.txt para subtipo exacto
        dtype = "dnerf"
        type_blob_name = f"{dataset}/data/dataset_type.txt"
        if type_blob_name in data_blobs:
            blob = client.bucket(BUCKET).blob(type_blob_name)
            dtype = blob.download_as_text().strip()
        print(f"Tipo detectado: '{dtype}' (transforms_train.json en data/)")
        return dtype

    raise ValueError(
        f"No se pudo detectar el tipo del dataset '{dataset}'. "
        f"Asegúrate de que existe:\n"
        f"  gs://{BUCKET}/{dataset}/raw/  (imágenes)\n"
        f"  gs://{BUCKET}/{dataset}/data/ (dataset pre-procesado con transforms.json)"
    )


@task(name="validate-raw-input", retries=2, retry_delay_seconds=10, log_prints=True)
def validate_raw_input(dataset: str) -> int:
    """Assert que gs://<BUCKET>/<dataset>/raw/ contiene al menos una imagen."""
    print(f"Validando raw input: gs://{BUCKET}/{dataset}/raw/")

    client = storage.Client()
    blobs = list(client.list_blobs(BUCKET, prefix=f"{dataset}/raw/", max_results=500))

    images = [b for b in blobs if any(b.name.endswith(ext) for ext in _IMAGE_EXTS)]

    if not images:
        raise ValueError(
            f"No se encontraron imágenes en gs://{BUCKET}/{dataset}/raw/ "
            f"(total objetos: {len(blobs)}). Sube imágenes antes de correr el pipeline."
        )

    print(f"Raw input OK — {len(images)} imagen(es) encontrada(s)")
    return len(images)


@task(name="validate-processed-output", retries=2, retry_delay_seconds=10, log_prints=True)
def validate_processed_output(dataset: str) -> None:
    """Assert que transforms.json existe en <dataset>/processed/ tras COLMAP."""
    print(f"Validando output COLMAP: gs://{BUCKET}/{dataset}/processed/")

    client = storage.Client()
    blobs = {b.name for b in client.list_blobs(BUCKET, prefix=f"{dataset}/processed/")}

    transforms = f"{dataset}/processed/transforms.json"
    if transforms not in blobs:
        raise FileNotFoundError(
            f"transforms.json no encontrado en gs://{BUCKET}/{transforms}. "
            f"COLMAP puede haber fallado silenciosamente."
        )

    print("Processed output OK — transforms.json encontrado")


@task(name="validate-exported-output", retries=2, retry_delay_seconds=10, log_prints=True)
def validate_exported_output(dataset: str) -> str:
    """Assert que existe un .ply en <dataset>/exported/ y retorna su GCS URI."""
    print(f"Validando export output: gs://{BUCKET}/{dataset}/exported/")

    client = storage.Client()
    blobs = list(client.list_blobs(BUCKET, prefix=f"{dataset}/exported/"))

    ply_blobs = [b for b in blobs if b.name.endswith(".ply")]
    if not ply_blobs:
        raise FileNotFoundError(
            f"No se encontró archivo .ply en gs://{BUCKET}/{dataset}/exported/. "
            f"El export puede haber fallado."
        )

    ply_uri = f"gs://{BUCKET}/{ply_blobs[0].name}"
    size_mb = ply_blobs[0].size / (1024 ** 2)
    print(f"Export OK — {ply_uri} ({size_mb:.1f} MB)")
    return ply_uri
