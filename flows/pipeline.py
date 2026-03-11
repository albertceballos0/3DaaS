"""
flows/pipeline.py
=================
Prefect flow principal para el pipeline 3DaaS Gaussian Splatting.

Stages:
  1. Validar imágenes raw en GCS
  2. Preprocessing (COLMAP) en Vertex AI
  3. Validar output de COLMAP
  4. Training + Export (splatfacto) en Vertex AI
  5. Validar .ply exportado

Deploy:
  prefect deploy --name prod

Trigger manual:
  prefect deployment run 'gaussian-pipeline/prod' -p run_id=xxx -p dataset=yyy
"""

from __future__ import annotations

from datetime import datetime, timezone

from prefect import flow, get_run_logger

import api.db as db
from flows.config import PreprocessParams, TrainParams
from flows.config import BUCKET
from flows.tasks.gcs import (
    detect_dataset_type,
    validate_exported_output,
    validate_processed_output,
    validate_raw_input,
)
from flows.tasks.vertex import poll_vertex_job, submit_preprocess_job, submit_train_job

# Tipos que no requieren COLMAP y leen los datos desde <dataset>/data/
_PREBUILT_TYPES = {"nerfstudio", "dnerf", "blender", "instant-ngp"}


def _resolve_training_path(dataset: str, dtype: str) -> str | None:
    """Devuelve la ruta de datos para el job de training.

    None → submit_train_job usa el default (<dataset>/processed/).
    """
    if dtype in _PREBUILT_TYPES:
        return f"/gcs/{BUCKET}/{dataset}/data"
    return None


def _resolve_dataparser(dtype: str, explicit: str) -> str:
    """Devuelve el argumento dataparser para ns-train.

    El dataparser explícito (pasado por el usuario) tiene prioridad.
    Para tipos pre-construidos distintos de nerfstudio, usa ``<dtype>-data``.
    """
    if explicit:
        return explicit
    if dtype in ("raw", "nerfstudio"):
        return ""
    return f"{dtype}-data"


@flow(name="gaussian-pipeline", log_prints=True)
def gaussian_pipeline(
    run_id: str,
    dataset: str,
    dataset_type: str = "auto",
    skip_preprocess: bool = False,
    preprocess_params: dict | None = None,
    train_params: dict | None = None,
) -> str:
    """Pipeline completo: COLMAP → splatfacto → export .ply.

    Args:
        run_id:           UUID del run (creado por la API).
        dataset:          Nombre del dataset en GCS.
        dataset_type:     Tipo de dataset. ``"auto"`` detecta automáticamente la
                          estructura GCS. Valores explícitos: ``"raw"``,
                          ``"nerfstudio"``, ``"dnerf"``, ``"blender"``, etc.
        skip_preprocess:  Fuerza saltarse COLMAP (ignorado si dataset_type != raw).
        preprocess_params: Overrides de PreprocessParams.
        train_params:     Overrides de TrainParams (incluye ``dataparser``).

    Returns:
        GCS URI del archivo .ply exportado.
    """
    logger = get_run_logger()

    try:
        db.update_run(run_id, {"status": "running"})

        # ── Detectar tipo de dataset ───────────────────────────────────────────
        if dataset_type == "auto":
            db.update_run(run_id, {"stage": "detecting_dataset"})
            dtype = detect_dataset_type(dataset)
        else:
            dtype = dataset_type
        logger.info(f"Dataset type: '{dtype}'")
        db.update_run(run_id, {"dataset_type": dtype})

        effective_skip = skip_preprocess or (dtype in _PREBUILT_TYPES)

        # ── Stage 1: Preprocessing ─────────────────────────────────────────────
        if not effective_skip:
            db.update_run(run_id, {"stage": "validating_raw"})
            image_count = validate_raw_input(dataset)
            logger.info(f"Dataset '{dataset}': {image_count} imágenes encontradas")

            db.update_run(run_id, {"stage": "preprocessing"})
            p_params = PreprocessParams(**(preprocess_params or {}))
            preprocess_resource = submit_preprocess_job(dataset, p_params)
            poll_vertex_job(preprocess_resource)

            db.update_run(run_id, {"stage": "validating_processed"})
            validate_processed_output(dataset)
        else:
            logger.info(f"Saltando preprocessing (dataset_type='{dtype}')")

        # ── Stage 2: Training + Export ─────────────────────────────────────────
        db.update_run(run_id, {"stage": "training"})

        # Aplicar dataparser por tipo si el usuario no lo ha especificado
        t_dict = dict(train_params or {})
        if "dataparser" not in t_dict:
            t_dict["dataparser"] = _resolve_dataparser(dtype, "")
        else:
            t_dict["dataparser"] = _resolve_dataparser(dtype, t_dict["dataparser"])

        t_params = TrainParams(**t_dict)
        data_path = _resolve_training_path(dataset, dtype)
        train_resource = submit_train_job(dataset, t_params, data_path=data_path)
        poll_vertex_job(train_resource)

        db.update_run(run_id, {"stage": "validating_exported"})
        ply_uri = validate_exported_output(dataset)

        # ── Done ───────────────────────────────────────────────────────────────
        db.update_run(run_id, {
            "status": "done",
            "stage": None,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "ply_uri": ply_uri,
        })
        logger.info(f"Pipeline completado. PLY: {ply_uri}")
        return ply_uri

    except Exception as exc:
        logger.error(f"Pipeline fallido: {exc}")
        db.update_run(run_id, {
            "status": "failed",
            "stage": None,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "error": str(exc),
        })
        raise
