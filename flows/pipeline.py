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
from utils.logger import get_logger, setup_logging

setup_logging("worker")
_log = get_logger("pipeline")
from flows.config import PreprocessParams, TrainParams
from flows.config import BUCKET, GCS_DATASET_PREFIX, PREBUILT_DATA_SUBFOLDER
from flows.tasks.gcs import (
    detect_dataset_type,
    validate_dataset_exists,
    validate_exported_output,
    validate_processed_output,
    validate_raw_input,
)
from flows.tasks.notify import send_webhook
from flows.tasks.vertex import (
    build_job_resource_name,
    check_vertex_job_state,
    poll_vertex_job,
    submit_preprocess_job,
    submit_train_job,
)

# Tipos que no requieren COLMAP y leen los datos desde <dataset>/data/
_PREBUILT_TYPES = {"nerfstudio", "dnerf", "blender", "instant-ngp"}

# Orden canónico de stages para la lógica de resume
_STAGE_ORDER = [
    "validating_dataset",
    "detecting_dataset",
    "validating_raw",
    "preprocessing",
    "validating_processed",
    "training",
    "validating_exported",
]

# Estados de Vertex AI que indican que el job sigue activo
_VERTEX_ACTIVE = {"JOB_STATE_RUNNING", "JOB_STATE_PENDING", "JOB_STATE_QUEUED"}


def _reaches(stage: str, resume_from: str | None) -> bool:
    """True si `stage` debe ejecutarse dado el punto de reanudación."""
    if not resume_from:
        return True
    try:
        return _STAGE_ORDER.index(stage) >= _STAGE_ORDER.index(resume_from)
    except ValueError:
        return True  # stage desconocido → ejecutar siempre


def _resolve_training_path(dataset: str, dtype: str) -> str | None:
    """Devuelve la ruta de datos para el job de training.

    None → submit_train_job usa el default (<dataset>/processed/).
    La subcarpeta para tipos pre-construidos se configura con PREBUILT_DATA_SUBFOLDER
    (default: "data").
    """
    if dtype in _PREBUILT_TYPES:
        return f"/gcs/{BUCKET}/{GCS_DATASET_PREFIX}/{dataset}/{PREBUILT_DATA_SUBFOLDER}"
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
    resume_from_stage: str | None = None,
) -> str:
    """Pipeline completo: COLMAP → splatfacto → export .ply.

    Args:
        run_id:             UUID del run (creado por la API).
        dataset:            Nombre del dataset en GCS.
        dataset_type:       Tipo de dataset. ``"auto"`` detecta automáticamente.
                            En resumes se pasa el tipo ya detectado.
        skip_preprocess:    Fuerza saltarse COLMAP (ignorado si dataset_type != raw).
        preprocess_params:  Overrides de PreprocessParams.
        train_params:       Overrides de TrainParams (incluye ``dataparser``).
        resume_from_stage:  Si se especifica, salta todos los stages anteriores.
                            Los jobs de Vertex AI activos se reanudan en vez de
                            re-submitirse.

    Returns:
        GCS URI del archivo .ply exportado.
    """
    logger = get_run_logger()

    def _stage(stage: str, **extra) -> None:
        """Actualiza Firestore y envía webhook de cambio de stage."""
        db.update_run(run_id, {"stage": stage})
        send_webhook("stage_changed", {
            "run_id": run_id,
            "dataset": dataset,
            "stage": stage,
            **extra,
        })

    # ── Cargar estado previo si es un resume ──────────────────────────────────
    image_count: int = 0
    existing_vertex_job_id: str | None = None
    if resume_from_stage:
        stored = db.get_run(run_id) or {}
        image_count = stored.get("image_count", 0)
        existing_vertex_job_id = stored.get("vertex_job_id")
        _log.info(
            f"[{run_id[:8]}] Reanudando desde stage '{resume_from_stage}' "
            f"vertex_job_id={existing_vertex_job_id}"
        )
    else:
        _log.info(f"[{run_id[:8]}] Iniciando pipeline dataset={dataset} type={dataset_type}")

    try:
        db.update_run(run_id, {"status": "running"})
        if not resume_from_stage:
            send_webhook("pipeline_started", {
                "run_id": run_id,
                "dataset": dataset,
                "dataset_type": dataset_type,
            })

        # ── Stage 0: Validar existencia del dataset ───────────────────────────
        if _reaches("validating_dataset", resume_from_stage):
            _stage("validating_dataset")
            validate_dataset_exists(dataset)

        # ── Detectar tipo de dataset ───────────────────────────────────────────
        if _reaches("detecting_dataset", resume_from_stage):
            if dataset_type == "auto":
                _stage("detecting_dataset")
                dtype = detect_dataset_type(dataset)
            else:
                dtype = dataset_type
            logger.info(f"Dataset type: '{dtype}'")
            db.update_run(run_id, {"dataset_type": dtype})
        else:
            # En resume, el recovery script pasa el tipo ya detectado
            dtype = dataset_type

        effective_skip = skip_preprocess or (dtype in _PREBUILT_TYPES)
        if effective_skip:
            logger.info(f"Saltando preprocessing (dataset_type='{dtype}')")

        # ── Stage 1: Validar raw ───────────────────────────────────────────────
        if not effective_skip and _reaches("validating_raw", resume_from_stage):
            _stage("validating_raw")
            image_count = validate_raw_input(dataset)
            logger.info(f"Dataset '{dataset}': {image_count} imágenes encontradas")
            db.update_run(run_id, {"image_count": image_count})

        # ── Stage 2: Preprocessing ─────────────────────────────────────────────
        if not effective_skip and _reaches("preprocessing", resume_from_stage):
            _stage("preprocessing", image_count=image_count)
            p_params = PreprocessParams(**(preprocess_params or {}))

            job_handled = False
            if resume_from_stage == "preprocessing" and existing_vertex_job_id:
                state = check_vertex_job_state(existing_vertex_job_id)
                if state == "JOB_STATE_SUCCEEDED":
                    logger.info(
                        f"Preprocess job {existing_vertex_job_id} ya completado. "
                        f"Saltando submit."
                    )
                    job_handled = True
                elif state in _VERTEX_ACTIVE:
                    logger.info(
                        f"Reanudando polling de preprocess job activo: "
                        f"{existing_vertex_job_id} ({state})"
                    )
                    poll_vertex_job(
                        build_job_resource_name(existing_vertex_job_id),
                        run_id=run_id,
                        job_type="preprocess",
                        is_resume=True,
                    )
                    job_handled = True
                else:
                    logger.info(
                        f"Preprocess job {existing_vertex_job_id} en estado {state}. "
                        f"Re-submitiendo."
                    )

            if not job_handled:
                preprocess_resource = submit_preprocess_job(dataset, p_params)
                db.update_run(run_id, {"vertex_job_id": preprocess_resource.split("/")[-1]})
                poll_vertex_job(preprocess_resource, run_id=run_id, job_type="preprocess")

        # ── Stage 3: Validar processed ─────────────────────────────────────────
        if not effective_skip and _reaches("validating_processed", resume_from_stage):
            _stage("validating_processed")
            validate_processed_output(dataset)

        # ── Stage 4: Training + Export ─────────────────────────────────────────
        t_dict = dict(train_params or {})
        if "dataparser" not in t_dict:
            t_dict["dataparser"] = _resolve_dataparser(dtype, "")
        else:
            t_dict["dataparser"] = _resolve_dataparser(dtype, t_dict["dataparser"])

        t_params = TrainParams(**t_dict)
        data_path = _resolve_training_path(dataset, dtype)

        if _reaches("training", resume_from_stage):
            _stage("training", max_iters=t_params.max_iters)

            job_handled = False
            if resume_from_stage == "training" and existing_vertex_job_id:
                state = check_vertex_job_state(existing_vertex_job_id)
                if state == "JOB_STATE_SUCCEEDED":
                    logger.info(
                        f"Train job {existing_vertex_job_id} ya completado. "
                        f"Saltando submit."
                    )
                    job_handled = True
                elif state in _VERTEX_ACTIVE:
                    logger.info(
                        f"Reanudando polling de train job activo: "
                        f"{existing_vertex_job_id} ({state})"
                    )
                    poll_vertex_job(
                        build_job_resource_name(existing_vertex_job_id),
                        run_id=run_id,
                        job_type="train",
                        is_resume=True,
                    )
                    job_handled = True
                else:
                    logger.info(
                        f"Train job {existing_vertex_job_id} en estado {state}. "
                        f"Re-submitiendo."
                    )

            if not job_handled:
                train_resource = submit_train_job(dataset, t_params, data_path=data_path)
                db.update_run(run_id, {"vertex_job_id": train_resource.split("/")[-1]})
                poll_vertex_job(train_resource, run_id=run_id, job_type="train")

        # ── Stage 5: Validar exported ──────────────────────────────────────────
        _stage("validating_exported")
        ply_uri = validate_exported_output(dataset)

        # ── Done ───────────────────────────────────────────────────────────────
        db.update_run(run_id, {
            "status": "done",
            "stage": None,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "ply_uri": ply_uri,
        })
        send_webhook("pipeline_done", {
            "run_id": run_id,
            "dataset": dataset,
            "ply_uri": ply_uri,
        })
        logger.info(f"Pipeline completado. PLY: {ply_uri}")
        return ply_uri

    except Exception as exc:
        _log.error(f"[{run_id[:8]}] Pipeline fallido: {exc}", exc_info=True)
        logger.error(f"Pipeline fallido: {exc}")
        db.update_run(run_id, {
            "status": "failed",
            "stage": None,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "error": str(exc),
        })
        send_webhook("pipeline_failed", {
            "run_id": run_id,
            "dataset": dataset,
            "error": str(exc),
        })
        raise
