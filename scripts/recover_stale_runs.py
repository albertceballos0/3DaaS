#!/usr/bin/env python3
"""
scripts/recover_stale_runs.py
==============================
Ejecutar al arrancar el worker para reanudar cualquier run de Firestore
que quedó en estado 'running' por un crash anterior del worker.

Estrategia por stage:
  - detecting_dataset / validating_raw / validating_processed /
    validating_exported  → re-ejecuta desde ese stage (operación rápida).
  - preprocessing / training  → comprueba el Vertex AI job almacenado:
      · JOB_STATE_SUCCEEDED  → reanuda desde el siguiente stage.
      · RUNNING/PENDING/QUEUED → reanuda el polling del job existente.
      · Cualquier otro estado  → re-submitirá el job.

Los runs en 'queued' no se tocan: Prefect los reejecutará al arrancar el worker.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent))

import api.db as db

PREFECT_API_URL = os.environ.get("PREFECT_API_URL", "http://localhost:4200/api")
PREFECT_API_KEY = os.environ.get("PREFECT_API_KEY", "")
PREFECT_DEPLOYMENT = os.environ.get("PREFECT_DEPLOYMENT", "gaussian-pipeline/prod")

# Si el worker crashed en un stage de Vertex AI y el job ya terminó con éxito,
# reanudamos desde el siguiente stage de validación.
_NEXT_STAGE_AFTER_VERTEX: dict[str, str] = {
    "preprocessing": "validating_processed",
    "training":      "validating_exported",
}


def _headers() -> dict:
    h = {"Content-Type": "application/json"}
    if PREFECT_API_KEY:
        h["Authorization"] = f"Bearer {PREFECT_API_KEY}"
    return h


def _get_deployment_id() -> str | None:
    try:
        resp = requests.get(
            f"{PREFECT_API_URL}/deployments/name/{PREFECT_DEPLOYMENT}",
            headers=_headers(),
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json()["id"]
        print(f"  ERROR: deployment '{PREFECT_DEPLOYMENT}' no encontrado ({resp.status_code})")
    except Exception as exc:
        print(f"  ERROR: no se pudo conectar a Prefect API: {exc}")
    return None


def _resolve_resume_stage(run: dict) -> str:
    """Determina el stage desde el que hay que reanudar.

    Para jobs de Vertex AI ya completados, avanza al siguiente stage de validación
    para no re-submitir trabajo ya hecho en GCP.
    """
    stage = run.get("stage") or "detecting_dataset"
    if stage not in _NEXT_STAGE_AFTER_VERTEX:
        return stage

    vertex_job_id = run.get("vertex_job_id")
    if not vertex_job_id:
        return stage

    # Importamos aquí para no fallar si GCP no está disponible en todos los entornos
    try:
        from flows.tasks.vertex import check_vertex_job_state
        state = check_vertex_job_state(vertex_job_id)
        if state == "JOB_STATE_SUCCEEDED":
            next_stage = _NEXT_STAGE_AFTER_VERTEX[stage]
            print(
                f"    Job {vertex_job_id} ya completado (SUCCEEDED). "
                f"Reanudando desde '{next_stage}'."
            )
            return next_stage
        print(f"    Job {vertex_job_id} en estado {state}. Reanudando desde '{stage}'.")
    except Exception as exc:
        print(f"    No se pudo verificar estado del job {vertex_job_id}: {exc}")

    return stage


def _cancel_prefect_run(flow_run_id: str) -> None:
    """Cancela el flow run anterior en Prefect para evitar runs duplicados."""
    if not flow_run_id:
        return
    try:
        resp = requests.post(
            f"{PREFECT_API_URL}/flow_runs/{flow_run_id}/set_state",
            headers=_headers(),
            json={"state": {"type": "CANCELLED", "name": "Cancelled", "message": "Cancelled by stale-run recovery on worker restart"}},
            timeout=10,
        )
        if resp.status_code in (200, 201):
            print(f"    Cancelled previous flow run {flow_run_id[:8]}")
        else:
            print(f"    Could not cancel flow run {flow_run_id[:8]}: {resp.status_code}")
    except Exception as exc:
        print(f"    WARNING: failed to cancel flow run {flow_run_id}: {exc}")


def _trigger_resume(deployment_id: str, run: dict, resume_stage: str) -> str | None:
    """Crea un nuevo Prefect flow run para reanudar el run interrumpido."""
    run_id = run["run_id"]
    dataset = run.get("dataset", "")
    params = run.get("params") or {}

    # Usar el dataset_type ya detectado para no re-detectar en el resume
    detected_type = run.get("dataset_type") or params.get("dataset_type", "auto")

    preprocess_params = {
        "matching_method": params.get("matching_method", "vocab_tree"),
        "sfm_tool":        params.get("sfm_tool", "colmap"),
        "feature_type":    params.get("feature_type", "sift"),
        "matcher_type":    params.get("matcher_type", "NN"),
        "num_downscales":  params.get("num_downscales_preprocess", 3),
        "skip_colmap":     params.get("skip_colmap", False),
    }
    train_params = {
        "dataparser":    params.get("dataparser", ""),
        "max_iters":     params.get("max_iters", 30000),
        "sh_degree":     params.get("sh_degree", 3),
        "num_random":    params.get("num_random", 100000),
        "num_downscales": params.get("num_downscales_train", 0),
        "res_schedule":  params.get("res_schedule", 250),
        "densify_grad":  params.get("densify_grad", 0.0002),
        "densify_size":  params.get("densify_size", 0.01),
        "n_split":       params.get("n_split", 2),
        "refine_every":  params.get("refine_every", 100),
        "cull_scale":    params.get("cull_scale", 0.5),
        "reset_alpha":   params.get("reset_alpha", 30),
        "use_wandb":     params.get("use_wandb", False),
    }

    try:
        resp = requests.post(
            f"{PREFECT_API_URL}/deployments/{deployment_id}/create_flow_run",
            headers=_headers(),
            json={
                "name": f"resume-{run_id[:8]}",
                "parameters": {
                    "run_id":            run_id,
                    "dataset":           dataset,
                    "dataset_type":      detected_type,
                    "skip_preprocess":   params.get("skip_preprocess", False),
                    "preprocess_params": preprocess_params,
                    "train_params":      train_params,
                    "resume_from_stage": resume_stage,
                },
            },
            timeout=10,
        )
        if resp.status_code in (200, 201):
            return resp.json()["id"]
        print(f"    ERROR al crear flow run: {resp.status_code} {resp.text}")
    except Exception as exc:
        print(f"    ERROR al llamar a Prefect API: {exc}")
    return None


def _mark_failed(run: dict, error: str) -> None:
    from flows.tasks.notify import send_webhook
    run_id = run["run_id"]
    dataset = run.get("dataset", "unknown")
    db.update_run(run_id, {
        "status": "failed",
        "stage": None,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "error": error,
    })
    send_webhook("pipeline_failed", {"run_id": run_id, "dataset": dataset, "error": error})
    print(f"  [failed]  {run_id} (dataset={dataset}) → {error}")


def recover_stale_runs() -> int:
    runs = db.list_runs(limit=500)
    stale = [r for r in runs if r.get("status") == "running"]

    if not stale:
        return 0

    deployment_id = _get_deployment_id()
    if not deployment_id:
        # Sin deployment no podemos re-triggear: marcamos como failed
        error = "Worker restarted — deployment not available for resume"
        for run in stale:
            _mark_failed(run, error)
        return len(stale)

    recovered = 0
    for run in stale:
        run_id = run["run_id"]
        dataset = run.get("dataset", "unknown")
        crashed_stage = run.get("stage") or "detecting_dataset"

        print(f"  [stale]   {run_id} (dataset={dataset}, stage={crashed_stage})")
        resume_stage = _resolve_resume_stage(run)

        # Cancela el flow run anterior en Prefect antes de crear uno nuevo
        _cancel_prefect_run(run.get("prefect_flow_run_id", ""))

        new_flow_run_id = _trigger_resume(deployment_id, run, resume_stage)
        if new_flow_run_id:
            db.update_run(run_id, {
                "status": "queued",
                "stage": None,
                "prefect_flow_run_id": new_flow_run_id,
                "error": None,
            })
            print(
                f"  [resume]  {run_id} → re-scheduled desde '{resume_stage}' "
                f"(flow_run={new_flow_run_id[:8]})"
            )
            recovered += 1
        else:
            _mark_failed(run, f"Worker restarted — failed to re-schedule from stage '{crashed_stage}'")

    return recovered


if __name__ == "__main__":
    print("==> Recovering stale runs...")
    count = recover_stale_runs()
    print(f"==> {count} stale run(s) recovered.")
