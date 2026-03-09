"""
flows/gaussian_pipeline.py
==========================
Prefect end-to-end pipeline for 3D Gaussian Splatting on GCP.

Stages:
  1. Validate raw input images in GCS
  2. Submit COLMAP preprocessing job → Vertex AI
  3. Poll until job terminal state
  4. Validate processed output (transforms.json)
  5. Submit splatfacto training + export job → Vertex AI
  6. Poll until job terminal state
  7. Validate exported .ply
  8. Notify Laravel on each stage via webhook

Run locally:
  python flows/gaussian_pipeline.py --dataset my_scene

Run via Prefect:
  prefect deployment run 'gaussian-splatting-pipeline/local' -p dataset=my_scene
"""

from __future__ import annotations

import argparse
from datetime import datetime
from typing import Optional

from prefect import flow, get_run_logger

from flows.config import (
    BUCKET, WEBHOOK_URL,
    PreprocessParams, TrainParams,
)
from flows.notify import notify, post_webhook
from flows.tasks import (
    validate_raw_input,
    check_processed_exists,
    validate_processed_output,
    validate_exported_output,
    submit_preprocess_job,
    submit_train_job,
    wait_for_vertex_job,
)


@flow(
    name="gaussian-splatting-pipeline",
    description="End-to-end Gaussian Splatting pipeline: COLMAP → splatfacto → .ply on GCP Vertex AI",
)
def gaussian_pipeline(
    dataset: str,
    # ── Prefect polling ────────────────────────────────────────────────────────
    poll_interval: int = 30,
    # ── Webhook / tracking ─────────────────────────────────────────────────────
    run_id: str = "",
    webhook_url: str = "",
    # ── Stage control ──────────────────────────────────────────────────────────
    skip_preprocess: bool = False,  # skip COLMAP, use existing processed data
    # ── Preprocessing (COLMAP) params ──────────────────────────────────────────
    matching_method: str  = "vocab_tree",
    sfm_tool:        str  = "colmap",
    feature_type:    str  = "sift",
    matcher_type:    str  = "NN",
    num_downscales_preprocess: int  = 3,
    skip_colmap:     bool = False,
    # ── Training (splatfacto) params ───────────────────────────────────────────
    max_iters:      int   = 30000,
    sh_degree:      int   = 3,
    num_random:     int   = 100000,
    num_downscales_train: int = 0,
    res_schedule:   int   = 250,
    densify_grad:   float = 0.0002,
    densify_size:   float = 0.01,
    n_split:        int   = 2,
    refine_every:   int   = 100,
    cull_scale:     float = 0.5,
    reset_alpha:    int   = 30,
) -> Optional[str]:
    """
    Full Gaussian Splatting pipeline.

    Returns the GCS URI of the exported .ply on success, or raises on failure.
    """
    logger = get_run_logger()

    _wh_url = webhook_url or WEBHOOK_URL
    _run_id = run_id or dataset

    def _wh(event: str, status: str, **extra) -> None:
        post_webhook(_wh_url, {
            "run_id": _run_id,
            "dataset": dataset,
            "event": event,
            "status": status,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            **extra,
        })

    pp = PreprocessParams(
        matching_method=matching_method,
        sfm_tool=sfm_tool,
        feature_type=feature_type,
        matcher_type=matcher_type,
        num_downscales=num_downscales_preprocess,
        skip_colmap=skip_colmap,
    )
    tp = TrainParams(
        max_iters=max_iters,
        sh_degree=sh_degree,
        num_random=num_random,
        num_downscales=num_downscales_train,
        res_schedule=res_schedule,
        densify_grad=densify_grad,
        densify_size=densify_size,
        n_split=n_split,
        refine_every=refine_every,
        cull_scale=cull_scale,
        reset_alpha=reset_alpha,
    )

    print()
    print("=" * 60)
    notify(f"Pipeline START — dataset: {dataset}", "START")
    print(f"  Bucket:        gs://{BUCKET}/{dataset}/")
    print(f"  skip_preprocess: {skip_preprocess}")
    if not skip_preprocess:
        print(f"  COLMAP:  matching={matching_method}, sfm={sfm_tool}, downscales={num_downscales_preprocess}")
    print(f"  Train:   iters={max_iters}, sh_degree={sh_degree}, densify_grad={densify_grad}")
    print("=" * 60)
    print()

    _wh("pipeline.started", "running")

    try:
        if skip_preprocess:
            # ── Skip Stage 1: verify processed data already exists ─────────────
            print()
            print("── Stage 1: SKIPPED (skip_preprocess=True) ────────────────")
            check_processed_exists(dataset)
            _wh("preprocess.skipped", "skipped")
        else:
            # ── Stage 0: Validate raw input ────────────────────────────────────
            validate_raw_input(dataset)

            # ── Stage 1: COLMAP preprocessing ─────────────────────────────────
            print()
            print("── Stage 1: COLMAP Preprocessing ─────────────────────────")
            preprocess_job = submit_preprocess_job(dataset, pp)
            _wh("preprocess.started", "running",
                vertex_job_id=preprocess_job.split("/")[-1])

            wait_for_vertex_job(preprocess_job, stage="Preprocess", poll_interval=poll_interval)
            validate_processed_output(dataset)
            _wh("preprocess.done", "succeeded")

        # ── Stage 2: splatfacto training + export ─────────────────────────────
        print()
        print("── Stage 2: Gaussian Splatting Training ───────────────────")
        train_job = submit_train_job(dataset, tp)
        _wh("training.started", "running",
            vertex_job_id=train_job.split("/")[-1])

        wait_for_vertex_job(train_job, stage="Train", poll_interval=poll_interval)
        ply_uri = validate_exported_output(dataset)

        # ── Done ──────────────────────────────────────────────────────────────
        print()
        print("=" * 60)
        notify(f"Pipeline COMPLETE — dataset: {dataset}", "OK")
        print(f"  Trained model: gs://{BUCKET}/{dataset}/trained/")
        print(f"  Exported .ply: {ply_uri}")
        print("=" * 60)
        print()

        _wh("pipeline.completed", "succeeded", ply_uri=ply_uri)
        logger.info("Pipeline complete. PLY: %s", ply_uri)
        return ply_uri

    except Exception as exc:
        _wh("pipeline.failed", "failed", error=str(exc))
        raise


# ── CLI entrypoint ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Gaussian Splatting Prefect pipeline locally"
    )
    parser.add_argument("--dataset",          required=True, help="Dataset name in GCS")
    parser.add_argument("--poll",             type=int, default=30, help="Vertex AI poll interval (s)")
    # Stage control
    parser.add_argument("--skip-preprocess",  action="store_true", help="Skip COLMAP, use existing processed data")
    # Preprocessing (ignored when --skip-preprocess)
    parser.add_argument("--matching-method",  default="vocab_tree")
    parser.add_argument("--sfm-tool",         default="colmap")
    parser.add_argument("--feature-type",     default="sift")
    parser.add_argument("--matcher-type",     default="NN")
    parser.add_argument("--num-downscales-pp",type=int, default=3)
    parser.add_argument("--skip-colmap",      action="store_true")
    # Training
    parser.add_argument("--max-iters",        type=int,   default=30000)
    parser.add_argument("--sh-degree",        type=int,   default=3)
    parser.add_argument("--num-random",       type=int,   default=100000)
    parser.add_argument("--num-downscales-train", type=int, default=0)
    parser.add_argument("--res-schedule",     type=int,   default=250)
    parser.add_argument("--densify-grad",     type=float, default=0.0002)
    parser.add_argument("--densify-size",     type=float, default=0.01)
    parser.add_argument("--n-split",          type=int,   default=2)
    parser.add_argument("--refine-every",     type=int,   default=100)
    parser.add_argument("--cull-scale",       type=float, default=0.5)
    parser.add_argument("--reset-alpha",      type=int,   default=30)

    args = parser.parse_args()

    gaussian_pipeline(
        dataset=args.dataset,
        poll_interval=args.poll,
        skip_preprocess=args.skip_preprocess,
        matching_method=args.matching_method,
        sfm_tool=args.sfm_tool,
        feature_type=args.feature_type,
        matcher_type=args.matcher_type,
        num_downscales_preprocess=args.num_downscales_pp,
        skip_colmap=args.skip_colmap,
        max_iters=args.max_iters,
        sh_degree=args.sh_degree,
        num_random=args.num_random,
        num_downscales_train=args.num_downscales_train,
        res_schedule=args.res_schedule,
        densify_grad=args.densify_grad,
        densify_size=args.densify_size,
        n_split=args.n_split,
        refine_every=args.refine_every,
        cull_scale=args.cull_scale,
        reset_alpha=args.reset_alpha,
    )
