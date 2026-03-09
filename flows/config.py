"""
flows/config.py
===============
Environment configuration and parameter dataclasses for the pipeline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")


def _require(var: str) -> str:
    value = os.environ.get(var)
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{var}' is not set. "
            f"Check your .env file."
        )
    return value


PROJECT_ID       = _require("GCP_PROJECT_ID")
REGION           = _require("GCP_REGION")
BUCKET           = _require("GCS_BUCKET")
SERVICE_ACCOUNT  = _require("GCP_SERVICE_ACCOUNT")
IMAGE_PREPROCESS = _require("IMAGE_PREPROCESS")
IMAGE_TRAIN      = _require("IMAGE_TRAIN")
WEBHOOK_URL      = os.environ.get("WEBHOOK_URL", "")

API_ENDPOINT   = f"{REGION}-aiplatform.googleapis.com"
VERTEX_CONSOLE = (
    f"https://console.cloud.google.com/vertex-ai/training/custom-jobs"
    f"?project={PROJECT_ID}"
)


@dataclass
class PreprocessParams:
    matching_method: str  = "vocab_tree"  # vocab_tree | exhaustive | sequential
    sfm_tool:        str  = "colmap"      # colmap | hloc
    feature_type:    str  = "sift"        # sift | superpoint | superpoint_aachen
    matcher_type:    str  = "NN"          # NN | superglue | supergluefast
    num_downscales:  int  = 3
    skip_colmap:     bool = False


@dataclass
class TrainParams:
    max_iters:      int   = 30000
    sh_degree:      int   = 3
    num_random:     int   = 100000
    num_downscales: int   = 0
    res_schedule:   int   = 250
    densify_grad:   float = 0.0002
    densify_size:   float = 0.01
    n_split:        int   = 2
    refine_every:   int   = 100
    cull_scale:     float = 0.5
    reset_alpha:    int   = 30
