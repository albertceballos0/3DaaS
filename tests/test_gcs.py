"""
tests/test_gcs.py
=================
Unit tests for flows/tasks/gcs.py

Each task is tested with mock GCS blobs, verifying:
  - Happy path: correct blobs present → task succeeds
  - Error path: required file/blobs missing → task raises
"""
import pytest
from unittest.mock import MagicMock

from tests.conftest import make_blob

# Import tasks (env vars already set in conftest)
from flows.tasks.gcs import (
    validate_raw_input,
    check_processed_exists,
    validate_processed_output,
    validate_exported_output,
)

DATASET = "test_scene"


# ── validate_raw_input ────────────────────────────────────────────────────────

class TestValidateRawInput:
    def test_ok_with_jpg_images(self, mock_gcs_client):
        mock_gcs_client.list_blobs.return_value = [
            make_blob(f"{DATASET}/raw/img_{i}.jpg") for i in range(5)
        ]
        result = validate_raw_input.fn(DATASET)
        assert result == 5

    def test_ok_with_png_images(self, mock_gcs_client):
        mock_gcs_client.list_blobs.return_value = [
            make_blob(f"{DATASET}/raw/frame_0.PNG"),
            make_blob(f"{DATASET}/raw/frame_1.PNG"),
        ]
        result = validate_raw_input.fn(DATASET)
        assert result == 2

    def test_ok_mixed_case_extensions(self, mock_gcs_client):
        mock_gcs_client.list_blobs.return_value = [
            make_blob(f"{DATASET}/raw/a.JPG"),
            make_blob(f"{DATASET}/raw/b.jpeg"),
            make_blob(f"{DATASET}/raw/c.JPEG"),
        ]
        result = validate_raw_input.fn(DATASET)
        assert result == 3

    def test_raises_when_no_images(self, mock_gcs_client):
        mock_gcs_client.list_blobs.return_value = []
        with pytest.raises(ValueError, match="No images found"):
            validate_raw_input.fn(DATASET)

    def test_raises_when_only_non_image_files(self, mock_gcs_client):
        mock_gcs_client.list_blobs.return_value = [
            make_blob(f"{DATASET}/raw/README.txt"),
            make_blob(f"{DATASET}/raw/data.csv"),
        ]
        with pytest.raises(ValueError, match="No images found"):
            validate_raw_input.fn(DATASET)

    def test_calls_correct_prefix(self, mock_gcs_client):
        mock_gcs_client.list_blobs.return_value = [make_blob(f"{DATASET}/raw/x.jpg")]
        validate_raw_input.fn(DATASET)
        mock_gcs_client.list_blobs.assert_called_once_with(
            "test-bucket", prefix=f"{DATASET}/raw/", max_results=500
        )


# ── check_processed_exists ────────────────────────────────────────────────────

class TestCheckProcessedExists:
    def test_ok_when_transforms_present(self, mock_gcs_client):
        mock_gcs_client.list_blobs.return_value = [
            make_blob(f"{DATASET}/processed/transforms.json"),
            make_blob(f"{DATASET}/processed/images/img_0.jpg"),
        ]
        check_processed_exists.fn(DATASET)  # should not raise

    def test_raises_when_transforms_missing(self, mock_gcs_client):
        mock_gcs_client.list_blobs.return_value = [
            make_blob(f"{DATASET}/processed/images/img_0.jpg"),
        ]
        with pytest.raises(FileNotFoundError, match="skip_preprocess=True"):
            check_processed_exists.fn(DATASET)

    def test_raises_when_folder_empty(self, mock_gcs_client):
        mock_gcs_client.list_blobs.return_value = []
        with pytest.raises(FileNotFoundError):
            check_processed_exists.fn(DATASET)

    def test_calls_correct_prefix(self, mock_gcs_client):
        mock_gcs_client.list_blobs.return_value = [
            make_blob(f"{DATASET}/processed/transforms.json")
        ]
        check_processed_exists.fn(DATASET)
        mock_gcs_client.list_blobs.assert_called_once_with(
            "test-bucket", prefix=f"{DATASET}/processed/"
        )


# ── validate_processed_output ─────────────────────────────────────────────────

class TestValidateProcessedOutput:
    def test_ok_when_transforms_present(self, mock_gcs_client):
        mock_gcs_client.list_blobs.return_value = [
            make_blob(f"{DATASET}/processed/transforms.json"),
            make_blob(f"{DATASET}/processed/images/img_0.jpg"),
        ]
        validate_processed_output.fn(DATASET)  # should not raise

    def test_raises_when_transforms_missing(self, mock_gcs_client):
        mock_gcs_client.list_blobs.return_value = [
            make_blob(f"{DATASET}/processed/images/img_0.jpg"),
        ]
        with pytest.raises(FileNotFoundError, match="transforms.json not found"):
            validate_processed_output.fn(DATASET)

    def test_raises_when_folder_empty(self, mock_gcs_client):
        mock_gcs_client.list_blobs.return_value = []
        with pytest.raises(FileNotFoundError):
            validate_processed_output.fn(DATASET)


# ── validate_exported_output ──────────────────────────────────────────────────

class TestValidateExportedOutput:
    def test_ok_returns_gcs_uri(self, mock_gcs_client):
        mock_gcs_client.list_blobs.return_value = [
            make_blob(f"{DATASET}/exported/splat.ply", size=50 * 1024 * 1024),
        ]
        uri = validate_exported_output.fn(DATASET)
        assert uri == f"gs://test-bucket/{DATASET}/exported/splat.ply"

    def test_returns_first_ply_when_multiple(self, mock_gcs_client):
        mock_gcs_client.list_blobs.return_value = [
            make_blob(f"{DATASET}/exported/point_cloud.ply"),
            make_blob(f"{DATASET}/exported/point_cloud_2.ply"),
        ]
        uri = validate_exported_output.fn(DATASET)
        assert uri.endswith(".ply")

    def test_raises_when_no_ply(self, mock_gcs_client):
        mock_gcs_client.list_blobs.return_value = [
            make_blob(f"{DATASET}/exported/config.yml"),
        ]
        with pytest.raises(FileNotFoundError, match="No .ply file found"):
            validate_exported_output.fn(DATASET)

    def test_raises_when_folder_empty(self, mock_gcs_client):
        mock_gcs_client.list_blobs.return_value = []
        with pytest.raises(FileNotFoundError):
            validate_exported_output.fn(DATASET)

    def test_calls_correct_prefix(self, mock_gcs_client):
        mock_gcs_client.list_blobs.return_value = [
            make_blob(f"{DATASET}/exported/splat.ply")
        ]
        validate_exported_output.fn(DATASET)
        mock_gcs_client.list_blobs.assert_called_once_with(
            "test-bucket", prefix=f"{DATASET}/exported/"
        )
