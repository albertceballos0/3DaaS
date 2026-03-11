"""
tests/test_gcs.py
=================
Unit tests for flows/tasks/gcs.py

Prefect tasks are invoked via .fn() to bypass Prefect machinery.
"""
import pytest

from tests.conftest import make_blob

from flows.tasks.gcs import (
    detect_dataset_type,
    validate_raw_input,
    validate_processed_output,
    validate_exported_output,
)

DATASET = "test_scene"


# ── detect_dataset_type ───────────────────────────────────────────────────────

class TestDetectDatasetType:
    def test_detects_raw_from_images(self, mock_gcs_client):
        mock_gcs_client.list_blobs.return_value = [
            make_blob(f"{DATASET}/raw/img_0.jpg"),
        ]
        assert detect_dataset_type.fn(DATASET) == "raw"

    def test_detects_nerfstudio_from_transforms_json(self, mock_gcs_client):
        mock_gcs_client.list_blobs.side_effect = [
            [],                                                         # raw/ — no images
            [make_blob(f"{DATASET}/data/transforms.json")],            # data/
        ]
        assert detect_dataset_type.fn(DATASET) == "nerfstudio"

    def test_detects_dnerf_from_transforms_train_json(self, mock_gcs_client):
        mock_gcs_client.list_blobs.side_effect = [
            [],                                                          # raw/ — no images
            [make_blob(f"{DATASET}/data/transforms_train.json")],       # data/
        ]
        assert detect_dataset_type.fn(DATASET) == "dnerf"

    def test_reads_dataset_type_txt_for_subtype(self, mock_gcs_client):
        mock_gcs_client.list_blobs.side_effect = [
            [],
            [
                make_blob(f"{DATASET}/data/transforms_train.json"),
                make_blob(f"{DATASET}/data/dataset_type.txt"),
            ],
        ]
        mock_gcs_client.bucket.return_value.blob.return_value.download_as_text.return_value = "blender"
        assert detect_dataset_type.fn(DATASET) == "blender"

    def test_raises_when_no_recognizable_structure(self, mock_gcs_client):
        mock_gcs_client.list_blobs.side_effect = [[], []]
        with pytest.raises(ValueError, match="No se pudo detectar"):
            detect_dataset_type.fn(DATASET)

    def test_raw_takes_priority_over_data(self, mock_gcs_client):
        # Si hay raw/ con imágenes, siempre gana aunque exista data/
        mock_gcs_client.list_blobs.return_value = [
            make_blob(f"{DATASET}/raw/img_0.jpg"),
        ]
        assert detect_dataset_type.fn(DATASET) == "raw"


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
        with pytest.raises(ValueError, match="No se encontraron"):
            validate_raw_input.fn(DATASET)

    def test_raises_when_only_non_image_files(self, mock_gcs_client):
        mock_gcs_client.list_blobs.return_value = [
            make_blob(f"{DATASET}/raw/README.txt"),
            make_blob(f"{DATASET}/raw/data.csv"),
        ]
        with pytest.raises(ValueError, match="No se encontraron"):
            validate_raw_input.fn(DATASET)

    def test_calls_correct_prefix(self, mock_gcs_client):
        mock_gcs_client.list_blobs.return_value = [make_blob(f"{DATASET}/raw/x.jpg")]
        validate_raw_input.fn(DATASET)
        mock_gcs_client.list_blobs.assert_called_once_with(
            "test-bucket", prefix=f"{DATASET}/raw/", max_results=500
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
        with pytest.raises(FileNotFoundError, match="transforms.json no encontrado"):
            validate_processed_output.fn(DATASET)

    def test_raises_when_folder_empty(self, mock_gcs_client):
        mock_gcs_client.list_blobs.return_value = []
        with pytest.raises(FileNotFoundError):
            validate_processed_output.fn(DATASET)

    def test_calls_correct_prefix(self, mock_gcs_client):
        mock_gcs_client.list_blobs.return_value = [
            make_blob(f"{DATASET}/processed/transforms.json")
        ]
        validate_processed_output.fn(DATASET)
        mock_gcs_client.list_blobs.assert_called_once_with(
            "test-bucket", prefix=f"{DATASET}/processed/"
        )


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
        with pytest.raises(FileNotFoundError, match="No se encontró archivo .ply"):
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
