"""Dependency-light tests for pinned, state-blind feature extraction."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from deja_cue.preprocessing import (
    FEATURE_DIMENSION,
    MODEL_ID,
    MODEL_REVISION,
    load_encoder_lock,
    validate_feature_manifest,
    validate_model_directory,
)


def _feature_manifest() -> dict:
    return {
        "schema_version": 1,
        "kind": "deja_cue_state_blind_feature_manifest",
        "state_reference_labels_included": False,
        "histories": [
            {
                "history_id": "H001",
                "sequence_id": "0001_cut_test_object",
                "target_lineage_label_ids": [3, 7],
                "frames": [
                    {
                        "evaluation_index": 0,
                        "source_frame_number": 0,
                        "rgb_path": "JPEGImages/0001/frame0000.jpg",
                        "mask_path": "Annotations/0001/frame0000.png",
                        "rgb_sha256": "a" * 64,
                        "mask_sha256": "b" * 64,
                    }
                ],
                "states": [
                    {"state_id": "pre", "descriptions": ["whole object"]},
                    {"state_id": "post", "descriptions": ["cut object"]},
                ],
            }
        ],
    }


def test_encoder_lock_pins_revision_runtime_and_all_inference_files() -> None:
    lock = load_encoder_lock()
    assert lock["model_id"] == MODEL_ID
    assert lock["revision"] == MODEL_REVISION
    assert lock["feature_dimension"] == FEATURE_DIMENSION == 768
    assert lock["transformers_version"] == "4.57.6"
    assert {row["path"] for row in lock["required_files"]} == {
        "config.json",
        "model.safetensors",
        "preprocessor_config.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "tokenizer.model",
    }


def test_model_directory_is_checked_before_loading(tmp_path: Path) -> None:
    first = tmp_path / "config.json"
    second = tmp_path / "model.safetensors"
    first.write_bytes(b"config")
    second.write_bytes(b"weights")
    lock = {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "required_files": [
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in (first, second)
        ],
    }
    model_info = validate_model_directory(tmp_path, lock=lock)
    assert model_info["revision"] == MODEL_REVISION
    second.write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs from lock"):
        validate_model_directory(tmp_path, lock=lock)


def test_feature_manifest_rejects_reference_leakage_and_unsafe_paths() -> None:
    normalized = validate_feature_manifest(_feature_manifest())
    assert normalized[0]["target_lineage_label_ids"] == [3, 7]

    leaked = _feature_manifest()
    leaked["histories"][0]["selected_event"] = {"pre_interval": [0, 4]}
    with pytest.raises(ValueError, match="forbidden"):
        validate_feature_manifest(leaked)

    actual_schema_leak = _feature_manifest()
    actual_schema_leak["histories"][0]["references"] = {"pre": [[0, 4]]}
    with pytest.raises(ValueError, match="forbidden"):
        validate_feature_manifest(actual_schema_leak)

    unsafe = _feature_manifest()
    unsafe["histories"][0]["frames"][0]["rgb_path"] = "../private.jpg"
    with pytest.raises(ValueError, match="relative path"):
        validate_feature_manifest(unsafe)

    missing = _feature_manifest()
    missing["histories"][0]["frames"][0].pop("mask_path")
    with pytest.raises(ValueError, match="relative path"):
        validate_feature_manifest(missing)
