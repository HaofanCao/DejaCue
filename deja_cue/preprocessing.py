"""Pinned SigLIP2 object-local feature extraction.

The visual manifest is state-reference blind: it may contain
source identities, RGB/mask paths, temporal indices, target-lineage labels,
and the closed text vocabulary, but never framewise state annotations or
reference intervals. The model directory is accepted only when every required
inference file matches the immutable lock in ``configs/siglip2_encoder.json``.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .vost_protocol import mask_union_padded_crop, prompt_forms


MODEL_ID = "google/siglip2-base-patch16-224"
MODEL_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
FEATURE_DIMENSION = 768
FORBIDDEN_REFERENCE_FIELDS = frozenset(
    {
        "annotation_labels",
        "candidate_events",
        "consensus_labels",
        "frame_labels",
        "ground_truth",
        "post_interval",
        "pre_interval",
        "reference_windows",
        "reference_intervals",
        "references",
        "selected_event",
        "state_moments",
        "transition_interval",
    }
)


def package_root() -> Path:
    """Resolve the repository/package root from this module."""

    return Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def load_encoder_lock(path: Path | None = None) -> dict[str, Any]:
    """Load and structurally validate the immutable SigLIP2 lock."""

    lock_path = (
        package_root() / "configs" / "siglip2_encoder.json"
        if path is None
        else Path(path)
    )
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("SigLIP2 encoder lock must be a JSON object")
    if (
        payload.get("kind") != "deja_cue_siglip2_encoder_lock"
        or payload.get("model_id") != MODEL_ID
        or payload.get("revision") != MODEL_REVISION
        or int(payload.get("feature_dimension", -1)) != FEATURE_DIMENSION
    ):
        raise ValueError("SigLIP2 encoder identity differs from the finalized lock")
    files = payload.get("required_files")
    if not isinstance(files, list) or len(files) != 7:
        raise ValueError("SigLIP2 encoder lock must list seven inference files")
    names: set[str] = set()
    for row in files:
        if not isinstance(row, Mapping):
            raise ValueError("SigLIP2 file lock rows must be objects")
        name = str(row.get("path", ""))
        digest = str(row.get("sha256", ""))
        size = row.get("size_bytes")
        if (
            not name
            or Path(name).name != name
            or name in names
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
        ):
            raise ValueError(f"Invalid SigLIP2 file lock row: {row}")
        names.add(name)
    return payload


def validate_model_directory(
    model_directory: Path, *, lock: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Verify every required local model file before importing Transformers."""

    model_root = Path(model_directory).resolve()
    if not model_root.is_dir():
        raise FileNotFoundError(f"SigLIP2 model directory does not exist: {model_directory}")
    encoder_lock = load_encoder_lock() if lock is None else dict(lock)
    if (
        encoder_lock.get("model_id") != MODEL_ID
        or encoder_lock.get("revision") != MODEL_REVISION
    ):
        raise ValueError("Supplied SigLIP2 lock has the wrong model identity")
    verified: list[dict[str, Any]] = []
    for row in encoder_lock["required_files"]:
        path = model_root / str(row["path"])
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"Missing regular SigLIP2 file: {row['path']}")
        observed_size = int(path.stat().st_size)
        observed_sha256 = sha256_file(path)
        if observed_size != int(row["size_bytes"]) or observed_sha256 != row["sha256"]:
            raise ValueError(f"SigLIP2 file differs from lock: {row['path']}")
        verified.append(
            {
                "path": str(row["path"]),
                "size_bytes": observed_size,
                "sha256": observed_sha256,
            }
        )
    return {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "feature_dimension": FEATURE_DIMENSION,
        "required_files": verified,
    }


def l2_normalize(values: np.ndarray, *, axis: int = -1) -> np.ndarray:
    """Normalize finite nonzero vectors and return float32 output."""

    array = np.asarray(values, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("Feature array contains non-finite values")
    norms = np.linalg.norm(array, axis=axis, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Cannot normalize a zero feature vector")
    return np.asarray(array / norms, dtype=np.float32)


def _tensor_to_numpy(value: Any) -> np.ndarray:
    tensor = value.float().detach().cpu()
    return np.asarray(tensor.numpy(), dtype=np.float32)


def encode_image_batches(
    model: Any,
    processor: Any,
    images: Sequence[Any],
    *,
    batch_size: int,
    device: str,
) -> np.ndarray:
    """Encode object-local RGB crops as unit-normalized projected features."""

    import torch

    if isinstance(batch_size, bool) or int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    rows: list[np.ndarray] = []
    for start in range(0, len(images), int(batch_size)):
        inputs = processor(
            images=list(images[start : start + int(batch_size)]),
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            features = model.get_image_features(**inputs)
        rows.append(_tensor_to_numpy(features))
    if not rows:
        raise ValueError("At least one visible crop is required")
    encoded = l2_normalize(np.concatenate(rows, axis=0), axis=1)
    if encoded.shape[1] != FEATURE_DIMENSION:
        raise ValueError(
            f"Expected {FEATURE_DIMENSION} visual dimensions, got {encoded.shape[1]}"
        )
    return encoded


def encode_description_batches(
    model: Any,
    processor: Any,
    descriptions: Sequence[str],
    *,
    batch_size: int,
    device: str,
) -> np.ndarray:
    """Encode raw/photo/definite forms, then average and renormalize each triplet."""

    import torch

    texts = [str(value).strip() for value in descriptions]
    if not texts or any(not value for value in texts):
        raise ValueError("Descriptions must be non-empty strings")
    prompts = [prompt for text in texts for prompt in prompt_forms(text)]
    prompt_rows: list[np.ndarray] = []
    for start in range(0, len(prompts), int(batch_size)):
        inputs = processor(
            text=prompts[start : start + int(batch_size)],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            features = model.get_text_features(**inputs)
        prompt_rows.append(_tensor_to_numpy(features))
    prompt_features = l2_normalize(np.concatenate(prompt_rows, axis=0), axis=1)
    grouped = prompt_features.reshape(len(texts), 3, -1)
    encoded = l2_normalize(grouped.mean(axis=1), axis=1)
    if encoded.shape[1] != FEATURE_DIMENSION:
        raise ValueError(
            f"Expected {FEATURE_DIMENSION} text dimensions, got {encoded.shape[1]}"
        )
    return encoded


def load_pinned_encoder(model_directory: Path, *, device: str) -> tuple[Any, Any, dict[str, Any]]:
    """Validate and load the exact local SigLIP2 checkpoint."""

    from transformers import AutoModel, AutoProcessor

    lock = load_encoder_lock()
    installed_transformers = importlib.metadata.version("transformers")
    if installed_transformers != str(lock["transformers_version"]):
        raise RuntimeError(
            "Transformers differs from the feature-extraction lock: "
            f"expected {lock['transformers_version']}, got {installed_transformers}"
        )
    model_info = {
        **validate_model_directory(model_directory, lock=lock),
        "transformers_version": installed_transformers,
    }
    model_root = Path(model_directory).resolve()
    model = AutoModel.from_pretrained(model_root, local_files_only=True).to(device).eval()
    processor = AutoProcessor.from_pretrained(
        model_root, local_files_only=True, use_fast=True
    )
    return model, processor, model_info


def _reject_reference_fields(value: Any, *, location: str = "manifest") -> None:
    if isinstance(value, Mapping):
        forbidden = sorted(set(value).intersection(FORBIDDEN_REFERENCE_FIELDS))
        if forbidden:
            raise ValueError(
                f"State-reference fields are forbidden in {location}: {forbidden}"
            )
        for key, nested in value.items():
            _reject_reference_fields(nested, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_reference_fields(nested, location=f"{location}[{index}]")


def _relative_source_path(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a normalized relative path")
    text = str(value).strip().replace("\\", "/")
    path = Path(text)
    if not text or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must be a normalized relative path")
    return path.as_posix()


def validate_feature_manifest(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate the state-blind source manifest consumed by feature extraction."""

    if payload.get("kind") != "deja_cue_state_blind_feature_manifest":
        raise ValueError("Unexpected feature-manifest kind")
    if payload.get("state_reference_labels_included") is not False:
        raise ValueError("Feature manifest must explicitly exclude reference labels")
    _reject_reference_fields(payload)
    histories = payload.get("histories")
    if not isinstance(histories, list) or not histories:
        raise ValueError("Feature manifest must contain histories")
    normalized: list[dict[str, Any]] = []
    seen_histories: set[str] = set()
    for history in histories:
        if not isinstance(history, Mapping):
            raise ValueError("Feature histories must be objects")
        history_id = str(history.get("history_id", "")).strip()
        sequence_id = str(history.get("sequence_id", "")).strip()
        if not history_id or not sequence_id or history_id in seen_histories:
            raise ValueError("Feature history identities must be unique and non-empty")
        seen_histories.add(history_id)
        lineage = history.get("target_lineage_label_ids")
        if (
            not isinstance(lineage, list)
            or not lineage
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 1 <= value <= 254
                for value in lineage
            )
        ):
            raise ValueError(f"Invalid lineage labels for {history_id}")
        lineage_ids = sorted(set(int(value) for value in lineage))

        frames = history.get("frames")
        if not isinstance(frames, list) or not frames:
            raise ValueError(f"Feature history {history_id} has no frames")
        normalized_frames: list[dict[str, Any]] = []
        for frame in frames:
            if not isinstance(frame, Mapping):
                raise ValueError(f"Feature frame in {history_id} is not an object")
            evaluation_index = frame.get("evaluation_index")
            source_frame_number = frame.get("source_frame_number")
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in (evaluation_index, source_frame_number)
            ):
                raise ValueError(f"Invalid temporal index in {history_id}")
            rgb_sha256 = str(frame.get("rgb_sha256", ""))
            mask_sha256 = str(frame.get("mask_sha256", ""))
            if any(
                len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in (rgb_sha256, mask_sha256)
            ):
                raise ValueError(f"Missing source hash in {history_id}")
            normalized_frames.append(
                {
                    "evaluation_index": int(evaluation_index),
                    "source_frame_number": int(source_frame_number),
                    "rgb_path": _relative_source_path(
                        frame.get("rgb_path"), field="rgb_path"
                    ),
                    "mask_path": _relative_source_path(
                        frame.get("mask_path"), field="mask_path"
                    ),
                    "rgb_sha256": rgb_sha256,
                    "mask_sha256": mask_sha256,
                }
            )
        evaluation_indices = [row["evaluation_index"] for row in normalized_frames]
        source_numbers = [row["source_frame_number"] for row in normalized_frames]
        if any(b <= a for a, b in zip(evaluation_indices, evaluation_indices[1:])):
            raise ValueError(f"Evaluation indices are not increasing in {history_id}")
        if any(b <= a for a, b in zip(source_numbers, source_numbers[1:])):
            raise ValueError(f"Source frame numbers are not increasing in {history_id}")

        states = history.get("states")
        if not isinstance(states, list) or len(states) < 2:
            raise ValueError(f"Feature history {history_id} requires a sibling vocabulary")
        normalized_states: list[dict[str, Any]] = []
        seen_states: set[str] = set()
        seen_descriptions: set[str] = set()
        for state in states:
            if not isinstance(state, Mapping):
                raise ValueError(f"State row in {history_id} is not an object")
            state_id = str(state.get("state_id", "")).strip()
            descriptions = state.get("descriptions")
            if (
                not state_id
                or state_id in seen_states
                or not isinstance(descriptions, list)
                or not descriptions
            ):
                raise ValueError(f"Invalid sibling state in {history_id}")
            texts = [str(value).strip() for value in descriptions]
            if any(not text or text in seen_descriptions for text in texts):
                raise ValueError(f"Descriptions must be non-empty and unique in {history_id}")
            seen_states.add(state_id)
            seen_descriptions.update(texts)
            normalized_states.append({"state_id": state_id, "descriptions": texts})
        normalized.append(
            {
                "history_id": history_id,
                "sequence_id": sequence_id,
                "target_lineage_label_ids": lineage_ids,
                "frames": normalized_frames,
                "states": normalized_states,
            }
        )
    return normalized


def _resolve_hashed_source(raw_root: Path, relative: str, expected_sha256: str) -> Path:
    root = Path(raw_root).resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Source path escapes the raw root: {relative}") from exc
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"Missing regular source file: {relative}")
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"Source hash differs: {relative}")
    return path


def extract_feature_manifest(
    manifest: Mapping[str, Any],
    *,
    raw_root: Path,
    output_root: Path,
    model: Any,
    processor: Any,
    model_info: Mapping[str, Any],
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    """Extract all histories and write relative-path feature metadata."""

    from PIL import Image

    histories = validate_feature_manifest(manifest)
    destination = Path(output_root)
    if destination.exists():
        raise FileExistsError(f"Feature output already exists: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    reports: list[dict[str, Any]] = []
    for history in histories:
        crops: list[Image.Image] = []
        evaluation_indices: list[int] = []
        source_frame_numbers: list[int] = []
        visibility: list[int] = []
        source_frames: list[dict[str, Any]] = []
        for frame in history["frames"]:
            rgb_path = _resolve_hashed_source(
                raw_root, frame["rgb_path"], frame["rgb_sha256"]
            )
            mask_path = _resolve_hashed_source(
                raw_root, frame["mask_path"], frame["mask_sha256"]
            )
            with Image.open(rgb_path) as image, Image.open(mask_path) as mask_image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
                label_mask = np.asarray(mask_image)
            crop = mask_union_padded_crop(
                rgb,
                label_mask,
                history["target_lineage_label_ids"],
                padding_fraction=0.20,
                minimum_padding=4,
                background_value=127,
                ignored_label=255,
            )
            union = np.isin(
                label_mask, np.asarray(history["target_lineage_label_ids"])
            ) & (label_mask != 255)
            crops.append(Image.fromarray(crop, mode="RGB"))
            evaluation_indices.append(frame["evaluation_index"])
            source_frame_numbers.append(frame["source_frame_number"])
            visibility.append(int(np.count_nonzero(union)))
            source_frames.append(dict(frame))

        visual_features = encode_image_batches(
            model,
            processor,
            crops,
            batch_size=batch_size,
            device=device,
        )
        state_ids: list[str] = []
        descriptions: list[str] = []
        for state in history["states"]:
            for description in state["descriptions"]:
                state_ids.append(state["state_id"])
                descriptions.append(description)
        text_features = encode_description_batches(
            model,
            processor,
            descriptions,
            batch_size=batch_size,
            device=device,
        )

        history_dir = destination / history["history_id"]
        history_dir.mkdir(parents=False, exist_ok=False)
        visual_path = history_dir / "visual_features.npz"
        text_path = history_dir / "text_features.npz"
        np.savez_compressed(
            visual_path,
            frame_indices=np.asarray(evaluation_indices, dtype=np.int64),
            visual_features=visual_features,
            visibility_count=np.asarray(visibility, dtype=np.int64),
        )
        np.savez_compressed(
            text_path,
            state_ids=np.asarray(state_ids, dtype=str),
            state_texts=np.asarray(descriptions, dtype=str),
            text_features=text_features,
        )
        metadata = {
            "schema_version": 1,
            "kind": "deja_cue_siglip2_feature_history",
            "history_id": history["history_id"],
            "sequence_id": history["sequence_id"],
            "state_reference_labels_accessed": False,
            "feature_dimension": FEATURE_DIMENSION,
            "frame_indices": evaluation_indices,
            "source_frame_numbers": source_frame_numbers,
            "target_lineage_label_ids": history["target_lineage_label_ids"],
            "crop": {
                "mask_composition": "union_of_target_lineage_labels",
                "padding_fraction": 0.20,
                "minimum_padding_pixels": 4,
                "outside_mask_rgb_value": 127,
                "ignored_mask_label": 255,
            },
            "text_prompt_forms": [
                "description",
                "a photo of [description]",
                "the [description]",
            ],
            "source_frames": source_frames,
            "files": {
                "visual_features": {
                    "path": "visual_features.npz",
                    "sha256": sha256_file(visual_path),
                },
                "text_features": {
                    "path": "text_features.npz",
                    "sha256": sha256_file(text_path),
                },
            },
        }
        metadata_path = history_dir / "metadata.json"
        metadata_path.write_bytes(_canonical_json_bytes(metadata))
        reports.append(
            {
                "history_id": history["history_id"],
                "relative_directory": history["history_id"],
                "visible_frames": len(evaluation_indices),
                "descriptions": len(descriptions),
                "metadata_sha256": sha256_file(metadata_path),
            }
        )

    summary = {
        "schema_version": 1,
        "kind": "deja_cue_siglip2_feature_summary",
        "state_reference_labels_accessed": False,
        "model": dict(model_info),
        "history_count": len(reports),
        "reports": reports,
    }
    (destination / "summary.json").write_bytes(_canonical_json_bytes(summary))
    return summary
