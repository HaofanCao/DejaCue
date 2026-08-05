"""Portable checkpoint envelope for the adapted-decoder experiment.

The envelope binds weights to the frozen model/seed configuration and retains
the scientific fields required to restore a decoder.  Parameter hashes are
computed from tensor content rather than serialization so they are stable
across archive containers and PyTorch save implementations.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .protocol import (
    DEVELOPMENT_POSITIVE_RECORD_COUNT,
    EVALUATION_HISTORY_COUNT,
    NUM_PROPOSALS,
    TEXT_TOKEN_DIM,
    VIDEO_TOKEN_DIM,
    TrainingConfig,
    decoder_spec,
)


CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_KIND = "deja_cue_learned_decoder_checkpoint"
PROTOCOL_ID = "seven_history_low_data_decoder_v1"

_REQUIRED_FIELDS = {
    "schema_version",
    "kind",
    "protocol_id",
    "model_id",
    "seed",
    "training_config",
    "training_config_sha256",
    "metadata",
    "parameter_sha256",
    "state_dict",
}
_ARCHITECTURE_METADATA_FIELDS = frozenset(
    (
        "attention_heads",
        "auxiliary_loss",
        "contrastive_dimension",
        "decoder_layers",
        "dropout",
        "encoder_layers",
        "feedforward_dimension",
        "hidden_dimension",
        "input_dropout",
        "maximum_video_tokens",
        "model_family",
        "native_config_lock_sha256",
        "native_source_commit",
        "native_source_lock_sha256",
        "position_embedding",
        "proposal_count",
        "span_loss_type",
        "temperature",
    )
)


def canonical_sha256(value: Any) -> str:
    """Hash a JSON-compatible value with stable key and separator ordering."""

    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _tensor_bytes(value: Any) -> tuple[str, list[int], bytes]:
    if hasattr(value, "detach"):
        tensor = value.detach().cpu().contiguous()
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError("PyTorch is required to hash tensor parameters") from exc
        raw = tensor.reshape(-1).view(dtype=torch.uint8).numpy().tobytes(order="C")
        return str(tensor.dtype), list(tensor.shape), raw
    array = np.ascontiguousarray(np.asarray(value))
    if array.dtype.hasobject:
        raise ValueError("Object arrays are not valid model parameters")
    return str(array.dtype), list(array.shape), array.tobytes(order="C")


def parameter_state_sha256(state_dict: Mapping[str, Any]) -> str:
    """Hash ordered parameter names, dtype/shape metadata, and raw bytes."""

    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("Model state dictionary is empty")
    digest = hashlib.sha256()
    for raw_name in sorted(state_dict):
        name = str(raw_name)
        if not name:
            raise ValueError("Model state dictionary contains an empty key")
        dtype, shape, raw = _tensor_bytes(state_dict[raw_name])
        metadata = {"name": name, "dtype": dtype, "shape": shape}
        digest.update(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode(
                "ascii"
            )
        )
        digest.update(b"\0")
        digest.update(raw)
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_architecture_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    unexpected = sorted(set(value).difference(_ARCHITECTURE_METADATA_FIELDS))
    if unexpected:
        raise ValueError(f"Architecture metadata contains unsupported fields: {unexpected}")
    try:
        normalized = json.loads(json.dumps(value, sort_keys=True, ensure_ascii=True))
    except (TypeError, ValueError) as exc:
        raise ValueError("Architecture metadata must be JSON-compatible") from exc
    if not isinstance(normalized, dict):
        raise ValueError("Architecture metadata must be a mapping")
    return normalized


def _default_metadata(config: TrainingConfig) -> dict[str, Any]:
    spec = decoder_spec(config.model_id)
    return {
        "training_scope": "five_development_histories",
        "positive_training_records": DEVELOPMENT_POSITIVE_RECORD_COUNT,
        "evaluation_history_count": EVALUATION_HISTORY_COUNT,
        "evaluation_histories_used_for_training": 0,
        "video_token_dimension": VIDEO_TOKEN_DIM,
        "text_token_dimension": TEXT_TOKEN_DIM,
        "proposal_count": NUM_PROPOSALS,
        "coordinate_semantics": "state_balanced_vocabulary_relative",
        "sibling_context": "complete_vocabulary_target_first",
        "batching_family": spec.batching_family,
        "native_target_position": spec.native_target_position,
        "inference_weight_source": config.inference_weight_source,
    }


def build_checkpoint(
    state_dict: Mapping[str, Any],
    config: TrainingConfig,
    *,
    architecture: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a validated in-memory checkpoint payload.

    ``architecture`` may contain numeric or categorical model settings.  Keys
    outside the supported numeric and categorical model settings are rejected.
    """

    configuration = config.to_dict()
    metadata = _default_metadata(config)
    if architecture is not None:
        metadata["architecture"] = _validate_architecture_metadata(architecture)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "kind": CHECKPOINT_KIND,
        "protocol_id": PROTOCOL_ID,
        "model_id": config.model_id,
        "seed": config.seed,
        "training_config": configuration,
        "training_config_sha256": canonical_sha256(configuration),
        "metadata": metadata,
        "parameter_sha256": parameter_state_sha256(state_dict),
        "state_dict": state_dict,
    }
    validate_checkpoint(payload, expected_model_id=config.model_id, expected_seed=config.seed)
    return payload


def validate_checkpoint(
    payload: Mapping[str, Any],
    *,
    expected_model_id: str | None = None,
    expected_seed: int | None = None,
) -> dict[str, Any]:
    """Validate identity, protocol configuration, metadata, and parameter bytes."""

    if not isinstance(payload, Mapping) or set(payload) != _REQUIRED_FIELDS:
        raise ValueError("Checkpoint envelope fields differ from schema version 1")
    if (
        payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
        or payload.get("kind") != CHECKPOINT_KIND
        or payload.get("protocol_id") != PROTOCOL_ID
    ):
        raise ValueError("Checkpoint protocol identity differs")
    model_id = str(payload.get("model_id", ""))
    seed = payload.get("seed")
    decoder_spec(model_id)
    if expected_model_id is not None and model_id != expected_model_id:
        raise ValueError("Checkpoint model identity differs")
    if expected_seed is not None and seed != expected_seed:
        raise ValueError("Checkpoint seed differs")

    raw_config = payload.get("training_config")
    if not isinstance(raw_config, Mapping):
        raise ValueError("Checkpoint training config is missing")
    try:
        config = TrainingConfig(**dict(raw_config))
    except (TypeError, ValueError) as exc:
        raise ValueError("Checkpoint training config is invalid") from exc
    if config.model_id != model_id or config.seed != seed:
        raise ValueError("Checkpoint config identity differs from its envelope")
    if payload.get("training_config_sha256") != canonical_sha256(config.to_dict()):
        raise ValueError("Checkpoint training config digest differs")

    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("Checkpoint metadata is missing")
    expected_metadata = _default_metadata(config)
    if any(metadata.get(key) != value for key, value in expected_metadata.items()):
        raise ValueError("Checkpoint shared protocol metadata differs")
    if set(metadata).difference({*expected_metadata, "architecture"}):
        raise ValueError("Checkpoint metadata contains an unexpected field")
    if "architecture" in metadata:
        if not isinstance(metadata["architecture"], Mapping):
            raise ValueError("Checkpoint architecture metadata must be a mapping")
        _validate_architecture_metadata(metadata["architecture"])

    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("Checkpoint state dictionary is missing")
    digest = parameter_state_sha256(state_dict)
    if payload.get("parameter_sha256") != digest:
        raise ValueError("Checkpoint parameter digest differs")
    return {
        "model_id": model_id,
        "seed": int(seed),
        "parameter_sha256": digest,
        "training_config_sha256": canonical_sha256(config.to_dict()),
    }


def save_checkpoint(
    path: Path,
    state_dict: Mapping[str, Any],
    config: TrainingConfig,
    *,
    architecture: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize one validated checkpoint envelope."""

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required to save learned checkpoints") from exc
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"Checkpoint already exists: {destination.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = build_checkpoint(state_dict, config, architecture=architecture)
    torch.save(payload, destination)
    return validate_checkpoint(payload)


def load_checkpoint(
    path: Path,
    *,
    expected_model_id: str,
    expected_seed: int,
    map_location: str = "cpu",
) -> dict[str, Any]:
    """Load and strictly validate a checkpoint before model-state restoration."""

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required to load learned checkpoints") from exc
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {source.name}")
    try:
        payload = torch.load(source, map_location=map_location, weights_only=True)
    except TypeError:
        payload = torch.load(source, map_location=map_location)
    validate_checkpoint(
        payload,
        expected_model_id=expected_model_id,
        expected_seed=expected_seed,
    )
    return dict(payload)
