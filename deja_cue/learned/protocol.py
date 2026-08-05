"""Frozen protocol for the seven-history learned-decoder experiment.

Only architecture-specific behavior that changes how the shared records are
consumed is represented here.  Data preparation, optimizer settings, training
duration, and evaluation inputs are common to all decoders.
"""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any, Literal, Mapping

import numpy as np


TRAINING_SEEDS = (3407, 3408, 3409)
DEVELOPMENT_HISTORY_COUNT = 5
DEVELOPMENT_STATE_COUNT = 13
DEVELOPMENT_DESCRIPTION_COUNT = 26
DEVELOPMENT_EPISODE_COUNT = 46
DEVELOPMENT_POSITIVE_RECORD_COUNT = 86
EVALUATION_HISTORY_COUNT = 7

VISUAL_FEATURE_DIM = 768
TEMPORAL_FEATURE_DIM = 2
VIDEO_TOKEN_DIM = VISUAL_FEATURE_DIM + TEMPORAL_FEATURE_DIM
TEXT_FEATURE_DIM = 768
TARGET_ROLE_DIM = 1
TEXT_TOKEN_DIM = TEXT_FEATURE_DIM + TARGET_ROLE_DIM
NUM_PROPOSALS = 10

EPOCHS = 200
# Architecture-specific schedulers may emit smaller batches; this is the cap.
BATCH_SIZE = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 0.1
CUBLAS_WORKSPACE_CONFIG = ":4096:8"

BatchingFamily = Literal[
    "stratified_exact_once",
    "cross_history_cycle",
    "history_unique",
    "unique_history_vtc",
]


@dataclass(frozen=True)
class DecoderSpec:
    """Architecture behavior required by the common training harness."""

    model_id: str
    display_name: str
    backend: Literal["lighthouse", "sim_detr"]
    batching_family: BatchingFamily
    native_target_position: Literal["first", "last"] = "first"
    maximum_video_tokens: int = 4096
    uses_ema: bool = False
    ema_decay: float | None = None
    lr_drop_epoch: int | None = None
    lr_drop_gamma: float | None = None
    ctc_loss_coefficient: float = 0.0
    vtc_loss_coefficient: float = 0.0

    def __post_init__(self) -> None:
        if not self.model_id or not self.display_name:
            raise ValueError("Decoder identifiers must be non-empty")
        if self.maximum_video_tokens <= 0:
            raise ValueError("maximum_video_tokens must be positive")
        if self.uses_ema != (self.ema_decay is not None):
            raise ValueError("EMA use and decay must be specified together")
        if self.ema_decay is not None and not 0.0 < self.ema_decay < 1.0:
            raise ValueError("ema_decay must lie in (0, 1)")
        if (self.lr_drop_epoch is None) != (self.lr_drop_gamma is None):
            raise ValueError("Learning-rate drop epoch and gamma must be paired")


_DECODER_SPECS = (
    DecoderSpec(
        "moment_detr", "Moment-DETR", "lighthouse", "stratified_exact_once"
    ),
    DecoderSpec(
        "qd_detr", "QD-DETR", "lighthouse", "cross_history_cycle"
    ),
    DecoderSpec("eatr", "EaTR", "lighthouse", "stratified_exact_once"),
    DecoderSpec(
        "cg_detr", "CG-DETR", "lighthouse", "cross_history_cycle"
    ),
    DecoderSpec("uvcom", "UVCOM", "lighthouse", "cross_history_cycle"),
    DecoderSpec(
        "tr_detr",
        "TR-DETR",
        "lighthouse",
        "history_unique",
        ctc_loss_coefficient=0.5,
        vtc_loss_coefficient=0.3,
    ),
    DecoderSpec(
        "taskweave_mr2hd",
        "TaskWeave",
        "lighthouse",
        "history_unique",
        maximum_video_tokens=75,
        uses_ema=True,
        ema_decay=0.9,
    ),
    DecoderSpec(
        "sim_detr",
        "Sim-DETR",
        "sim_detr",
        "unique_history_vtc",
        native_target_position="last",
        lr_drop_epoch=100,
        lr_drop_gamma=0.1,
        ctc_loss_coefficient=0.5,
        vtc_loss_coefficient=0.3,
    ),
)

DECODER_REGISTRY: Mapping[str, DecoderSpec] = MappingProxyType(
    {spec.model_id: spec for spec in _DECODER_SPECS}
)


def decoder_spec(model_id: str) -> DecoderSpec:
    """Return the settings for one supported decoder."""

    try:
        return DECODER_REGISTRY[model_id]
    except KeyError as exc:
        raise ValueError(f"Unsupported decoder: {model_id!r}") from exc


@dataclass(frozen=True)
class TrainingConfig:
    """Complete, deterministic training configuration for one model and seed."""

    model_id: str
    seed: int
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    gradient_clip: float = GRADIENT_CLIP
    optimizer: str = "AdamW"
    # These machine labels match the training records.  With exactly two
    # descriptions per state, query-mean centering is the state-balanced
    # vocabulary-relative coordinate described in the paper.
    text_coordinates: str = "query_centered"
    text_context_mode: str = "target_first_siblings"
    training_run_policy: str = "positive_only"
    checkpoint_selection: str = "final_checkpoint_only"
    development_record_count: int = DEVELOPMENT_POSITIVE_RECORD_COUNT
    lr_drop_epoch: int | None = None
    lr_drop_gamma: float | None = None
    inference_weight_source: str = "final_model"

    def __post_init__(self) -> None:
        spec = decoder_spec(self.model_id)
        if self.seed not in TRAINING_SEEDS:
            raise ValueError(f"Unexpected training seed: {self.seed}")
        fixed = {
            "epochs": (self.epochs, EPOCHS),
            "batch_size": (self.batch_size, BATCH_SIZE),
            "learning_rate": (self.learning_rate, LEARNING_RATE),
            "weight_decay": (self.weight_decay, WEIGHT_DECAY),
            "gradient_clip": (self.gradient_clip, GRADIENT_CLIP),
            "optimizer": (self.optimizer, "AdamW"),
            "text_coordinates": (
                self.text_coordinates,
                "query_centered",
            ),
            "text_context_mode": (
                self.text_context_mode,
                "target_first_siblings",
            ),
            "training_run_policy": (self.training_run_policy, "positive_only"),
            "checkpoint_selection": (
                self.checkpoint_selection,
                "final_checkpoint_only",
            ),
            "development_record_count": (
                self.development_record_count,
                DEVELOPMENT_POSITIVE_RECORD_COUNT,
            ),
            "lr_drop_epoch": (self.lr_drop_epoch, spec.lr_drop_epoch),
            "lr_drop_gamma": (self.lr_drop_gamma, spec.lr_drop_gamma),
            "inference_weight_source": (
                self.inference_weight_source,
                "final_ema" if spec.uses_ema else "final_model",
            ),
        }
        mismatches = {
            name: actual
            for name, (actual, expected) in fixed.items()
            if actual != expected
        }
        if mismatches:
            raise ValueError(f"Training config differs from the protocol: {mismatches}")

    def to_dict(self) -> dict[str, Any]:
        """Return all locked training fields as a JSON-compatible mapping."""

        return asdict(self)


def make_training_config(model_id: str, seed: int) -> TrainingConfig:
    """Build the fixed configuration for one decoder and seed."""

    spec = decoder_spec(model_id)
    return TrainingConfig(
        model_id=model_id,
        seed=seed,
        lr_drop_epoch=spec.lr_drop_epoch,
        lr_drop_gamma=spec.lr_drop_gamma,
        inference_weight_source="final_ema" if spec.uses_ema else "final_model",
    )


def epoch_seed(training_seed: int, zero_based_epoch: int) -> int:
    """Seed used to derive an epoch's deterministic batch schedule."""

    if training_seed not in TRAINING_SEEDS:
        raise ValueError(f"Unexpected training seed: {training_seed}")
    if zero_based_epoch < 0:
        raise ValueError("zero_based_epoch must be non-negative")
    return training_seed + zero_based_epoch


def saliency_seed(
    training_seed: int, zero_based_epoch: int, batch_index: int
) -> int:
    """Base seed for native saliency-pair sampling in one training batch."""

    if training_seed not in TRAINING_SEEDS:
        raise ValueError(f"Unexpected training seed: {training_seed}")
    if zero_based_epoch < 0 or batch_index < 0:
        raise ValueError("Epoch and batch indices must be non-negative")
    return training_seed * 1_000_003 + zero_based_epoch * 10_007 + batch_index * 101


def configure_deterministic_cuda_workspace() -> str:
    """Fix the cuBLAS workspace required by deterministic CUDA matmul.

    PyTorch requires this process environment variable before the first cuBLAS
    operation when deterministic algorithms are enabled.  Refusing a different
    pre-existing value ensures that the public training entry point uses the
    documented CUDA algorithm set.
    """

    existing = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if existing is None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
    elif existing != CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG must be "
            f"{CUBLAS_WORKSPACE_CONFIG!r}, observed {existing!r}"
        )
    return CUBLAS_WORKSPACE_CONFIG


def set_deterministic_seed(seed: int, *, torch_module: Any | None = None) -> dict[str, Any]:
    """Seed Python, NumPy, and, when available, PyTorch deterministically.

    ``torch_module`` permits dependency injection in lightweight tests.  If it
    is omitted, PyTorch is imported when installed; record preparation remains
    usable in NumPy-only environments.
    """

    workspace_config = configure_deterministic_cuda_workspace()
    random.seed(seed)
    np.random.seed(seed)
    torch_available = True
    if torch_module is None:
        try:
            import torch as torch_module  # type: ignore[no-redef]
        except ModuleNotFoundError:
            torch_available = False
    if torch_available:
        torch_module.manual_seed(seed)
        if torch_module.cuda.is_available():
            torch_module.cuda.manual_seed_all(seed)
        torch_module.use_deterministic_algorithms(True)
        torch_module.backends.cudnn.benchmark = False
        torch_module.backends.cudnn.deterministic = True
    return {
        "seed": int(seed),
        "python_seeded": True,
        "numpy_seeded": True,
        "torch_seeded": torch_available,
        "deterministic_algorithms": torch_available,
        "cublas_workspace_config": workspace_config,
    }


def protocol_summary() -> dict[str, Any]:
    """Return the shared training protocol as structured data."""

    return {
        "development": {
            "histories": DEVELOPMENT_HISTORY_COUNT,
            "states": DEVELOPMENT_STATE_COUNT,
            "descriptions": DEVELOPMENT_DESCRIPTION_COUNT,
            "episodes": DEVELOPMENT_EPISODE_COUNT,
            "positive_records": DEVELOPMENT_POSITIVE_RECORD_COUNT,
        },
        "evaluation_histories": EVALUATION_HISTORY_COUNT,
        "seeds": list(TRAINING_SEEDS),
        "coordinate_semantics": "state_balanced_vocabulary_relative",
        "sibling_context": "target_first_complete_vocabulary",
        "feature_dimensions": {
            "visual": VISUAL_FEATURE_DIM,
            "temporal": TEMPORAL_FEATURE_DIM,
            "video_token": VIDEO_TOKEN_DIM,
            "text": TEXT_FEATURE_DIM,
            "text_with_target_role": TEXT_TOKEN_DIM,
        },
        "decoders": [asdict(spec) for spec in _DECODER_SPECS],
    }
