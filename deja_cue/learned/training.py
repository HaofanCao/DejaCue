"""Deterministic executor for the shared learned-decoder training protocol.

Decoder-specific code constructs the native model and objective through a
small adapter.  This module owns every operation shared by the eight reported
decoders: record validation, batch scheduling, saliency-pair sampling, text
ordering, optimization, gradient clipping, learning-rate scheduling, EMA
weight selection, and checkpoint creation.

The fixed settings implement the paper appendix, "Low-Data Decoder Adaptation / Shared
Adaptation Protocol."
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .batching import build_epoch_batches, summarize_batches
from .checkpoint import parameter_state_sha256, save_checkpoint
from .protocol import (
    DEVELOPMENT_DESCRIPTION_COUNT,
    DEVELOPMENT_HISTORY_COUNT,
    DEVELOPMENT_POSITIVE_RECORD_COUNT,
    DEVELOPMENT_STATE_COUNT,
    DecoderSpec,
    TrainingConfig,
    decoder_spec,
    saliency_seed,
    set_deterministic_seed,
)
from .records import (
    MAX_RUN_TOKENS,
    LearnedRunRecord,
    PreparedBatch,
    collate_records,
    reorder_sim_detr_text,
    sample_saliency_pairs,
)


@dataclass(frozen=True)
class TrainingStepContext:
    """Frozen protocol fields visible to one native objective call."""

    config: TrainingConfig
    decoder: DecoderSpec
    zero_based_epoch: int
    batch_index: int
    device: Any


@dataclass(frozen=True)
class LossOutput:
    """Scalar optimization loss and named scalar components for reporting."""

    total_loss: Any
    components: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DecoderTrainingComponents:
    """Native objects constructed after deterministic seeds have been fixed.

    ``objective`` receives the model, the shared NumPy batch, and its exact
    protocol context.  It is responsible only for architecture-native tensor
    conversion, forward propagation, matching, and loss composition.
    ``update_ema`` and ``load_ema_weights`` are required only for the TaskWeave
    configuration.
    """

    model: Any
    objective: Callable[[Any, PreparedBatch, TrainingStepContext], LossOutput]
    architecture_metadata: Mapping[str, Any] | None = None
    update_ema: Callable[[Any, int], None] | None = None
    load_ema_weights: Callable[[Any], None] | None = None


@dataclass(frozen=True)
class DecoderTrainingAdapter:
    """Factory connecting one supported decoder ID to its native components."""

    model_id: str
    build: Callable[[TrainingConfig, Any], DecoderTrainingComponents]


@dataclass(frozen=True)
class EpochSummary:
    """Deterministic summary of one completed training epoch."""

    epoch: int
    learning_rate: float
    batches: int
    exposures: int
    duplicate_exposures: int
    mean_total_loss: float
    mean_components: Mapping[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible epoch record."""

        return asdict(self)


@dataclass(frozen=True)
class TrainingSummary:
    """Configuration, parameter hashes, and per-epoch optimization summaries."""

    model_id: str
    seed: int
    epochs: int
    optimizer_steps: int
    record_counts: Mapping[str, int]
    seed_state: Mapping[str, Any]
    initial_parameter_sha256: str
    final_parameter_sha256: str
    inference_weight_source: str
    epoch_summaries: tuple[EpochSummary, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible complete training summary."""

        payload = asdict(self)
        payload["epoch_summaries"] = [
            summary.to_dict() for summary in self.epoch_summaries
        ]
        return payload


@dataclass(frozen=True)
class TrainingResult:
    """Trained native model together with its training summary."""

    model: Any
    architecture_metadata: Mapping[str, Any] | None
    summary: TrainingSummary


def validate_training_records(
    records: Sequence[LearnedRunRecord], *, model_id: str
) -> dict[str, int]:
    """Validate the expected counts for five-history positive training records."""

    selected = tuple(records)
    spec = decoder_spec(model_id)
    histories = {record.history_id for record in selected}
    states = {(record.history_id, record.state_id) for record in selected}
    descriptions = {
        (record.history_id, record.query_index) for record in selected
    }
    record_ids = {record.record_id for record in selected}
    counts = {
        "records": len(selected),
        "histories": len(histories),
        "states": len(states),
        "descriptions": len(descriptions),
    }
    expected = {
        "records": DEVELOPMENT_POSITIVE_RECORD_COUNT,
        "histories": DEVELOPMENT_HISTORY_COUNT,
        "states": DEVELOPMENT_STATE_COUNT,
        "descriptions": DEVELOPMENT_DESCRIPTION_COUNT,
    }
    if counts != expected:
        raise ValueError(
            f"Training-record counts differ: expected={expected}, "
            f"observed={counts}"
        )
    if len(record_ids) != len(selected):
        raise ValueError("Training records contain a repeated history/query/run key")
    if not all(record.has_target for record in selected):
        raise ValueError("The training set must contain positive records only")
    # TaskWeave consumes a fixed 75-token resampling of each run.  The 75-token
    # native horizon is not an input-data exclusion rule; all decoders receive
    # the same positive run roster before architecture-native conversion.
    if any(record.num_frames > MAX_RUN_TOKENS for record in selected):
        raise ValueError(
            f"A training run exceeds the shared {MAX_RUN_TOKENS}-token limit"
        )
    return counts


def prepare_training_batch(
    records: Sequence[LearnedRunRecord],
    *,
    config: TrainingConfig,
    zero_based_epoch: int,
    batch_index: int,
) -> PreparedBatch:
    """Collate one batch and apply its exact stochastic and ordering rules."""

    if zero_based_epoch < 0 or batch_index < 0:
        raise ValueError("Epoch and batch indices must be non-negative")
    selected = tuple(records)
    batch = collate_records(selected)
    base_seed = saliency_seed(config.seed, zero_based_epoch, batch_index)
    positive_pairs = []
    negative_pairs = []
    for record_position, record in enumerate(selected):
        positive, negative = sample_saliency_pairs(
            record, seed=base_seed + record_position
        )
        positive_pairs.append(positive)
        negative_pairs.append(negative)
    batch = replace(
        batch,
        saliency_positive=np.asarray(positive_pairs, dtype=np.int64),
        saliency_negative=np.asarray(negative_pairs, dtype=np.int64),
    )

    # Sim-DETR consumes the queried text at the final valid position.  Only the
    # physical token order changes; the complete sibling set and role channel
    # remain identical to the target-first representation used elsewhere.
    if decoder_spec(config.model_id).native_target_position == "last":
        batch = replace(batch, text_tokens=reorder_sim_detr_text(batch))
    return batch


def _validate_components(
    components: DecoderTrainingComponents, spec: DecoderSpec
) -> None:
    if components.model is None or not callable(components.objective):
        raise ValueError("The decoder adapter did not construct a model and objective")
    ema_hooks = (components.update_ema, components.load_ema_weights)
    if spec.uses_ema and not all(callable(hook) for hook in ema_hooks):
        raise ValueError("The TaskWeave decoder requires EMA update and restore hooks")
    if not spec.uses_ema and any(hook is not None for hook in ema_hooks):
        raise ValueError("EMA hooks are not used by this decoder")


def _component_values(values: Mapping[str, Any]) -> dict[str, float]:
    output: dict[str, float] = {}
    for raw_name, raw_value in values.items():
        name = str(raw_name)
        if not name or name in output:
            raise ValueError("Loss component names must be unique and non-empty")
        if hasattr(raw_value, "detach"):
            detached = raw_value.detach()
            if getattr(detached, "numel", lambda: 0)() != 1:
                raise ValueError(f"Loss component {name!r} is not scalar")
            value = float(detached.cpu().item())
        else:
            value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"Loss component {name!r} is not finite")
        output[name] = value
    return output


def train_decoder(
    records: Sequence[LearnedRunRecord],
    adapter: DecoderTrainingAdapter,
    config: TrainingConfig,
    *,
    device: str = "cpu",
) -> TrainingResult:
    """Train one supported decoder under the fixed 200-epoch protocol."""

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required to train learned decoders") from exc

    if adapter.model_id != config.model_id:
        raise ValueError("Decoder adapter and training configuration differ")
    spec = decoder_spec(config.model_id)
    record_counts = validate_training_records(records, model_id=config.model_id)
    seed_state = set_deterministic_seed(config.seed, torch_module=torch)
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    # Model and criterion construction happens after all random generators are
    # fixed, so parameter initialization is part of the seeded experiment.
    components = adapter.build(config, resolved_device)
    _validate_components(components, spec)
    model = components.model.to(resolved_device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("The decoder model has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = None
    if config.lr_drop_epoch is not None:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.lr_drop_epoch,
            gamma=float(config.lr_drop_gamma),
        )

    initial_hash = parameter_state_sha256(model.state_dict())
    epoch_summaries: list[EpochSummary] = []
    optimizer_steps = 0
    expected_component_names: tuple[str, ...] | None = None
    model.train()
    selected = tuple(records)
    for zero_based_epoch in range(config.epochs):
        schedule = build_epoch_batches(
            selected,
            model_id=config.model_id,
            training_seed=config.seed,
            zero_based_epoch=zero_based_epoch,
            batch_size=config.batch_size,
        )
        batch_summary = summarize_batches(selected, schedule)
        if not all(
            bool(batch_summary[key])
            for key in (
                "all_records_covered",
                "all_exposures_positive",
                "all_batches_have_two_records",
            )
        ):
            raise RuntimeError("Generated epoch batches violate the protocol")

        learning_rate = float(optimizer.param_groups[0]["lr"])
        total_losses: list[float] = []
        component_totals: dict[str, float] = {}
        for batch_index, indices in enumerate(schedule):
            batch_records = tuple(selected[index] for index in indices)
            batch = prepare_training_batch(
                batch_records,
                config=config,
                zero_based_epoch=zero_based_epoch,
                batch_index=batch_index,
            )
            context = TrainingStepContext(
                config=config,
                decoder=spec,
                zero_based_epoch=zero_based_epoch,
                batch_index=batch_index,
                device=resolved_device,
            )
            optimizer.zero_grad(set_to_none=True)
            output = components.objective(model, batch, context)
            if not isinstance(output, LossOutput):
                raise TypeError("Decoder objective must return LossOutput")
            loss = output.total_loss
            if (
                not torch.is_tensor(loss)
                or loss.ndim != 0
                or not bool(torch.isfinite(loss).item())
                or not loss.requires_grad
            ):
                raise ValueError("Decoder objective returned an invalid scalar loss")
            component_values = _component_values(output.components)
            names = tuple(sorted(component_values))
            if expected_component_names is None:
                expected_component_names = names
            elif names != expected_component_names:
                raise ValueError("Loss component fields changed between batches")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                trainable, config.gradient_clip, error_if_nonfinite=True
            )
            optimizer.step()
            optimizer_steps += 1
            total_losses.append(float(loss.detach().cpu().item()))
            for name, value in component_values.items():
                component_totals[name] = component_totals.get(name, 0.0) + value

        if components.update_ema is not None:
            components.update_ema(model, zero_based_epoch)
        epoch_summaries.append(
            EpochSummary(
                epoch=zero_based_epoch + 1,
                learning_rate=learning_rate,
                batches=int(batch_summary["num_batches"]),
                exposures=int(batch_summary["num_exposures"]),
                duplicate_exposures=int(
                    batch_summary["num_duplicate_exposures"]
                ),
                mean_total_loss=float(np.mean(total_losses)),
                mean_components={
                    name: value / len(total_losses)
                    for name, value in sorted(component_totals.items())
                },
            )
        )
        if scheduler is not None:
            scheduler.step()

    if components.load_ema_weights is not None:
        components.load_ema_weights(model)
    model.eval()
    final_hash = parameter_state_sha256(model.state_dict())
    summary = TrainingSummary(
        model_id=config.model_id,
        seed=config.seed,
        epochs=config.epochs,
        optimizer_steps=optimizer_steps,
        record_counts=record_counts,
        seed_state=seed_state,
        initial_parameter_sha256=initial_hash,
        final_parameter_sha256=final_hash,
        inference_weight_source=config.inference_weight_source,
        epoch_summaries=tuple(epoch_summaries),
    )
    return TrainingResult(
        model=model,
        architecture_metadata=components.architecture_metadata,
        summary=summary,
    )


def train_and_save_checkpoint(
    path: Path,
    records: Sequence[LearnedRunRecord],
    adapter: DecoderTrainingAdapter,
    config: TrainingConfig,
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run adaptation and save the final inference weights."""

    result = train_decoder(records, adapter, config, device=device)
    checkpoint = save_checkpoint(
        Path(path),
        result.model.state_dict(),
        config,
        architecture=result.architecture_metadata,
    )
    return {"training": result.summary.to_dict(), "checkpoint": checkpoint}
