"""Deterministic tracking perturbations for robustness experiments.

Operations are pure: the input history and distractor arrays are never
modified.  Missing observations are deleted, contamination uses only
exact-time distractors, and identity switches replace only an exact-time
suffix. All randomness is derived from stable history identifiers.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np

from ..data import History
from ..seven_history import SevenHistoryDistractor


REPLICATE_SEEDS = (3407, 3408, 3409)
RANDOM_MISSING_RATES = (0.10, 0.25, 0.50, 0.75)
CONTIGUOUS_MISSING_SPAN_FRACTIONS = (0.10, 0.25, 0.50)
DISTRACTOR_CONTAMINATION_FRACTIONS = (0.25, 0.50, 1.00)
DISTRACTOR_MIXING_WEIGHTS = (0.25, 0.50, 1.00)
IDENTITY_SWITCH_POINT_FRACTIONS = (0.25, 0.50, 0.75)


@dataclass(frozen=True)
class PerturbationOutcome:
    """A perturbed history and details of the applied changes."""

    history: History
    details: dict[str, Any]


def _format_fraction(value: float) -> str:
    return f"{int(round(100.0 * value)):03d}"


def perturbation_cells() -> tuple[dict[str, Any], ...]:
    """Return the complete grid of perturbation conditions."""

    rows: list[dict[str, Any]] = []
    for rate in RANDOM_MISSING_RATES:
        rows.append(
            {
                "condition_id": f"random_missing_{_format_fraction(rate)}pct",
                "family": "random_missing_frames",
                "parameters": {"missing_fraction": rate},
                "replicate_seeds": list(REPLICATE_SEEDS),
            }
        )
    for fraction in CONTIGUOUS_MISSING_SPAN_FRACTIONS:
        rows.append(
            {
                "condition_id": (
                    f"contiguous_missing_span_{_format_fraction(fraction)}pct"
                ),
                "family": "contiguous_missing_frames",
                "parameters": {"missing_span_fraction": fraction},
                "replicate_seeds": list(REPLICATE_SEEDS),
            }
        )
    for affected_fraction in DISTRACTOR_CONTAMINATION_FRACTIONS:
        for mixing_weight in DISTRACTOR_MIXING_WEIGHTS:
            rows.append(
                {
                    "condition_id": (
                        "distractor_contamination_"
                        f"{_format_fraction(affected_fraction)}pct_"
                        f"mix{_format_fraction(mixing_weight)}pct"
                    ),
                    "family": "distractor_contamination",
                    "parameters": {
                        "affected_fraction": affected_fraction,
                        "mixing_weight": mixing_weight,
                    },
                    "replicate_seeds": list(REPLICATE_SEEDS),
                }
            )
    for switch_point in IDENTITY_SWITCH_POINT_FRACTIONS:
        rows.append(
            {
                "condition_id": (
                    f"identity_switch_after_{_format_fraction(switch_point)}pct"
                ),
                "family": "identity_switch",
                "parameters": {"switch_point_fraction": switch_point},
                # The operation itself is deterministic, but each condition
                # is evaluated with the same three seeds as the other families.
                "replicate_seeds": list(REPLICATE_SEEDS),
            }
        )
    return tuple(rows)


def perturbation_protocol() -> dict[str, Any]:
    """Return the perturbation protocol as structured data."""

    cells = perturbation_cells()
    return {
        "schema_version": 1,
        "coordinate_system": "inclusive_absolute_frame_indices",
        "randomness": "sha256-derived per replicate, condition, and history_id",
        "missing_frame_semantics": (
            "removed observations form validity gaps that cannot be smoothed or scanned across"
        ),
        "contamination_semantics": (
            "only exact-time distractor features are mixed and then L2-normalized"
        ),
        "identity_switch_semantics": (
            "maximum exact-time suffix coverage with track-id tie breaking"
        ),
        "num_conditions": len(cells),
        "cells": [dict(row) for row in cells],
    }


def canonical_sha256(value: Any) -> str:
    """Hash a JSON-compatible value using a stable canonical serialization."""

    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def derived_seed(replicate_seed: int, condition_id: str, object_group_id: str) -> int:
    """Derive an order-independent RNG seed from the source object group."""

    payload = f"{int(replicate_seed)}\0{condition_id}\0{object_group_id}".encode(
        "ascii"
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _rng(
    history: History, condition: Mapping[str, Any], replicate_seed: int
) -> np.random.Generator:
    return np.random.default_rng(
        derived_seed(
            replicate_seed,
            str(condition["condition_id"]),
            history.source_component_id,
        )
    )


def _base_details(
    history: History, condition: Mapping[str, Any], replicate_seed: int
) -> dict[str, Any]:
    return {
        "history_id": history.history_id,
        "condition_id": str(condition["condition_id"]),
        "family": str(condition["family"]),
        "parameters": dict(condition["parameters"]),
        "replicate_seed": int(replicate_seed),
        "derived_seed": derived_seed(
            replicate_seed,
            str(condition["condition_id"]),
            history.source_component_id,
        ),
        "num_target_observations_before": int(len(history.frame_indices)),
        "target_support_before": [
            int(history.frame_indices[0]),
            int(history.frame_indices[-1]),
        ],
    }


def _finalize_details(details: dict[str, Any]) -> dict[str, Any]:
    changes = details.get("changes", [])
    details["num_modified_observations"] = len(changes)
    details["applied"] = bool(changes)
    details["changes_sha256"] = canonical_sha256(changes)
    return details


def _subset_history(history: History, keep: np.ndarray) -> History:
    if keep.dtype != np.bool_ or keep.shape != history.frame_indices.shape:
        raise ValueError("keep must be a boolean vector aligned with the history")
    if not bool(keep.any()):
        raise ValueError("A perturbation must retain at least one observation")
    return replace(
        history,
        frame_indices=history.frame_indices[keep].copy(),
        visual_features=history.visual_features[keep].copy(),
        visibility_count=history.visibility_count[keep].copy(),
    )


def _normalized(rows: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(rows, axis=-1, keepdims=True)
    if bool((norms <= 1e-12).any()):
        raise ValueError("Feature mixing produced a zero-norm vector")
    result = rows / norms
    if not bool(np.isfinite(result).all()):
        raise ValueError("Feature mixing produced non-finite values")
    return result.astype(np.float32)


def _random_missing(
    history: History,
    condition: Mapping[str, Any],
    replicate_seed: int,
) -> PerturbationOutcome:
    details = _base_details(history, condition, replicate_seed)
    count = len(history.frame_indices)
    rate = float(condition["parameters"]["missing_fraction"])
    if count <= 1:
        details["ineligible_reason"] = "fewer_than_two_target_observations"
        return PerturbationOutcome(history, _finalize_details(details))
    remove_count = min(count - 1, max(1, int(round(rate * count))))
    removed_indices = np.sort(
        _rng(history, condition, replicate_seed).choice(
            count, size=remove_count, replace=False
        )
    )
    keep = np.ones(count, dtype=bool)
    keep[removed_indices] = False
    removed_frames = history.frame_indices[removed_indices]
    details.update(
        {
            "requested_missing_fraction": rate,
            "achieved_missing_fraction": float(remove_count / count),
            "changes": [
                {"frame_index": int(frame), "operation": "delete"}
                for frame in removed_frames.tolist()
            ],
        }
    )
    return PerturbationOutcome(
        _subset_history(history, keep), _finalize_details(details)
    )


def _contiguous_missing(
    history: History,
    condition: Mapping[str, Any],
    replicate_seed: int,
) -> PerturbationOutcome:
    details = _base_details(history, condition, replicate_seed)
    frames = history.frame_indices
    if len(frames) <= 1:
        details["ineligible_reason"] = "fewer_than_two_target_observations"
        return PerturbationOutcome(history, _finalize_details(details))
    support_span = int(frames[-1] - frames[0] + 1)
    fraction = float(condition["parameters"]["missing_span_fraction"])
    interval_length = min(
        support_span - 1, max(1, int(round(fraction * support_span)))
    )
    starts = np.arange(
        int(frames[0]), int(frames[-1]) - interval_length + 2, dtype=np.int64
    )
    candidates: list[tuple[int, np.ndarray]] = []
    for start in starts.tolist():
        removed = (frames >= start) & (frames < start + interval_length)
        if bool(removed.any()) and not bool(removed.all()):
            candidates.append((int(start), removed))
    if not candidates:
        details["ineligible_reason"] = "no_nontrivial_contiguous_interval"
        return PerturbationOutcome(history, _finalize_details(details))
    rng = _rng(history, condition, replicate_seed)
    start, removed = candidates[int(rng.integers(0, len(candidates)))]
    end = start + interval_length - 1
    removed_frames = frames[removed]
    details.update(
        {
            "requested_missing_span_fraction": fraction,
            "missing_interval": [start, end],
            "missing_interval_length": interval_length,
            "achieved_span_fraction": float(interval_length / support_span),
            "achieved_observation_missing_fraction": float(removed.sum() / len(frames)),
            "changes": [
                {"frame_index": int(frame), "operation": "delete"}
                for frame in removed_frames.tolist()
            ],
        }
    )
    return PerturbationOutcome(
        _subset_history(history, ~removed), _finalize_details(details)
    )


def _exact_time_distractors(
    distractors: Sequence[SevenHistoryDistractor],
) -> dict[int, list[tuple[int, np.ndarray]]]:
    rows: dict[int, list[tuple[int, np.ndarray]]] = {}
    for distractor in sorted(distractors, key=lambda item: item.track_id):
        for frame, feature in zip(
            distractor.frame_indices.tolist(), distractor.visual_features
        ):
            rows.setdefault(int(frame), []).append(
                (int(distractor.track_id), np.asarray(feature, dtype=np.float32))
            )
    return rows


def _distractor_contamination(
    history: History,
    distractors: Sequence[SevenHistoryDistractor],
    condition: Mapping[str, Any],
    replicate_seed: int,
) -> PerturbationOutcome:
    details = _base_details(history, condition, replicate_seed)
    by_frame = _exact_time_distractors(distractors)
    eligible_indices = np.asarray(
        [
            index
            for index, frame in enumerate(history.frame_indices.tolist())
            if int(frame) in by_frame
        ],
        dtype=np.int64,
    )
    if not len(eligible_indices):
        details["ineligible_reason"] = "no_exact_time_distractor_overlap"
        return PerturbationOutcome(history, _finalize_details(details))
    affected_fraction = float(condition["parameters"]["affected_fraction"])
    mixing_weight = float(condition["parameters"]["mixing_weight"])
    affected_count = min(
        len(eligible_indices),
        max(1, int(round(affected_fraction * len(eligible_indices)))),
    )
    rng = _rng(history, condition, replicate_seed)
    affected_indices = np.sort(
        rng.choice(eligible_indices, size=affected_count, replace=False)
    )
    features = history.visual_features.copy()
    changes: list[dict[str, Any]] = []
    for target_index in affected_indices.tolist():
        frame = int(history.frame_indices[target_index])
        sources = by_frame[frame]
        source_track_id, source_feature = sources[int(rng.integers(0, len(sources)))]
        mixed = (
            (1.0 - mixing_weight) * features[target_index]
            + mixing_weight * source_feature
        )
        features[target_index] = _normalized(mixed[None, :])[0]
        changes.append(
            {
                "frame_index": frame,
                "operation": "mix_exact_time_distractor",
                "source_track_id": source_track_id,
                "mixing_weight": mixing_weight,
            }
        )
    details.update(
        {
            "num_exact_time_eligible_observations": int(len(eligible_indices)),
            "requested_affected_fraction": affected_fraction,
            "achieved_affected_fraction_of_eligible": float(
                affected_count / len(eligible_indices)
            ),
            "achieved_affected_fraction_of_target": float(
                affected_count / len(history.frame_indices)
            ),
            "mixing_weight": mixing_weight,
            "exact_time_alignment": True,
            "changes": changes,
        }
    )
    return PerturbationOutcome(
        replace(history, visual_features=features), _finalize_details(details)
    )


def _identity_switch(
    history: History,
    distractors: Sequence[SevenHistoryDistractor],
    condition: Mapping[str, Any],
    replicate_seed: int,
) -> PerturbationOutcome:
    details = _base_details(history, condition, replicate_seed)
    target_frames = history.frame_indices
    switch_fraction = float(condition["parameters"]["switch_point_fraction"])
    span = int(target_frames[-1] - target_frames[0] + 1)
    switch_frame = int(target_frames[0] + round(switch_fraction * (span - 1)))
    target_lookup = {
        int(frame): index for index, frame in enumerate(target_frames.tolist())
    }
    candidates: list[tuple[int, SevenHistoryDistractor, list[int]]] = []
    for distractor in distractors:
        shared = sorted(
            int(frame)
            for frame in distractor.frame_indices.tolist()
            if int(frame) >= switch_frame and int(frame) in target_lookup
        )
        if shared:
            candidates.append((len(shared), distractor, shared))
    if not candidates:
        details["ineligible_reason"] = "no_exact_time_distractor_suffix_overlap"
        return PerturbationOutcome(history, _finalize_details(details))
    _, selected, shared_frames = sorted(
        candidates, key=lambda row: (-row[0], int(row[1].track_id))
    )[0]
    source_lookup = {
        int(frame): np.asarray(feature, dtype=np.float32)
        for frame, feature in zip(
            selected.frame_indices.tolist(), selected.visual_features
        )
    }
    features = history.visual_features.copy()
    changes: list[dict[str, Any]] = []
    for frame in shared_frames:
        features[target_lookup[frame]] = source_lookup[frame]
        changes.append(
            {
                "frame_index": frame,
                "operation": "replace_exact_time_identity",
                "source_track_id": int(selected.track_id),
            }
        )
    details.update(
        {
            "switch_point_fraction": switch_fraction,
            "switch_frame": switch_frame,
            "selected_source_track_id": int(selected.track_id),
            "selection_rule": (
                "maximum_exact_time_suffix_coverage_then_lowest_track_id"
            ),
            "exact_time_alignment": True,
            "achieved_replacement_fraction_of_target": float(
                len(shared_frames) / len(target_frames)
            ),
            "changes": changes,
        }
    )
    return PerturbationOutcome(
        replace(history, visual_features=features), _finalize_details(details)
    )


def _validate_outcome(original: History, outcome: PerturbationOutcome) -> None:
    perturbed = outcome.history
    if perturbed.history_id != original.history_id:
        raise ValueError("Perturbation changed the history identifier")
    if (
        perturbed.queries is not original.queries
        or perturbed.references is not original.references
    ):
        raise ValueError("Perturbation changed queries or state annotations")
    if not len(perturbed.frame_indices) or bool(
        np.any(np.diff(perturbed.frame_indices) <= 0)
    ):
        raise ValueError("Perturbed frames are empty or unordered")
    if perturbed.visual_features.shape[0] != len(perturbed.frame_indices):
        raise ValueError("Perturbed feature rows are misaligned")
    if perturbed.visibility_count.shape != perturbed.frame_indices.shape:
        raise ValueError("Perturbed visibility rows are misaligned")
    if not bool(np.isfinite(perturbed.visual_features).all()):
        raise ValueError("Perturbed features are non-finite")
    if not bool(
        np.allclose(
            np.linalg.norm(perturbed.visual_features, axis=1), 1.0, atol=2e-3
        )
    ):
        raise ValueError("Perturbed features are not unit normalized")
    changes = outcome.details.get("changes", [])
    if outcome.details.get("applied") != bool(changes):
        raise ValueError("Perturbation applied flag is inconsistent")
    if outcome.details.get("changes_sha256") != canonical_sha256(changes):
        raise ValueError("Perturbation change digest differs")


def apply_perturbation(
    history: History,
    distractors: Sequence[SevenHistoryDistractor],
    condition: Mapping[str, Any],
    *,
    replicate_seed: int,
) -> PerturbationOutcome:
    """Apply one selected condition without mutating any input array."""

    family = str(condition.get("family", ""))
    if family == "random_missing_frames":
        outcome = _random_missing(history, condition, replicate_seed)
    elif family == "contiguous_missing_frames":
        outcome = _contiguous_missing(history, condition, replicate_seed)
    elif family == "distractor_contamination":
        outcome = _distractor_contamination(
            history, distractors, condition, replicate_seed
        )
    elif family == "identity_switch":
        outcome = _identity_switch(history, distractors, condition, replicate_seed)
    else:
        raise ValueError(f"Unsupported perturbation family: {family}")
    outcome.details["num_target_observations_after"] = int(
        len(outcome.history.frame_indices)
    )
    outcome.details["target_support_after"] = [
        int(outcome.history.frame_indices[0]),
        int(outcome.history.frame_indices[-1]),
    ]
    _validate_outcome(history, outcome)
    return outcome
