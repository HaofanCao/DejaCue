"""Diagnostic analyses for experiment results.

The caller supplies arrays or JSON rows for the selected cohort. The functions
use inclusive frame intervals and aggregate descriptions within states, then
states within histories.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from numbers import Integral
from statistics import fmean
from typing import Any, Mapping, Sequence

import numpy as np

from ..robustness import temporal_iou


RECURRENCE_THRESHOLDS = (0.3, 0.5, 0.7)
SENSITIVITY_METRICS = (
    "state_macro_top1_tiou",
    "state_macro_r1_tiou_0.5",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mean(values: Sequence[float], *, label: str) -> float:
    _require(bool(values), f"Cannot average an empty population: {label}")
    array = np.asarray(values, dtype=np.float64)
    _require(bool(np.isfinite(array).all()), f"Non-finite value in {label}")
    return float(array.mean())


def _window(value: Sequence[int], *, label: str) -> tuple[int, int]:
    _require(
        not isinstance(value, (str, bytes, Mapping))
        and hasattr(value, "__len__")
        and hasattr(value, "__iter__"),
        f"{label} must be a two-endpoint sequence",
    )
    _require(len(value) == 2, f"{label} must contain two endpoints")
    start, end = value
    _require(
        isinstance(start, Integral)
        and not isinstance(start, bool)
        and isinstance(end, Integral)
        and not isinstance(end, bool),
        f"{label} endpoints must be integers",
    )
    _require(0 <= start <= end, f"{label} must be a non-negative inclusive interval")
    return int(start), int(end)


def _windows(values: Sequence[Sequence[int]], *, label: str) -> tuple[tuple[int, int], ...]:
    _require(
        not isinstance(values, (str, bytes, Mapping))
        and hasattr(values, "__len__")
        and hasattr(values, "__iter__"),
        f"{label} must be an interval sequence",
    )
    result = tuple(_window(value, label=f"{label}[{index}]") for index, value in enumerate(values))
    _require(bool(result), f"{label} must contain at least one interval")
    return result


def _best_reference_iou(window: tuple[int, int], references: Sequence[tuple[int, int]]) -> float:
    return max(temporal_iou(window, reference) for reference in references)


def _intersects(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return max(left[0], right[0]) <= min(left[1], right[1])


def cohen_kappa(pass_a: Sequence[str], pass_b: Sequence[str]) -> dict[str, float]:
    """Compute Cohen's kappa and its observed/chance agreement components."""

    left_values, right_values = list(pass_a), list(pass_b)
    _require(
        len(left_values) == len(right_values) and bool(left_values),
        "Annotation passes differ or are empty",
    )
    _require(
        all(
            isinstance(value, str) and value
            for value in (*left_values, *right_values)
        ),
        "Annotation labels must be non-empty strings",
    )
    observed = fmean(
        float(left == right)
        for left, right in zip(left_values, right_values, strict=True)
    )
    labels = sorted(set(left_values) | set(right_values))
    left_counts, right_counts = Counter(left_values), Counter(right_values)
    total = len(left_values)
    chance = sum(
        (left_counts[label] / total) * (right_counts[label] / total)
        for label in labels
    )
    # A single-category population has an undefined 0/0 form. Two identical
    # constant annotation passes are treated as perfect agreement.
    if math.isclose(chance, 1.0, rel_tol=0.0, abs_tol=1e-15):
        kappa = 1.0 if math.isclose(observed, 1.0, rel_tol=0.0, abs_tol=1e-15) else 0.0
    else:
        kappa = (observed - chance) / (1.0 - chance)
    return {
        "observed_agreement": float(observed),
        "chance_agreement": float(chance),
        "cohen_kappa": float(kappa),
    }


def compare_annotation_passes(pass_a: Sequence[str], pass_b: Sequence[str]) -> dict[str, Any]:
    """Compare two independent frame-label passes over the same population."""

    left_values, right_values = list(pass_a), list(pass_b)
    kappa = cohen_kappa(left_values, right_values)
    disagreement_count = sum(
        left != right
        for left, right in zip(left_values, right_values, strict=True)
    )
    exact_count = len(left_values) - disagreement_count
    return {
        "frame_count": len(left_values),
        "exact_agreement_count": exact_count,
        "disagreement_count": disagreement_count,
        "exact_agreement_rate": float(exact_count / len(left_values)),
        "kappa": kappa,
    }


def summarize_annotation_agreement(histories: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize selected-frame and stable-valid annotation agreement.

    Each input row must provide ``history_id``, ``pass_A``, ``pass_B``, and a
    Boolean ``stable_valid_mask``. The caller supplies the mask so this
    implementation never guesses which label names denote
    transition or unknown visibility.
    """

    _require(bool(histories), "Annotation agreement requires at least one history")
    history_rows: list[dict[str, Any]] = []
    pooled_a: list[str] = []
    pooled_b: list[str] = []
    pooled_stable_a: list[str] = []
    pooled_stable_b: list[str] = []
    seen: set[str] = set()
    for row in histories:
        history_id = str(row.get("history_id", ""))
        _require(bool(history_id) and history_id not in seen, "History identifiers must be unique")
        seen.add(history_id)
        pass_a = row.get("pass_A")
        pass_b = row.get("pass_B")
        stable = row.get("stable_valid_mask")
        _require(isinstance(pass_a, list) and isinstance(pass_b, list), f"Missing passes for {history_id}")
        _require(isinstance(stable, list), f"Missing stable-valid mask for {history_id}")
        _require(
            len(pass_a) == len(pass_b) == len(stable) and bool(pass_a),
            f"Annotation populations differ for {history_id}",
        )
        _require(all(isinstance(value, bool) for value in stable), f"Stable mask is not Boolean for {history_id}")
        stable_a = [label for label, keep in zip(pass_a, stable, strict=True) if keep]
        stable_b = [label for label, keep in zip(pass_b, stable, strict=True) if keep]
        _require(bool(stable_a), f"Stable-valid population is empty for {history_id}")
        history_rows.append(
            {
                "history_id": history_id,
                "selected_frames": compare_annotation_passes(pass_a, pass_b),
                "stable_valid_frames": compare_annotation_passes(stable_a, stable_b),
            }
        )
        pooled_a.extend(pass_a)
        pooled_b.extend(pass_b)
        pooled_stable_a.extend(stable_a)
        pooled_stable_b.extend(stable_b)

    history_rows.sort(key=lambda item: item["history_id"])
    return {
        "schema_version": 1,
        "kind": "deja_cue_annotation_agreement",
        "histories": history_rows,
        "reported": {
            "global_pooled": {
                "selected_frames": compare_annotation_passes(pooled_a, pooled_b),
                "stable_valid_frames": compare_annotation_passes(
                    pooled_stable_a, pooled_stable_b
                ),
            },
            "history_macro": {
                scope: _mean(
                    [float(row[scope]["exact_agreement_rate"]) for row in history_rows],
                    label=f"annotation/{scope}",
                )
                for scope in ("selected_frames", "stable_valid_frames")
            },
        },
    }


def _reference_index(rows: Sequence[Mapping[str, Any]], *, label: str) -> dict[tuple[str, str], tuple[tuple[int, int], ...]]:
    result: dict[tuple[str, str], tuple[tuple[int, int], ...]] = {}
    for index, row in enumerate(rows):
        _require(isinstance(row, Mapping), f"Malformed reference row at {label}[{index}]")
        history_id, state_id = str(row.get("history_id", "")), str(row.get("state_id", ""))
        _require(bool(history_id) and bool(state_id), f"Missing reference identity at {label}[{index}]")
        key = (history_id, state_id)
        _require(key not in result, f"Duplicate reference state at {label}/{history_id}/{state_id}")
        result[key] = _windows(row.get("windows", ()), label=f"{label}/{history_id}/{state_id}")
    _require(bool(result), f"Reference set is empty: {label}")
    return result


def _prediction_index(rows: Sequence[Mapping[str, Any]], *, label: str) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for index, row in enumerate(rows):
        _require(isinstance(row, Mapping), f"Malformed prediction row at {label}[{index}]")
        history_id = str(row.get("history_id", ""))
        query_id = str(row.get("query_id", ""))
        state_id = str(row.get("state_id", ""))
        _require(all((history_id, query_id, state_id)), f"Missing prediction identity at {label}[{index}]")
        key = (history_id, query_id)
        _require(key not in result, f"Duplicate prediction at {label}/{history_id}/{query_id}")
        result[key] = {
            "history_id": history_id,
            "query_id": query_id,
            "state_id": state_id,
            "window": _window(row.get("window", ()), label=f"{label}/{history_id}/{query_id}"),
        }
    _require(bool(result), f"Prediction set is empty: {label}")
    return result


def score_predictions_against_references(
    predictions: Sequence[Mapping[str, Any]],
    references: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Score fixed one-window predictions against one boundary set.

    Description-level scores are averaged within state, then state-level scores
    are averaged within history.  The final reported values are unweighted
    history means, matching the boundary-sensitivity calculation.
    """

    prediction_index = _prediction_index(predictions, label="predictions")
    reference_index = _reference_index(references, label="references")
    query_rows: list[dict[str, Any]] = []
    by_state: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for prediction in prediction_index.values():
        state_key = (prediction["history_id"], prediction["state_id"])
        _require(state_key in reference_index, f"Prediction has no reference state: {state_key}")
        overlap = _best_reference_iou(prediction["window"], reference_index[state_key])
        result = {
            **prediction,
            "window": list(prediction["window"]),
            "top1_tiou": overlap,
            "r1_tiou_0.5": float(overlap >= 0.5),
        }
        query_rows.append(result)
        by_state[state_key].append(result)
    _require(set(by_state) == set(reference_index), "Prediction and reference state sets differ")

    state_rows: list[dict[str, Any]] = []
    for (history_id, state_id), rows in sorted(by_state.items()):
        state_rows.append(
            {
                "history_id": history_id,
                "state_id": state_id,
                "num_descriptions": len(rows),
                "top1_tiou": _mean([float(row["top1_tiou"]) for row in rows], label="state Top-1 tIoU"),
                "r1_tiou_0.5": _mean([float(row["r1_tiou_0.5"]) for row in rows], label="state R@1"),
            }
        )
    by_history: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in state_rows:
        by_history[row["history_id"]].append(row)
    history_rows = [
        {
            "history_id": history_id,
            "num_states": len(rows),
            "state_macro_top1_tiou": _mean([float(row["top1_tiou"]) for row in rows], label="history Top-1 tIoU"),
            "state_macro_r1_tiou_0.5": _mean([float(row["r1_tiou_0.5"]) for row in rows], label="history R@1"),
        }
        for history_id, rows in sorted(by_history.items())
    ]
    return {
        "query_rows": sorted(query_rows, key=lambda row: (row["history_id"], row["query_id"])),
        "state_rows": state_rows,
        "history_rows": history_rows,
        "reported_history_macro": {
            metric: _mean([float(row[metric]) for row in history_rows], label=f"boundary/{metric}")
            for metric in SENSITIVITY_METRICS
        },
    }


def reference_boundary_sensitivity(
    predictions_by_condition: Mapping[str, Sequence[Mapping[str, Any]]],
    references_by_set: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    baseline_condition: str,
) -> dict[str, Any]:
    """Rescore unchanged condition predictions under multiple boundary sets."""

    _require(baseline_condition in predictions_by_condition, "Baseline condition is absent")
    _require(bool(references_by_set), "No boundary sets were supplied")
    prediction_keys: set[tuple[str, str, str]] | None = None
    normalized_predictions: dict[str, Sequence[Mapping[str, Any]]] = {}
    for condition, rows in sorted(predictions_by_condition.items()):
        indexed = _prediction_index(rows, label=f"predictions/{condition}")
        current_keys = {
            (row["history_id"], row["query_id"], row["state_id"])
            for row in indexed.values()
        }
        if prediction_keys is None:
            prediction_keys = current_keys
        else:
            _require(
                current_keys == prediction_keys,
                "Prediction set changes across conditions",
            )
        normalized_predictions[condition] = rows

    scores: dict[str, dict[str, Any]] = {}
    deltas: dict[str, dict[str, dict[str, float]]] = {}
    for reference_name, references in sorted(references_by_set.items()):
        condition_scores = {
            condition: score_predictions_against_references(rows, references)
            for condition, rows in normalized_predictions.items()
        }
        scores[reference_name] = condition_scores
        baseline = condition_scores[baseline_condition]["reported_history_macro"]
        deltas[reference_name] = {}
        for condition, result in condition_scores.items():
            if condition == baseline_condition:
                continue
            deltas[reference_name][f"{condition}_minus_{baseline_condition}"] = {
                metric: float(result["reported_history_macro"][metric] - baseline[metric])
                for metric in SENSITIVITY_METRICS
            }
    return {
        "schema_version": 1,
        "kind": "deja_cue_annotation_sensitivity",
        "scope": "unchanged_predictions_across_reference_sets",
        "scores": scores,
        "reported_coordinate_deltas": deltas,
    }


def contiguous_index_runs(frame_indices: Sequence[int]) -> tuple[tuple[int, int], ...]:
    """Return half-open array ranges for maximal consecutive frame runs."""

    frames = np.asarray(frame_indices, dtype=np.int64)
    _require(
        frames.ndim == 1 and len(frames) > 0 and bool(np.all(np.diff(frames) > 0)),
        "frame_indices must be a non-empty strictly increasing vector",
    )
    boundaries = np.flatnonzero(np.diff(frames) != 1) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(frames)]))
    return tuple((int(start), int(end)) for start, end in zip(starts, ends))


def _within_run_shift_plan(frame_indices: Sequence[int], *, seed: int) -> tuple[tuple[int, int, int], ...]:
    _require(
        isinstance(seed, Integral) and not isinstance(seed, bool),
        "Shift seed must be an integer",
    )
    seed = int(seed)
    rng = np.random.default_rng(seed)
    plan = []
    for left, right in contiguous_index_runs(frame_indices):
        length = right - left
        shift = 0 if length <= 1 else int(rng.integers(1, length))
        plan.append((left, right, shift))
    return tuple(plan)


def circular_shift_within_runs(
    values: Sequence[Any] | np.ndarray,
    frame_indices: Sequence[int],
    *,
    seed: int,
) -> np.ndarray:
    """Apply one independent non-identity circular shift inside each valid run."""

    array = np.asarray(values)
    _require(array.ndim >= 1, "Shift values must have a frame axis")
    _require(array.shape[0] == len(frame_indices), "Values and frame indices differ")
    shifted = array.copy()
    for left, right, offset in _within_run_shift_plan(frame_indices, seed=seed):
        if offset:
            shifted[left:right] = np.roll(array[left:right], shift=offset, axis=0)
    return shifted


def run_circular_shift_control(
    values: Sequence[Any] | np.ndarray,
    frame_indices: Sequence[int],
    *,
    seed: int,
) -> dict[str, Any]:
    """Return shifted values and details for each contiguous run."""

    frames = np.asarray(frame_indices, dtype=np.int64)
    plan = _within_run_shift_plan(frames, seed=seed)
    shifted = circular_shift_within_runs(values, frames, seed=seed)
    return {
        "schema_version": 1,
        "kind": "deja_cue_within_run_circular_shift",
        "seed": seed,
        "frame_indices": frames.tolist(),
        "shifted_values": shifted.tolist(),
        "runs": [
            {
                "array_start": left,
                "array_end_exclusive": right,
                "frame_start": int(frames[left]),
                "frame_end": int(frames[right - 1]),
                "shift": offset,
            }
            for left, right, offset in plan
        ],
    }


def permute_state_assignments(
    query_rows: Sequence[Mapping[str, Any]],
    source_to_target: Mapping[str, str],
    *,
    require_derangement: bool = True,
) -> list[dict[str, Any]]:
    """Exchange sibling-state assignments while preserving every query payload.

    ``source_to_target`` maps each query's original state to its assigned state.
    It must be a bijection over the complete state set. The default control
    uses a derangement, so fixed points are rejected.
    """

    _require(bool(query_rows), "State permutation requires query rows")
    states = {str(row.get("state_id", "")) for row in query_rows}
    _require("" not in states, "Every query row needs a state_id")
    mapping = {str(source): str(target) for source, target in source_to_target.items()}
    _require(
        set(mapping) == states and set(mapping.values()) == states,
        "State mapping is not a permutation of the provided states",
    )
    if require_derangement:
        _require(all(source != target for source, target in mapping.items()), "State control requires a derangement")
    result = []
    for row in query_rows:
        copied = dict(row)
        copied["state_id"] = mapping[str(row["state_id"])]
        result.append(copied)
    return result


def _candidate_rows(values: Sequence[Mapping[str, Any]], *, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        _require(isinstance(value, Mapping), f"Malformed candidate at {label}[{index}]")
        window = _window((value.get("start"), value.get("end")), label=f"{label}[{index}]")
        score = float(value.get("score", math.nan))
        _require(math.isfinite(score), f"Candidate score is non-finite at {label}[{index}]")
        rows.append({"start": window[0], "end": window[1], "score": score})
    return rows


def _best_candidate(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    return max(rows, key=lambda row: float(row["score"]), default=None)


def hard_negative_margins(
    target_candidates: Sequence[Mapping[str, Any]],
    queried_references: Sequence[Sequence[int]],
    sibling_references: Sequence[Sequence[int]],
    auxiliary_tracks: Sequence[Mapping[str, Any]],
    *,
    positive_tiou: float = 0.5,
) -> dict[str, Any]:
    """Compute positive, sibling, and joint hard-negative margins.

    Auxiliary-track rows contain ``track_id``, ``candidates``, and the queried
    state's ``co_visible_intervals``.  A joint negative is eligible only when
    its inclusive window does not intersect any such interval for its track.
    Missing eligible sets produce ``None`` instead of an invented score.
    """

    _require(0.0 < positive_tiou <= 1.0, "positive_tiou must lie in (0, 1]")
    target = _candidate_rows(target_candidates, label="target_candidates")
    queried = _windows(queried_references, label="queried_references")
    siblings = _windows(sibling_references, label="sibling_references")
    positives = [
        row
        for row in target
        if _best_reference_iou((row["start"], row["end"]), queried) >= positive_tiou
    ]
    sibling_negatives = [
        row
        for row in target
        if _best_reference_iou((row["start"], row["end"]), siblings) >= positive_tiou
        and not any(_intersects((row["start"], row["end"]), reference) for reference in queried)
    ]

    joint_negatives: list[dict[str, Any]] = []
    seen_tracks: set[str] = set()
    for track_index, track in enumerate(auxiliary_tracks):
        track_id = str(track.get("track_id", ""))
        _require(bool(track_id) and track_id not in seen_tracks, "Auxiliary track identifiers must be unique")
        seen_tracks.add(track_id)
        co_visible = tuple(
            _window(value, label=f"auxiliary_tracks[{track_index}]/co_visible_intervals")
            for value in track.get("co_visible_intervals", ())
        )
        candidates = _candidate_rows(
            track.get("candidates", ()), label=f"auxiliary_tracks[{track_index}]/candidates"
        )
        for candidate in candidates:
            window = (candidate["start"], candidate["end"])
            if not any(_intersects(window, interval) for interval in co_visible):
                joint_negatives.append({**candidate, "track_id": track_id})

    positive = _best_candidate(positives)
    sibling = _best_candidate(sibling_negatives)
    joint = _best_candidate(joint_negatives)
    positive_score = float(positive["score"]) if positive is not None else None
    sibling_score = float(sibling["score"]) if sibling is not None else None
    joint_score = float(joint["score"]) if joint is not None else None
    return {
        "positive": dict(positive) if positive is not None else None,
        "sibling_negative": dict(sibling) if sibling is not None else None,
        "joint_negative": dict(joint) if joint is not None else None,
        "temporal_margin": (
            float(positive_score - sibling_score)
            if positive_score is not None and sibling_score is not None
            else None
        ),
        "double_margin": (
            float(positive_score - joint_score)
            if positive_score is not None and joint_score is not None
            else None
        ),
    }


def _paired_rows(rows: Sequence[Mapping[str, Any]], *, label: str) -> dict[tuple[str, str], Mapping[str, Any]]:
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        key = (str(row.get("history_id", "")), str(row.get("query_id", "")))
        _require(all(key) and key not in result, f"Invalid paired identity at {label}[{index}]")
        result[key] = row
    _require(bool(result), f"Paired result is empty: {label}")
    return result


def paired_margin_summary(
    baseline_rows: Sequence[Mapping[str, Any]],
    treatment_rows: Sequence[Mapping[str, Any]],
    *,
    margin_field: str,
    seed: int = 3407,
    bootstrap_resamples: int = 10_000,
) -> dict[str, Any]:
    """Aggregate within-query margin changes by history median and bootstrap."""

    _require(bool(margin_field), "A margin field is required")
    _require(bootstrap_resamples > 0, "Bootstrap resamples must be positive")
    baseline = _paired_rows(baseline_rows, label="baseline")
    treatment = _paired_rows(treatment_rows, label="treatment")
    _require(set(baseline) == set(treatment), "Baseline and treatment query sets differ")
    by_history: dict[str, list[float]] = defaultdict(list)
    missing_pairs = 0
    positive_pairs = 0
    for key in sorted(baseline):
        left, right = baseline[key].get(margin_field), treatment[key].get(margin_field)
        if left is None or right is None:
            missing_pairs += 1
            continue
        left_value, right_value = float(left), float(right)
        _require(math.isfinite(left_value) and math.isfinite(right_value), f"Non-finite paired margin at {key}")
        delta = right_value - left_value
        by_history[key[0]].append(delta)
        positive_pairs += int(delta > 0.0)
    _require(bool(by_history), f"No paired numeric values for {margin_field}")
    _require(all(values for values in by_history.values()), "A history has no paired margins")
    history_rows = [
        {"history_id": history_id, "delta": float(np.median(np.asarray(values, dtype=np.float64)))}
        for history_id, values in sorted(by_history.items())
    ]
    history_values = np.asarray([row["delta"] for row in history_rows], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = history_values[
        rng.integers(0, len(history_values), size=(bootstrap_resamples, len(history_values)))
    ].mean(axis=1)
    return {
        "schema_version": 1,
        "kind": "deja_cue_paired_margin_summary",
        "field": margin_field,
        "num_pairs": int(sum(len(values) for values in by_history.values())),
        "num_missing_pairs": missing_pairs,
        "num_positive_pairs": positive_pairs,
        "history_median_deltas": history_rows,
        "reported_history_macro_delta": float(history_values.mean()),
        "reported_history_bootstrap_95ci": np.quantile(draws, [0.025, 0.975]).tolist(),
        "bootstrap": {
            "unit": "history",
            "num_resamples": bootstrap_resamples,
            "seed": seed,
        },
    }


def paraphrase_window_consistency(
    prediction_rows: Sequence[Mapping[str, Any]],
    *,
    component_key: str = "source_component_id",
) -> dict[str, Any]:
    """Measure same-state agreement between exactly two descriptions."""

    _require(bool(prediction_rows), "Paraphrase consistency requires predictions")
    grouped: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for index, row in enumerate(prediction_rows):
        component = str(row.get(component_key, ""))
        state_id = str(row.get("state_id", ""))
        _require(bool(component) and bool(state_id), f"Missing paraphrase identity at row {index}")
        grouped[(component, state_id)].append(
            _window(row.get("window", ()), label=f"paraphrase[{index}]")
        )
    state_rows: list[dict[str, Any]] = []
    for (component, state_id), windows in sorted(grouped.items()):
        _require(len(windows) == 2, f"State must have exactly two descriptions: {component}/{state_id}")
        left, right = windows
        state_rows.append(
            {
                component_key: component,
                "state_id": state_id,
                "paraphrase_window_tiou": temporal_iou(left, right),
                "paraphrase_exact_agreement": float(left == right),
            }
        )
    by_component: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in state_rows:
        by_component[row[component_key]].append(row)
    component_rows = [
        {
            component_key: component,
            "paraphrase_window_tiou": _mean(
                [float(row["paraphrase_window_tiou"]) for row in rows],
                label="component paraphrase tIoU",
            ),
            "paraphrase_exact_agreement": _mean(
                [float(row["paraphrase_exact_agreement"]) for row in rows],
                label="component paraphrase exact agreement",
            ),
        }
        for component, rows in sorted(by_component.items())
    ]
    return {
        "schema_version": 1,
        "kind": "deja_cue_paraphrase_consistency",
        "aggregation": "unweighted_source_component_mean",
        "state_rows": state_rows,
        "source_components": component_rows,
        "reported_source_component_macro": {
            metric: _mean([float(row[metric]) for row in component_rows], label=f"paraphrase/{metric}")
            for metric in ("paraphrase_window_tiou", "paraphrase_exact_agreement")
        },
    }


def average_precision(
    predictions: Sequence[Mapping[str, Any]],
    references: Sequence[Sequence[int]],
    *,
    threshold: float,
) -> float:
    """Compute one-to-one ranked temporal AP for one recurrence query."""

    _require(0.0 < threshold <= 1.0, "AP threshold must lie in (0, 1]")
    predicted = _candidate_rows(predictions, label="recurrence_predictions")
    reference_windows = _windows(references, label="recurrence_references")
    ordered = sorted(predicted, key=lambda row: float(row["score"]), reverse=True)
    matched: set[int] = set()
    precision_sum = 0.0
    true_positives = 0
    for rank, prediction in enumerate(ordered, start=1):
        window = (prediction["start"], prediction["end"])
        candidates = [
            (temporal_iou(window, reference), index)
            for index, reference in enumerate(reference_windows)
            if index not in matched
        ]
        best_iou, best_index = max(candidates, default=(0.0, -1))
        if best_iou >= threshold:
            matched.add(best_index)
            true_positives += 1
            precision_sum += true_positives / rank
    return float(precision_sum / len(reference_windows))


def recurrence_metrics(
    predictions: Sequence[Mapping[str, Any]],
    references: Sequence[Sequence[int]],
) -> dict[str, float]:
    """Compute recurrence mAP, recall, precision, and count error."""

    predicted = _candidate_rows(predictions, label="recurrence_predictions")
    reference_windows = _windows(references, label="recurrence_references")
    aps = [
        average_precision(predicted, reference_windows, threshold=threshold)
        for threshold in RECURRENCE_THRESHOLDS
    ]
    matched: set[int] = set()
    true_positives = 0
    for prediction in sorted(predicted, key=lambda row: float(row["score"]), reverse=True):
        window = (prediction["start"], prediction["end"])
        candidates = [
            (temporal_iou(window, reference), index)
            for index, reference in enumerate(reference_windows)
            if index not in matched
        ]
        best_iou, best_index = max(candidates, default=(0.0, -1))
        if best_iou >= 0.5:
            matched.add(best_index)
            true_positives += 1
    return {
        "multi_window_map_tiou_0.3_0.5_0.7": _mean(aps, label="recurrence AP thresholds"),
        "multi_window_recall_tiou_0.5": float(true_positives / len(reference_windows)),
        "multi_window_precision_tiou_0.5": (
            float(true_positives / len(predicted)) if predicted else 0.0
        ),
        "episode_count_absolute_error": float(abs(len(predicted) - len(reference_windows))),
    }


def summarize_recurrence_conditions(
    queries: Sequence[Mapping[str, Any]],
    references: Mapping[str, Sequence[Sequence[int]]],
    predictions_by_condition: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, dict[str, float]]:
    """Aggregate recurrence metrics over the provided queries."""

    query_states: dict[str, str] = {}
    for index, query in enumerate(queries):
        query_id, state_id = str(query.get("query_id", "")), str(query.get("state_id", ""))
        _require(bool(query_id) and bool(state_id) and query_id not in query_states, f"Invalid recurrence query at row {index}")
        _require(state_id in references, f"Recurrence state has no references: {state_id}")
        query_states[query_id] = state_id
    _require(bool(query_states), "Recurrence query set is empty")

    result: dict[str, dict[str, float]] = {}
    for condition, rows in sorted(predictions_by_condition.items()):
        by_query: dict[str, Sequence[Mapping[str, Any]]] = {}
        for index, row in enumerate(rows):
            query_id = str(row.get("query_id", ""))
            _require(query_id in query_states and query_id not in by_query, f"Invalid recurrence prediction at {condition}[{index}]")
            predicted = row.get("predicted_windows")
            _require(isinstance(predicted, list), f"Missing predicted windows at {condition}/{query_id}")
            by_query[query_id] = predicted
        _require(
            set(by_query) == set(query_states),
            f"Recurrence query set differs for {condition}",
        )
        per_query = [
            recurrence_metrics(by_query[query_id], references[state_id])
            for query_id, state_id in sorted(query_states.items())
        ]
        result[condition] = {
            "query_macro_multi_window_map_tiou_0.3_0.5_0.7": _mean(
                [row["multi_window_map_tiou_0.3_0.5_0.7"] for row in per_query],
                label="query recurrence mAP",
            ),
            "query_macro_multi_window_recall_tiou_0.5": _mean(
                [row["multi_window_recall_tiou_0.5"] for row in per_query],
                label="query recurrence recall",
            ),
            "query_macro_multi_window_precision_tiou_0.5": _mean(
                [row["multi_window_precision_tiou_0.5"] for row in per_query],
                label="query recurrence precision",
            ),
            "query_macro_episode_count_absolute_error": _mean(
                [row["episode_count_absolute_error"] for row in per_query],
                label="query recurrence count error",
            ),
        }
    _require(bool(result), "No recurrence conditions were supplied")
    return result
