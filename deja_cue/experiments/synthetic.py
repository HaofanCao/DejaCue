"""Synthetic duration/grid experiment.

The default configuration describes the published v3 design. The generator
returns results for all 270 simulation conditions.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


FINAL_CONFIG_FIELDS = (
    "seed",
    "sequence_length",
    "boundary_margin",
    "true_durations",
    "grid_bases",
    "grid_ratios",
    "maximum_candidate_duration",
    "candidate_grid_implementation",
    "signal_means",
    "ar1_correlations",
    "trials_per_noise_cell",
    "smoothing_kernel_size",
    "smoothing_normalization",
    "frame_standardization",
    "mad_floor",
    "normalizations",
    "design_cell_bootstrap_resamples",
    "uncertainty_method",
    "bootstrap_seed",
)

DEFAULT_CONFIG = {
    "schema_version": 3,
    "kind": "deja_cue_synthetic_duration_grid",
    "status": "published_v3",
    "seed": 3407,
    "sequence_length": 192,
    "boundary_margin": 24,
    "true_durations": [8, 12, 20, 32, 48],
    "grid_bases": [4, 5, 6],
    "grid_ratios": [1.25, 1.5, 2.0],
    "maximum_candidate_duration": 72,
    "candidate_grid_implementation": (
        "production_iterative_integer_rounding_with_upper_cap"
    ),
    "signal_means": [0.5, 1.0, 1.5],
    "ar1_correlations": [0.0, 0.6],
    "trials_per_noise_cell": 400,
    "smoothing_kernel_size": 3,
    "smoothing_normalization": "local_squared_kernel_norm",
    "frame_standardization": "median_and_normalized_mad",
    "mad_floor": 0.001,
    "normalizations": ["sum", "mean", "sqrt_valid_count"],
    "metrics": ["r1_tiou_0.5", "top1_tiou", "absolute_duration_error"],
    "design_cell_bootstrap_resamples": 10000,
    "uncertainty_method": (
        "stratified_paired_trial_block_bootstrap_within_30_noise_cells"
    ),
    "bootstrap_seed": 3407,
}


def canonical_json_bytes(value: Any, *, pretty: bool = True) -> bytes:
    """Encode finite experiment output deterministically as ASCII JSON."""

    if pretty:
        text = json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
    else:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    return (text + "\n").encode("ascii")


def write_synthetic_result(payload: Mapping[str, Any], path: Path) -> None:
    """Write a result with stable canonical JSON encoding."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))


def robust_standardize(values: np.ndarray, floor: float) -> np.ndarray:
    """Apply median and normalized-MAD calibration."""

    center = float(np.median(values))
    scale = 1.4826 * float(np.median(np.abs(values - center)))
    return (values - center) / max(scale, floor)


def smooth_scores(values: np.ndarray, kernel_size: int) -> np.ndarray:
    """Smooth scores with the squared-kernel normalization."""

    kernel = np.ones(kernel_size, dtype=np.float64)
    numerator = np.convolve(values, kernel, mode="same")
    squared_norm = np.convolve(
        np.ones_like(values), np.square(kernel), mode="same"
    )
    return numerator / np.sqrt(np.maximum(squared_norm, 1e-12))


def geometric_duration_grid(base: int, ratio: float, maximum: int) -> list[int]:
    """Construct an iterative integer grid with an upper cap."""

    if base <= 0 or ratio <= 1.0 or maximum < base:
        raise ValueError("Invalid geometric-grid parameters")
    values = [int(base)]
    while values[-1] < maximum:
        next_value = max(values[-1] + 1, int(round(values[-1] * ratio)))
        values.append(min(next_value, int(maximum)))
    return values


def select_window(
    evidence: np.ndarray,
    durations: Sequence[int],
    normalization: str,
) -> tuple[int, int, float]:
    """Select one inclusive window with deterministic tie rules."""

    prefix = np.concatenate((np.zeros(1), np.cumsum(evidence, dtype=np.float64)))
    best: tuple[float, int, int] | None = None
    for duration in durations:
        sums = prefix[duration:] - prefix[:-duration]
        if normalization == "sum":
            scores = sums
        elif normalization == "mean":
            scores = sums / duration
        elif normalization == "sqrt_valid_count":
            scores = sums / math.sqrt(duration)
        else:
            raise ValueError(f"Unknown normalization: {normalization}")
        local_start = int(np.argmax(scores))
        row = (float(scores[local_start]), local_start, local_start + duration - 1)
        if best is None or (row[0], -row[1], -row[2]) > (
            best[0],
            -best[1],
            -best[2],
        ):
            best = row
    if best is None:
        raise ValueError("No candidate window was generated")
    return best[1], best[2], best[0]


def temporal_iou(left: tuple[int, int], right: tuple[int, int]) -> float:
    """Return inclusive temporal intersection over union."""

    intersection = max(0, min(left[1], right[1]) - max(left[0], right[0]) + 1)
    union = (left[1] - left[0] + 1) + (right[1] - right[0] + 1) - intersection
    return float(intersection / union)


def ar1_noise(
    length: int, correlation: float, rng: np.random.Generator
) -> np.ndarray:
    """Draw a stationary unit-variance AR(1) noise sequence."""

    innovations = rng.normal(size=length)
    values = np.empty(length, dtype=np.float64)
    values[0] = innovations[0]
    scale = math.sqrt(max(1.0 - correlation * correlation, 0.0))
    for index in range(1, length):
        values[index] = correlation * values[index - 1] + scale * innovations[index]
    return values


def _stratified_paired_trial_bootstrap(
    grouped_deltas: Sequence[np.ndarray],
    *,
    seed: int,
    resamples: int,
) -> list[float]:
    if not grouped_deltas or resamples <= 0:
        raise ValueError("Grouped paired bootstrap requires data and resamples")
    rng = np.random.default_rng(seed)
    draws = np.zeros(resamples, dtype=np.float64)
    for values in grouped_deltas:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 1 or not len(values) or not bool(np.isfinite(values).all()):
            raise ValueError("Each bootstrap stratum must be a finite vector")
        indices = rng.integers(0, len(values), size=(resamples, len(values)))
        draws += values[indices].mean(axis=1)
    draws /= len(grouped_deltas)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def _validate_config(
    config: Mapping[str, Any], *, require_default_config: bool
) -> None:
    if config.get("schema_version") != 3:
        raise ValueError("Synthetic duration experiment requires schema version 3")
    if config.get("kind") != "deja_cue_synthetic_duration_grid":
        raise ValueError("Unexpected synthetic duration config kind")
    if config.get("status") != "published_v3":
        raise ValueError("Synthetic duration config must use the published v3 settings")
    if config.get("candidate_grid_implementation") != (
        "production_iterative_integer_rounding_with_upper_cap"
    ):
        raise ValueError("Unsupported candidate grid implementation")
    if config.get("smoothing_normalization") != "local_squared_kernel_norm":
        raise ValueError("Unsupported smoothing normalization")
    if config.get("frame_standardization") != "median_and_normalized_mad":
        raise ValueError("Unsupported frame standardization")
    if config.get("normalizations") != ["sum", "mean", "sqrt_valid_count"]:
        raise ValueError("Unsupported normalization set")
    if config.get("uncertainty_method") != (
        "stratified_paired_trial_block_bootstrap_within_30_noise_cells"
    ):
        raise ValueError("Unsupported uncertainty method")
    if require_default_config:
        mismatches = [
            key for key, value in DEFAULT_CONFIG.items() if config.get(key) != value
        ]
        if mismatches:
            raise ValueError(
                "Configuration differs from the published v3 settings: "
                + ", ".join(mismatches)
            )

    required_sequences = (
        "true_durations",
        "grid_bases",
        "grid_ratios",
        "signal_means",
        "ar1_correlations",
    )
    if any(not config.get(field) for field in required_sequences):
        raise ValueError("Synthetic design axes must be non-empty")
    length = int(config["sequence_length"])
    margin = int(config["boundary_margin"])
    durations = [int(value) for value in config["true_durations"]]
    if length <= 0 or margin < 0 or max(durations) > length - 2 * margin:
        raise ValueError("Sequence length, margin, and true durations are inconsistent")
    if any(int(value) <= 0 for value in config["grid_bases"]):
        raise ValueError("Grid bases must be positive")
    if any(float(value) <= 1.0 for value in config["grid_ratios"]):
        raise ValueError("Grid ratios must exceed one")
    if int(config["maximum_candidate_duration"]) > length:
        raise ValueError("Maximum candidate duration exceeds the sequence")
    if any(abs(float(value)) >= 1.0 for value in config["ar1_correlations"]):
        raise ValueError("AR(1) correlations must lie strictly inside (-1, 1)")
    kernel = int(config["smoothing_kernel_size"])
    if kernel <= 0 or kernel % 2 == 0:
        raise ValueError("Smoothing kernel size must be a positive odd integer")
    if int(config["trials_per_noise_cell"]) <= 0:
        raise ValueError("Trials per noise cell must be positive")
    if int(config["design_cell_bootstrap_resamples"]) <= 0:
        raise ValueError("Bootstrap resamples must be positive")


def _length_match(
    true_duration: int, candidates: Sequence[int]
) -> dict[str, float | int]:
    best = max(
        candidates,
        key=lambda value: (
            min(true_duration, value) / max(true_duration, value),
            -abs(value - true_duration),
            -value,
        ),
    )
    return {
        "nearest_candidate_duration": int(best),
        "nearest_length_tiou": float(
            min(true_duration, best) / max(true_duration, best)
        ),
        "absolute_log_length_mismatch": float(abs(math.log(best / true_duration))),
    }


def _summarize(values: Mapping[str, list[float]]) -> dict[str, float | int]:
    return {
        "trials": len(values["tiou"]),
        "top1_tiou": float(np.mean(values["tiou"])),
        "r1_tiou_0.5": float(np.mean(values["recall"])),
        "mean_absolute_duration_error": float(np.mean(values["duration_error"])),
    }


def run_synthetic_duration(
    config: Mapping[str, Any],
    *,
    require_default_config: bool = True,
) -> dict[str, Any]:
    """Run the v3 paired synthetic study and return its result records.

    Set ``require_default_config=False`` for smaller custom designs. The
    equations, candidate construction, pairing, and uncertainty procedure stay
    unchanged in reduced mode.
    """

    _validate_config(config, require_default_config=require_default_config)
    rng = np.random.default_rng(int(config["seed"]))
    normalizations = [str(value) for value in config["normalizations"]]
    grids = {
        (int(base), float(ratio)): geometric_duration_grid(
            int(base), float(ratio), int(config["maximum_candidate_duration"])
        )
        for base in config["grid_bases"]
        for ratio in config["grid_ratios"]
    }
    cell_values: dict[
        tuple[int, float, float, float, int, str], dict[str, list[float]]
    ] = {}
    for base, ratio in grids:
        for correlation in config["ar1_correlations"]:
            for signal in config["signal_means"]:
                for duration in config["true_durations"]:
                    for method in normalizations:
                        cell_values[
                            (
                                base,
                                ratio,
                                float(correlation),
                                float(signal),
                                int(duration),
                                method,
                            )
                        ] = {"tiou": [], "recall": [], "duration_error": []}

    length = int(config["sequence_length"])
    margin = int(config["boundary_margin"])
    for correlation in config["ar1_correlations"]:
        for signal in config["signal_means"]:
            for raw_duration in config["true_durations"]:
                duration = int(raw_duration)
                maximum_start = length - margin - duration
                for _ in range(int(config["trials_per_noise_cell"])):
                    start = int(rng.integers(margin, maximum_start + 1))
                    end = start + duration - 1
                    evidence = ar1_noise(length, float(correlation), rng)
                    evidence[start : end + 1] += float(signal)
                    evidence = smooth_scores(
                        robust_standardize(evidence, float(config["mad_floor"])),
                        int(config["smoothing_kernel_size"]),
                    )
                    for (base, ratio), candidates in grids.items():
                        for method in normalizations:
                            predicted_start, predicted_end, _ = select_window(
                                evidence, candidates, method
                            )
                            overlap = temporal_iou(
                                (predicted_start, predicted_end), (start, end)
                            )
                            values = cell_values[
                                (
                                    base,
                                    ratio,
                                    float(correlation),
                                    float(signal),
                                    duration,
                                    method,
                                )
                            ]
                            values["tiou"].append(overlap)
                            values["recall"].append(float(overlap >= 0.5))
                            values["duration_error"].append(
                                abs((predicted_end - predicted_start + 1) - duration)
                            )

    full_cells: list[dict[str, Any]] = []
    for base, ratio, correlation, signal, duration in sorted(
        {key[:-1] for key in cell_values}
    ):
        candidates = grids[(base, ratio)]
        full_cells.append(
            {
                "grid_base": base,
                "grid_ratio": ratio,
                "candidate_durations": candidates,
                "ar1_correlation": correlation,
                "signal_mean": signal,
                "true_duration": duration,
                **_length_match(duration, candidates),
                "methods": {
                    method: _summarize(
                        cell_values[
                            (base, ratio, correlation, signal, duration, method)
                        ]
                    )
                    for method in normalizations
                },
            }
        )

    by_method: dict[str, dict[str, list[float]]] = {
        method: defaultdict(list) for method in normalizations
    }
    metrics = (
        "top1_tiou",
        "r1_tiou_0.5",
        "mean_absolute_duration_error",
    )
    for row in full_cells:
        for method in normalizations:
            for metric in metrics:
                by_method[method][metric].append(float(row["methods"][method][metric]))
    equal_cell_summary = {
        method: {
            "design_cells": len(full_cells),
            **{metric: float(np.mean(values)) for metric, values in values.items()},
        }
        for method, values in by_method.items()
    }

    paired: dict[str, dict[str, Any]] = {}
    for baseline in ("sum", "mean"):
        paired_rows: dict[str, Any] = {}
        for metric, raw_metric in (("top1_tiou", "tiou"), ("r1_tiou_0.5", "recall")):
            deltas = np.asarray(by_method["sqrt_valid_count"][metric]) - np.asarray(
                by_method[baseline][metric]
            )
            grouped_trial_deltas = []
            for correlation in config["ar1_correlations"]:
                for signal in config["signal_means"]:
                    for duration in config["true_durations"]:
                        sqrt_trials = np.mean(
                            [
                                np.asarray(
                                    cell_values[
                                        (
                                            base,
                                            ratio,
                                            float(correlation),
                                            float(signal),
                                            int(duration),
                                            "sqrt_valid_count",
                                        )
                                    ][raw_metric],
                                    dtype=np.float64,
                                )
                                for base, ratio in grids
                            ],
                            axis=0,
                        )
                        baseline_trials = np.mean(
                            [
                                np.asarray(
                                    cell_values[
                                        (
                                            base,
                                            ratio,
                                            float(correlation),
                                            float(signal),
                                            int(duration),
                                            baseline,
                                        )
                                    ][raw_metric],
                                    dtype=np.float64,
                                )
                                for base, ratio in grids
                            ],
                            axis=0,
                        )
                        grouped_trial_deltas.append(sqrt_trials - baseline_trials)
            paired_rows[f"{metric}_delta"] = float(deltas.mean())
            paired_rows[
                f"{metric}_conditional_stratified_paired_trial_bootstrap_95ci"
            ] = _stratified_paired_trial_bootstrap(
                grouped_trial_deltas,
                seed=int(config["bootstrap_seed"]),
                resamples=int(config["design_cell_bootstrap_resamples"]),
            )
            paired_rows[f"{metric}_paired_noise_sequence_blocks"] = int(
                sum(len(values) for values in grouped_trial_deltas)
            )
            paired_rows[f"{metric}_cells_sqrt_better"] = int((deltas > 0).sum())
            paired_rows[f"{metric}_cells_tied"] = int((deltas == 0).sum())
            paired_rows[f"{metric}_cells_baseline_better"] = int((deltas < 0).sum())
        paired[f"sqrt_valid_count_minus_{baseline}"] = paired_rows

    by_duration: dict[str, dict[str, dict[str, float]]] = {}
    for value in sorted({row["true_duration"] for row in full_cells}):
        selected = [row for row in full_cells if row["true_duration"] == value]
        by_duration[str(value)] = {
            method: {
                metric: float(
                    np.mean([row["methods"][method][metric] for row in selected])
                )
                for metric in ("top1_tiou", "r1_tiou_0.5")
            }
            for method in normalizations
        }

    result_cells = [
        {
            key: value
            for key, value in row.items()
            if key
            not in {
                "candidate_durations",
                "nearest_candidate_duration",
                "nearest_length_tiou",
                "absolute_log_length_mismatch",
            }
        }
        for row in full_cells
    ]
    return {
        "schema_version": 1,
        "kind": "deja_cue_synthetic_duration_evidence",
        "design": {field: config[field] for field in FINAL_CONFIG_FIELDS},
        "cells": result_cells,
        "reported_equal_cell_summary": equal_cell_summary,
        "reported_paired_comparisons": paired,
        "reported_by_true_duration": by_duration,
    }
