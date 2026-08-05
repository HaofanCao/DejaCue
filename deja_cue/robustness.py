"""Recalculate the bundled robustness and diagnostic evidence summaries.

The JSON assets retain the lowest practical aggregation level for each study:
simulation cells, perturbation replicate-by-history rows, temporal windows,
annotation counts, or history-level paired effects. These functions recalculate
the reported paper values from those rows and enforce the documented study
designs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REFERENCE_DIR = Path("data") / "reference" / "robustness"
SYNTHETIC_METRICS = (
    "r1_tiou_0.5",
    "top1_tiou",
    "mean_absolute_duration_error",
)
TRACKING_METRICS = (
    "state_macro_target_top1_tiou",
    "state_macro_target_r1_tiou_0.5",
    "query_macro_joint_identity_accuracy",
)
PARAPHRASE_METRICS = (
    "paraphrase_window_tiou",
    "paraphrase_exact_agreement",
)


def _root(root: Path | None) -> Path:
    return Path(root) if root is not None else Path(__file__).resolve().parents[1]


def _read(name: str, root: Path) -> dict[str, Any]:
    path = root / REFERENCE_DIR / name
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _close(left: float, right: float, *, tolerance: float = 1e-12) -> bool:
    return bool(np.isclose(float(left), float(right), atol=tolerance, rtol=0.0))


def _mean(values: Sequence[float]) -> float:
    _require(len(values) > 0, "Cannot average an empty sequence")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _validate_finite(value: Any, *, label: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _validate_finite(child, label=f"{label}/{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_finite(child, label=f"{label}/{index}")
    elif isinstance(value, float):
        _require(math.isfinite(value), f"Non-finite value at {label}")


def temporal_iou(left: Sequence[int], right: Sequence[int]) -> float:
    """Return inclusive temporal intersection-over-union for two windows."""

    _require(len(left) == 2 and len(right) == 2, "Temporal windows need two endpoints")
    left_start, left_end = int(left[0]), int(left[1])
    right_start, right_end = int(right[0]), int(right[1])
    _require(left_end >= left_start and right_end >= right_start, "Window endpoints differ")
    intersection = max(0, min(left_end, right_end) - max(left_start, right_start) + 1)
    union = (left_end - left_start + 1) + (right_end - right_start + 1) - intersection
    return float(intersection / union)


def _average_precision(
    predictions: Sequence[Mapping[str, Any]],
    references: Sequence[Sequence[int]],
    *,
    threshold: float,
) -> float:
    """Compute one-to-one ranked AP for a single recurrence query."""

    if not references:
        return 0.0
    ordered = sorted(predictions, key=lambda row: float(row["score"]), reverse=True)
    matched: set[int] = set()
    precision_sum = 0.0
    true_positives = 0
    for rank, prediction in enumerate(ordered, start=1):
        window = (int(prediction["start"]), int(prediction["end"]))
        candidates = [
            (temporal_iou(window, reference), index)
            for index, reference in enumerate(references)
            if index not in matched
        ]
        best_iou, best_index = max(candidates, default=(0.0, -1))
        if best_iou >= threshold:
            matched.add(best_index)
            true_positives += 1
            precision_sum += true_positives / rank
    return float(precision_sum / len(references))


def _recurrence_metrics(
    predictions: Sequence[Mapping[str, Any]],
    references: Sequence[Sequence[int]],
) -> dict[str, float]:
    thresholds = (0.3, 0.5, 0.7)
    average_precisions = [
        _average_precision(predictions, references, threshold=threshold)
        for threshold in thresholds
    ]
    matches_at_half = 0
    matched: set[int] = set()
    for prediction in sorted(
        predictions, key=lambda row: float(row["score"]), reverse=True
    ):
        window = (int(prediction["start"]), int(prediction["end"]))
        candidates = [
            (temporal_iou(window, reference), index)
            for index, reference in enumerate(references)
            if index not in matched
        ]
        best_iou, best_index = max(candidates, default=(0.0, -1))
        if best_iou >= 0.5:
            matched.add(best_index)
            matches_at_half += 1
    return {
        "multi_window_map_tiou_0.3_0.5_0.7": _mean(average_precisions),
        "multi_window_recall_tiou_0.5": float(matches_at_half / len(references)),
        "multi_window_precision_tiou_0.5": (
            float(matches_at_half / len(predictions)) if predictions else 0.0
        ),
        "episode_count_absolute_error": float(abs(len(predictions) - len(references))),
    }


def _validate_synthetic(root: Path) -> dict[str, int]:
    payload = _read("synthetic_duration.json", root)
    _require(
        payload.get("kind") == "deja_cue_synthetic_duration_evidence",
        "Unexpected synthetic-duration evidence kind",
    )
    design = payload["design"]
    cells = payload["cells"]
    _require(len(cells) == 270, "Synthetic design must contain 270 cells")
    expected_design = {
        "true_durations": {8, 12, 20, 32, 48},
        "signal_means": {0.5, 1.0, 1.5},
        "ar1_correlations": {0.0, 0.6},
        "grid_bases": {4, 5, 6},
        "grid_ratios": {1.25, 1.5, 2.0},
    }
    for field, expected in expected_design.items():
        _require(set(design[field]) == expected, f"Synthetic {field} differs")
    combinations = {
        (
            row["true_duration"],
            row["signal_mean"],
            row["ar1_correlation"],
            row["grid_base"],
            row["grid_ratio"],
        )
        for row in cells
    }
    _require(len(combinations) == 270, "Synthetic design cells are not unique")

    methods = {method for row in cells for method in row["methods"]}
    _require(methods == {"sum", "mean", "sqrt_valid_count"}, "Method roster differs")
    reported = payload["reported_equal_cell_summary"]
    for method in sorted(methods):
        for metric in SYNTHETIC_METRICS:
            calculated = _mean(
                [float(row["methods"][method][metric]) for row in cells]
            )
            _require(
                _close(calculated, reported[method][metric]),
                f"Synthetic mean differs for {method}/{metric}",
            )
        _require(
            all(
                int(row["methods"][method]["trials"])
                == int(design["trials_per_noise_cell"])
                for row in cells
            ),
            f"Synthetic trial count differs for {method}",
        )

    expected_summary = {
        "sqrt_valid_count": (0.5768703703703704, 0.5077334844385318, 9.599231481481482),
        "sum": (0.3811666666666667, 0.40069811672540445, 30.269620370370372),
        "mean": (0.1113888888888889, 0.17959264552750423, 18.94840740740741),
    }
    for method, values in expected_summary.items():
        for metric, expected in zip(SYNTHETIC_METRICS, values):
            _require(
                _close(reported[method][metric], expected),
                f"Paper synthetic value differs for {method}/{metric}",
            )

    comparisons = payload["reported_paired_comparisons"]
    expected_deltas = {
        "sqrt_valid_count_minus_sum": {
            "r1_tiou_0.5": 0.1957037037037037,
            "top1_tiou": 0.10703536771312733,
        },
        "sqrt_valid_count_minus_mean": {
            "r1_tiou_0.5": 0.4654814814814815,
            "top1_tiou": 0.32814083891102763,
        },
    }
    for comparison, expected_by_metric in expected_deltas.items():
        baseline = comparison.removeprefix("sqrt_valid_count_minus_")
        for metric, expected in expected_by_metric.items():
            delta = reported["sqrt_valid_count"][metric] - reported[baseline][metric]
            _require(
                _close(delta, comparisons[comparison][f"{metric}_delta"]),
                f"Synthetic comparison differs for {comparison}/{metric}",
            )
            _require(
                _close(delta, expected),
                f"Paper synthetic delta differs for {comparison}/{metric}",
            )
    _validate_finite(payload, label="synthetic_duration")
    return {"cells": len(cells), "methods": len(methods)}


def _validate_tracking(root: Path) -> dict[str, int]:
    payload = _read("tracking_perturbations.json", root)
    _require(
        payload.get("kind") == "deja_cue_tracking_perturbation_evidence",
        "Unexpected tracking evidence kind",
    )
    _require(tuple(payload["metrics"]) == TRACKING_METRICS, "Tracking metrics differ")
    expected_deltas = {
        "random_missing_010pct": (-0.32176775046779466, -0.375, -0.10416666666666667),
        "contiguous_missing_span_010pct": (-0.06483561305018763, -0.125, 0.0),
        "distractor_contamination_025pct_mix050pct": (0.0034027953321735063, 0.0, 0.0),
        "distractor_contamination_100pct_mix100pct": (
            -0.105584682431072,
            -0.08333333333333333,
            0.0,
        ),
        "identity_switch_after_050pct": (-0.034884946420788765, 0.0, 0.0),
    }
    _require(set(payload["conditions"]) == set(expected_deltas), "Tracking cells differ")
    total_replicates = 0
    for condition_id, expected_values in expected_deltas.items():
        condition = payload["conditions"][condition_id]
        replicates = condition["replicates"]
        total_replicates += len(replicates)
        _require(
            condition["replicate_seeds"] == [3407, 3408, 3409],
            f"Finalized tracking seed roster differs for {condition_id}",
        )
        _require(
            [row["seed"] for row in replicates] == condition["replicate_seeds"],
            f"Tracking seeds differ for {condition_id}",
        )
        for metric, expected in zip(TRACKING_METRICS, expected_values):
            baseline = _mean(
                [
                    float(scene["metrics"][metric])
                    for replicate in replicates
                    for scene in replicate["baseline"]
                ]
            )
            perturbed = _mean(
                [
                    float(scene["metrics"][metric])
                    for replicate in replicates
                    for scene in replicate["perturbed"]
                ]
            )
            observed = condition["reported"]["observed"][metric]
            _require(
                _close(baseline, observed["baseline"]),
                f"Baseline differs for {condition_id}/{metric}",
            )
            _require(
                _close(perturbed, observed["perturbed"]),
                f"Perturbed mean differs for {condition_id}/{metric}",
            )
            _require(
                _close(perturbed - baseline, observed["delta"]),
                f"Delta differs for {condition_id}/{metric}",
            )
            _require(
                _close(observed["delta"], expected),
                f"Paper tracking value differs for {condition_id}/{metric}",
            )
    _validate_finite(payload, label="tracking_perturbations")
    return {"conditions": len(expected_deltas), "replicates": total_replicates}


def _validate_dnerf(root: Path) -> dict[str, int]:
    payload = _read("dnerf_recurrence.json", root)
    _require(
        payload.get("kind") == "deja_cue_recurrence_stress_evidence",
        "Unexpected recurrence evidence kind",
    )
    _require(payload["sequence_frames"] == 201, "Recurrence frame count differs")
    _require(payload["primary_min_run_frames"] == 3, "Recurrence minimum run differs")
    queries = {row["query_id"]: row for row in payload["queries"]}
    _require(len(queries) == 4, "Recurrence query count differs")
    for condition, prediction_rows in payload["predictions"].items():
        _require(
            {row["query_id"] for row in prediction_rows} == set(queries),
            f"Query roster differs for {condition}",
        )
        per_query = []
        for row in prediction_rows:
            state_id = queries[row["query_id"]]["state_id"]
            per_query.append(
                _recurrence_metrics(row["predicted_windows"], payload["references"][state_id])
            )
        calculated = {
            "query_macro_multi_window_map_tiou_0.3_0.5_0.7": _mean(
                [row["multi_window_map_tiou_0.3_0.5_0.7"] for row in per_query]
            ),
            "query_macro_multi_window_recall_tiou_0.5": _mean(
                [row["multi_window_recall_tiou_0.5"] for row in per_query]
            ),
            "query_macro_multi_window_precision_tiou_0.5": _mean(
                [row["multi_window_precision_tiou_0.5"] for row in per_query]
            ),
            "query_macro_episode_count_absolute_error": _mean(
                [row["episode_count_absolute_error"] for row in per_query]
            ),
        }
        for metric, value in calculated.items():
            _require(
                _close(value, payload["reported_metrics"][condition][metric]),
                f"Recurrence metric differs for {condition}/{metric}",
            )

    primary_name = "development_selected_dual_centered_locked_scan"
    primary = payload["reported_metrics"][primary_name]
    _require(
        _close(primary["query_macro_multi_window_map_tiou_0.3_0.5_0.7"], 1.0 / 12.0),
        "Paper recurrence mAP differs",
    )
    _require(
        _close(primary["query_macro_multi_window_recall_tiou_0.5"], 0.125),
        "Paper recurrence recall differs",
    )
    primary_predictions = payload["predictions"][primary_name]
    predicted = [window for row in primary_predictions for window in row["predicted_windows"]]
    _require(
        [(row["start"], row["end"]) for row in predicted] == [(1, 18)],
        "Primary recurrence prediction differs",
    )
    return {"queries": len(queries), "predicted_episodes": len(predicted)}


def _kappa(observed: float, chance: float) -> float:
    _require(chance < 1.0, "Kappa chance agreement must be below one")
    return float((observed - chance) / (1.0 - chance))


def _validate_annotation_agreement(root: Path) -> dict[str, int]:
    payload = _read("annotation_agreement.json", root)
    _require(
        payload.get("kind") == "deja_cue_annotation_agreement_evidence",
        "Unexpected annotation-agreement evidence kind",
    )
    histories = payload["histories"]
    _require(len(histories) == 3, "Annotation history count differs")
    for history in histories:
        for scope in ("selected_frames", "stable_valid_frames"):
            row = history[scope]
            _require(
                row["frame_count"]
                == row["exact_agreement_count"] + row["disagreement_count"],
                f"Annotation counts differ for {history['history_id']}/{scope}",
            )
            rate = row["exact_agreement_count"] / row["frame_count"]
            _require(
                _close(rate, row["exact_agreement_rate"]),
                f"Agreement rate differs for {history['history_id']}/{scope}",
            )
            kappa = row["kappa"]
            _require(
                _close(
                    _kappa(kappa["observed_agreement"], kappa["chance_agreement"]),
                    kappa["cohen_kappa"],
                ),
                f"Kappa differs for {history['history_id']}/{scope}",
            )

    for scope in ("selected_frames", "stable_valid_frames"):
        reported = payload["reported"]["global_pooled"][scope]
        frames = sum(int(row[scope]["frame_count"]) for row in histories)
        exact = sum(int(row[scope]["exact_agreement_count"]) for row in histories)
        disagreements = sum(int(row[scope]["disagreement_count"]) for row in histories)
        _require(
            (frames, exact, disagreements)
            == (
                reported["frame_count"],
                reported["exact_agreement_count"],
                reported["disagreement_count"],
            ),
            f"Pooled counts differ for {scope}",
        )
        _require(
            _close(exact / frames, reported["exact_agreement_rate"]),
            f"Pooled rate differs for {scope}",
        )
        kappa = reported["kappa"]
        _require(
            _close(
                _kappa(kappa["observed_agreement"], kappa["chance_agreement"]),
                kappa["cohen_kappa"],
            ),
            f"Pooled kappa differs for {scope}",
        )
        history_macro = _mean([float(row[scope]["exact_agreement_rate"]) for row in histories])
        _require(
            _close(history_macro, payload["reported"]["history_macro"][scope]),
            f"History macro differs for {scope}",
        )

    selected = payload["reported"]["global_pooled"]["selected_frames"]
    stable = payload["reported"]["global_pooled"]["stable_valid_frames"]
    _require(
        selected["frame_count"] == 1883 and stable["frame_count"] == 1700,
        "Paper annotation coverage differs",
    )
    _require(
        _close(selected["exact_agreement_rate"], 0.605416887944769),
        "Paper pooled agreement differs",
    )
    _require(
        _close(selected["kappa"]["cohen_kappa"], 0.477819625623721),
        "Paper pooled kappa differs",
    )
    _require(
        _close(
            payload["reported"]["history_macro"]["selected_frames"],
            0.7640077304358103,
        ),
        "Paper history-macro agreement differs",
    )
    by_label = {row["label"]: row for row in histories}
    _require(
        _close(
            by_label["cross_hands"]["stable_valid_frames"]["exact_agreement_rate"],
            0.49659348978046935,
        ),
        "Cross-hands agreement differs",
    )
    _require(
        _close(
            by_label["coffee_martini"]["stable_valid_frames"][
                "exact_agreement_rate"
            ],
            0.98,
        ),
        "Coffee-martini agreement differs",
    )
    adjudication = payload["reported"]["adjudication"]
    _require(adjudication["frame_disagreement_count"] == 743, "Disagreement count differs")
    _require(
        _close(adjudication["adjudication_rate_per_disagreement"], 1.0),
        "Adjudication rate differs",
    )
    return {"histories": len(histories), "selected_frames": selected["frame_count"]}


def _validate_annotation_sensitivity(root: Path) -> dict[str, int]:
    payload = _read("annotation_sensitivity.json", root)
    _require(
        payload.get("kind") == "deja_cue_annotation_sensitivity_evidence",
        "Unexpected annotation-sensitivity evidence kind",
    )
    conditions = {
        "absolute_coordinates",
        "vocabulary_centered_coordinates",
        "dual_centered_coordinates",
    }
    for annotation_set, results in payload["scores"].items():
        _require(set(results) == conditions, f"Coordinate roster differs for {annotation_set}")
        for condition, result in results.items():
            rows = result["history_rows"]
            _require(len(rows) == 3, f"History count differs for {annotation_set}/{condition}")
            for metric in ("state_macro_top1_tiou", "state_macro_r1_tiou_0.5"):
                value = _mean([float(row[metric]) for row in rows])
                _require(
                    _close(value, result["reported_history_macro"][metric]),
                    "Annotation sensitivity mean differs for "
                    f"{annotation_set}/{condition}/{metric}",
                )

        expected_treatments = {
            "vocabulary_centered_minus_absolute": "vocabulary_centered_coordinates",
            "dual_centered_minus_absolute": "dual_centered_coordinates",
        }
        for comparison, treatment in expected_treatments.items():
            for metric in ("state_macro_top1_tiou", "state_macro_r1_tiou_0.5"):
                delta = (
                    results[treatment]["reported_history_macro"][metric]
                    - results["absolute_coordinates"]["reported_history_macro"][metric]
                )
                reported = payload["reported_coordinate_deltas"][annotation_set][comparison][metric]
                _require(
                    _close(delta, reported),
                    "Annotation sensitivity delta differs for "
                    f"{annotation_set}/{comparison}/{metric}",
                )

    expected = {
        ("adjudicated", "vocabulary_centered_minus_absolute"): (
            -0.1290528791189984,
            -0.19444444444444442,
        ),
        ("adjudicated", "dual_centered_minus_absolute"): (
            -0.11070339415615871,
            -0.27777777777777773,
        ),
        ("pass_A", "vocabulary_centered_minus_absolute"): (
            -0.1290528791189984,
            -0.19444444444444442,
        ),
        ("pass_A", "dual_centered_minus_absolute"): (-0.11070339415615871, -0.27777777777777773),
        ("pass_B", "vocabulary_centered_minus_absolute"): (
            -0.06592732360008202,
            -0.08333333333333331,
        ),
        ("pass_B", "dual_centered_minus_absolute"): (-0.07148770450504704, -0.08333333333333331),
    }
    for (annotation_set, comparison), values in expected.items():
        for metric, value in zip(("state_macro_top1_tiou", "state_macro_r1_tiou_0.5"), values):
            _require(
                _close(
                    payload["reported_coordinate_deltas"][annotation_set][comparison][
                        metric
                    ],
                    value,
                ),
                "Paper annotation sensitivity differs for "
                f"{annotation_set}/{comparison}/{metric}",
            )
    return {"annotation_sets": len(payload["scores"]), "coordinate_conditions": len(conditions)}


def _validate_hard_negatives(root: Path) -> dict[str, int]:
    payload = _read("hard_negative_margins.json", root)
    _require(
        payload.get("kind") == "deja_cue_hard_negative_margin_evidence",
        "Unexpected hard-negative evidence kind",
    )
    identity = payload["identity_accuracy"]
    _require(
        identity["absolute_k3"]["num_correct"] == 15
        and identity["absolute_k3"]["num_queries"] == 16,
        "Absolute identity count differs",
    )
    _require(
        identity["query_only_k3"]["num_correct"] == 16
        and identity["query_only_k3"]["num_queries"] == 16,
        "Vocabulary identity count differs",
    )
    for row in identity.values():
        _require(
            _close(
                100.0 * row["num_correct"] / row["num_queries"],
                row["accuracy_percent"],
            ),
            "Identity percentage differs",
        )

    expected = {
        "sibling_state": (10, 2.3092881441116333, (-2.2949554920196533, 7.751538872718811)),
        "joint_identity_state": (9, -1.3406186997890472, (-13.813504338264465, 7.702645689249039)),
    }
    for margin_name, (positive, point, interval) in expected.items():
        result = payload["paired_margins"][margin_name]
        values = np.asarray(
            [float(row["delta"]) for row in result["history_median_deltas"]],
            dtype=np.float64,
        )
        _require(len(values) == 4, f"History count differs for {margin_name}")
        _require(
            _close(float(values.mean()), result["reported_history_macro_delta"]),
            f"Hard-negative mean differs for {margin_name}",
        )
        bootstrap = result["bootstrap"]
        rng = np.random.default_rng(int(bootstrap["seed"]))
        draws = values[
            rng.integers(0, len(values), size=(int(bootstrap["num_resamples"]), len(values)))
        ].mean(axis=1)
        bootstrap_interval = np.quantile(draws, [0.025, 0.975])
        _require(
            np.allclose(
                bootstrap_interval,
                result["reported_history_bootstrap_95ci"],
                atol=1e-12,
                rtol=0.0,
            ),
            f"Hard-negative interval differs for {margin_name}",
        )
        _require(
            result["num_positive_pairs"] == positive,
            f"Positive-pair count differs for {margin_name}",
        )
        _require(
            _close(result["reported_history_macro_delta"], point),
            f"Paper hard-negative point differs for {margin_name}",
        )
        _require(
            np.allclose(
                result["reported_history_bootstrap_95ci"],
                interval,
                atol=1e-12,
                rtol=0.0,
            ),
            f"Paper hard-negative interval differs for {margin_name}",
        )
    return {"queries": 16, "margin_families": len(expected)}


def _validate_development_controls(root: Path) -> dict[str, int]:
    payload = _read("development_controls.json", root)
    _require(
        payload.get("kind") == "deja_cue_development_and_control_evidence",
        "Unexpected development-control evidence kind",
    )
    development = payload["development_selection"]
    _require(development["smoothing_kernel_size"] == 3, "Selected smoothing support differs")
    _require(development["optimized_scalar_count"] == 0, "Optimized scalar count differs")
    folds = development["folds"]
    _require(len(folds) == 5, "Development fold count differs")
    _require(
        (
            sum(row["states"] for row in folds),
            sum(row["descriptions"] for row in folds),
            sum(row["episodes"] for row in folds),
        )
        == (13, 26, 46),
        "Development counts differ",
    )
    calculated = {
        "r1_tiou_0.5": _mean([row["r1_tiou_0.5"] for row in folds]),
        "candidate_oracle_top1_tiou": _mean(
            [row["candidate_oracle_top1_tiou"] for row in folds]
        ),
        "permuted_r1_tiou_0.5": _mean(
            [row["permuted_r1_tiou_0.5"] for row in folds]
        ),
    }
    expected = {
        "r1_tiou_0.5": 0.5333333333333334,
        "candidate_oracle_top1_tiou": 0.9733732084433239,
        "permuted_r1_tiou_0.5": 0.03333333333333333,
    }
    for metric, value in calculated.items():
        _require(
            _close(value, development["reported_macro"][metric]),
            f"Development macro differs for {metric}",
        )
        _require(_close(value, expected[metric]), f"Paper development value differs for {metric}")

    expected_rows = {
        "hand": (1.0, 1.0, 0.0),
        "banana": (0.25, 0.9352226853370667, 1.0 / 12.0),
        "lemon": (1.0 / 6.0, 0.971794863541921, 1.0 / 12.0),
        "cookie": (0.25, 1.0, 0.0),
        "toy_container": (1.0, 0.9598484933376312, 0.0),
    }
    for row in folds:
        expected_row = expected_rows[row["label"]]
        actual = (
            row["r1_tiou_0.5"],
            row["candidate_oracle_top1_tiou"],
            row["permuted_r1_tiou_0.5"],
        )
        _require(
            np.allclose(actual, expected_row, atol=1e-12, rtol=0.0),
            f"Development row differs for {row['label']}",
        )

    control = payload["circular_shift"]
    unshifted = _mean([row["r1_tiou_0.5"] for row in control["unshifted"]["histories"]])
    _require(
        _close(
            unshifted,
            control["unshifted"]["reported_macro_r1_tiou_0.5"],
        ),
        "Unshifted macro differs",
    )
    shift_macros = []
    for shift in control["shifts"]:
        value = _mean([row["r1_tiou_0.5"] for row in shift["histories"]])
        _require(
            _close(value, shift["reported_macro_r1_tiou_0.5"]),
            f"Shift macro differs for {shift['shift_index']}",
        )
        shift_macros.append(value)
    shift_mean = _mean(shift_macros)
    _require(_close(shift_mean, control["reported_shift_mean_r1_tiou_0.5"]), "Shift mean differs")
    _require(
        _close(unshifted - shift_mean, control["reported_drop_r1_tiou_0.5"]),
        "Shift drop differs",
    )
    _require(_close(unshifted, 0.4375), "Paper unshifted recall differs")
    _require(_close(shift_mean, 0.140625), "Paper shift recall differs")
    _require(_close(control["reported_drop_r1_tiou_0.5"], 0.296875), "Paper shift drop differs")
    return {"development_folds": len(folds), "circular_shifts": len(control["shifts"])}


def _validate_vost_paraphrase(root: Path) -> dict[str, int]:
    payload = _read("vost_paraphrase.json", root)
    _require(
        payload.get("kind") == "deja_cue_vost_paraphrase_evidence",
        "Unexpected VOST paraphrase evidence kind",
    )
    expected = {
        "absolute_coordinates": (0.6311847902834415, 0.32051282051282054),
        "vocabulary_centered_coordinates": (0.13321064490204057, 0.02564102564102564),
    }
    for condition, expected_values in expected.items():
        result = payload["conditions"][condition]
        rows = result["source_components"]
        _require(len(rows) == 78, f"VOST paraphrase roster differs for {condition}")
        _require(
            len({row["source_component_id"] for row in rows}) == 78,
            f"VOST component identifiers repeat for {condition}",
        )
        for metric, expected_value in zip(PARAPHRASE_METRICS, expected_values):
            value = _mean([float(row[metric]) for row in rows])
            _require(
                _close(value, result["reported_source_component_macro"][metric]),
                f"VOST paraphrase mean differs for {condition}/{metric}",
            )
            _require(
                _close(value, expected_value),
                f"Paper VOST paraphrase value differs for {condition}/{metric}",
            )
    return {"conditions": len(expected), "source_components": 78}


def validate_robustness_results(root: Path | None = None) -> dict[str, dict[str, int]]:
    """Validate and recalculate every bundled robustness evidence family."""

    package = _root(root)
    return {
        "synthetic_duration": _validate_synthetic(package),
        "tracking_perturbations": _validate_tracking(package),
        "dnerf_recurrence": _validate_dnerf(package),
        "annotation_agreement": _validate_annotation_agreement(package),
        "annotation_sensitivity": _validate_annotation_sensitivity(package),
        "hard_negative_margins": _validate_hard_negatives(package),
        "development_controls": _validate_development_controls(package),
        "vost_paraphrase": _validate_vost_paraphrase(package),
    }
