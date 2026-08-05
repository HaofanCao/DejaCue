"""Comparison helpers for frozen query windows and reported aggregate values."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .data import package_root


def load_reference(name: str, root: Path | None = None) -> dict[str, Any]:
    """Load one bundled result reference by its filename stem."""

    archive_root = package_root() if root is None else Path(root)
    path = archive_root / "data" / "reference" / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _prediction_map_from_result(
    histories: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, str], tuple[int, int]]:
    return {
        (str(history["history_id"]), str(query["state_id"]), str(query["text"])): tuple(
            int(value) for value in query["window"]
        )
        for history in histories
        for query in history["queries"]
    }


def _prediction_map_from_reference(
    histories: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, str], tuple[int, int]]:
    return {
        (str(history["history_id"]), str(query["state_id"]), str(query["text"])): tuple(
            int(value) for value in query["window"]
        )
        for history in histories
        for query in history["queries"]
    }


def compare_condition(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Compare all query intervals, primary metrics, and bootstrap intervals."""

    observed_windows = _prediction_map_from_result(observed["histories"])
    expected_windows = _prediction_map_from_reference(expected["predictions"])
    missing = sorted(set(expected_windows).difference(observed_windows))
    unexpected = sorted(set(observed_windows).difference(expected_windows))
    changed = [
        {
            "key": list(key),
            "observed": list(observed_windows[key]),
            "expected": list(expected_windows[key]),
        }
        for key in sorted(set(observed_windows).intersection(expected_windows))
        if observed_windows[key] != expected_windows[key]
    ]

    aggregate_errors = {}
    observed_aggregate = observed["aggregate"]["source_component_macro"]
    for metric, target in expected["aggregate"].items():
        value = float(observed_aggregate[metric])
        if not np.isclose(value, float(target), atol=tolerance, rtol=0.0):
            aggregate_errors[metric] = {"observed": value, "expected": float(target)}

    interval_errors = {}
    observed_intervals = observed["aggregate"]["source_component_bootstrap_95ci"]
    for metric, target in expected["bootstrap_95ci"].items():
        values = [float(value) for value in observed_intervals[metric]]
        if not np.allclose(values, target, atol=tolerance, rtol=0.0):
            interval_errors[metric] = {"observed": values, "expected": target}
    passed = (
        not missing
        and not unexpected
        and not changed
        and not aggregate_errors
        and not interval_errors
    )
    return {
        "passed": passed,
        "queries_checked": len(expected_windows),
        "missing_queries": [list(key) for key in missing],
        "unexpected_queries": [list(key) for key in unexpected],
        "changed_windows": changed,
        "aggregate_errors": aggregate_errors,
        "bootstrap_interval_errors": interval_errors,
    }


def compare_main(
    result: Mapping[str, Any], reference: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare every primary method window, metric, and confidence interval."""

    checks = {
        name: compare_condition(result["methods"][name], expected)
        for name, expected in reference["methods"].items()
    }
    return {
        "passed": all(check["passed"] for check in checks.values()),
        "methods": checks,
        "total_query_windows_checked": sum(
            check["queries_checked"] for check in checks.values()
        ),
    }


def compare_extended(
    result: Mapping[str, Any], reference: Mapping[str, Any], *, tolerance: float = 1e-8
) -> dict[str, Any]:
    """Compare all extended conditions, rank summaries, and duration strata."""

    condition_checks: dict[str, Any] = {}
    for coordinate, expected in reference["adaptive_peak"].items():
        condition_checks[f"adaptive_peak/{coordinate}"] = compare_condition(
            result["adaptive_peak"][coordinate], expected, tolerance=tolerance
        )
    for variant, coordinates in reference["prompt_sensitivity"].items():
        for coordinate, expected in coordinates.items():
            condition_checks[f"prompt_sensitivity/{variant}/{coordinate}"] = (
                compare_condition(
                    result["prompt_sensitivity"][variant][coordinate],
                    expected,
                    tolerance=tolerance,
                )
            )
    for normalization, coordinates in reference["normalization_sensitivity"].items():
        for coordinate, expected in coordinates.items():
            condition_checks[
                f"normalization_sensitivity/{normalization}/{coordinate}"
            ] = compare_condition(
                result["normalization_sensitivity"][normalization][coordinate],
                expected,
                tolerance=tolerance,
            )

    ranking_errors = {}
    for condition, expected in reference["candidate_ranking"].items():
        observed = result["candidate_ranking"]["conditions"][condition]["aggregate"][
            "history_macro"
        ]
        errors = {
            metric: {"observed": float(observed[metric]), "expected": float(target)}
            for metric, target in expected.items()
            if not np.isclose(
                float(observed[metric]), float(target), atol=tolerance, rtol=0.0
            )
        }
        if errors:
            ranking_errors[condition] = errors

    strata_errors = {}
    for condition, expected in reference["duration_strata"].items():
        observed = result["duration_strata"][condition]
        errors = []
        if not np.allclose(
            observed["duration_quartile_boundaries"],
            expected["duration_quartile_boundaries"],
            atol=tolerance,
            rtol=0.0,
        ):
            errors.append("duration_quartile_boundaries")
        for grouping in ("by_duration",):
            if set(observed[grouping]) != set(expected[grouping]):
                errors.append(f"{grouping}/labels")
                continue
            for label, expected_row in expected[grouping].items():
                observed_row = observed[grouping][label]
                for metric, target in expected_row.items():
                    value = observed_row.get(metric)
                    if metric == "num_descriptions":
                        matches = int(value) == int(target)
                    else:
                        matches = np.isclose(
                            float(value), float(target), atol=tolerance, rtol=0.0
                        )
                    if not matches:
                        errors.append(f"{grouping}/{label}/{metric}")
        if errors:
            strata_errors[condition] = sorted(set(errors))
    return {
        "passed": all(check["passed"] for check in condition_checks.values())
        and not ranking_errors
        and not strata_errors,
        "conditions": condition_checks,
        "candidate_ranking_errors": ranking_errors,
        "duration_strata_errors": strata_errors,
        "total_query_windows_checked": sum(
            check["queries_checked"] for check in condition_checks.values()
        ),
    }


def require_reference_match(check: Mapping[str, Any], *, label: str) -> None:
    """Raise with a stable workflow label when a reference comparison fails."""

    if check.get("passed") is not True:
        raise RuntimeError(
            f"{label} differs from the bundled reference; inspect the JSON report"
        )
