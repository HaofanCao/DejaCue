"""Inclusive interval metrics and source-component statistical summaries."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np


PRIMARY_METRICS = (
    "state_macro_target_r1_tiou_0.3",
    "state_macro_target_r1_tiou_0.5",
    "state_macro_target_top1_tiou",
)


def finite_mean(values: Sequence[float | None]) -> float | None:
    """Return the mean of finite non-null values, or ``None`` when empty."""

    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return float(np.mean(finite)) if finite else None


def summarize_history(
    *,
    history_id: str,
    sequence_id: str,
    source_component_id: str,
    query_rows: Sequence[Mapping[str, Any]],
    state_order: Sequence[str],
) -> dict[str, Any]:
    """Average descriptions within state before forming one history row."""

    states = []
    for state_id in state_order:
        members = [row for row in query_rows if row["state_id"] == state_id]
        if not members:
            raise ValueError(f"No descriptions for state {state_id}")
        states.append(
            {
                "state_id": state_id,
                "target_top1_tiou": float(
                    np.mean([row["top1_tiou"] for row in members])
                ),
                "target_r1_tiou_0.3": float(
                    np.mean([row["r1_tiou_0.3"] for row in members])
                ),
                "target_r1_tiou_0.5": float(
                    np.mean([row["r1_tiou_0.5"] for row in members])
                ),
            }
        )
    summary = {
        "state_macro_target_top1_tiou": float(
            np.mean([row["target_top1_tiou"] for row in states])
        ),
        "state_macro_target_r1_tiou_0.3": float(
            np.mean([row["target_r1_tiou_0.3"] for row in states])
        ),
        "state_macro_target_r1_tiou_0.5": float(
            np.mean([row["target_r1_tiou_0.5"] for row in states])
        ),
    }
    return {
        "history_id": history_id,
        "sequence_id": sequence_id,
        "source_component_id": source_component_id,
        "states": states,
        "queries": list(query_rows),
        "summary": summary,
    }


def aggregate_histories(
    histories: Sequence[Mapping[str, Any]],
    *,
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Apply history-to-source-component aggregation and bootstrap CIs."""

    if not histories:
        raise ValueError("Cannot aggregate an empty result")
    by_component: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for history in histories:
        by_component[str(history["source_component_id"])].append(history)
    component_rows = []
    for component_id, rows in sorted(by_component.items()):
        component_rows.append(
            {
                "source_component_id": component_id,
                "history_ids": sorted(str(row["history_id"]) for row in rows),
                "summary": {
                    metric: finite_mean([row["summary"][metric] for row in rows])
                    for metric in PRIMARY_METRICS
                },
            }
        )
    macro = {
        metric: finite_mean([row["summary"][metric] for row in component_rows])
        for metric in PRIMARY_METRICS
    }
    intervals: dict[str, list[float] | None] = {
        metric: None for metric in PRIMARY_METRICS
    }
    if bootstrap_resamples:
        rng = np.random.default_rng(seed)
        indices = rng.integers(
            0,
            len(component_rows),
            size=(bootstrap_resamples, len(component_rows)),
        )
        for metric in PRIMARY_METRICS:
            values = np.asarray(
                [row["summary"][metric] for row in component_rows], dtype=np.float64
            )
            draws = values[indices].mean(axis=1)
            intervals[metric] = [
                float(np.quantile(draws, 0.025)),
                float(np.quantile(draws, 0.975)),
            ]
    return {
        "aggregation": (
            "descriptions_to_state_then_states_to_history_then_histories_to_"
            "source_component_then_uniform_source_component_macro"
        ),
        "num_histories": len(histories),
        "num_source_components": len(component_rows),
        "source_component_macro": macro,
        "source_component_bootstrap_95ci": intervals,
        "per_source_component": component_rows,
        "bootstrap": {
            "num_resamples": bootstrap_resamples,
            "seed": seed,
            "sampling": "source_components_with_replacement",
        },
    }


def component_metric_map(result: Mapping[str, Any], metric: str) -> dict[str, float]:
    """Index one aggregate metric by its independent source component."""

    return {
        str(row["source_component_id"]): float(row["summary"][metric])
        for row in result["aggregate"]["per_source_component"]
    }


def paired_cluster_bootstrap(
    left: Mapping[str, float],
    right: Mapping[str, float],
    *,
    seed: int,
    resamples: int,
) -> list[float]:
    """Bootstrap paired source-component differences and return a 95% interval."""

    if set(left) != set(right):
        raise ValueError("Paired source-component keys differ")
    keys = sorted(left)
    deltas = np.asarray([left[key] - right[key] for key in keys], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = deltas[rng.integers(0, len(deltas), size=(resamples, len(deltas)))].mean(
        axis=1
    )
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def paired_sign_flip(
    left: Mapping[str, float],
    right: Mapping[str, float],
    *,
    seed: int,
    assignments: int,
) -> dict[str, Any]:
    """Monte Carlo two-sided paired sign-flip test with plus-one correction."""

    if set(left) != set(right):
        raise ValueError("Paired source-component keys differ")
    keys = sorted(left)
    deltas = np.asarray([left[key] - right[key] for key in keys], dtype=np.float64)
    if len(deltas) < 2 or not np.isfinite(deltas).all():
        raise ValueError("Paired test requires at least two finite pairs")
    observed = float(deltas.mean())
    rng = np.random.default_rng(seed)
    exceed = 0
    batch_size = 10_000
    for offset in range(0, assignments, batch_size):
        count = min(batch_size, assignments - offset)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(count, len(deltas)))
        draws = (signs * deltas).mean(axis=1)
        exceed += int(np.count_nonzero(np.abs(draws) >= abs(observed) - 1e-15))
    return {
        "observed_delta": observed,
        "source_component_deltas": {
            key: float(delta) for key, delta in zip(keys, deltas)
        },
        "two_sided_p": float((exceed + 1) / (assignments + 1)),
        "assignments": assignments,
        "seed": seed,
    }


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Return monotone Holm-adjusted p-values with deterministic tie order."""

    ordered = sorted(p_values, key=lambda key: (float(p_values[key]), key))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, key in enumerate(ordered):
        running = max(running, (count - rank) * float(p_values[key]))
        adjusted[key] = min(1.0, running)
    return adjusted
