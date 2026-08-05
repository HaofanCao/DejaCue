"""Tests for tracking perturbations and paired aggregation."""

from __future__ import annotations

import numpy as np

from deja_cue.experiments.tracking import (
    DEFAULT_CONDITION_IDS,
    TRACKING_METRICS,
    _reported_summary,
)


def test_default_tracking_conditions_and_metrics_are_stable() -> None:
    assert len(DEFAULT_CONDITION_IDS) == 5
    assert len(set(DEFAULT_CONDITION_IDS)) == 5
    assert TRACKING_METRICS == (
        "state_macro_target_top1_tiou",
        "state_macro_target_r1_tiou_0.5",
        "query_macro_joint_identity_accuracy",
    )


def test_tracking_summary_pairs_scenes_and_replicates() -> None:
    def row(scene: str, value: float) -> dict[str, object]:
        return {
            "scene": scene,
            "metrics": {metric: value for metric in TRACKING_METRICS},
        }

    replicates = [
        {
            "seed": 3407,
            "baseline": [row("a", 0.5), row("b", 0.75)],
            "perturbed": [row("a", 0.25), row("b", 0.5)],
        },
        {
            "seed": 3408,
            "baseline": [row("a", 0.5), row("b", 0.75)],
            "perturbed": [row("a", 0.0), row("b", 0.75)],
        },
    ]
    result = _reported_summary(replicates, bootstrap_resamples=100, seed=3407)
    for metric in TRACKING_METRICS:
        assert result["observed"][metric] == {
            "baseline": 0.625,
            "perturbed": 0.375,
            "delta": -0.25,
        }
        interval = result["paired_delta_scene_cluster_bootstrap_95ci"][metric]
        assert len(interval) == 2 and np.isfinite(interval).all()
