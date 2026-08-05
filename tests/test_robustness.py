"""Arithmetic tests for the bundled robustness evidence."""

import pytest

from deja_cue.robustness import temporal_iou, validate_robustness_results


def test_inclusive_temporal_iou() -> None:
    assert temporal_iou((1, 18), (0, 28)) == pytest.approx(18.0 / 29.0)
    assert temporal_iou((1, 18), (179, 200)) == 0.0
    assert temporal_iou((4, 4), (4, 4)) == 1.0


def test_robustness_result_statistics_are_recalculated() -> None:
    checks = validate_robustness_results()
    assert checks == {
        "synthetic_duration": {"cells": 270, "methods": 3},
        "tracking_perturbations": {"conditions": 5, "replicates": 15},
        "dnerf_recurrence": {"queries": 4, "predicted_episodes": 1},
        "annotation_agreement": {"histories": 3, "selected_frames": 1883},
        "annotation_sensitivity": {
            "annotation_sets": 3,
            "coordinate_conditions": 3,
        },
        "hard_negative_margins": {"queries": 16, "margin_families": 2},
        "development_controls": {
            "development_folds": 5,
            "circular_shifts": 8,
        },
        "vost_paraphrase": {"conditions": 2, "source_components": 78},
    }
