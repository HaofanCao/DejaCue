"""Unit tests for robust evidence calibration and temporal decoding rules."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from deja_cue.evaluation import candidate_rank_row
from deja_cue.scan import (
    _median,
    robust_standardize,
    select_window_index,
    smooth_within_runs,
    temporal_iou,
)


def test_median_averages_two_middle_values() -> None:
    values = torch.tensor([[4.0, 1.0, 3.0, 2.0]])
    assert _median(values, dim=1).item() == pytest.approx(2.5)


def test_robust_standardization_uses_visible_values_and_mad_floor() -> None:
    evidence = torch.tensor([[1.0, 100.0, 3.0]])
    valid = torch.tensor([True, False, True])
    standardized, centers, scales = robust_standardize(evidence, valid)
    assert centers.item() == pytest.approx(2.0)
    assert scales.item() == pytest.approx(1.4826)
    assert standardized[0, 1].item() == 0.0


def test_smoothing_does_not_cross_missing_frames() -> None:
    evidence = torch.tensor([[1.0, 100.0, 1.0]])
    valid = torch.tensor([True, False, True])
    smoothed = smooth_within_runs(evidence, valid, kernel_size=3)
    torch.testing.assert_close(smoothed, torch.tensor([[1.0, 0.0, 1.0]]))


def test_boundary_smoothing_normalizes_active_taps() -> None:
    evidence = torch.ones((1, 3))
    valid = torch.ones(3, dtype=torch.bool)
    smoothed = smooth_within_runs(evidence, valid, kernel_size=3)
    assert smoothed[0, 0].item() == pytest.approx(math.sqrt(2.0))
    assert smoothed[0, 1].item() == pytest.approx(math.sqrt(3.0))


def test_exact_score_tie_uses_earliest_start_then_end() -> None:
    scores = torch.tensor([5.0, 5.0, 5.0])
    starts = torch.tensor([2, 1, 1])
    ends = torch.tensor([2, 2, 1])
    assert select_window_index(scores, starts, ends) == 2


def test_temporal_iou_uses_inclusive_endpoints() -> None:
    assert temporal_iou((0, 1), ((1, 2),)) == pytest.approx(1.0 / 3.0)


def test_candidate_ranking_reports_topk_oracle_gap_closure() -> None:
    row = candidate_rank_row(
        np.asarray([3.0, 2.0, 1.0]),
        np.asarray([0, 1, 2]),
        np.asarray([0, 1, 2]),
        ((2, 2),),
        topk=(1, 2, 3),
    )
    assert row["oracle_gap"] == pytest.approx(1.0)
    assert row["oracle_gap_closed_at_1"] == pytest.approx(0.0)
    assert row["oracle_gap_closed_at_3"] == pytest.approx(1.0)
