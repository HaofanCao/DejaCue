"""Focused tests for diagnostic analyses."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from deja_cue.experiments.diagnostics import (
    average_precision,
    circular_shift_within_runs,
    cohen_kappa,
    hard_negative_margins,
    paired_margin_summary,
    paraphrase_window_consistency,
    permute_state_assignments,
    recurrence_metrics,
    reference_boundary_sensitivity,
    summarize_annotation_agreement,
    summarize_recurrence_conditions,
)


ROOT = Path(__file__).resolve().parents[1]


def test_annotation_agreement_summarizes_pooled_and_stable_populations() -> None:
    left = ["pre", "pre", "transition", "post"]
    right = ["pre", "transition", "transition", "post"]
    kappa = cohen_kappa(left, right)
    assert kappa["observed_agreement"] == pytest.approx(0.75)
    assert kappa["chance_agreement"] == pytest.approx(0.3125)
    assert kappa["cohen_kappa"] == pytest.approx(7.0 / 11.0)

    result = summarize_annotation_agreement(
        [
            {
                "history_id": "A01",
                "pass_A": left,
                "pass_B": right,
                "stable_valid_mask": [True, False, True, True],
            },
            {
                "history_id": "A02",
                "pass_A": ["state_a", "state_b"],
                "pass_B": ["state_a", "state_b"],
                "stable_valid_mask": [True, True],
            },
        ]
    )
    pooled = result["reported"]["global_pooled"]
    assert pooled["selected_frames"]["frame_count"] == 6
    assert pooled["selected_frames"]["disagreement_count"] == 1
    assert pooled["stable_valid_frames"]["exact_agreement_rate"] == 1.0
    assert result["reported"]["history_macro"]["selected_frames"] == pytest.approx(0.875)


def _reference_rows(first: tuple[int, int], second: tuple[int, int]) -> list[dict[str, object]]:
    return [
        {"history_id": "H01", "state_id": "state_a", "windows": [list(first)]},
        {"history_id": "H02", "state_id": "state_b", "windows": [list(second)]},
    ]


def _prediction_rows(windows: tuple[tuple[int, int], ...]) -> list[dict[str, object]]:
    identities = (
        ("H01", "Q01", "state_a"),
        ("H01", "Q02", "state_a"),
        ("H02", "Q03", "state_b"),
        ("H02", "Q04", "state_b"),
    )
    return [
        {
            "history_id": history_id,
            "query_id": query_id,
            "state_id": state_id,
            "window": list(window),
        }
        for (history_id, query_id, state_id), window in zip(identities, windows, strict=True)
    ]


def test_boundary_sensitivity_keeps_predictions_fixed_across_reference_sets() -> None:
    baseline = _prediction_rows(((0, 4), (0, 4), (10, 14), (10, 14)))
    treatment = _prediction_rows(((0, 1), (20, 21), (10, 14), (10, 14)))
    result = reference_boundary_sensitivity(
        {"absolute": baseline, "vocabulary": treatment},
        {
            "pass_A": _reference_rows((0, 4), (10, 14)),
            "pass_B": _reference_rows((0, 1), (10, 14)),
        },
        baseline_condition="absolute",
    )
    pass_a = result["scores"]["pass_A"]
    assert pass_a["absolute"]["reported_history_macro"] == {
        "state_macro_top1_tiou": 1.0,
        "state_macro_r1_tiou_0.5": 1.0,
    }
    assert pass_a["vocabulary"]["reported_history_macro"] == pytest.approx(
        {
            "state_macro_top1_tiou": 0.6,
            "state_macro_r1_tiou_0.5": 0.5,
        }
    )
    assert result["reported_coordinate_deltas"]["pass_A"][
        "vocabulary_minus_absolute"
    ] == pytest.approx(
        {
            "state_macro_top1_tiou": -0.4,
            "state_macro_r1_tiou_0.5": -0.5,
        }
    )
    # The same treatment windows score differently when only the reference
    # boundaries change, which is the intended sensitivity intervention.
    assert (
        result["scores"]["pass_B"]["vocabulary"]["reported_history_macro"]
        ["state_macro_top1_tiou"]
        > pass_a["vocabulary"]["reported_history_macro"]["state_macro_top1_tiou"]
    )


def test_within_run_shift_and_state_permutation_preserve_control_invariants() -> None:
    frames = np.asarray([0, 1, 2, 7, 8, 9], dtype=np.int64)
    values = np.arange(12, dtype=np.float32).reshape(6, 2)
    first = circular_shift_within_runs(values, frames, seed=3407)
    second = circular_shift_within_runs(values, frames, seed=3407)
    np.testing.assert_array_equal(first, second)
    for left, right in ((0, 3), (3, 6)):
        assert sorted(map(tuple, first[left:right])) == sorted(map(tuple, values[left:right]))
        assert not np.array_equal(first[left:right], values[left:right])

    queries = [
        {"query_id": "Q01", "state_id": "state_a", "text_feature": [1.0, 0.0]},
        {"query_id": "Q02", "state_id": "state_b", "text_feature": [0.0, 1.0]},
    ]
    permuted = permute_state_assignments(
        queries, {"state_a": "state_b", "state_b": "state_a"}
    )
    assert [row["state_id"] for row in permuted] == ["state_b", "state_a"]
    assert [row["text_feature"] for row in permuted] == [
        [1.0, 0.0],
        [0.0, 1.0],
    ]
    assert [row["state_id"] for row in queries] == ["state_a", "state_b"]
    with pytest.raises(ValueError, match="derangement"):
        permute_state_assignments(
            queries, {"state_a": "state_a", "state_b": "state_b"}
        )


def test_hard_negative_selection_and_paired_history_aggregation() -> None:
    result = hard_negative_margins(
        target_candidates=[
            {"start": 0, "end": 4, "score": 5.0},
            {"start": 10, "end": 14, "score": 4.0},
            {"start": 30, "end": 34, "score": 20.0},
        ],
        queried_references=[[0, 4]],
        sibling_references=[[10, 14]],
        auxiliary_tracks=[
            {
                "track_id": "T01",
                "co_visible_intervals": [[0, 4]],
                "candidates": [
                    {"start": 0, "end": 4, "score": 9.0},
                    {"start": 20, "end": 22, "score": 3.0},
                ],
            }
        ],
    )
    assert result["positive"]["score"] == 5.0
    assert result["sibling_negative"]["score"] == 4.0
    assert result["joint_negative"]["score"] == 3.0
    assert result["temporal_margin"] == 1.0
    assert result["double_margin"] == 2.0

    baseline = [
        {"history_id": "H01", "query_id": "Q01", "temporal_margin": 0.0},
        {"history_id": "H01", "query_id": "Q02", "temporal_margin": 1.0},
        {"history_id": "H02", "query_id": "Q03", "temporal_margin": 3.0},
        {"history_id": "H02", "query_id": "Q04", "temporal_margin": 5.0},
    ]
    treatment = [
        {"history_id": "H01", "query_id": "Q01", "temporal_margin": 2.0},
        {"history_id": "H01", "query_id": "Q02", "temporal_margin": 5.0},
        {"history_id": "H02", "query_id": "Q03", "temporal_margin": 2.0},
        {"history_id": "H02", "query_id": "Q04", "temporal_margin": 6.0},
    ]
    summary = paired_margin_summary(
        baseline,
        treatment,
        margin_field="temporal_margin",
        bootstrap_resamples=200,
    )
    assert summary["history_median_deltas"] == [
        {"history_id": "H01", "delta": 3.0},
        {"history_id": "H02", "delta": 0.0},
    ]
    assert summary["reported_history_macro_delta"] == 1.5
    assert summary["num_positive_pairs"] == 3


def test_paraphrase_consistency_uses_state_then_source_component_aggregation() -> None:
    rows = [
        {"source_component_id": "C01", "state_id": "a", "window": [0, 3]},
        {"source_component_id": "C01", "state_id": "a", "window": [0, 3]},
        {"source_component_id": "C01", "state_id": "b", "window": [0, 3]},
        {"source_component_id": "C01", "state_id": "b", "window": [2, 5]},
        {"source_component_id": "C02", "state_id": "a", "window": [0, 1]},
        {"source_component_id": "C02", "state_id": "a", "window": [3, 4]},
    ]
    result = paraphrase_window_consistency(rows)
    assert result["source_components"][0] == pytest.approx(
        {
            "source_component_id": "C01",
            "paraphrase_window_tiou": 2.0 / 3.0,
            "paraphrase_exact_agreement": 0.5,
        }
    )
    assert result["reported_source_component_macro"] == pytest.approx(
        {
            "paraphrase_window_tiou": 1.0 / 3.0,
            "paraphrase_exact_agreement": 0.25,
        }
    )


def test_recurrence_metrics_use_ranked_one_to_one_episode_matching() -> None:
    references = [[0, 4], [10, 14]]
    predictions = [
        {"start": 0, "end": 4, "score": 0.9},
        {"start": 0, "end": 4, "score": 0.8},
        {"start": 10, "end": 14, "score": 0.7},
    ]
    assert average_precision(predictions, references, threshold=0.5) == pytest.approx(5.0 / 6.0)
    metrics = recurrence_metrics(predictions, references)
    assert metrics == pytest.approx(
        {
            "multi_window_map_tiou_0.3_0.5_0.7": 5.0 / 6.0,
            "multi_window_recall_tiou_0.5": 1.0,
            "multi_window_precision_tiou_0.5": 2.0 / 3.0,
            "episode_count_absolute_error": 1.0,
        }
    )


def test_recurrence_api_matches_reference_results() -> None:
    payload = json.loads(
        (ROOT / "data" / "reference" / "robustness" / "dnerf_recurrence.json").read_text(
            encoding="utf-8"
        )
    )
    calculated = summarize_recurrence_conditions(
        payload["queries"], payload["references"], payload["predictions"]
    )
    assert set(calculated) == set(payload["reported_metrics"])
    for condition, metrics in calculated.items():
        assert metrics == pytest.approx(payload["reported_metrics"][condition])
