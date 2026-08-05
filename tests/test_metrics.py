"""Focused tests for hierarchical aggregation and multiplicity correction."""

from __future__ import annotations

import pytest

from deja_cue.metrics import aggregate_histories, holm_adjust, summarize_history


def test_description_then_state_aggregation_order() -> None:
    queries = [
        {"state_id": "a", "top1_tiou": 0.0, "r1_tiou_0.3": 0.0, "r1_tiou_0.5": 0.0},
        {"state_id": "a", "top1_tiou": 1.0, "r1_tiou_0.3": 1.0, "r1_tiou_0.5": 1.0},
        {"state_id": "b", "top1_tiou": 1.0, "r1_tiou_0.3": 1.0, "r1_tiou_0.5": 1.0},
    ]
    row = summarize_history(
        history_id="H001",
        sequence_id="sequence",
        source_component_id="component",
        query_rows=queries,
        state_order=("a", "b"),
    )
    assert row["summary"]["state_macro_target_top1_tiou"] == pytest.approx(0.75)
    aggregate = aggregate_histories([row], bootstrap_resamples=5, seed=3407)
    assert aggregate["source_component_macro"][
        "state_macro_target_top1_tiou"
    ] == pytest.approx(0.75)


def test_holm_adjustment_is_monotone_in_sorted_order() -> None:
    adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})
