"""Regression tests for roster loading and feature-condition alignment."""

from __future__ import annotations

import numpy as np

from deja_cue.data import load_histories, load_protocol


def test_primary_roster_and_protocol_counts() -> None:
    histories = load_histories()
    protocol = load_protocol()
    assert [history.history_id for history in histories] == [
        f"H{index:03d}" for index in range(1, 79)
    ]
    assert sum(len(history.references) for history in histories) == 156
    assert sum(len(history.queries) for history in histories) == 312
    assert len({history.source_component_id for history in histories}) == 78
    assert len(protocol["window_schedule"]) == 33


def test_feature_conditions_preserve_query_and_frame_rosters() -> None:
    primary = load_histories()
    prompted = load_histories(prompt_variant="ensemble")
    for baseline, prompt in zip(primary, prompted):
        baseline_queries = [(query.state_id, query.text) for query in baseline.queries]
        assert [
            (query.state_id, query.text) for query in prompt.queries
        ] == baseline_queries
        assert (prompt.frame_indices == baseline.frame_indices).all()
        np.testing.assert_array_equal(
            np.stack([query.embedding for query in prompt.queries]),
            np.stack([query.embedding for query in baseline.queries]),
        )
