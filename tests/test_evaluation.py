"""Focused regression tests for evaluation-only regrouping."""

from __future__ import annotations

import numpy as np
import pytest

from deja_cue.data import History, Query
from deja_cue.evaluation import summarize_duration_strata


def test_duration_strata_use_included_reference_episodes() -> None:
    state_ids = ("pre_a", "pre_b", "post_c", "post_d")
    history = History(
        history_id="H001",
        sequence_id="sequence",
        source_component_id="component",
        frame_indices=np.arange(4),
        visual_features=np.eye(4, dtype=np.float32),
        visibility_count=np.ones(4, dtype=np.int64),
        queries=tuple(
            Query(state_id, f"query_{index}", np.ones(4, dtype=np.float32) / 2.0)
            for index, state_id in enumerate(state_ids)
        ),
        references={
            "pre_a": ((0, 0),),
            "pre_b": ((0, 1),),
            "post_c": ((0, 2),),
            "post_d": ((0, 3),),
        },
    )
    query_rows = [
        {
            "state_id": state_id,
            "top1_tiou": value,
            "r1_tiou_0.3": float(value >= 0.3),
            "r1_tiou_0.5": float(value >= 0.5),
        }
        for state_id, value in zip(state_ids, (0.0, 0.4, 0.6, 1.0))
    ]
    result = summarize_duration_strata(
        (history,), {"histories": [{"history_id": "H001", "queries": query_rows}]}
    )
    assert result["duration_quartile_boundaries"] == pytest.approx([1.75, 2.5, 3.25])
    assert [
        result["by_duration"][name]["num_descriptions"]
        for name in ("q1_shortest", "q2", "q3", "q4_longest")
    ] == [1, 1, 1, 1]
