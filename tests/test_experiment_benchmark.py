"""Tests for the scan benchmark schema."""

from __future__ import annotations

import numpy as np
import pytest

from deja_cue.data import History, Query
from deja_cue.experiments.benchmark import benchmark_scan, build_benchmark_payload


def _history() -> History:
    return History(
        history_id="H001",
        sequence_id="sequence_001",
        source_component_id="component_001",
        frame_indices=np.arange(6, dtype=np.int64),
        visual_features=np.asarray(
            [[1.0, 0.0], [1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [0.0, 1.0], [0.6, 0.8]],
            dtype=np.float32,
        ),
        visibility_count=np.ones(6, dtype=np.int64),
        queries=(
            Query("state_a", "state a", np.asarray([1.0, 0.0], dtype=np.float32)),
            Query("state_b", "state b", np.asarray([0.0, 1.0], dtype=np.float32)),
        ),
        references={"state_a": ((0, 2),), "state_b": ((3, 5),)},
    )


def test_benchmark_payload_schema_and_summary() -> None:
    payload = build_benchmark_payload(
        [1.0, 2.0, 3.0],
        device="cpu",
        history_count=2,
        query_count=4,
        warmup_repetitions=1,
        window_count=33,
    )
    assert payload["measurement"]["median_ms_per_query"] == 2.0
    assert set(payload) == {
        "schema_version",
        "kind",
        "device",
        "unit",
        "measurement",
        "method",
    }


def test_scan_benchmark_executes_decoder() -> None:
    pytest.importorskip("torch")
    payload = benchmark_scan(
        (_history(),),
        (1, 2, 3),
        device="cpu",
        warmup_repetitions=0,
        timed_repetitions=1,
    )
    assert payload["device"] == "cpu"
    assert payload["measurement"]["history_count"] == 1
    assert payload["measurement"]["query_count_per_repetition"] == 2
    assert payload["measurement"]["median_ms_per_query"] > 0.0
