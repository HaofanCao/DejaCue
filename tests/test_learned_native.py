"""Dependency-light tests for native decoder source and configuration locks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from deja_cue.data import History, Query
from deja_cue.learned.development import load_development_histories
from deja_cue.learned.native import (
    LIGHTHOUSE_COMMIT,
    SIM_DETR_COMMIT,
    native_training_adapter,
    resolve_native_config,
    verify_vendored_source,
)
from deja_cue.learned.protocol import DECODER_REGISTRY
from deja_cue.learned.records import (
    build_evaluation_run_records,
    build_positive_run_records,
)
from deja_cue.learned.training import validate_training_records


MAXIMUM_DEVELOPMENT_RUN_TOKENS = 204


def _unit(value: int) -> np.ndarray:
    vector = np.zeros(768, dtype=np.float32)
    vector[value] = 1.0
    return vector


def test_bundled_native_sources_match_immutable_locks() -> None:
    lighthouse = verify_vendored_source("lighthouse")
    sim_detr = verify_vendored_source("sim_detr")
    assert lighthouse["upstream_commit"] == LIGHTHOUSE_COMMIT
    assert sim_detr["upstream_commit"] == SIM_DETR_COMMIT
    assert lighthouse["file_count"] >= 20
    assert sim_detr["file_count"] >= 10


def test_packaged_development_data_produces_exact_training_records() -> None:
    histories = load_development_histories()
    records = build_positive_run_records(histories)
    assert len(histories) == 5
    assert sum(len(history.references) for history in histories) == 13
    assert sum(len(history.queries) for history in histories) == 26
    assert sum(
        len(intervals)
        for history in histories
        for intervals in history.references.values()
    ) == 46
    assert sum(len(history.frame_indices) for history in histories) == 3559
    assert len(records) == 86
    assert [history.sequence_id for history in histories] == [
        "hand",
        "banana",
        "lemon",
        "cookie",
        "toy_container",
    ]


def test_all_eight_native_configs_are_resolved_from_the_paper_lock() -> None:
    for model_id, spec in DECODER_REGISTRY.items():
        config = resolve_native_config(
            model_id,
            device="cuda:0",
            maximum_video_tokens=(
                MAXIMUM_DEVELOPMENT_RUN_TOKENS
                if model_id == "sim_detr"
                else None
            ),
        )
        assert config["v_feat_dim"] == 770
        assert config["t_feat_dim"] == 769
        assert config["span_loss_type"] == "l1"
        assert config["num_queries"] == (30 if model_id == "uvcom" else 10)
        assert spec.backend in {"lighthouse", "sim_detr"}
    assert resolve_native_config(
        "taskweave_mr2hd", device="cuda:0"
    )["max_v_l"] == 75
    assert resolve_native_config(
        "sim_detr",
        device="cuda:0",
        maximum_video_tokens=MAXIMUM_DEVELOPMENT_RUN_TOKENS,
    )["dec_layers"] == 4


def test_sim_adapter_requires_the_observed_development_horizon() -> None:
    with pytest.raises(ValueError, match="maximum development run length"):
        native_training_adapter("sim_detr")
    assert native_training_adapter(
        "sim_detr", maximum_video_tokens=MAXIMUM_DEVELOPMENT_RUN_TOKENS
    ).model_id == "sim_detr"


@dataclass(frozen=True)
class _ExampleRecord:
    history_id: str
    state_id: str
    query_index: int
    index: int
    num_frames: int

    @property
    def record_id(self) -> str:
        return f"record-{self.index}"

    @property
    def has_target(self) -> bool:
        return True


def test_taskweave_horizon_does_not_exclude_long_shared_records() -> None:
    records = []
    state_counts = (3, 3, 3, 2, 2)
    record_counts = (18, 18, 18, 16, 16)
    index = 0
    for history_index, (states, count) in enumerate(
        zip(state_counts, record_counts)
    ):
        for local_index in range(count):
            query_index = local_index % (states * 2)
            records.append(
                _ExampleRecord(
                    history_id=f"D{history_index + 1:02d}",
                    state_id=f"state-{query_index // 2}",
                    query_index=query_index,
                    index=index,
                    num_frames=(
                        MAXIMUM_DEVELOPMENT_RUN_TOKENS if index == 0 else 8
                    ),
                )
            )
            index += 1
    record_counts = validate_training_records(records, model_id="taskweave_mr2hd")
    assert record_counts == {
        "records": 86,
        "histories": 5,
        "states": 13,
        "descriptions": 26,
    }


def test_evaluation_record_construction_is_label_free_and_covers_all_runs() -> None:
    history = History(
        history_id="S01",
        sequence_id="example",
        source_component_id="dataset/example/object",
        frame_indices=np.asarray([0, 1, 4, 5], dtype=np.int64),
        visual_features=np.stack([_unit(0), _unit(1), _unit(2), _unit(3)]),
        visibility_count=np.ones(4, dtype=np.int64),
        queries=(
            Query("before", "before state", _unit(4)),
            Query("after", "after state", _unit(5)),
        ),
        references={"before": ((0, 1),), "after": ((4, 5),)},
    )
    records = build_evaluation_run_records((history,))
    assert len(records) == 4
    assert {(record.query_index, record.run_index) for record in records} == {
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    }
    assert not any(record.has_target for record in records)
    assert {record.native_history_id for record in records} == {
        "dataset/example/object"
    }
