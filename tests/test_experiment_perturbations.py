"""Tests for deterministic tracking perturbations."""

from __future__ import annotations

import numpy as np

from deja_cue.data import History, Query
from deja_cue.experiments.perturbations import (
    apply_perturbation,
    perturbation_cells,
    perturbation_protocol,
)
from deja_cue.seven_history import SevenHistoryDistractor


def _history() -> History:
    return History(
        history_id="H001",
        sequence_id="sequence_001",
        source_component_id="component_001",
        frame_indices=np.arange(6, dtype=np.int64),
        visual_features=np.tile(
            np.asarray([[1.0, 0.0]], dtype=np.float32), (6, 1)
        ),
        visibility_count=np.ones(6, dtype=np.int64),
        queries=(
            Query("state_a", "state a", np.asarray([1.0, 0.0], dtype=np.float32)),
            Query("state_b", "state b", np.asarray([0.0, 1.0], dtype=np.float32)),
        ),
        references={"state_a": ((0, 2),), "state_b": ((3, 5),)},
    )


def _distractor(track_id: int, feature: tuple[float, float]) -> SevenHistoryDistractor:
    frames = np.arange(2, 6, dtype=np.int64)
    return SevenHistoryDistractor(
        track_id=track_id,
        object_label=f"track_{track_id}",
        frame_indices=frames,
        visual_features=np.tile(np.asarray([feature], dtype=np.float32), (4, 1)),
        visibility_count=np.ones(4, dtype=np.int64),
        co_visible_moments={},
    )


def _condition(family: str, condition_id: str, **parameters: float) -> dict[str, object]:
    return {
        "condition_id": condition_id,
        "family": family,
        "parameters": parameters,
    }


def test_perturbation_matrix_is_complete_and_uses_three_seeds() -> None:
    cells = perturbation_cells()
    assert len(cells) == 19
    assert len({row["condition_id"] for row in cells}) == 19
    protocol = perturbation_protocol()
    assert protocol["num_conditions"] == 19
    assert set(protocol) == {
        "schema_version",
        "coordinate_system",
        "randomness",
        "missing_frame_semantics",
        "contamination_semantics",
        "identity_switch_semantics",
        "num_conditions",
        "cells",
    }
    assert all(row["replicate_seeds"] == [3407, 3408, 3409] for row in cells)


def test_random_missing_is_deterministic_and_does_not_mutate_input() -> None:
    history = _history()
    original = history.visual_features.copy()
    condition = _condition(
        "random_missing_frames", "random_missing_050pct", missing_fraction=0.5
    )
    first = apply_perturbation(history, (), condition, replicate_seed=3407)
    second = apply_perturbation(history, (), condition, replicate_seed=3407)
    np.testing.assert_array_equal(first.history.frame_indices, second.history.frame_indices)
    assert first.details == second.details
    assert len(first.history.frame_indices) == 3
    np.testing.assert_array_equal(history.visual_features, original)
    assert first.history.queries is history.queries
    assert first.history.references is history.references


def test_contiguous_missing_deletes_only_observations_inside_one_interval() -> None:
    condition = _condition(
        "contiguous_missing_frames",
        "contiguous_missing_span_050pct",
        missing_span_fraction=0.5,
    )
    outcome = apply_perturbation(_history(), (), condition, replicate_seed=3407)
    interval = outcome.details["missing_interval"]
    removed = {
        row["frame_index"] for row in outcome.details["changes"]
    }
    assert removed
    assert all(interval[0] <= frame <= interval[1] for frame in removed)
    assert removed.isdisjoint(set(outcome.history.frame_indices.tolist()))


def test_contamination_uses_exact_time_features_and_renormalizes() -> None:
    condition = _condition(
        "distractor_contamination",
        "distractor_contamination_100pct_mix100pct",
        affected_fraction=1.0,
        mixing_weight=1.0,
    )
    distractor = _distractor(8, (0.0, 1.0))
    outcome = apply_perturbation(
        _history(), (distractor,), condition, replicate_seed=3407
    )
    np.testing.assert_allclose(
        outcome.history.visual_features[2:],
        np.tile(np.asarray([[0.0, 1.0]], dtype=np.float32), (4, 1)),
    )
    np.testing.assert_allclose(
        np.linalg.norm(outcome.history.visual_features, axis=1), 1.0
    )
    assert outcome.details["num_exact_time_eligible_observations"] == 4


def test_identity_switch_prefers_lowest_track_id_after_coverage_tie() -> None:
    condition = _condition(
        "identity_switch",
        "identity_switch_after_050pct",
        switch_point_fraction=0.5,
    )
    distractors = (_distractor(8, (0.0, 1.0)), _distractor(7, (-1.0, 0.0)))
    outcome = apply_perturbation(
        _history(), distractors, condition, replicate_seed=3407
    )
    assert outcome.details["selected_source_track_id"] == 7
    assert all(
        row["source_track_id"] == 7
        for row in outcome.details["changes"]
    )
