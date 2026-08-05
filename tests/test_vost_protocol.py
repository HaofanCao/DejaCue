"""Tests for the VOST cohort rule and model-independent preprocessing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from deja_cue.vost_protocol import (
    apply_neutral_background,
    average_prompt_embeddings,
    derive_qualifying_events,
    label_runs,
    load_cohort_asset,
    mask_union_padded_crop,
    padded_mask_bounds,
    prompt_forms,
    select_designated_event,
    union_lineage_mask,
    validate_cohort_asset,
)


ROOT = Path(__file__).resolve().parents[1]


def test_event_requires_adjacent_five_frame_stable_runs() -> None:
    labels = ["pre"] * 5 + ["transition"] * 2 + ["post"] * 5
    frames = [index * 12 for index in range(len(labels))]
    events = derive_qualifying_events(labels, frames)
    assert len(events) == 1
    assert events[0].pre.length == 5
    assert events[0].transition is not None
    assert events[0].post.length == 5
    assert events[0].post.source_frame_start == 84

    # An unobserved run breaks adjacency; short stable sides also fail.
    assert not derive_qualifying_events(
        ["pre"] * 5 + ["unobserved"] + ["post"] * 5
    )
    assert not derive_qualifying_events(["pre"] * 5 + ["post"] * 5)
    assert not derive_qualifying_events(
        ["pre"] * 4 + ["transition"] + ["post"] * 5
    )
    assert not derive_qualifying_events(
        ["pre"] * 5 + ["transition"] + ["post"] * 4
    )


def test_earliest_event_is_the_designated_reference() -> None:
    labels = (
        ["pre"] * 5
        + ["transition"]
        + ["post"] * 5
        + ["unobserved"]
        + ["pre"] * 6
        + ["transition"]
        + ["post"] * 7
    )
    events = derive_qualifying_events(labels)
    assert len(events) == 2
    assert select_designated_event(tuple(reversed(events))) == events[0]
    assert select_designated_event(()) is None
    assert [(run.label, run.length) for run in label_runs(labels)][:3] == [
        ("pre", 5),
        ("transition", 1),
        ("post", 5),
    ]


def test_lineage_union_crop_and_neutral_background() -> None:
    rgb = np.full((12, 14, 3), 200, dtype=np.uint8)
    labels = np.zeros((12, 14), dtype=np.uint8)
    labels[5:7, 6:8] = 3
    labels[6, 8] = 7
    labels[5, 7] = 255

    union = union_lineage_mask(labels, [3, 7, 255])
    assert union.sum() == 4
    assert not union[5, 7]
    assert padded_mask_bounds(union) == (1, 11, 2, 13)

    crop = mask_union_padded_crop(rgb, labels, [3, 7])
    assert crop.shape == (10, 11, 3)
    assert np.all(crop[4, 4] == 200)
    assert np.all(crop[0, 0] == 127)
    assert np.all(rgb == 200), "Preprocessing must not mutate the source RGB array"

    with pytest.raises(ValueError, match="empty"):
        padded_mask_bounds(np.zeros((3, 3), dtype=bool))
    with pytest.raises(ValueError, match="geometry"):
        apply_neutral_background(rgb, np.zeros((2, 2), dtype=bool))


def test_three_form_prompt_average() -> None:
    assert prompt_forms("open jar") == (
        "open jar",
        "a photo of open jar",
        "the open jar",
    )
    vectors = np.asarray([[2.0, 0.0], [1.0, 0.0], [0.0, 3.0]], dtype=np.float32)
    observed = average_prompt_embeddings(vectors)
    expected = np.asarray([2.0, 1.0], dtype=np.float32) / np.sqrt(5.0)
    np.testing.assert_allclose(observed, expected, atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(observed), 1.0, atol=1e-7)


def test_bundled_cohort_is_the_fixed_100_to_78_partition() -> None:
    payload = load_cohort_asset()
    assert validate_cohort_asset(payload) == {
        "sampled_histories": 100,
        "retained_histories": 78,
        "excluded_histories": 22,
    }
    assert payload["counts"]["retained_states"] == 156
    assert payload["counts"]["retained_descriptions"] == 312
    assert payload["counts"]["unique_sequences"] == 100
    assert payload["counts"]["unique_action_object_pairs"] == 100
    assert payload["protocol"]["eligibility"]["required_run_order"] == [
        "pre",
        "transition",
        "post",
    ]
    assert payload["excluded_history_ids"] == [
        "H004", "H008", "H009", "H010", "H013", "H014", "H027", "H029",
        "H030", "H036", "H041", "H042", "H047", "H048", "H049", "H054",
        "H067", "H069", "H070", "H079", "H081", "H092",
    ]

    assert set(payload) == {
        "schema_version",
        "kind",
        "protocol",
        "counts",
        "retained_history_ids",
        "excluded_history_ids",
        "roster",
    }
    assert set(payload["roster"][0]) == {
        "history_id",
        "evaluation_history_id",
        "sequence_id",
        "action",
        "target_noun",
        "status",
        "qualifying_event_count",
        "selected_event_index",
        "selected_event",
        "exclusion_reason",
        "eligibility_failure",
        "run_summary",
    }
    retained_rows = [row for row in payload["roster"] if row["status"] == "retained"]
    assert [row["evaluation_history_id"] for row in retained_rows] == [
        f"H{index:03d}" for index in range(1, 79)
    ]
    failures = {
        row["history_id"]: row["eligibility_failure"]
        for row in payload["roster"]
        if row["status"] == "excluded"
    }
    assert failures["H004"] == "stable_sides_not_adjacent_in_required_order"
    assert failures["H009"] == "pre_side_absent_or_short"
    assert failures["H013"] == "post_side_absent_or_short"
    assert failures["H030"] == "both_stable_sides_absent_or_short"

    assert (ROOT / "data" / "reference" / "vost_cohort.json").is_file()
