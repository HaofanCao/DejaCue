"""Regression tests for source-only VOST round-robin selection."""

from __future__ import annotations

import json
from pathlib import Path

from deja_cue.vost_sampling import (
    build_cohort_selection,
    parse_sequence_id,
    validate_selection_against_cohort,
)


ROOT = Path(__file__).resolve().parents[1]


def test_packaged_splits_reproduce_the_fixed_100_history_roster() -> None:
    selection = build_cohort_selection(
        ROOT / "data" / "protocol" / "vost" / "train.txt",
        ROOT / "data" / "protocol" / "vost" / "validation.txt",
    )
    cohort = json.loads(
        (ROOT / "data" / "reference" / "vost_cohort.json").read_text(
            encoding="utf-8"
        )
    )
    validate_selection_against_cohort(selection, cohort)
    assert selection["seed"] == 3407
    assert selection["state_or_reference_data_accessed"] is False
    assert selection["model_scores_accessed"] is False
    assert selection["counts"] == {
        "candidate_sequences": 642,
        "primary_histories": 100,
        "primary_action_object_pairs": 100,
        "reserve_sequences": 542,
    }
    primary = selection["primary_histories"]
    assert [row["history_id"] for row in primary] == [
        f"H{index:03d}" for index in range(1, 101)
    ]
    assert len({(row["action"], row["target_noun_token"]) for row in primary}) == 100


def test_vost_sequence_parser_keeps_noun_suffixes() -> None:
    assert parse_sequence_id("3511_unscrew_screw_top_jar") == (
        "unscrew",
        "screw_top_jar",
    )
