"""Tests for complete independent-pass and consensus enforcement."""

from __future__ import annotations

import copy

import pytest

from deja_cue.vost_annotation import (
    build_consensus_template,
    derive_consensus_cohort,
    validate_consensus,
)


def _pass(pass_id: str, annotator: str, labels: list[str]) -> dict:
    return {
        "schema_version": 1,
        "kind": "deja_cue_vost_annotation_pass",
        "pass_id": pass_id,
        "annotator_token": annotator,
        "status": "complete",
        "method_scores_visible": False,
        "histories": [
            {
                "history_id": "H001",
                "sequence_id": "0001_cut_test_object",
                "source_frame_numbers": [index * 12 for index in range(len(labels))],
                "labels": labels,
            }
        ],
    }


def _completed_consensus() -> tuple[dict, dict, dict]:
    labels_a = ["pre"] * 5 + ["transition"] + ["post"] * 5
    labels_b = ["pre"] * 6 + ["post"] * 5
    pass_a = _pass("A", "annotator-A", labels_a)
    pass_b = _pass("B", "annotator-B", labels_b)
    consensus = build_consensus_template(pass_a, pass_b)
    row = consensus["histories"][0]
    consensus["status"] = "complete"
    consensus["adjudicator_token"] = "adjudicator"
    row["labels"] = labels_a
    row["adjudicated_indices"] = [5]
    row["review_complete"] = True
    return pass_a, pass_b, consensus


def test_consensus_requires_every_disagreement_and_derives_earliest_event() -> None:
    pass_a, pass_b, consensus = _completed_consensus()
    histories = validate_consensus(pass_a, pass_b, consensus)
    cohort = derive_consensus_cohort(histories)
    assert cohort["counts"] == {
        "sampled_histories": 1,
        "retained_histories": 1,
        "excluded_histories": 0,
    }
    event = cohort["roster"][0]["selected_event"]
    assert event["pre_interval"]["length"] == 5
    assert event["transition_interval"]["source_frame_start"] == 60
    assert event["post_interval"]["length"] == 5


def test_incomplete_or_same_annotator_consensus_is_rejected() -> None:
    pass_a, pass_b, consensus = _completed_consensus()
    missing = copy.deepcopy(consensus)
    missing["histories"][0]["adjudicated_indices"] = []
    with pytest.raises(ValueError, match="coverage differs"):
        validate_consensus(pass_a, pass_b, missing)

    same_annotator = copy.deepcopy(pass_b)
    same_annotator["annotator_token"] = pass_a["annotator_token"]
    with pytest.raises(ValueError, match="distinct annotator"):
        validate_consensus(pass_a, same_annotator, consensus)

    template = build_consensus_template(pass_a, pass_b)
    assert template["disagreement_count"] == 1
    assert template["histories"][0]["labels"][5] is None
    with pytest.raises(ValueError, match="not complete"):
        validate_consensus(pass_a, pass_b, template)
