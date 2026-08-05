"""Strict two-pass annotation and consensus validation for VOST.

The validator requires complete adjudication. Agreement frames are copied
unchanged, every disagreement index must be listed as adjudicated, and every
consensus label must be one of the four labels in the finalized paper. Distinct
annotator tokens identify the two independent passes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .vost_protocol import (
    FRAME_LABELS,
    derive_qualifying_events,
    select_designated_event,
)


@dataclass(frozen=True)
class AnnotationHistory:
    """One complete framewise label sequence from an annotation file."""

    history_id: str
    sequence_id: str
    source_frame_numbers: tuple[int, ...]
    labels: tuple[str, ...]


def _history_rows(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = payload.get("histories")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Annotation file must contain non-empty histories")
    if any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("Every annotation history must be an object")
    return rows


def _parse_history(row: Mapping[str, Any]) -> AnnotationHistory:
    history_id = str(row.get("history_id", "")).strip()
    sequence_id = str(row.get("sequence_id", "")).strip()
    frames = row.get("source_frame_numbers")
    labels = row.get("labels")
    if not history_id or not sequence_id:
        raise ValueError("Annotation history identities must be non-empty")
    if not isinstance(frames, list) or not isinstance(labels, list):
        raise ValueError(f"History {history_id} lacks frame or label arrays")
    if not frames or len(frames) != len(labels):
        raise ValueError(f"History {history_id} has incomplete framewise labels")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in frames):
        raise ValueError(f"History {history_id} has non-integer source frames")
    frame_numbers = tuple(int(value) for value in frames)
    if any(b <= a for a, b in zip(frame_numbers, frame_numbers[1:])):
        raise ValueError(f"History {history_id} source frames are not increasing")
    normalized_labels = tuple(str(value) for value in labels)
    unknown = sorted(set(normalized_labels).difference(FRAME_LABELS))
    if unknown:
        raise ValueError(f"History {history_id} has unknown labels: {unknown}")
    return AnnotationHistory(
        history_id=history_id,
        sequence_id=sequence_id,
        source_frame_numbers=frame_numbers,
        labels=normalized_labels,
    )


def validate_annotation_pass(
    payload: Mapping[str, Any], *, expected_pass_id: str | None = None
) -> tuple[str, str, dict[str, AnnotationHistory]]:
    """Validate one complete, score-blind independent annotation pass."""

    if payload.get("kind") != "deja_cue_vost_annotation_pass":
        raise ValueError("Unexpected VOST annotation-pass kind")
    pass_id = str(payload.get("pass_id", "")).strip()
    if pass_id not in {"A", "B"}:
        raise ValueError("Annotation pass_id must be A or B")
    if expected_pass_id is not None and pass_id != expected_pass_id:
        raise ValueError(f"Expected annotation pass {expected_pass_id}, got {pass_id}")
    annotator_token = str(payload.get("annotator_token", "")).strip()
    if not annotator_token:
        raise ValueError("Annotation pass requires a non-empty annotator_token")
    if payload.get("status") != "complete":
        raise ValueError(f"Annotation pass {pass_id} is not complete")
    if payload.get("method_scores_visible") is not False:
        raise ValueError(f"Annotation pass {pass_id} was not score-blind")

    histories: dict[str, AnnotationHistory] = {}
    for row in _history_rows(payload):
        history = _parse_history(row)
        if history.history_id in histories:
            raise ValueError(f"Duplicate annotation history: {history.history_id}")
        histories[history.history_id] = history
    return pass_id, annotator_token, histories


def validate_consensus(
    pass_a: Mapping[str, Any],
    pass_b: Mapping[str, Any],
    consensus: Mapping[str, Any],
) -> dict[str, AnnotationHistory]:
    """Validate two distinct passes and complete disagreement reconciliation."""

    _, annotator_a, histories_a = validate_annotation_pass(
        pass_a, expected_pass_id="A"
    )
    _, annotator_b, histories_b = validate_annotation_pass(
        pass_b, expected_pass_id="B"
    )
    if annotator_a == annotator_b:
        raise ValueError("Pass A and pass B must use distinct annotator tokens")
    if set(histories_a) != set(histories_b):
        raise ValueError("Annotation passes cover different history rosters")
    if consensus.get("kind") != "deja_cue_vost_annotation_consensus":
        raise ValueError("Unexpected VOST consensus kind")
    if consensus.get("status") != "complete":
        raise ValueError("Consensus file is not complete")
    if consensus.get("method_scores_visible") is not False:
        raise ValueError("Consensus review was not score-blind")

    consensus_rows = {
        str(row.get("history_id", "")): row for row in _history_rows(consensus)
    }
    if len(consensus_rows) != len(_history_rows(consensus)):
        raise ValueError("Consensus file repeats a history ID")
    if set(consensus_rows) != set(histories_a):
        raise ValueError("Consensus does not cover both complete pass rosters")

    validated: dict[str, AnnotationHistory] = {}
    for history_id in sorted(histories_a):
        left = histories_a[history_id]
        right = histories_b[history_id]
        if (
            left.sequence_id != right.sequence_id
            or left.source_frame_numbers != right.source_frame_numbers
        ):
            raise ValueError(f"Pass geometry differs for {history_id}")
        row = consensus_rows[history_id]
        if str(row.get("sequence_id", "")) != left.sequence_id:
            raise ValueError(f"Consensus sequence identity differs for {history_id}")
        labels = row.get("labels")
        adjudicated = row.get("adjudicated_indices")
        if row.get("review_complete") is not True:
            raise ValueError(f"Consensus review is incomplete for {history_id}")
        if not isinstance(labels, list) or len(labels) != len(left.labels):
            raise ValueError(f"Consensus labels are incomplete for {history_id}")
        normalized = tuple(str(value) for value in labels)
        if set(normalized).difference(FRAME_LABELS):
            raise ValueError(f"Consensus contains an unknown label for {history_id}")
        if not isinstance(adjudicated, list) or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in adjudicated
        ):
            raise ValueError(f"Consensus adjudication indices are invalid for {history_id}")
        disagreements = {
            index
            for index, (label_a, label_b) in enumerate(
                zip(left.labels, right.labels)
            )
            if label_a != label_b
        }
        observed_indices = [int(value) for value in adjudicated]
        if len(observed_indices) != len(set(observed_indices)):
            raise ValueError(f"Consensus repeats an adjudication index for {history_id}")
        if set(observed_indices) != disagreements:
            missing = sorted(disagreements.difference(observed_indices))
            extra = sorted(set(observed_indices).difference(disagreements))
            raise ValueError(
                f"Consensus disagreement coverage differs for {history_id}: "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
        for index, (label_a, label_b, final_label) in enumerate(
            zip(left.labels, right.labels, normalized)
        ):
            if label_a == label_b and final_label != label_a:
                raise ValueError(
                    f"Consensus changed an agreement frame in {history_id} at {index}"
                )
        validated[history_id] = AnnotationHistory(
            history_id=history_id,
            sequence_id=left.sequence_id,
            source_frame_numbers=left.source_frame_numbers,
            labels=normalized,
        )
    return validated


def build_consensus_template(
    pass_a: Mapping[str, Any], pass_b: Mapping[str, Any]
) -> dict[str, Any]:
    """Create a review template while leaving every disagreement unresolved."""

    _, annotator_a, histories_a = validate_annotation_pass(
        pass_a, expected_pass_id="A"
    )
    _, annotator_b, histories_b = validate_annotation_pass(
        pass_b, expected_pass_id="B"
    )
    if annotator_a == annotator_b:
        raise ValueError("Pass A and pass B must use distinct annotator tokens")
    if set(histories_a) != set(histories_b):
        raise ValueError("Annotation passes cover different history rosters")
    rows: list[dict[str, Any]] = []
    disagreement_count = 0
    for history_id in sorted(histories_a):
        left = histories_a[history_id]
        right = histories_b[history_id]
        if (
            left.sequence_id != right.sequence_id
            or left.source_frame_numbers != right.source_frame_numbers
        ):
            raise ValueError(f"Pass geometry differs for {history_id}")
        labels: list[str | None] = []
        disagreement_indices: list[int] = []
        for index, (label_a, label_b) in enumerate(zip(left.labels, right.labels)):
            if label_a == label_b:
                labels.append(label_a)
            else:
                labels.append(None)
                disagreement_indices.append(index)
        disagreement_count += len(disagreement_indices)
        rows.append(
            {
                "history_id": history_id,
                "sequence_id": left.sequence_id,
                "source_frame_numbers": list(left.source_frame_numbers),
                "pass_A_labels": list(left.labels),
                "pass_B_labels": list(right.labels),
                "labels": labels,
                "adjudicated_indices": [],
                "disagreement_indices": disagreement_indices,
                "review_complete": False,
            }
        )
    return {
        "schema_version": 1,
        "kind": "deja_cue_vost_annotation_consensus",
        "status": "incomplete",
        "method_scores_visible": False,
        "adjudicator_token": None,
        "disagreement_count": disagreement_count,
        "histories": rows,
    }


def derive_consensus_cohort(
    histories: Mapping[str, AnnotationHistory],
    *,
    minimum_stable_frames: int = 5,
) -> dict[str, Any]:
    """Derive eligibility and the earliest event from validated consensus labels."""

    rows: list[dict[str, Any]] = []
    retained = 0
    for history_id in sorted(histories):
        history = histories[history_id]
        events = derive_qualifying_events(
            history.labels,
            history.source_frame_numbers,
            minimum_stable_frames=minimum_stable_frames,
        )
        selected = select_designated_event(events)
        evaluation_history_id = None
        if selected is not None:
            retained += 1
            evaluation_history_id = f"H{retained:03d}"
        rows.append(
            {
                "history_id": history_id,
                "evaluation_history_id": evaluation_history_id,
                "sequence_id": history.sequence_id,
                "status": "retained" if selected is not None else "excluded",
                "qualifying_event_count": len(events),
                "selected_event_index": 0 if selected is not None else None,
                "selected_event": selected.to_dict() if selected is not None else None,
                "exclusion_reason": (
                    None if selected is not None else "no_qualifying_event"
                ),
            }
        )
    return {
        "schema_version": 1,
        "kind": "deja_cue_vost_consensus_cohort",
        "designated_event": "earliest_qualifying_event",
        "minimum_stable_frames_per_side": int(minimum_stable_frames),
        "counts": {
            "sampled_histories": len(rows),
            "retained_histories": retained,
            "excluded_histories": len(rows) - retained,
        },
        "roster": rows,
    }


def validate_derived_cohort_against_reference(
    derived: Mapping[str, Any], reference: Mapping[str, Any]
) -> None:
    """Require consensus-derived eligibility and intervals to match the release asset."""

    observed_rows = derived.get("roster")
    expected_rows = reference.get("roster")
    if not isinstance(observed_rows, list) or not isinstance(expected_rows, list):
        raise ValueError("Derived or reference cohort lacks a roster")
    if len(observed_rows) != len(expected_rows):
        raise ValueError("Derived and reference cohort sizes differ")
    keys = (
        "history_id",
        "evaluation_history_id",
        "sequence_id",
        "status",
        "qualifying_event_count",
        "selected_event_index",
        "selected_event",
        "exclusion_reason",
    )
    for observed, expected in zip(observed_rows, expected_rows):
        if any(observed.get(key) != expected.get(key) for key in keys):
            raise ValueError(
                "Consensus-derived cohort differs from the bundled reference at "
                f"{expected.get('history_id')}"
            )
