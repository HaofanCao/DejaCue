"""Pure VOST cohort and object-local preprocessing utilities.

The functions in this module implement the fixed preprocessing steps used by
the VOST evaluation. They operate on arrays and labels supplied by the caller;
the caller handles model loading and experiment orchestration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


FRAME_LABELS = ("pre", "transition", "post", "unobserved")
MINIMUM_STABLE_FRAMES = 5
PROMPT_COUNT = 3


@dataclass(frozen=True)
class LabelRun:
    """One maximal, contiguous run of a framewise annotation label."""

    label: str
    start: int
    end: int

    @property
    def length(self) -> int:
        """Inclusive run length in evaluation frames."""

        return self.end - self.start + 1


@dataclass(frozen=True)
class FrameInterval:
    """Inclusive evaluation and source-frame bounds for one label run."""

    evaluation_start: int
    evaluation_end: int
    source_frame_start: int
    source_frame_end: int

    @property
    def length(self) -> int:
        """Inclusive interval length in evaluation frames."""

        return self.evaluation_end - self.evaluation_start + 1

    def to_dict(self, *, include_length: bool = True) -> dict[str, int]:
        """Serialize paired evaluation/source bounds with an optional length."""

        result = {
            "evaluation_start": self.evaluation_start,
            "evaluation_end": self.evaluation_end,
            "source_frame_start": self.source_frame_start,
            "source_frame_end": self.source_frame_end,
        }
        if include_length:
            result["length"] = self.length
        return result


@dataclass(frozen=True)
class QualifyingEvent:
    """An adjacent stable pre-state, transition, and stable post-state."""

    pre: FrameInterval
    transition: FrameInterval
    post: FrameInterval

    def to_dict(self) -> dict[str, Any]:
        """Serialize the event using explicit inclusive interval fields."""

        return {
            "pre_interval": self.pre.to_dict(),
            "transition_interval": self.transition.to_dict(include_length=False),
            "post_interval": self.post.to_dict(),
        }


def _validate_labels(labels: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(label) for label in labels)
    if not normalized:
        raise ValueError("At least one frame label is required")
    unknown = sorted(set(normalized).difference(FRAME_LABELS))
    if unknown:
        raise ValueError(f"Unknown frame labels: {unknown}")
    return normalized


def label_runs(labels: Sequence[str]) -> tuple[LabelRun, ...]:
    """Return all maximal label runs in temporal order.

    ``unobserved`` and ``transition`` runs are retained because either one
    breaks direct pre/post adjacency.  This prevents an event from silently
    spanning an occlusion or another ambiguous interval.
    """

    normalized = _validate_labels(labels)
    runs: list[LabelRun] = []
    start = 0
    for index in range(1, len(normalized) + 1):
        if index == len(normalized) or normalized[index] != normalized[start]:
            runs.append(LabelRun(normalized[start], start, index - 1))
            start = index
    return tuple(runs)


def stable_runs(labels: Sequence[str], label: str) -> tuple[LabelRun, ...]:
    """Return the maximal runs for one stable-state label."""

    if label not in {"pre", "post"}:
        raise ValueError("stable_runs accepts only 'pre' or 'post'")
    return tuple(run for run in label_runs(labels) if run.label == label)


def _frame_interval(run: LabelRun, frame_numbers: Sequence[int]) -> FrameInterval:
    return FrameInterval(
        evaluation_start=run.start,
        evaluation_end=run.end,
        source_frame_start=int(frame_numbers[run.start]),
        source_frame_end=int(frame_numbers[run.end]),
    )


def derive_qualifying_events(
    labels: Sequence[str],
    frame_numbers: Sequence[int] | None = None,
    *,
    minimum_stable_frames: int = MINIMUM_STABLE_FRAMES,
) -> tuple[QualifyingEvent, ...]:
    """Derive all events satisfying the fixed VOST eligibility rule.

    A qualifying event is a maximal ``pre`` run followed immediately by a
    ``transition`` run and then a maximal ``post`` run. Both stable
    runs must contain at least ``minimum_stable_frames`` sampled frames.
    """

    normalized = _validate_labels(labels)
    if (
        isinstance(minimum_stable_frames, bool)
        or not isinstance(minimum_stable_frames, Integral)
        or minimum_stable_frames <= 0
    ):
        raise ValueError("minimum_stable_frames must be a positive integer")
    if frame_numbers is None:
        sources = tuple(range(len(normalized)))
    else:
        if len(frame_numbers) != len(normalized):
            raise ValueError("labels and frame_numbers differ in length")
        sources = tuple(int(value) for value in frame_numbers)
        if any(later <= earlier for earlier, later in zip(sources, sources[1:])):
            raise ValueError("frame_numbers must be strictly increasing")

    runs = label_runs(normalized)
    events: list[QualifyingEvent] = []
    for run_index, pre_run in enumerate(runs):
        if pre_run.label != "pre" or pre_run.length < minimum_stable_frames:
            continue

        next_index = run_index + 1
        if next_index >= len(runs) or runs[next_index].label != "transition":
            continue
        transition_run = runs[next_index]
        next_index += 1
        if next_index >= len(runs):
            continue

        post_run = runs[next_index]
        if post_run.label != "post" or post_run.length < minimum_stable_frames:
            continue
        events.append(
            QualifyingEvent(
                pre=_frame_interval(pre_run, sources),
                transition=_frame_interval(transition_run, sources),
                post=_frame_interval(post_run, sources),
            )
        )
    return tuple(events)


def select_designated_event(
    events: Sequence[QualifyingEvent],
) -> QualifyingEvent | None:
    """Select the earliest qualifying event, or ``None`` when none exists."""

    if not events:
        return None
    ordered = sorted(events, key=lambda event: event.pre.evaluation_start)
    return ordered[0]


def union_lineage_mask(
    label_mask: np.ndarray,
    lineage_label_ids: Sequence[int],
    *,
    ignored_label: int = 255,
) -> np.ndarray:
    """Union all supplied target-lineage labels into one boolean mask.

    The ignored label is always removed, even if it is accidentally supplied
    as a lineage identifier.  Descendant labels therefore form one tracked
    object observation without admitting unrelated or void pixels.
    """

    labels = np.asarray(label_mask)
    if labels.ndim != 2:
        raise ValueError("label_mask must be a two-dimensional array")
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError("label_mask must contain integer instance labels")
    if any(
        isinstance(value, bool) or not isinstance(value, Integral)
        for value in lineage_label_ids
    ):
        raise TypeError("Lineage labels must be integers")
    lineage = tuple(int(value) for value in lineage_label_ids)
    if not lineage:
        raise ValueError("At least one lineage label is required")
    if any(value < 0 for value in lineage):
        raise ValueError("Lineage labels must be non-negative integers")
    selected = np.isin(labels, np.asarray(lineage, dtype=labels.dtype))
    return np.asarray(selected & (labels != ignored_label), dtype=bool)


def padded_mask_bounds(
    mask: np.ndarray,
    *,
    padding_fraction: float = 0.20,
    minimum_padding: int = 4,
) -> tuple[int, int, int, int]:
    """Return clipped ``(y0, y1, x0, x1)`` bounds around a visible mask.

    Padding is measured from the larger side of the tight target box.  The
    fixed four-pixel floor matches preprocessing for small target instances.
    End coordinates follow NumPy's exclusive slicing convention.
    """

    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("mask must be a two-dimensional array")
    if not np.isfinite(padding_fraction) or padding_fraction < 0:
        raise ValueError("padding_fraction must be finite and non-negative")
    if (
        isinstance(minimum_padding, bool)
        or not isinstance(minimum_padding, Integral)
        or minimum_padding < 0
    ):
        raise ValueError("minimum_padding must be a non-negative integer")
    ys, xs = np.where(binary)
    if len(xs) == 0:
        raise ValueError("Cannot crop an empty target mask")

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    padding = max(
        int(minimum_padding),
        int(round(max(x1 - x0, y1 - y0) * float(padding_fraction))),
    )
    height, width = binary.shape
    return (
        max(0, y0 - padding),
        min(height, y1 + padding),
        max(0, x0 - padding),
        min(width, x1 + padding),
    )


def apply_neutral_background(
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    background_value: int = 127,
) -> np.ndarray:
    """Replace pixels outside ``mask`` with a fixed neutral RGB value."""

    image = np.asarray(rgb)
    binary = np.asarray(mask, dtype=bool)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("rgb must have shape (height, width, 3)")
    if binary.shape != image.shape[:2]:
        raise ValueError("rgb and mask geometry differ")
    if image.dtype != np.uint8:
        raise TypeError("rgb must use uint8 values")
    if (
        isinstance(background_value, bool)
        or not isinstance(background_value, Integral)
        or not 0 <= background_value <= 255
    ):
        raise ValueError("background_value must be an integer in [0, 255]")
    result = image.copy()
    result[~binary] = np.uint8(background_value)
    return result


def mask_union_padded_crop(
    rgb: np.ndarray,
    label_mask: np.ndarray,
    lineage_label_ids: Sequence[int],
    *,
    padding_fraction: float = 0.20,
    minimum_padding: int = 4,
    background_value: int = 127,
    ignored_label: int = 255,
) -> np.ndarray:
    """Create the object-local RGB crop used before visual encoding."""

    image = np.asarray(rgb)
    union = union_lineage_mask(
        label_mask, lineage_label_ids, ignored_label=ignored_label
    )
    if image.shape[:2] != union.shape:
        raise ValueError("rgb and label_mask geometry differ")
    y0, y1, x0, x1 = padded_mask_bounds(
        union,
        padding_fraction=padding_fraction,
        minimum_padding=minimum_padding,
    )
    return apply_neutral_background(
        image[y0:y1, x0:x1],
        union[y0:y1, x0:x1],
        background_value=background_value,
    )


def prompt_forms(description: str) -> tuple[str, str, str]:
    """Return the three fixed text forms used for one state description."""

    value = str(description).strip()
    if not value:
        raise ValueError("description must be non-empty")
    return value, f"a photo of {value}", f"the {value}"


def average_prompt_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Normalize, average, and renormalize one three-form prompt ensemble."""

    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != PROMPT_COUNT or values.shape[1] == 0:
        raise ValueError("embeddings must have shape (3, feature_dimension)")
    if not np.isfinite(values).all():
        raise ValueError("embeddings contain non-finite values")
    row_norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(row_norms <= 0):
        raise ValueError("Each prompt embedding must have non-zero norm")
    unit_rows = values / row_norms
    mean = unit_rows.mean(axis=0)
    mean_norm = float(np.linalg.norm(mean))
    if mean_norm <= 0:
        raise ValueError("The prompt ensemble has a zero mean vector")
    return np.asarray(mean / mean_norm, dtype=np.float32)


def package_root() -> Path:
    """Resolve the installed archive root from this module location."""

    return Path(__file__).resolve().parents[1]


def validate_cohort_asset(payload: Mapping[str, Any]) -> dict[str, int]:
    """Validate the bundled 100-history cohort data and return its counts."""

    if payload.get("kind") != "deja_cue_vost_cohort":
        raise ValueError("Unexpected VOST cohort asset kind")
    roster = payload.get("roster")
    retained_ids = payload.get("retained_history_ids")
    excluded_ids = payload.get("excluded_history_ids")
    if (
        not isinstance(roster, list)
        or not isinstance(retained_ids, list)
        or not isinstance(excluded_ids, list)
    ):
        raise ValueError("Cohort roster and ID sets must be lists")

    expected_ids = [f"H{index:03d}" for index in range(1, 101)]
    observed_ids = [str(row.get("history_id", "")) for row in roster if isinstance(row, Mapping)]
    if len(observed_ids) != len(roster) or observed_ids != expected_ids:
        raise ValueError("Cohort roster must contain H001 through H100 in order")
    if len(set(retained_ids).intersection(excluded_ids)) != 0:
        raise ValueError("Retained and excluded history IDs overlap")
    if sorted([*retained_ids, *excluded_ids]) != expected_ids:
        raise ValueError("Retained and excluded IDs do not partition the roster")
    evaluation_ids = [
        str(row.get("evaluation_history_id", ""))
        for row in roster
        if row.get("status") == "retained"
    ]
    if evaluation_ids != [f"H{index:03d}" for index in range(1, 79)]:
        raise ValueError("Retained rows do not map to H001 through H078")

    for row in roster:
        history_id = str(row["history_id"])
        event_count = int(row.get("qualifying_event_count", -1))
        if history_id in retained_ids:
            event = row.get("selected_event")
            if (
                row.get("status") != "retained"
                or event_count < 1
                or not isinstance(event, Mapping)
                or row.get("selected_event_index") != 0
                or row.get("exclusion_reason") is not None
                or row.get("eligibility_failure") is not None
            ):
                raise ValueError(f"Retained history lacks an event: {history_id}")
            if (
                int(event["pre_interval"]["length"]) < MINIMUM_STABLE_FRAMES
                or int(event["post_interval"]["length"]) < MINIMUM_STABLE_FRAMES
            ):
                raise ValueError(
                    f"Retained history violates the stable-frame rule: {history_id}"
                )
        else:
            if (
                row.get("status") != "excluded"
                or row.get("evaluation_history_id") is not None
                or event_count != 0
                or row.get("selected_event_index") is not None
                or row.get("selected_event") is not None
                or row.get("exclusion_reason") != "no_qualifying_event"
                or row.get("eligibility_failure")
                not in {
                    "both_stable_sides_absent_or_short",
                    "pre_side_absent_or_short",
                    "post_side_absent_or_short",
                    "stable_sides_not_adjacent_in_required_order",
                }
            ):
                raise ValueError(
                    f"Excluded history has an inconsistent reason: {history_id}"
                )

    counts = payload.get("counts")
    observed = {
        "sampled_histories": len(roster),
        "retained_histories": len(retained_ids),
        "excluded_histories": len(excluded_ids),
    }
    if not isinstance(counts, Mapping) or any(
        int(counts.get(key, -1)) != value for key, value in observed.items()
    ):
        raise ValueError("Cohort counts differ from the roster")
    protocol = payload.get("protocol")
    eligibility = (
        protocol.get("eligibility") if isinstance(protocol, Mapping) else None
    )
    if (
        not isinstance(protocol, Mapping)
        or protocol.get("designated_event") != "earliest_qualifying_event"
        or not isinstance(eligibility, Mapping)
        or eligibility.get("minimum_stable_frames_per_side")
        != MINIMUM_STABLE_FRAMES
        or eligibility.get("required_run_order") != ["pre", "transition", "post"]
        or eligibility.get("other_runs_between_sides_allowed") is not False
    ):
        raise ValueError("Cohort eligibility protocol differs")
    return observed


def load_cohort_asset(path: Path | None = None) -> dict[str, Any]:
    """Load and validate the bundled VOST cohort reference asset."""

    asset_path = (
        package_root() / "data" / "reference" / "vost_cohort.json"
        if path is None
        else Path(path)
    )
    payload = json.loads(asset_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("VOST cohort asset must be a JSON object")
    validate_cohort_asset(payload)
    return payload
