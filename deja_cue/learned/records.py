"""Shared record, coordinate, target, and proposal transformations.

The learned comparison changes decoder architecture while holding this module's
output fixed.  Reference intervals are used only to construct development
training targets; label-free evaluation records use the same visual and text
representation with empty targets.

These records implement the common representation defined in the paper appendix,
"Low-Data Decoder Adaptation / Shared Adaptation Protocol."
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .protocol import (
    NUM_PROPOSALS,
    TEXT_FEATURE_DIM,
    TEXT_TOKEN_DIM,
    VIDEO_TOKEN_DIM,
    VISUAL_FEATURE_DIM,
)


_EPS = 1e-8
MAX_RUN_TOKENS = 4096


def _unit_rows(values: np.ndarray, *, label: str) -> np.ndarray:
    rows = np.asarray(values, dtype=np.float32)
    if rows.ndim != 2 or not len(rows) or not np.isfinite(rows).all():
        raise ValueError(f"{label} must be a non-empty finite matrix")
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    if np.any(norms <= _EPS):
        raise ValueError(f"{label} contains a zero vector")
    return np.asarray(rows / norms, dtype=np.float32)


def state_balanced_centroid(
    embeddings: np.ndarray, state_ids: Sequence[str]
) -> np.ndarray:
    """Compute an equal-state centroid from normalized descriptions.

    Description vectors are first normalized.  Descriptions of the same state
    are averaged, and state prototypes then receive equal weight.  The reported
    development set has two descriptions for every state, so this is exactly
    the query-mean centering used by the training executor while making the
    intended state-balanced semantics explicit.
    """

    normalized = _unit_rows(embeddings, label="Text embeddings")
    states = tuple(str(value) for value in state_ids)
    if len(states) != len(normalized) or any(not value for value in states):
        raise ValueError("state_ids must align with all text embeddings")
    ordered_states = tuple(dict.fromkeys(states))
    prototypes = np.stack(
        [
            normalized[np.asarray([value == state for value in states])].mean(axis=0)
            for state in ordered_states
        ]
    )
    return np.asarray(prototypes.mean(axis=0), dtype=np.float32)


def state_balanced_coordinates(
    embeddings: np.ndarray, state_ids: Sequence[str]
) -> np.ndarray:
    """Return unit vocabulary-relative coordinates for all descriptions."""

    normalized = _unit_rows(embeddings, label="Text embeddings")
    centroid = state_balanced_centroid(normalized, state_ids)
    centered = normalized - centroid[None, :]
    centered_norms = np.linalg.norm(centered, axis=1, keepdims=True)

    # A vector can equal the centroid in a degenerate vocabulary.  Retaining its
    # normalized absolute vector is deterministic and avoids division by zero;
    # non-degenerate rows use the vocabulary-relative direction.
    output = normalized.copy()
    stable = centered_norms[:, 0] > _EPS
    output[stable] = centered[stable] / centered_norms[stable]
    return np.asarray(output, dtype=np.float32)


def contiguous_run_slices(frame_indices: np.ndarray) -> tuple[tuple[int, int], ...]:
    """Split observed frame indices into half-open slices of contiguous runs."""

    frames = np.asarray(frame_indices, dtype=np.int64)
    if frames.ndim != 1 or not len(frames) or np.any(np.diff(frames) <= 0):
        raise ValueError("frame_indices must be a non-empty increasing vector")
    boundaries = np.flatnonzero(np.diff(frames) != 1) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(frames)]))
    return tuple((int(start), int(end)) for start, end in zip(starts, ends))


def encode_inclusive_moments(
    moments: Sequence[tuple[int, int]], absolute_frames: np.ndarray
) -> tuple[tuple[tuple[int, int], ...], np.ndarray]:
    """Encode inclusive absolute intervals as local center-width targets.

    Local token spans are half open.  A reference may be outside a visible run,
    but it may not cross a run boundary: such a crossing would silently train on
    a truncated target and therefore fails validation.
    """

    frames = np.asarray(absolute_frames, dtype=np.int64)
    if frames.ndim != 1 or not len(frames) or np.any(np.diff(frames) != 1):
        raise ValueError("absolute_frames must describe one contiguous run")
    run_start, run_end = int(frames[0]), int(frames[-1])
    spans: list[tuple[int, int]] = []
    for raw_start, raw_end in moments:
        start, end = int(raw_start), int(raw_end)
        if start > end:
            raise ValueError(f"Invalid inclusive interval: {(start, end)}")
        overlap_start, overlap_end = max(start, run_start), min(end, run_end)
        if overlap_start > overlap_end:
            continue
        if overlap_start != start or overlap_end != end:
            raise ValueError(
                f"Reference {(start, end)} crosses visible run {(run_start, run_end)}"
            )
        spans.append((start - run_start, end - run_start + 1))
    if len(spans) > NUM_PROPOSALS:
        raise ValueError(f"A run cannot exceed {NUM_PROPOSALS} target intervals")
    length = len(frames)
    normalized = np.empty((len(spans), 2), dtype=np.float32)
    for index, (start, end) in enumerate(spans):
        normalized[index] = (
            (start + end) / (2.0 * length),
            (end - start) / length,
        )
    return tuple(spans), normalized


@dataclass(frozen=True)
class LearnedRunRecord:
    """One description on one contiguous observed run."""

    history_id: str
    state_id: str
    text: str
    query_index: int
    run_index: int
    absolute_frames: np.ndarray
    visual_features: np.ndarray
    text_embedding: np.ndarray
    text_context_embeddings: np.ndarray
    text_context_query_indices: tuple[int, ...]
    target_token_spans: tuple[tuple[int, int], ...]
    normalized_spans: np.ndarray
    native_history_id: str | None = None

    @property
    def num_frames(self) -> int:
        """Number of contiguous observed frames represented by this record."""

        return int(len(self.absolute_frames))

    @property
    def has_target(self) -> bool:
        """Whether at least one reference span lies inside this observed run."""

        return bool(self.target_token_spans)

    @property
    def record_id(self) -> str:
        """Stable identifier for this history/query/run key."""

        identity = (
            f"{self.native_history_id or self.history_id}\0{self.state_id}\0"
            f"{self.query_index}\0{self.run_index}"
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _validate_record(record: LearnedRunRecord) -> None:
    frames = np.asarray(record.absolute_frames)
    visual = np.asarray(record.visual_features)
    text = np.asarray(record.text_embedding)
    context = np.asarray(record.text_context_embeddings)
    if (
        not record.history_id
        or not record.state_id
        or record.query_index < 0
        or record.run_index < 0
    ):
        raise ValueError("Run record identity is invalid")
    if (
        frames.ndim != 1
        or not len(frames)
        or np.any(np.diff(frames) != 1)
        or len(frames) > MAX_RUN_TOKENS
    ):
        raise ValueError("Run record frames are not one supported contiguous run")
    if visual.shape != (len(frames), VISUAL_FEATURE_DIM) or not np.isfinite(visual).all():
        raise ValueError("Run record visual features have the wrong shape")
    if text.shape != (TEXT_FEATURE_DIM,) or not np.isfinite(text).all():
        raise ValueError("Run record target text has the wrong shape")
    if (
        context.ndim != 2
        or context.shape[1] != TEXT_FEATURE_DIM
        or len(context) != len(record.text_context_query_indices)
        or not np.isfinite(context).all()
    ):
        raise ValueError("Run record sibling context has the wrong shape")
    if not record.text_context_query_indices:
        raise ValueError("Run record sibling context is empty")
    if record.text_context_query_indices[0] != record.query_index:
        raise ValueError("The target description must be the first logical text token")
    if len(set(record.text_context_query_indices)) != len(
        record.text_context_query_indices
    ):
        raise ValueError("Sibling context repeats a query")
    if record.normalized_spans.shape != (len(record.target_token_spans), 2):
        raise ValueError("Run record span targets are inconsistent")
    for start, end in record.target_token_spans:
        if not 0 <= start < end <= len(frames):
            raise ValueError("Run record contains an invalid target token span")


def build_run_records(
    histories: Sequence[Any],
    *,
    include_targets: bool,
    positive_only: bool,
) -> tuple[LearnedRunRecord, ...]:
    """Create run records with complete sibling context.

    ``histories`` follows the public :class:`deja_cue.data.History` interface:
    each history supplies observed frames/features, ordered text queries, and a
    mapping from state ID to inclusive reference intervals.  Development
    training sets both flags.  Evaluation sets both flags to false so reference
    intervals cannot influence which visible runs reach the decoder.
    """

    if positive_only and not include_targets:
        raise ValueError("Positive-only records require target construction")

    output: list[LearnedRunRecord] = []
    for history in histories:
        frames = np.asarray(history.frame_indices, dtype=np.int64)
        visual = np.asarray(history.visual_features, dtype=np.float32)
        queries = tuple(history.queries)
        if (
            not queries
            or visual.shape != (len(frames), VISUAL_FEATURE_DIM)
            or not np.isfinite(visual).all()
        ):
            raise ValueError(f"Invalid learned history: {history.history_id}")
        raw_text = np.stack(
            [np.asarray(query.embedding, dtype=np.float32) for query in queries]
        )
        state_ids = tuple(str(query.state_id) for query in queries)
        coordinates = state_balanced_coordinates(raw_text, state_ids)
        run_slices = contiguous_run_slices(frames)

        for query_index, query in enumerate(queries):
            context_indices = (query_index,) + tuple(
                index for index in range(len(queries)) if index != query_index
            )
            context = coordinates[np.asarray(context_indices, dtype=np.int64)].copy()
            if include_targets:
                try:
                    moments = history.references[str(query.state_id)]
                except KeyError as exc:
                    raise ValueError(
                        f"Missing references for {history.history_id}/{query.state_id}"
                    ) from exc
            else:
                moments = ()
            for run_index, (left, right) in enumerate(run_slices):
                run_frames = frames[left:right]
                spans, normalized = encode_inclusive_moments(moments, run_frames)
                if positive_only and not spans:
                    continue
                record = LearnedRunRecord(
                    history_id=str(history.history_id),
                    state_id=str(query.state_id),
                    text=str(query.text),
                    query_index=query_index,
                    run_index=run_index,
                    absolute_frames=run_frames.copy(),
                    visual_features=visual[left:right].copy(),
                    text_embedding=coordinates[query_index].copy(),
                    text_context_embeddings=context,
                    text_context_query_indices=context_indices,
                    target_token_spans=spans,
                    normalized_spans=normalized,
                    native_history_id=str(
                        getattr(history, "source_component_id", history.history_id)
                    ),
                )
                _validate_record(record)
                output.append(record)
    if not output:
        raise ValueError("No learned-decoder run records were constructed")
    return tuple(output)


def build_positive_run_records(histories: Sequence[Any]) -> tuple[LearnedRunRecord, ...]:
    """Create the positive-only five-history development training records."""

    return build_run_records(histories, include_targets=True, positive_only=True)


def build_evaluation_run_records(
    histories: Sequence[Any],
) -> tuple[LearnedRunRecord, ...]:
    """Create label-free records for every contiguous observed evaluation run."""

    return build_run_records(histories, include_targets=False, positive_only=False)


def _saliency_targets(
    record: LearnedRunRecord,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    labels = np.zeros(record.num_frames, dtype=np.float32)
    for start, end in record.target_token_spans:
        labels[start:end] = 1.0
    positive = np.flatnonzero(labels > 0)
    if not len(positive):
        return labels, (0, 0), (0, 0)
    positive_pair = (int(positive[0]), int(positive[-1]))
    negative = np.flatnonzero(labels <= 0)
    negative_pair = (
        (int(negative[0]), int(negative[-1]))
        if len(negative)
        else positive_pair
    )
    return labels, positive_pair, negative_pair


def sample_saliency_pairs(
    record: LearnedRunRecord, *, seed: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Sample native saliency endpoints from the full target-token union."""

    labels, fallback_positive, _fallback_negative = _saliency_targets(record)
    positive = np.flatnonzero(labels > 0)
    negative = np.flatnonzero(labels <= 0)
    if not len(positive):
        raise ValueError("Positive-only training record has no target tokens")
    rng = np.random.default_rng(seed)

    def pair(indices: np.ndarray) -> tuple[int, int]:
        """Sample two distinct tokens, duplicating only a singleton set."""

        if len(indices) >= 2:
            sampled = rng.choice(indices, size=2, replace=False)
            return int(sampled[0]), int(sampled[1])
        return int(indices[0]), int(indices[0])

    positive_pair = pair(positive) if len(positive) else fallback_positive
    # A full-run target has no valid temporal negative.  Reusing the positive
    # pair makes the corresponding margin term constant rather than inventing a
    # false negative token.
    negative_pair = pair(negative) if len(negative) else positive_pair
    return positive_pair, negative_pair


@dataclass(frozen=True)
class PreparedBatch:
    """Padded NumPy representation shared by every decoder adapter."""

    video_tokens: np.ndarray
    video_mask: np.ndarray
    text_tokens: np.ndarray
    text_mask: np.ndarray
    span_targets: tuple[np.ndarray, ...]
    saliency_labels: np.ndarray
    saliency_positive: np.ndarray
    saliency_negative: np.ndarray
    records: tuple[LearnedRunRecord, ...]

    def as_torch(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Convert the prepared batch to the tensor dictionaries used by models."""

        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError("PyTorch is required to execute learned decoders") from exc
        model_inputs = {
            "src_vid": torch.from_numpy(self.video_tokens.copy()),
            "src_vid_mask": torch.from_numpy(self.video_mask.copy()),
            "src_txt": torch.from_numpy(self.text_tokens.copy()),
            "src_txt_mask": torch.from_numpy(self.text_mask.copy()),
        }
        targets = {
            "span_labels": [
                {"spans": torch.from_numpy(value.copy())}
                for value in self.span_targets
            ],
            "saliency_all_labels": torch.from_numpy(self.saliency_labels.copy()),
            "saliency_pos_labels": torch.from_numpy(self.saliency_positive.copy()),
            "saliency_neg_labels": torch.from_numpy(self.saliency_negative.copy()),
        }
        return model_inputs, targets


def collate_records(
    records: Sequence[LearnedRunRecord], *, require_positive: bool = True
) -> PreparedBatch:
    """Pad records and append temporal and target-role channels."""

    selected = tuple(records)
    if not selected:
        raise ValueError("Cannot collate an empty record set")
    for record in selected:
        _validate_record(record)
    if require_positive and not any(record.has_target for record in selected):
        raise ValueError("A training batch must contain a positive record")

    batch_size = len(selected)
    max_video = max(record.num_frames for record in selected)
    max_text = max(len(record.text_context_embeddings) for record in selected)
    video = np.zeros((batch_size, max_video, VIDEO_TOKEN_DIM), dtype=np.float32)
    video_mask = np.zeros((batch_size, max_video), dtype=np.float32)
    text = np.zeros((batch_size, max_text, TEXT_TOKEN_DIM), dtype=np.float32)
    text_mask = np.zeros((batch_size, max_text), dtype=np.float32)
    saliency = np.zeros((batch_size, max_video), dtype=np.float32)
    positive_pairs: list[tuple[int, int]] = []
    negative_pairs: list[tuple[int, int]] = []
    span_targets: list[np.ndarray] = []

    for batch_index, record in enumerate(selected):
        length = record.num_frames
        positions = np.arange(length, dtype=np.float32)
        temporal = np.stack(
            (positions / length, (positions + 1.0) / length), axis=1
        )
        video[batch_index, :length] = np.concatenate(
            (record.visual_features.astype(np.float32, copy=False), temporal), axis=1
        )
        video_mask[batch_index, :length] = 1.0

        text_length = len(record.text_context_embeddings)
        text[batch_index, :text_length, :TEXT_FEATURE_DIM] = (
            record.text_context_embeddings
        )
        text[batch_index, 0, TEXT_FEATURE_DIM] = 1.0
        text_mask[batch_index, :text_length] = 1.0

        labels, positive, negative = _saliency_targets(record)
        saliency[batch_index, :length] = labels
        positive_pairs.append(positive)
        negative_pairs.append(negative)
        span_targets.append(record.normalized_spans.astype(np.float32, copy=True))

    return PreparedBatch(
        video_tokens=video,
        video_mask=video_mask,
        text_tokens=text,
        text_mask=text_mask,
        span_targets=tuple(span_targets),
        saliency_labels=saliency,
        saliency_positive=np.asarray(positive_pairs, dtype=np.int64),
        saliency_negative=np.asarray(negative_pairs, dtype=np.int64),
        records=selected,
    )


def target_last_permutation(
    target_roles: Sequence[float], valid_mask: Sequence[float]
) -> tuple[int, ...]:
    """Move the logical target-first token to the final valid text position."""

    roles = np.asarray(target_roles, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=np.float64)
    if roles.ndim != 1 or mask.shape != roles.shape:
        raise ValueError("Target-role and valid masks must align")
    valid = np.flatnonzero(mask > 0.5)
    if not len(valid) or not np.array_equal(valid, np.arange(len(valid))):
        raise ValueError("Valid text tokens must form a non-empty prefix")
    if not np.all(np.isclose(roles[valid], 0.0) | np.isclose(roles[valid], 1.0)):
        raise ValueError("Target-role values must be binary")
    if tuple(np.flatnonzero(np.isclose(roles[valid], 1.0))) != (0,):
        raise ValueError("Logical text context must contain one target-first token")
    return tuple(int(index) for index in (*valid[1:], valid[0]))


def reorder_sim_detr_text(batch: PreparedBatch) -> np.ndarray:
    """Return Sim-DETR's native target-last view without changing sibling content."""

    reordered = batch.text_tokens.copy()
    for batch_index in range(len(batch.records)):
        permutation = target_last_permutation(
            batch.text_tokens[batch_index, :, -1], batch.text_mask[batch_index]
        )
        reordered[batch_index, : len(permutation)] = batch.text_tokens[
            batch_index, np.asarray(permutation, dtype=np.int64)
        ]
    return reordered


def build_target_masks(
    records: Sequence[LearnedRunRecord], *, maximum_video_tokens: int | None = None
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Build Sim-DETR's union mask and one binary mask per target interval."""

    selected = tuple(records)
    if not selected:
        raise ValueError("Cannot build target masks for an empty record set")
    maximum = maximum_video_tokens or max(record.num_frames for record in selected)
    if maximum <= 0:
        raise ValueError("maximum_video_tokens must be positive")
    union = np.zeros((len(selected), maximum), dtype=np.float32)
    masks: list[np.ndarray] = []
    for batch_index, record in enumerate(selected):
        if record.num_frames > maximum:
            raise ValueError("A record exceeds the target-mask padding length")
        per_span = np.zeros((len(record.target_token_spans), maximum), dtype=np.float32)
        for span_index, (start, end) in enumerate(record.target_token_spans):
            per_span[span_index, start:end] = 1.0
            union[batch_index, start:end] = 1.0
        masks.append(per_span)
    return union, tuple(masks)


@dataclass(frozen=True)
class DecodedProposal:
    """One scored proposal decoded to inclusive absolute source frames."""

    history_id: str
    state_id: str
    query_index: int
    run_index: int
    proposal_index: int
    start: int
    end: int
    score: float


def _nearest_half_open_boundaries(
    center: float, width: float, length: int
) -> tuple[int, int]:
    if not math.isfinite(center) or not math.isfinite(width):
        raise ValueError("Proposal coordinates must be finite")
    if width < 0 or length <= 0:
        raise ValueError("Proposal width and run length are invalid")
    # The four-decimal quantization is part of the decoder adapter.  It removes
    # backend-dependent noise immediately around half-integer frame boundaries.
    raw = np.asarray(
        [
            float(f"{(center - width / 2.0) * length:.4f}"),
            float(f"{(center + width / 2.0) * length:.4f}"),
        ],
        dtype=np.float32,
    )
    rounded = np.clip(np.rint(raw), 0, length).astype(np.int64)
    start = min(length - 1, max(0, int(rounded[0])))
    end = min(length, max(start + 1, int(rounded[1])))
    return start, end


def _softmax_foreground(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float32)
    shifted = values - values.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return np.asarray(exp[:, 0] / exp.sum(axis=-1), dtype=np.float32)


def decode_proposals(
    record: LearnedRunRecord,
    pred_spans: np.ndarray,
    pred_logits: np.ndarray,
    *,
    sim_detr_iou_scores: np.ndarray | None = None,
) -> tuple[DecodedProposal, ...]:
    """Decode center-width predictions to inclusive absolute-frame proposals."""

    spans = np.asarray(pred_spans, dtype=np.float32)
    logits = np.asarray(pred_logits, dtype=np.float32)
    if spans.ndim != 2 or spans.shape[1] != 2:
        raise ValueError("pred_spans must have shape [num_proposals, 2]")
    if logits.shape != (len(spans), 2):
        raise ValueError("pred_logits must have shape [num_proposals, 2]")
    if not np.isfinite(spans).all() or not np.isfinite(logits).all():
        raise ValueError("Proposal predictions must be finite")
    scores = _softmax_foreground(logits)
    if sim_detr_iou_scores is not None:
        iou = np.asarray(sim_detr_iou_scores, dtype=np.float32)
        if iou.shape == (len(spans), 1):
            iou = iou[:, 0]
        if iou.shape != (len(spans),) or not np.isfinite(iou).all():
            raise ValueError("Sim-DETR IoU scores must align with proposals")
        scores = scores * (1.0 / (1.0 + np.exp(-iou)))

    proposals = []
    for proposal_index, (center, width) in enumerate(spans.tolist()):
        local_start, local_end = _nearest_half_open_boundaries(
            float(center), float(width), record.num_frames
        )
        proposals.append(
            DecodedProposal(
                history_id=record.history_id,
                state_id=record.state_id,
                query_index=record.query_index,
                run_index=record.run_index,
                proposal_index=proposal_index,
                start=int(record.absolute_frames[local_start]),
                end=int(record.absolute_frames[local_end - 1]),
                score=float(np.float32(scores[proposal_index])),
            )
        )
    return tuple(proposals)


def select_best_proposal(
    proposals: Sequence[DecodedProposal],
) -> DecodedProposal:
    """Apply the deterministic score and boundary tie break used at evaluation."""

    values = tuple(proposals)
    if not values:
        raise ValueError("Cannot select from an empty proposal set")
    identities = {
        (value.history_id, value.state_id, value.query_index) for value in values
    }
    if len(identities) != 1:
        raise ValueError("Proposals must describe one history-state query")
    return min(
        values,
        key=lambda value: (
            -value.score,
            value.start,
            value.end,
            value.run_index,
            value.proposal_index,
        ),
    )
