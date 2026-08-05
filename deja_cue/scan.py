"""Core vocabulary-relative evidence construction and temporal decoders.

The functions in this module mirror the equations in the paper directly. The
implementation keeps all candidate ordering and tie rules explicit because the
reproduced output must include intervals, not only rounded aggregates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from .data import History


WindowNormalization = Literal["sqrt_valid_count", "sum", "mean"]


@dataclass(frozen=True)
class ScanOutput:
    """All values needed for retrieval, diagnostics, and candidate ranking."""

    windows: tuple[tuple[int, int], ...]
    selected_scores: tuple[float, ...]
    candidate_scores: np.ndarray
    candidate_valid: np.ndarray
    candidate_starts: np.ndarray
    candidate_ends: np.ndarray
    candidate_lengths: np.ndarray
    smoothed_evidence: np.ndarray
    dense_valid: np.ndarray
    first_frame: int
    query_residual_norms: np.ndarray
    query_residual_fallback: np.ndarray
    evidence_centers: np.ndarray
    evidence_scales: np.ndarray


@dataclass(frozen=True)
class SimpleOutput:
    """Prediction rows for one non-learned comparison readout."""

    windows: tuple[tuple[int, int], ...]
    selected_scores: tuple[float, ...]


def _median(values: Tensor, dim: int) -> Tensor:
    """Torch median with the conventional average of two middle values."""

    ordered = values.sort(dim=dim).values
    count = int(ordered.shape[dim])
    midpoint = count // 2
    if count % 2:
        return ordered.select(dim, midpoint)
    return 0.5 * (ordered.select(dim, midpoint - 1) + ordered.select(dim, midpoint))


def contiguous_valid_runs(valid: Tensor) -> tuple[tuple[int, int], ...]:
    """Return half-open maximal runs from a rank-one boolean mask."""

    if valid.ndim != 1 or valid.dtype is not torch.bool:
        raise ValueError("valid must be a rank-one boolean tensor")
    padded = F.pad(valid.to(torch.int8), (1, 1))
    changes = padded[1:] - padded[:-1]
    starts = torch.nonzero(changes == 1, as_tuple=False).flatten().tolist()
    ends = torch.nonzero(changes == -1, as_tuple=False).flatten().tolist()
    return tuple((int(start), int(end)) for start, end in zip(starts, ends))


def contiguous_index_runs(frame_indices: np.ndarray) -> tuple[tuple[int, int], ...]:
    """Return half-open runs in a strictly increasing observed-frame vector."""

    frames = np.asarray(frame_indices, dtype=np.int64)
    if frames.ndim != 1 or not len(frames) or np.any(np.diff(frames) <= 0):
        raise ValueError("frame_indices must be non-empty and strictly increasing")
    boundaries = np.flatnonzero(np.diff(frames) != 1) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(frames)]))
    return tuple((int(start), int(end)) for start, end in zip(starts, ends))


def enumerate_windows(
    history: History, window_sizes: Sequence[int]
) -> tuple[tuple[int, int, int, int], ...]:
    """Enumerate run-local windows as array bounds and inclusive frame bounds."""

    sizes = tuple(dict.fromkeys(int(size) for size in window_sizes))
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("window sizes must be positive")
    windows: list[tuple[int, int, int, int]] = []
    for run_start, run_end in contiguous_index_runs(history.frame_indices):
        run_length = run_end - run_start
        for size in sizes:
            if size > run_length:
                continue
            for local_start in range(run_start, run_end - size + 1):
                local_end = local_start + size
                windows.append(
                    (
                        local_start,
                        local_end,
                        int(history.frame_indices[local_start]),
                        int(history.frame_indices[local_end - 1]),
                    )
                )
    if not windows:
        raise ValueError(f"{history.history_id} has no fitting candidate window")
    return tuple(windows)


def state_balanced_weights(history: History, *, device: torch.device) -> Tensor:
    """Give every state equal total mass, then split it over descriptions."""

    counts: dict[str, int] = {}
    for query in history.queries:
        counts[query.state_id] = counts.get(query.state_id, 0) + 1
    values = np.asarray(
        [1.0 / counts[query.state_id] for query in history.queries], dtype=np.float32
    )
    values /= values.sum()
    return torch.from_numpy(values).to(device=device)


def robust_standardize(
    evidence: Tensor,
    valid: Tensor,
    *,
    mad_factor: float = 1.4826,
    mad_floor: float = 1e-3,
) -> tuple[Tensor, Tensor, Tensor]:
    """Apply per-query median/MAD calibration over visible frames only."""

    values = evidence[:, valid]
    centers = _median(values, dim=1)
    deviations = (values - centers.unsqueeze(1)).abs()
    scales = (mad_factor * _median(deviations, dim=1)).clamp_min(mad_floor)
    standardized = (evidence - centers.unsqueeze(1)) / scales.unsqueeze(1)
    standardized = standardized.masked_fill(~valid.unsqueeze(0), 0.0)
    return standardized, centers.detach(), scales.detach()


def smooth_within_runs(evidence: Tensor, valid: Tensor, kernel_size: int = 3) -> Tensor:
    """Smooth each visible run and normalize the active tap variance.

    For evidence z and uniform weights h, each output is
    `sum(h*z) / sqrt(sum(h^2))`. Near a run boundary only in-run taps enter both
    sums, which prevents evidence leakage and boundary-dependent noise scale.
    """

    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    kernel = torch.full(
        (1, 1, kernel_size),
        1.0 / kernel_size,
        dtype=evidence.dtype,
        device=evidence.device,
    )
    squared = kernel.square()
    padding = kernel_size // 2
    smoothed = torch.zeros_like(evidence)
    for start, end in contiguous_valid_runs(valid):
        run = evidence[:, start:end].reshape(evidence.shape[0], 1, end - start)
        numerator = F.conv1d(run, kernel, padding=padding)
        local_norm2 = F.conv1d(
            torch.ones(
                (1, 1, end - start), dtype=evidence.dtype, device=evidence.device
            ),
            squared,
            padding=padding,
        )
        smoothed[:, start:end] = (
            numerator / local_norm2.clamp_min(1e-12).sqrt()
        ).reshape(evidence.shape[0], end - start)
    return smoothed.masked_fill(~valid.unsqueeze(0), 0.0)


def multiscale_scan(
    evidence: Tensor,
    valid: Tensor,
    window_sizes: Sequence[int],
    *,
    normalization: WindowNormalization,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Score every full-visibility interval in scale-major candidate order."""

    if normalization not in {"sqrt_valid_count", "sum", "mean"}:
        raise ValueError(f"Unsupported normalization: {normalization}")
    total_frames = int(evidence.shape[1])
    evidence_prefix = F.pad(evidence.cumsum(dim=-1), (1, 0))
    valid_prefix = F.pad(valid.to(evidence.dtype).cumsum(dim=-1), (1, 0))
    score_rows: list[Tensor] = []
    valid_rows: list[Tensor] = []
    starts: list[Tensor] = []
    ends: list[Tensor] = []
    lengths: list[Tensor] = []
    for raw_size in window_sizes:
        size = int(raw_size)
        if size <= 0:
            raise ValueError("Window sizes must be positive")
        if size > total_frames:
            continue
        evidence_sum = evidence_prefix[:, size:] - evidence_prefix[:, :-size]
        valid_count = valid_prefix[size:] - valid_prefix[:-size]
        candidate_valid = valid_count >= float(size) - 1e-6
        if normalization == "sqrt_valid_count":
            denominator = valid_count.clamp_min(1.0).sqrt()
        elif normalization == "mean":
            denominator = valid_count.clamp_min(1.0)
        else:
            denominator = torch.ones_like(valid_count)
        scores = evidence_sum / denominator.unsqueeze(0)
        scores = scores.masked_fill(~candidate_valid.unsqueeze(0), -torch.inf)
        count = total_frames - size + 1
        start = torch.arange(count, dtype=torch.long, device=evidence.device)
        score_rows.append(scores)
        valid_rows.append(candidate_valid)
        starts.append(start)
        ends.append(start + size - 1)
        lengths.append(
            torch.full((count,), size, dtype=torch.long, device=evidence.device)
        )
    if not score_rows:
        raise ValueError("No window size fits the history")
    candidate_valid = torch.cat(valid_rows)
    if not bool(candidate_valid.any()):
        raise ValueError("History has no full-visibility candidate")
    return (
        torch.cat(score_rows, dim=1),
        candidate_valid,
        torch.cat(starts),
        torch.cat(ends),
        torch.cat(lengths),
    )


def select_window_index(scores: Tensor, starts: Tensor, ends: Tensor) -> int:
    """Select an exact maximum, breaking ties by earliest start then end."""

    maximum = torch.max(scores)
    candidates = torch.nonzero(scores == maximum, as_tuple=False).flatten().tolist()
    if not candidates:
        raise ValueError("No maximum-scoring candidate is available")
    return min(
        (int(index) for index in candidates),
        key=lambda index: (int(starts[index]), int(ends[index]), index),
    )


def _resolve_device(value: str | torch.device) -> torch.device:
    if isinstance(value, torch.device):
        return value
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if value == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return torch.device(value)


@torch.no_grad()
def run_scan(
    history: History,
    window_sizes: Sequence[int],
    *,
    visual_centering: float,
    query_centering: float,
    normalization: WindowNormalization = "sqrt_valid_count",
    normalize_query_residual: bool = True,
    device: str | torch.device = "cpu",
    mad_factor: float = 1.4826,
    mad_floor: float = 1e-3,
    minimum_query_residual_norm: float = 1e-8,
) -> ScanOutput:
    """Run one fixed Deja Cue coordinate condition on one history."""

    if not 0.0 <= visual_centering <= 1.0:
        raise ValueError("visual_centering must lie in [0, 1]")
    if not 0.0 <= query_centering <= 1.0:
        raise ValueError("query_centering must lie in [0, 1]")
    torch_device = _resolve_device(device)
    first_frame, dense_features, dense_valid = history.dense()
    visual = torch.from_numpy(dense_features).to(torch_device).unsqueeze(0)
    valid = torch.from_numpy(dense_valid).to(torch_device)
    text = torch.from_numpy(
        np.stack([query.embedding for query in history.queries]).astype(np.float32)
    ).to(torch_device)

    # Normalize before defining either coordinate, exactly as in the method.
    visual = F.normalize(visual, dim=-1)[0]
    text = F.normalize(text, dim=-1)
    prototype = _median(visual[valid], dim=0).detach()
    visual_residual = F.normalize(
        visual - visual_centering * prototype.unsqueeze(0), dim=-1
    )
    visual_residual = visual_residual.masked_fill(~valid.unsqueeze(1), 0.0)

    weights = state_balanced_weights(history, device=torch_device)
    query_origin = torch.einsum("q,qd->d", weights, text)
    conditioned = text - query_centering * query_origin.unsqueeze(0)
    residual_norms = torch.linalg.vector_norm(conditioned, dim=-1, keepdim=True)
    fallback = residual_norms.squeeze(1) <= minimum_query_residual_norm
    conditioned = torch.where(~fallback.unsqueeze(1), conditioned, text)
    if normalize_query_residual:
        conditioned = F.normalize(conditioned, dim=-1)

    evidence = torch.einsum("td,qd->qt", visual_residual, conditioned)
    evidence = evidence.masked_fill(~valid.unsqueeze(0), 0.0)
    standardized, centers, scales = robust_standardize(
        evidence, valid, mad_factor=mad_factor, mad_floor=mad_floor
    )
    smoothed = smooth_within_runs(standardized, valid, kernel_size=3)
    scores, candidate_valid, starts, ends, lengths = multiscale_scan(
        smoothed, valid, window_sizes, normalization=normalization
    )
    absolute_starts = starts + first_frame
    absolute_ends = ends + first_frame
    indices = [
        select_window_index(scores[index], absolute_starts, absolute_ends)
        for index in range(scores.shape[0])
    ]
    windows = tuple(
        (
            int(absolute_starts[index].item()),
            int(absolute_ends[index].item()),
        )
        for index in indices
    )
    selected_scores = tuple(
        float(scores[q, index].item()) for q, index in enumerate(indices)
    )
    return ScanOutput(
        windows=windows,
        selected_scores=selected_scores,
        candidate_scores=scores.detach().cpu().numpy(),
        candidate_valid=candidate_valid.detach().cpu().numpy(),
        candidate_starts=absolute_starts.detach().cpu().numpy(),
        candidate_ends=absolute_ends.detach().cpu().numpy(),
        candidate_lengths=lengths.detach().cpu().numpy(),
        smoothed_evidence=smoothed.detach().cpu().numpy(),
        dense_valid=dense_valid.copy(),
        first_frame=first_frame,
        query_residual_norms=residual_norms.squeeze(1).detach().cpu().numpy(),
        query_residual_fallback=fallback.detach().cpu().numpy(),
        evidence_centers=centers.detach().cpu().numpy(),
        evidence_scales=scales.detach().cpu().numpy(),
    )


def _boxcar_scores(
    history: History, query: np.ndarray, kernel_size: int = 3
) -> np.ndarray:
    raw = history.visual_features @ np.asarray(query, dtype=np.float32)
    smoothed = np.zeros_like(raw)
    radius = kernel_size // 2
    for left, right in contiguous_index_runs(history.frame_indices):
        for index in range(left, right):
            start = max(left, index - radius)
            end = min(right, index + radius + 1)
            smoothed[index] = float(np.mean(raw[start:end]))
    return smoothed


def run_simple(
    history: History, window_sizes: Sequence[int], *, method: str
) -> SimpleOutput:
    """Evaluate one of the three fixed simple frozen readouts."""

    if method not in {"meanpool_abs", "boxcar3_abs", "maxpool_abs"}:
        raise ValueError(f"Unknown simple readout: {method}")
    candidates = enumerate_windows(history, window_sizes)
    windows: list[tuple[int, int]] = []
    selected_scores: list[float] = []
    for query in history.queries:
        query_vector = np.asarray(query.embedding, dtype=np.float32)
        raw = history.visual_features @ query_vector
        smoothed = (
            _boxcar_scores(history, query_vector, kernel_size=3)
            if method == "boxcar3_abs"
            else raw
        )
        scored: list[tuple[float, int, int]] = []
        for left, right, start, end in candidates:
            if method == "meanpool_abs":
                pooled = history.visual_features[left:right].mean(axis=0)
                pooled = pooled / max(float(np.linalg.norm(pooled)), 1e-8)
                score = float(pooled @ query_vector)
            elif method == "boxcar3_abs":
                score = float(np.mean(smoothed[left:right]))
            else:
                score = float(np.max(raw[left:right]))
            scored.append((score, start, end))
        score, start, end = max(scored, key=lambda row: (row[0], -row[1], -row[2]))
        windows.append((start, end))
        selected_scores.append(score)
    return SimpleOutput(tuple(windows), tuple(selected_scores))


def select_peak_span(
    evidence: np.ndarray,
    valid: np.ndarray,
    *,
    first_frame: int,
    threshold_ratio: float,
) -> tuple[int, int]:
    """Expand the earliest global evidence peak over a relative threshold."""

    values = np.asarray(evidence, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if values.ndim != 1 or mask.shape != values.shape or not mask.any():
        raise ValueError("Peak evidence is invalid")
    valid_indices = np.flatnonzero(mask)
    peak_value = float(values[valid_indices].max())
    peak_index = int(
        valid_indices[np.flatnonzero(values[valid_indices] == peak_value)[0]]
    )
    if peak_value <= 0.0:
        return first_frame + peak_index, first_frame + peak_index
    threshold = threshold_ratio * peak_value
    left = peak_index
    right = peak_index
    while left > 0 and mask[left - 1] and float(values[left - 1]) >= threshold:
        left -= 1
    while (
        right + 1 < len(values)
        and mask[right + 1]
        and float(values[right + 1]) >= threshold
    ):
        right += 1
    return first_frame + left, first_frame + right


def temporal_iou(
    window: tuple[int, int], references: Sequence[tuple[int, int]]
) -> float:
    """Best inclusive temporal IoU against one or more reference intervals."""

    best = 0.0
    for start, end in references:
        intersection = max(0, min(window[1], end) - max(window[0], start) + 1)
        union = (window[1] - window[0] + 1) + (end - start + 1) - intersection
        best = max(best, intersection / union if union else 0.0)
    return float(best)


def candidate_oracle_tiou(
    history: History,
    references: Sequence[tuple[int, int]],
    window_sizes: Sequence[int],
) -> float:
    """Return the best tIoU attainable by any legal run-contained candidate."""

    return max(
        temporal_iou((start, end), references)
        for _left, _right, start, end in enumerate_windows(history, window_sizes)
    )


def valid_run_edge_hit(history: History, window: tuple[int, int]) -> float:
    """Return 1 when a valid prediction touches its observed-run boundary."""

    for left, right in contiguous_index_runs(history.frame_indices):
        run_start = int(history.frame_indices[left])
        run_end = int(history.frame_indices[right - 1])
        if run_start <= window[0] <= window[1] <= run_end:
            return float(window[0] == run_start or window[1] == run_end)
    raise ValueError(
        f"Prediction lies outside every run: {history.history_id}/{window}"
    )
