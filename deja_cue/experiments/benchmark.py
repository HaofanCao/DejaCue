"""Timing harness for the Deja Cue scan."""

from __future__ import annotations

import time
from typing import Any, Sequence

import numpy as np

from ..data import History


def _resolve_device(value: str) -> Any:
    import torch

    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if value == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return torch.device(value)


def _synchronize(device: Any) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_benchmark_payload(
    milliseconds_per_query: Sequence[float],
    *,
    device: str,
    history_count: int,
    query_count: int,
    warmup_repetitions: int,
    window_count: int,
) -> dict[str, Any]:
    """Summarize timing samples and evaluation dimensions."""

    values = np.asarray(milliseconds_per_query, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not bool(np.isfinite(values).all()):
        raise ValueError("Timing samples must be a non-empty finite vector")
    if bool((values <= 0.0).any()):
        raise ValueError("Timing samples must be positive")
    return {
        "schema_version": 1,
        "kind": "deja_cue_scan_latency_benchmark",
        "device": device,
        "unit": "milliseconds_per_query",
        "measurement": {
            "history_count": int(history_count),
            "query_count_per_repetition": int(query_count),
            "window_schedule_size": int(window_count),
            "warmup_repetitions": int(warmup_repetitions),
            "timed_repetitions": int(len(values)),
            "median_ms_per_query": float(np.median(values)),
            "quartile_25_ms_per_query": float(np.quantile(values, 0.25)),
            "quartile_75_ms_per_query": float(np.quantile(values, 0.75)),
        },
        "method": {
            "visual_centering": 1.0,
            "query_centering": 1.0,
            "window_normalization": "sqrt_valid_count",
            "smoothing_kernel_size": 3,
        },
    }


def benchmark_scan(
    histories: Sequence[History],
    window_sizes: Sequence[int],
    *,
    device: str = "auto",
    warmup_repetitions: int = 3,
    timed_repetitions: int = 10,
) -> dict[str, Any]:
    """Measure end-to-end scan latency and report the median per query."""

    if not histories:
        raise ValueError("At least one history is required")
    if not window_sizes or any(int(size) <= 0 for size in window_sizes):
        raise ValueError("Window sizes must be non-empty and positive")
    if warmup_repetitions < 0 or timed_repetitions <= 0:
        raise ValueError("Warmup must be non-negative and repetitions positive")
    from ..scan import run_scan

    torch_device = _resolve_device(device)
    query_count = sum(len(history.queries) for history in histories)
    if query_count <= 0:
        raise ValueError("Histories contain no queries")

    def run_once() -> float:
        _synchronize(torch_device)
        start = time.perf_counter()
        for history in histories:
            run_scan(
                history,
                window_sizes,
                visual_centering=1.0,
                query_centering=1.0,
                normalization="sqrt_valid_count",
                device=torch_device,
            )
        _synchronize(torch_device)
        return 1000.0 * (time.perf_counter() - start) / query_count

    for _ in range(warmup_repetitions):
        run_once()
    timings = [run_once() for _ in range(timed_repetitions)]
    return build_benchmark_payload(
        timings,
        device=torch_device.type,
        history_count=len(histories),
        query_count=query_count,
        warmup_repetitions=warmup_repetitions,
        window_count=len(tuple(window_sizes)),
    )
