"""End-to-end evaluation for the primary and extended VOST experiments."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from .data import History
from .metrics import (
    PRIMARY_METRICS,
    aggregate_histories,
    component_metric_map,
    holm_adjust,
    paired_cluster_bootstrap,
    paired_sign_flip,
    summarize_history,
)
from .scan import (
    ScanOutput,
    candidate_oracle_tiou,
    run_scan,
    run_simple,
    select_peak_span,
    temporal_iou,
    valid_run_edge_hit,
)


MAIN_METHODS: Mapping[str, Mapping[str, Any]] = {
    "meanpool_abs": {"kind": "simple"},
    "boxcar3_abs": {"kind": "simple"},
    "maxpool_abs": {"kind": "simple"},
    "scan_abs": {
        "kind": "scan",
        "visual_centering": 0.0,
        "query_centering": 0.0,
    },
    "scan_visual": {
        "kind": "scan",
        "visual_centering": 1.0,
        "query_centering": 0.0,
    },
    "scan_vocab": {
        "kind": "scan",
        "visual_centering": 0.0,
        "query_centering": 1.0,
    },
    "scan_dual": {
        "kind": "scan",
        "visual_centering": 1.0,
        "query_centering": 1.0,
    },
}


def _query_rows(
    history: History,
    windows: Sequence[tuple[int, int]],
    scores: Sequence[float],
    window_sizes: Sequence[int],
) -> list[dict[str, Any]]:
    rows = []
    for query, window, score in zip(history.queries, windows, scores):
        overlap = temporal_iou(window, history.references[query.state_id])
        rows.append(
            {
                "state_id": query.state_id,
                "text": query.text,
                "window": [int(window[0]), int(window[1])],
                "score": float(score),
                "top1_tiou": overlap,
                "r1_tiou_0.3": float(overlap >= 0.3),
                "r1_tiou_0.5": float(overlap >= 0.5),
                "candidate_oracle_tiou": candidate_oracle_tiou(
                    history, history.references[query.state_id], window_sizes
                ),
                "valid_run_edge_hit": valid_run_edge_hit(history, window),
            }
        )
    return rows


def evaluate_condition(
    histories: Sequence[History],
    window_sizes: Sequence[int],
    *,
    visual_centering: float,
    query_centering: float,
    normalization: str = "sqrt_valid_count",
    normalize_query_residual: bool = True,
    device: str = "cpu",
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Evaluate one matched coordinate and window-statistic condition."""

    history_rows = []
    residual_rows = []
    for history in histories:
        output = run_scan(
            history,
            window_sizes,
            visual_centering=visual_centering,
            query_centering=query_centering,
            normalization=normalization,
            normalize_query_residual=normalize_query_residual,
            device=device,
        )
        queries = _query_rows(
            history, output.windows, output.selected_scores, window_sizes
        )
        history_rows.append(
            summarize_history(
                history_id=history.history_id,
                sequence_id=history.sequence_id,
                source_component_id=history.source_component_id,
                query_rows=queries,
                state_order=tuple(history.references),
            )
        )
        for query, norm, fallback, scale in zip(
            history.queries,
            output.query_residual_norms,
            output.query_residual_fallback,
            output.evidence_scales,
        ):
            residual_rows.append(
                {
                    "history_id": history.history_id,
                    "state_id": query.state_id,
                    "text": query.text,
                    "residual_norm": float(norm),
                    "residual_fallback": bool(fallback),
                    "evidence_scale": float(scale),
                    "mad_floor_active": bool(scale <= 1e-3),
                }
            )
    norms = np.asarray([row["residual_norm"] for row in residual_rows])
    return {
        "condition": {
            "visual_centering": visual_centering,
            "query_centering": query_centering,
            "normalization": normalization,
            "normalize_query_residual": normalize_query_residual,
        },
        "histories": history_rows,
        "aggregate": aggregate_histories(
            history_rows, bootstrap_resamples=bootstrap_resamples, seed=seed
        ),
        "residual_diagnostics": {
            "num_queries": len(residual_rows),
            "minimum": float(norms.min()),
            "median": float(np.median(norms)),
            "maximum": float(norms.max()),
            "num_residual_fallbacks": sum(
                int(row["residual_fallback"]) for row in residual_rows
            ),
            "num_mad_floor_active": sum(
                int(row["mad_floor_active"]) for row in residual_rows
            ),
        },
    }


def evaluate_simple_condition(
    histories: Sequence[History],
    window_sizes: Sequence[int],
    *,
    method: str,
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Evaluate one fixed non-scan baseline with the benchmark hierarchy."""

    history_rows = []
    for history in histories:
        output = run_simple(history, window_sizes, method=method)
        queries = _query_rows(
            history, output.windows, output.selected_scores, window_sizes
        )
        history_rows.append(
            summarize_history(
                history_id=history.history_id,
                sequence_id=history.sequence_id,
                source_component_id=history.source_component_id,
                query_rows=queries,
                state_order=tuple(history.references),
            )
        )
    return {
        "condition": {"method": method},
        "histories": history_rows,
        "aggregate": aggregate_histories(
            history_rows, bootstrap_resamples=bootstrap_resamples, seed=seed
        ),
    }


def _paired_comparison(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    metrics: Sequence[str],
    seed: int,
    bootstrap_resamples: int,
    sign_flip_assignments: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for metric in metrics:
        left_values = component_metric_map(left, metric)
        right_values = component_metric_map(right, metric)
        test = paired_sign_flip(
            left_values,
            right_values,
            seed=seed,
            assignments=sign_flip_assignments,
        )
        test["paired_source_component_bootstrap_95ci"] = paired_cluster_bootstrap(
            left_values,
            right_values,
            seed=seed,
            resamples=bootstrap_resamples,
        )
        output[metric] = test
    adjusted = holm_adjust(
        {metric: float(output[metric]["two_sided_p"]) for metric in metrics}
    )
    for metric, value in adjusted.items():
        output[metric]["holm_adjusted_p"] = value
    return {
        "metrics": output,
        "bootstrap_resamples": bootstrap_resamples,
        "sign_flip_assignments": sign_flip_assignments,
        "seed": seed,
        "multiple_comparison_correction": "holm",
    }


def summarize_duration_strata(
    histories: Sequence[History], condition: Mapping[str, Any]
) -> dict[str, Any]:
    """Aggregate description retrieval by reference-duration quartile.

    Quartile boundaries use the included reference episodes, while each description
    contributes one retrieval observation. Requiring one episode per state keeps
    the stratification unit fixed to the primary benchmark protocol.
    """

    history_index = {history.history_id: history for history in histories}
    lengths = sorted(
        end - start + 1
        for history in histories
        for references in history.references.values()
        for start, end in references
    )
    boundaries = [float(np.quantile(lengths, value)) for value in (0.25, 0.5, 0.75)]
    rows = []
    for result_history in condition["histories"]:
        history_id = str(result_history["history_id"])
        history = history_index[history_id]
        for query in result_history["queries"]:
            state_id = str(query["state_id"])
            references = history.references[state_id]
            if len(references) != 1:
                raise ValueError(
                    f"Duration strata require one reference for {history_id}/{state_id}"
                )
            start, end = references[0]
            length = end - start + 1
            q1, q2, q3 = boundaries
            duration_bucket = (
                "q1_shortest"
                if length <= q1
                else "q2" if length <= q2 else "q3" if length <= q3 else "q4_longest"
            )
            rows.append(
                {
                    "duration_bucket": duration_bucket,
                    "r1_tiou_0.3": float(query["r1_tiou_0.3"]),
                    "r1_tiou_0.5": float(query["r1_tiou_0.5"]),
                    "top1_tiou": float(query["top1_tiou"]),
                }
            )

    def summarize(field: str) -> dict[str, Any]:
        """Average description outcomes within each named duration bucket."""

        output = {}
        for label in sorted({str(row[field]) for row in rows}):
            members = [row for row in rows if row[field] == label]
            output[label] = {
                "num_descriptions": len(members),
                "r1_tiou_0.3": float(np.mean([row["r1_tiou_0.3"] for row in members])),
                "r1_tiou_0.5": float(np.mean([row["r1_tiou_0.5"] for row in members])),
                "top1_tiou": float(np.mean([row["top1_tiou"] for row in members])),
            }
        return output

    return {
        "duration_quartile_boundaries": boundaries,
        "by_duration": summarize("duration_bucket"),
    }


def evaluate_main(
    histories: Sequence[History],
    protocol: Mapping[str, Any],
    *,
    device: str,
) -> dict[str, Any]:
    """Reproduce the seven rows in the primary 78-history comparison."""

    schedule = tuple(int(value) for value in protocol["window_schedule"])
    resamples = int(protocol["bootstrap_resamples"])
    assignments = int(protocol["sign_flip_assignments"])
    seed = int(protocol["seed"])
    methods: dict[str, dict[str, Any]] = {}
    for name, specification in MAIN_METHODS.items():
        if specification["kind"] == "simple":
            methods[name] = evaluate_simple_condition(
                histories,
                schedule,
                method=name,
                bootstrap_resamples=resamples,
                seed=seed,
            )
        else:
            methods[name] = evaluate_condition(
                histories,
                schedule,
                visual_centering=float(specification["visual_centering"]),
                query_centering=float(specification["query_centering"]),
                device=device,
                bootstrap_resamples=resamples,
                seed=seed,
            )

    comparisons = {}
    for baseline in ("meanpool_abs", "boxcar3_abs", "maxpool_abs", "scan_abs"):
        comparisons[f"scan_dual_minus_{baseline}"] = _paired_comparison(
            methods["scan_dual"],
            methods[baseline],
            metrics=PRIMARY_METRICS,
            seed=seed,
            bootstrap_resamples=resamples,
            sign_flip_assignments=assignments,
        )
    coordinate_pairs = (
        ("scan_vocab", "scan_abs"),
        ("scan_visual", "scan_abs"),
        ("scan_dual", "scan_vocab"),
        ("scan_dual", "scan_visual"),
    )
    for left, right in coordinate_pairs:
        comparisons[f"{left}_minus_{right}"] = _paired_comparison(
            methods[left],
            methods[right],
            metrics=PRIMARY_METRICS,
            seed=seed,
            bootstrap_resamples=resamples,
            sign_flip_assignments=assignments,
        )
    return {
        "schema_version": 1,
        "kind": "deja_cue_primary_reproduction",
        "counts": {
            "histories": len(histories),
            "states": sum(len(history.references) for history in histories),
            "descriptions": sum(len(history.queries) for history in histories),
            "source_components": len(
                {history.source_component_id for history in histories}
            ),
        },
        "window_schedule": list(schedule),
        "methods": methods,
        "paired_comparisons": comparisons,
    }


def evaluate_peak_condition(
    histories: Sequence[History],
    window_sizes: Sequence[int],
    *,
    coordinate: str,
    threshold_ratio: float,
    device: str,
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Evaluate threshold-based expansion around each query evidence peak."""

    if coordinate not in {"absolute", "vocabulary"}:
        raise ValueError(f"Unknown peak coordinate: {coordinate}")
    history_rows = []
    for history in histories:
        output = run_scan(
            history,
            window_sizes,
            visual_centering=0.0,
            query_centering=1.0 if coordinate == "vocabulary" else 0.0,
            device=device,
        )
        windows = tuple(
            select_peak_span(
                output.smoothed_evidence[index],
                output.dense_valid,
                first_frame=output.first_frame,
                threshold_ratio=threshold_ratio,
            )
            for index in range(len(history.queries))
        )
        query_rows = _query_rows(
            history,
            windows,
            tuple(
                float(output.smoothed_evidence[index].max())
                for index in range(len(windows))
            ),
            window_sizes,
        )
        history_rows.append(
            summarize_history(
                history_id=history.history_id,
                sequence_id=history.sequence_id,
                source_component_id=history.source_component_id,
                query_rows=query_rows,
                state_order=tuple(history.references),
            )
        )
    return {
        "condition": {
            "method": "adaptive_peak_expansion",
            "coordinate": coordinate,
            "threshold_ratio": threshold_ratio,
        },
        "histories": history_rows,
        "aggregate": aggregate_histories(
            history_rows, bootstrap_resamples=bootstrap_resamples, seed=seed
        ),
    }


def candidate_rank_row(
    scores: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    references: Sequence[tuple[int, int]],
    *,
    topk: Sequence[int] = (1, 5, 10, 50, 100),
) -> dict[str, Any]:
    """Summarize where tIoU-qualified candidates appear in score order."""

    lengths = ends - starts + 1
    overlaps = np.zeros(len(starts), dtype=np.float64)
    for ref_start, ref_end in references:
        intersection = np.maximum(
            0, np.minimum(ends, ref_end) - np.maximum(starts, ref_start) + 1
        )
        union = lengths + (ref_end - ref_start + 1) - intersection
        overlaps = np.maximum(overlaps, intersection / union)
    order = np.argsort(-scores, kind="stable")
    ordered_overlap = overlaps[order]
    best_overlap = float(overlaps.max())
    top1_overlap = float(ordered_overlap[0])
    inverse_rank = np.empty(len(order), dtype=np.int64)
    inverse_rank[order] = np.arange(len(order), dtype=np.int64)
    oracle_candidates = np.flatnonzero(np.isclose(overlaps, best_overlap, atol=1e-12))
    best_overlap_rank = int(inverse_rank[oracle_candidates].min()) + 1
    row: dict[str, Any] = {
        "num_candidates": len(order),
        "top1_tiou": top1_overlap,
        "oracle_tiou": best_overlap,
        "oracle_gap": best_overlap - top1_overlap,
        "best_overlap_rank": best_overlap_rank,
        "best_overlap_normalized_rank": float(
            (best_overlap_rank - 1) / max(len(order) - 1, 1)
        ),
    }
    for threshold_name, threshold in (("03", 0.3), ("05", 0.5)):
        positives = np.flatnonzero(ordered_overlap >= threshold)
        first_rank = int(positives[0]) + 1 if len(positives) else None
        row[f"first_positive_rank_{threshold_name}"] = first_rank
        row[f"mrr_{threshold_name}"] = 0.0 if first_rank is None else 1.0 / first_rank
        for k in topk:
            limit = min(int(k), len(order))
            row[f"recall_at_{k}_{threshold_name}"] = float(
                np.any(ordered_overlap[:limit] >= threshold)
            )
            topk_overlap = float(ordered_overlap[:limit].max())
            gap = best_overlap - top1_overlap
            row[f"oracle_gap_closed_at_{k}"] = (
                1.0 if gap <= 1e-12 else float((topk_overlap - top1_overlap) / gap)
            )
    return row


def _aggregate_rank_rows(
    rows: Sequence[Mapping[str, Any]], *, bootstrap_resamples: int, seed: int
) -> dict[str, Any]:
    metric_names = (
        "top1_tiou",
        "oracle_tiou",
        "oracle_gap",
        "best_overlap_normalized_rank",
        "mrr_03",
        "mrr_05",
        "recall_at_5_05",
        "recall_at_10_05",
        "recall_at_50_05",
        "oracle_gap_closed_at_10",
    )
    by_history_state: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_history_state[(str(row["history_id"]), str(row["state_id"]))].append(row)
    state_rows = []
    for (history_id, state_id), members in sorted(by_history_state.items()):
        state_rows.append(
            {
                "history_id": history_id,
                "state_id": state_id,
                **{
                    metric: float(np.mean([row[metric] for row in members]))
                    for metric in metric_names
                },
            }
        )
    per_history = []
    for history_id in sorted({row["history_id"] for row in state_rows}):
        members = [row for row in state_rows if row["history_id"] == history_id]
        per_history.append(
            {
                "history_id": history_id,
                **{
                    metric: float(np.mean([row[metric] for row in members]))
                    for metric in metric_names
                },
            }
        )
    macro = {
        metric: float(np.mean([row[metric] for row in per_history]))
        for metric in metric_names
    }
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(per_history), size=(bootstrap_resamples, len(per_history))
    )
    intervals = {}
    for metric in metric_names:
        values = np.asarray([row[metric] for row in per_history], dtype=np.float64)
        draws = values[indices].mean(axis=1)
        intervals[metric] = [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ]
    return {
        "history_macro": macro,
        "history_bootstrap_95ci": intervals,
        "per_history": per_history,
    }


def evaluate_candidate_ranking(
    histories: Sequence[History],
    window_sizes: Sequence[int],
    *,
    device: str,
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Aggregate candidate-rank diagnostics for the three coordinate choices."""

    specifications = {
        "absolute": (0.0, 0.0),
        "vocabulary_only": (0.0, 1.0),
        "dual_centered": (1.0, 1.0),
    }
    conditions = {}
    for name, (visual_centering, query_centering) in specifications.items():
        rows = []
        for history in histories:
            output = run_scan(
                history,
                window_sizes,
                visual_centering=visual_centering,
                query_centering=query_centering,
                device=device,
            )
            valid = output.candidate_valid
            starts = output.candidate_starts[valid]
            ends = output.candidate_ends[valid]
            score_matrix = output.candidate_scores[:, valid]
            for index, query in enumerate(history.queries):
                rows.append(
                    {
                        "history_id": history.history_id,
                        "state_id": query.state_id,
                        "text": query.text,
                        **candidate_rank_row(
                            score_matrix[index],
                            starts,
                            ends,
                            history.references[query.state_id],
                        ),
                    }
                )
        conditions[name] = {
            "coordinate": {
                "visual_centering": visual_centering,
                "query_centering": query_centering,
            },
            "aggregate": _aggregate_rank_rows(
                rows, bootstrap_resamples=bootstrap_resamples, seed=seed
            ),
        }
    return {
        "rank_origin": "one_based_descending_candidate_score",
        "topk": [1, 5, 10, 50, 100],
        "conditions": conditions,
    }


def evaluate_extended(
    primary_histories: Sequence[History],
    prompt_histories: Mapping[str, Sequence[History]],
    protocol: Mapping[str, Any],
    *,
    device: str,
) -> dict[str, Any]:
    """Run the VOST robustness experiments with every non-target factor fixed."""

    schedule = tuple(int(value) for value in protocol["window_schedule"])
    resamples = int(protocol["bootstrap_resamples"])
    assignments = int(protocol["sign_flip_assignments"])
    seed = int(protocol["seed"])
    ratio = float(protocol["adaptive_peak_threshold_ratio"])

    peak = {
        coordinate: evaluate_peak_condition(
            primary_histories,
            schedule,
            coordinate=coordinate,
            threshold_ratio=ratio,
            device=device,
            bootstrap_resamples=resamples,
            seed=seed,
        )
        for coordinate in ("absolute", "vocabulary")
    }
    peak["vocabulary_minus_absolute"] = _paired_comparison(
        peak["vocabulary"],
        peak["absolute"],
        metrics=PRIMARY_METRICS,
        seed=seed,
        bootstrap_resamples=resamples,
        sign_flip_assignments=assignments,
    )

    prompts: dict[str, Any] = {}
    for variant, histories in prompt_histories.items():
        conditions = {
            "absolute": evaluate_condition(
                histories,
                schedule,
                visual_centering=0.0,
                query_centering=0.0,
                device=device,
                bootstrap_resamples=resamples,
                seed=seed,
            ),
            "vocabulary": evaluate_condition(
                histories,
                schedule,
                visual_centering=0.0,
                query_centering=1.0,
                device=device,
                bootstrap_resamples=resamples,
                seed=seed,
            ),
        }
        conditions["vocabulary_minus_absolute"] = _paired_comparison(
            conditions["vocabulary"],
            conditions["absolute"],
            metrics=PRIMARY_METRICS,
            seed=seed,
            bootstrap_resamples=resamples,
            sign_flip_assignments=assignments,
        )
        prompts[variant] = conditions

    normalization = {}
    for score_normalization in ("sqrt_valid_count", "sum"):
        normalization[score_normalization] = {}
        for name, (visual_centering, query_centering) in {
            "absolute": (0.0, 0.0),
            "vocabulary": (0.0, 1.0),
            "dual": (1.0, 1.0),
        }.items():
            normalization[score_normalization][name] = evaluate_condition(
                primary_histories,
                schedule,
                visual_centering=visual_centering,
                query_centering=query_centering,
                normalization=score_normalization,
                device=device,
                bootstrap_resamples=resamples,
                seed=seed,
            )

    ranking = evaluate_candidate_ranking(
        primary_histories,
        schedule,
        device=device,
        bootstrap_resamples=resamples,
        seed=seed,
    )
    duration_strata = {
        "scan_absolute": summarize_duration_strata(
            primary_histories, normalization["sqrt_valid_count"]["absolute"]
        ),
        "scan_vocabulary": summarize_duration_strata(
            primary_histories, normalization["sqrt_valid_count"]["vocabulary"]
        ),
    }
    return {
        "schema_version": 1,
        "kind": "deja_cue_extended_reproduction",
        "adaptive_peak": peak,
        "prompt_sensitivity": prompts,
        "normalization_sensitivity": normalization,
        "candidate_ranking": ranking,
        "duration_strata": duration_strata,
    }
