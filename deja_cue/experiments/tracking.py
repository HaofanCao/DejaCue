"""End-to-end tracking perturbation experiment.

The experiment keeps the selected target queries, references, temporal
indices, and 33-length schedule fixed. Each perturbation changes only the
target observations. Target and auxiliary tracks are then scored separately
with the same trajectory- and vocabulary-centered scan; the best per-track
score determines the identity decision.
"""

from __future__ import annotations

from statistics import fmean
from typing import Any, Mapping, Sequence

import numpy as np

from ..data import History
from ..seven_history import SevenHistoryDistractor, SevenHistoryRecord
from .perturbations import apply_perturbation, perturbation_cells


TRACKING_METRICS = (
    "state_macro_target_top1_tiou",
    "state_macro_target_r1_tiou_0.5",
    "query_macro_joint_identity_accuracy",
)
DEFAULT_CONDITION_IDS = (
    "random_missing_010pct",
    "contiguous_missing_span_010pct",
    "distractor_contamination_025pct_mix050pct",
    "distractor_contamination_100pct_mix100pct",
    "identity_switch_after_050pct",
)


def _track_history(
    target: History, distractor: SevenHistoryDistractor
) -> History:
    """Expose one auxiliary track through the common scan interface."""

    return History(
        history_id=f"{target.history_id}_T{distractor.track_id}",
        sequence_id=target.sequence_id,
        source_component_id=target.source_component_id,
        frame_indices=distractor.frame_indices,
        visual_features=distractor.visual_features,
        visibility_count=distractor.visibility_count,
        queries=target.queries,
        # References are never used to score the auxiliary track.  Retaining
        # the mapping keeps one immutable History interface for all tracks.
        references=target.references,
    )


def evaluate_tracking_history(
    history: History,
    distractors: Sequence[SevenHistoryDistractor],
    window_sizes: Sequence[int],
    *,
    device: str = "cpu",
) -> dict[str, float]:
    """Evaluate the three tracking metrics for one original history."""

    # Keep aggregation and setup inspection available in NumPy-only
    # environments; the executable scan imports PyTorch only when requested.
    from ..scan import run_scan, temporal_iou

    target = run_scan(
        history,
        window_sizes,
        visual_centering=1.0,
        query_centering=1.0,
        normalization="sqrt_valid_count",
        device=device,
    )
    auxiliary = [
        run_scan(
            _track_history(history, distractor),
            window_sizes,
            visual_centering=1.0,
            query_centering=1.0,
            normalization="sqrt_valid_count",
            device=device,
        )
        for distractor in distractors
    ]

    per_query_tiou: list[float] = []
    identity: list[float] = []
    for query_index, query in enumerate(history.queries):
        overlap = temporal_iou(target.windows[query_index], history.references[query.state_id])
        per_query_tiou.append(overlap)
        candidates = [float(target.selected_scores[query_index])] + [
            float(output.selected_scores[query_index]) for output in auxiliary
        ]
        maximum = max(candidates)
        winners = np.flatnonzero(
            np.isclose(candidates, maximum, atol=1e-8, rtol=1e-7)
        ).tolist()
        identity.append(float(winners == [0]))

    state_top1: list[float] = []
    state_rhalf: list[float] = []
    for state_id in history.references:
        indices = [
            index
            for index, query in enumerate(history.queries)
            if query.state_id == state_id
        ]
        if not indices:
            raise ValueError(f"State has no matching query: {state_id}")
        state_top1.append(float(fmean(per_query_tiou[index] for index in indices)))
        state_rhalf.append(
            float(fmean(float(per_query_tiou[index] >= 0.5) for index in indices))
        )
    return {
        "state_macro_target_top1_tiou": float(fmean(state_top1)),
        "state_macro_target_r1_tiou_0.5": float(fmean(state_rhalf)),
        "query_macro_joint_identity_accuracy": float(fmean(identity)),
    }


def _scene_rows(
    values: Sequence[tuple[str, Mapping[str, float]]]
) -> list[dict[str, Any]]:
    """Create one result row per scene."""

    return [
        {
            "scene": scene,
            "metrics": {metric: float(metrics[metric]) for metric in TRACKING_METRICS},
        }
        for scene, metrics in sorted(values, key=lambda row: row[0])
    ]


def _reported_summary(
    replicates: Sequence[Mapping[str, Any]],
    *,
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Compute the paired scene-cluster bootstrap summary."""

    if not replicates or bootstrap_resamples <= 0:
        raise ValueError("Tracking summary requires replicates and bootstrap draws")
    normalized: list[tuple[dict[str, Any], dict[str, Any]]] = []
    expected_scenes: tuple[str, ...] | None = None
    for replicate in replicates:
        baseline = {str(row["scene"]): row for row in replicate["baseline"]}
        perturbed = {str(row["scene"]): row for row in replicate["perturbed"]}
        scenes = tuple(sorted(baseline))
        if set(baseline) != set(perturbed):
            raise ValueError("Baseline and perturbed scene sets differ")
        if expected_scenes is None:
            expected_scenes = scenes
        elif scenes != expected_scenes:
            raise ValueError("Affected scene set changes across replicates")
        normalized.append((baseline, perturbed))
    if not expected_scenes:
        raise ValueError("No scene supports the perturbation")

    observed: dict[str, dict[str, float]] = {}
    for metric in TRACKING_METRICS:
        before = [
            float(left[scene]["metrics"][metric])
            for left, _right in normalized
            for scene in expected_scenes
        ]
        after = [
            float(right[scene]["metrics"][metric])
            for _left, right in normalized
            for scene in expected_scenes
        ]
        observed[metric] = {
            "baseline": float(fmean(before)),
            "perturbed": float(fmean(after)),
            "delta": float(fmean(after) - fmean(before)),
        }

    rng = np.random.default_rng(seed)
    draws = {metric: [] for metric in TRACKING_METRICS}
    for _ in range(bootstrap_resamples):
        selected_scenes = rng.choice(
            expected_scenes, size=len(expected_scenes), replace=True
        ).tolist()
        values = {metric: [] for metric in TRACKING_METRICS}
        for scene in selected_scenes:
            left, right = normalized[int(rng.integers(0, len(normalized)))]
            for metric in TRACKING_METRICS:
                values[metric].append(
                    float(right[str(scene)]["metrics"][metric])
                    - float(left[str(scene)]["metrics"][metric])
                )
        for metric in TRACKING_METRICS:
            draws[metric].append(float(fmean(values[metric])))
    return {
        "observed": observed,
        "paired_delta_scene_cluster_bootstrap_95ci": {
            metric: [
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            ]
            for metric, values in draws.items()
        },
        "bootstrap": {"num_resamples": bootstrap_resamples, "seed": seed},
    }


def run_tracking_perturbations(
    records: Sequence[SevenHistoryRecord],
    window_sizes: Sequence[int],
    *,
    device: str = "cpu",
    condition_ids: Sequence[str] = DEFAULT_CONDITION_IDS,
    bootstrap_resamples: int = 10_000,
    seed: int = 3407,
) -> dict[str, Any]:
    """Run the selected conditions on the four original histories."""

    originals = tuple(record for record in records if record.stratum == "original")
    if len(originals) != 4:
        raise ValueError("Tracking perturbations require four original histories")
    cell_index = {str(row["condition_id"]): row for row in perturbation_cells()}
    requested = tuple(str(value) for value in condition_ids)
    if len(set(requested)) != len(requested) or set(requested) - set(cell_index):
        raise ValueError("Tracking condition selection is invalid")
    clean = {
        record.history.history_id: evaluate_tracking_history(
            record.history, record.distractors, window_sizes, device=device
        )
        for record in originals
    }

    conditions: dict[str, Any] = {}
    for condition_id in requested:
        cell = cell_index[condition_id]
        replicates = []
        affected_ids: set[str] | None = None
        for replicate_seed in cell["replicate_seeds"]:
            perturbed_rows: list[tuple[str, Mapping[str, float]]] = []
            current_ids: set[str] = set()
            for record in originals:
                outcome = apply_perturbation(
                    record.history,
                    record.distractors,
                    cell,
                    replicate_seed=int(replicate_seed),
                )
                if not outcome.details["applied"]:
                    continue
                current_ids.add(record.history.history_id)
                perturbed_rows.append(
                    (
                        record.scene,
                        evaluate_tracking_history(
                            outcome.history,
                            record.distractors,
                            window_sizes,
                            device=device,
                        ),
                    )
                )
            if not current_ids:
                raise ValueError(f"No history supports {condition_id}")
            if affected_ids is None:
                affected_ids = current_ids
            elif current_ids != affected_ids:
                raise ValueError(f"Affected history set changes for {condition_id}")
            baseline_rows = [
                (record.scene, clean[record.history.history_id])
                for record in originals
                if record.history.history_id in current_ids
            ]
            replicates.append(
                {
                    "seed": int(replicate_seed),
                    "baseline": _scene_rows(baseline_rows),
                    "perturbed": _scene_rows(perturbed_rows),
                }
            )
        assert affected_ids is not None
        conditions[condition_id] = {
            "family": cell["family"],
            "parameters": dict(cell["parameters"]),
            "replicate_seeds": list(cell["replicate_seeds"]),
            "replicates": replicates,
            "reported": _reported_summary(
                replicates,
                bootstrap_resamples=bootstrap_resamples,
                seed=seed,
            ),
            "support": {
                "num_affected_histories": len(affected_ids),
                "num_affected_scenes": len(affected_ids),
            },
        }
    return {
        "schema_version": 1,
        "kind": "deja_cue_tracking_perturbation_evidence",
        "protocol": {
            "coordinate_system": "inclusive_absolute_frame_indices",
            "eligibility": (
                "metrics use only object histories with at least one actual feature or "
                "support modification; the paired clean baseline uses the identical subset"
            ),
            "missing_frame_semantics": (
                "removed observations are deleted from FeatureTrack, so the scanner sees a "
                "true validity gap and cannot smooth or scan across it"
            ),
            "contamination_semantics": (
                "only exact-time co-visible distractor features are mixed; each mixed "
                "feature is L2-normalized"
            ),
            "identity_switch_semantics": (
                "one distractor track, chosen by maximum exact-time suffix coverage with "
                "track-id tie breaking, replaces the target suffix at exact shared frames"
            ),
            "method": {
                "features": "union",
                "kernel_size": 3,
                "query_centering": 1.0,
                "visual_centering": 1.0,
            },
        },
        "metrics": list(TRACKING_METRICS),
        "conditions": conditions,
    }
