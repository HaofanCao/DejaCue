"""Load and verify the bundled reference-result summaries."""

from __future__ import annotations

import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .data import package_root


SCENES = (
    "interp_torchocolate",
    "misc_espresso",
    "misc_tamping",
    "misc_americano",
    "coffee_martini",
    "misc_cross-hands",
    "interp_slice-banana",
)
STRATA = {
    scene: "original" if index < 4 else "extension_v2"
    for index, scene in enumerate(SCENES)
}
MODELS = (
    "moment_detr",
    "qd_detr",
    "eatr",
    "cg_detr",
    "uvcom",
    "tr_detr",
    "taskweave_mr2hd",
    "sim_detr",
)
SEEDS = (3407, 3408, 3409)
FAMILY_METRICS = {
    "L-R": ("state_macro_r1_tiou_0.5", "r1_tiou_0.5"),
    "L-I": ("state_macro_top1_tiou", "top1_tiou"),
}


def _read_reference(name: str, root: Path) -> dict[str, Any]:
    path = root / "data" / "reference" / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _close(left: float, right: float, *, tolerance: float = 1e-12) -> bool:
    return bool(np.isclose(float(left), float(right), atol=tolerance, rtol=0.0))


def _validate_finite_tree(value: Any, *, label: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _validate_finite_tree(child, label=f"{label}/{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_finite_tree(child, label=f"{label}/{index}")
    elif isinstance(value, float):
        _require(math.isfinite(value), f"Non-finite value at {label}")


def _validate_seven_history(root: Path) -> dict[str, int]:
    manifest = json.loads(
        (root / "data" / "seven_history" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    summary = _read_reference("seven_history_summary", root)
    _require(
        manifest.get("kind") == "deja_cue_seven_history_feature_manifest",
        "Unexpected seven-history manifest kind",
    )
    histories = manifest.get("histories")
    _require(isinstance(histories, list), "Seven-history roster is not a list")
    _require(
        [row.get("history_id") for row in histories]
        == [f"S{index:02d}" for index in range(1, 8)],
        "Seven-history identifiers differ",
    )
    _require(
        tuple(str(row.get("scene")) for row in histories) == SCENES,
        "Seven-history scene roster differs",
    )
    counts = {
        "histories": len(histories),
        "states": sum(len(row.get("states", [])) for row in histories),
        "descriptions": sum(
            len(state.get("descriptions", []))
            for row in histories
            for state in row.get("states", [])
        ),
        "episodes": sum(
            len(state.get("references", []))
            for row in histories
            for state in row.get("states", [])
        ),
    }
    _require(
        counts == {"histories": 7, "states": 16, "descriptions": 32, "episodes": 59},
        "Seven-history counts differ",
    )
    for row in histories:
        state_ids = []
        for state in row["states"]:
            state_id = str(state.get("state_id", ""))
            state_ids.append(state_id)
            _require(state_id != "", "Seven-history state identifier is empty")
            descriptions = state.get("descriptions")
            _require(
                isinstance(descriptions, list)
                and len(descriptions) == 2
                and all(isinstance(text, str) and text for text in descriptions),
                f"Description format differs for {row['history_id']}/{state_id}",
            )
            references = state.get("references")
            _require(
                isinstance(references, list) and references, "Reference list is empty"
            )
            for interval in references:
                _require(
                    isinstance(interval, list)
                    and len(interval) == 2
                    and int(interval[0]) >= 0
                    and int(interval[1]) >= int(interval[0]),
                    f"Invalid interval for {row['history_id']}/{state_id}",
                )
        _require(len(state_ids) == len(set(state_ids)), "State identifiers repeat")

    _require(
        summary.get("kind") == "deja_cue_seven_history_v2_result_summary",
        "Unexpected seven-history result kind",
    )
    _require(
        summary.get("inventory") == counts, "Manifest and result counts differ"
    )
    _require(
        summary.get("protocol", {}).get("strata") == STRATA,
        "Seven-history strata differ",
    )
    _require(
        summary.get("original_n4_prediction_reproduction")
        == {"exact": True, "num_queries": 16},
        "Original prediction reproduction differs",
    )
    _require(
        summary.get("raw_residual_prediction_match") is True, "Residual check differs"
    )
    _validate_finite_tree(summary, label="seven_history_summary")
    return counts


def _exact_sign_flip(values: Sequence[float]) -> dict[str, float | int]:
    _require(len(values) == 7, "Exact sign flip requires seven values")
    observed = abs(float(np.mean(values)))
    statistics = (
        abs(float(np.mean([sign * value for sign, value in zip(signs, values)])))
        for signs in itertools.product((-1.0, 1.0), repeat=7)
    )
    extreme = sum(value >= observed - 1e-15 for value in statistics)
    return {
        "observed_absolute_mean": observed,
        "enumerations": 128,
        "extreme_assignments": extreme,
        "exact_two_sided_p": float(extreme / 128.0),
    }


def _stratified_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    scene_order: Sequence[str],
    resamples: int,
    seed: int,
) -> tuple[float, list[float]]:
    by_scene = {str(row["scene"]): row for row in rows}
    _require(set(by_scene) == set(scene_order), "Bootstrap scene order differs")
    grouped: dict[str, list[float]] = defaultdict(list)
    for scene in scene_order:
        row = by_scene[scene]
        grouped[str(row["stratum"])].append(float(row["delta"]))
    _require(
        sorted(len(values) for values in grouped.values()) == [3, 4],
        "Learned bootstrap strata differ",
    )
    ordered = [np.asarray(grouped[key], dtype=np.float64) for key in sorted(grouped)]
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    for draw_index in range(resamples):
        samples = [
            values[rng.integers(0, len(values), size=len(values))] for values in ordered
        ]
        draws[draw_index] = float(np.concatenate(samples).mean())
    point = float(np.mean([float(row["delta"]) for row in rows]))
    interval = [
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    ]
    return point, interval


def _holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=lambda key: (float(p_values[key]), key))
    adjusted = {}
    running = 0.0
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * p_values[key]))
        adjusted[key] = running
    return adjusted


def _validate_learned(root: Path) -> dict[str, int]:
    payload = _read_reference("seven_history_learned", root)
    _require(
        payload.get("kind") == "deja_cue_seven_history_learned_decoder_summary",
        "Unexpected learned result kind",
    )
    _require(
        payload.get("inventory") == {"models": 8, "seeds": list(SEEDS), "histories": 7},
        "Learned result counts differ",
    )
    models = payload.get("models", {})
    _require(set(models) == set(MODELS), "Learned model roster differs")
    for model_id, model in models.items():
        per_seed = model.get("per_seed", {})
        _require(
            set(per_seed) == {str(seed) for seed in SEEDS},
            f"Seed roster differs for {model_id}",
        )
        metric_names = set(next(iter(per_seed.values())))
        for metric in metric_names:
            values = [float(per_seed[str(seed)][metric]) for seed in SEEDS]
            _require(
                _close(model["mean"][metric], np.mean(values)),
                f"Mean differs for {model_id}/{metric}",
            )
            _require(
                _close(model["sample_std"][metric], np.std(values, ddof=1)),
                f"Standard deviation differs for {model_id}/{metric}",
            )

    families = payload.get("families", {})
    bootstrap_scene_order = payload.get("statistical_protocol", {}).get(
        "bootstrap_scene_order"
    )
    _require(
        isinstance(bootstrap_scene_order, list)
        and set(bootstrap_scene_order) == set(SCENES),
        "Bootstrap scene protocol differs",
    )
    _require(
        payload.get("statistical_protocol", {}).get("training_free_baseline")
        == "vocabulary_centered_coordinates",
        "Learned comparison baseline differs",
    )
    _require(set(families) == set(FAMILY_METRICS), "Learned comparison families differ")
    for family_id, (expected_metric, model_metric) in FAMILY_METRICS.items():
        family = families[family_id]
        rows = family.get("rows", [])
        _require(
            family.get("metric") == expected_metric, f"Metric differs for {family_id}"
        )
        _require(
            family.get("family_size") == 8 and len(rows) == 8,
            f"Family size differs for {family_id}",
        )
        expected_order = [f"vocabulary_centered_minus_{model}" for model in MODELS]
        _require(
            family.get("comparison_order") == expected_order,
            f"Comparison order differs for {family_id}",
        )
        raw_p = {}
        for row, model_id in zip(rows, MODELS):
            _require(
                row.get("model_id") == model_id, f"Model order differs for {family_id}"
            )
            histories = row.get("per_history", [])
            _require(
                [item.get("scene") for item in histories] == list(SCENES),
                f"History roster differs for {family_id}/{model_id}",
            )
            for item in histories:
                scene = str(item["scene"])
                _require(
                    item.get("stratum") == STRATA[scene], f"Stratum differs for {scene}"
                )
                seed_values = item.get("learned_seed_values", {})
                _require(
                    set(seed_values) == {str(seed) for seed in SEEDS},
                    f"Seed values differ for {scene}",
                )
                learned_mean = float(
                    np.mean([float(seed_values[str(seed)]) for seed in SEEDS])
                )
                _require(
                    _close(item["learned_three_seed_mean"], learned_mean),
                    f"History mean differs for {scene}",
                )
                _require(
                    _close(
                        item["delta"],
                        float(item["training_free_vocabulary"]) - learned_mean,
                    ),
                    f"History delta differs for {scene}",
                )
            for seed in SEEDS:
                history_mean = float(
                    np.mean(
                        [item["learned_seed_values"][str(seed)] for item in histories]
                    )
                )
                _require(
                    _close(
                        models[model_id]["per_seed"][str(seed)][model_metric],
                        history_mean,
                    ),
                    f"Seed aggregate differs for {family_id}/{model_id}/{seed}",
                )
            point, interval = _stratified_bootstrap(
                histories,
                scene_order=bootstrap_scene_order,
                resamples=int(row["bootstrap_resamples"]),
                seed=int(row["bootstrap_seed"]),
            )
            _require(
                _close(row["point_estimate"], point),
                f"Point estimate differs for {family_id}/{model_id}",
            )
            _require(
                np.allclose(row["ci95"], interval, atol=1e-12, rtol=0.0),
                f"Bootstrap interval differs for {family_id}/{model_id}",
            )
            sign = _exact_sign_flip([float(item["delta"]) for item in histories])
            reference_sign = row["exact_sign_flip"]
            for key, value in sign.items():
                _require(
                    _close(reference_sign[key], value),
                    f"Sign flip differs for {family_id}/{model_id}/{key}",
                )
            comparison_id = f"vocabulary_centered_minus_{model_id}"
            raw_p[comparison_id] = float(sign["exact_two_sided_p"])
            _require(
                _close(row["raw_exact_two_sided_p"], raw_p[comparison_id]),
                f"Raw p-value differs for {family_id}/{model_id}",
            )
        adjusted = _holm_adjust(raw_p)
        for row, model_id in zip(rows, MODELS):
            comparison_id = f"vocabulary_centered_minus_{model_id}"
            _require(
                _close(row["holm8_adjusted_p"], adjusted[comparison_id]),
                f"Holm value differs for {family_id}/{model_id}",
            )

    _validate_finite_tree(payload, label="seven_history_learned")
    return {"models": len(models), "seeds_per_model": len(SEEDS), "comparisons": 16}


def _validate_vocabulary(root: Path) -> dict[str, int]:
    payload = _read_reference("vocabulary_stress", root)
    _require(
        payload.get("kind") == "deja_cue_vocabulary_composition_summary",
        "Unexpected vocabulary result kind",
    )
    scope = payload.get("evaluation_scope", {})
    _require(
        scope.get("composition_table")
        == {"histories": 4, "original_descriptions": 16},
        "Vocabulary-composition evaluation scope differs",
    )
    _require(
        scope.get("duplication_stress")
        == {"histories": 7, "original_descriptions": 32},
        "Duplication-stress evaluation scope differs",
    )
    expected_counts = {
        "symmetric_four_per_state": 32,
        "one_sided_one": 4,
        "one_sided_two": 8,
        "one_sided_four": 16,
    }
    table = payload.get("table", {})
    _require(set(table) == set(expected_counts), "Vocabulary table rows differ")
    for name, expected_count in expected_counts.items():
        row = table[name]
        _require(
            row.get("num_new_descriptions") == expected_count,
            f"Vocabulary count differs for {name}",
        )
        for metric in (
            "new_description_r1_tiou_0.5",
            "new_description_top1_tiou",
            "new_description_identity_accuracy",
            "original_window_mean_tiou",
            "original_window_exact_rate",
        ):
            _require(
                0.0 <= float(row[metric]) <= 1.0,
                f"Vocabulary metric is out of range for {name}/{metric}",
            )
        _require(
            float(row["added_residual_norm_median"]) >= 0.0,
            f"Residual norm differs for {name}",
        )
    duplication = payload.get("duplication_stress", {})
    _require(set(duplication) == {"2", "4", "8"}, "Duplication factors differ")
    expected_changes = {"2": 13, "4": 14, "8": 16}
    for factor, row in duplication.items():
        balanced = row["state_balanced"]
        _require(
            balanced.get("exact_prediction_reproduction") is True,
            f"Balanced reproduction differs for factor {factor}",
        )
        stability = balanced["stability"]
        _require(
            stability.get("num_original_queries") == 32,
            f"Balanced query count differs for factor {factor}",
        )
        _require(
            stability.get("joint_identity_changes") == 0,
            f"Balanced state changes differ for factor {factor}",
        )
        _require(
            stability.get("joint_window_changes") == 0
            and stability.get("target_window_changes") == 0,
            f"Balanced windows differ for factor {factor}",
        )
        query_uniform = row["query_uniform"]["stability"]
        _require(
            query_uniform.get("num_original_queries") == 32
            and query_uniform.get("target_window_changes")
            == expected_changes[factor],
            f"Query-uniform duplication result differs for factor {factor}",
        )
    _validate_finite_tree(payload, label="vocabulary_stress")
    return {"table_rows": len(table), "duplication_factors": len(duplication)}


def validate_reference_results(root: Path | None = None) -> dict[str, Any]:
    """Recalculate result statistics and return the verified counts."""

    repository_root = package_root() if root is None else Path(root).resolve(strict=True)
    return {
        "seven_history": _validate_seven_history(repository_root),
        "learned": _validate_learned(repository_root),
        "vocabulary": _validate_vocabulary(repository_root),
    }
