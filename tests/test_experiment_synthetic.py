"""Tests for the synthetic duration experiment."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from deja_cue.experiments.synthetic import (
    DEFAULT_CONFIG,
    geometric_duration_grid,
    run_synthetic_duration,
)


ROOT = Path(__file__).resolve().parents[1]


def _tiny_config() -> dict[str, object]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    config.update(
        {
            "sequence_length": 40,
            "boundary_margin": 4,
            "true_durations": [4, 8],
            "grid_bases": [3],
            "grid_ratios": [1.5],
            "maximum_candidate_duration": 12,
            "signal_means": [1.0],
            "ar1_correlations": [0.0],
            "trials_per_noise_cell": 3,
            "design_cell_bootstrap_resamples": 20,
        }
    )
    return config


def test_duration_grid_uses_iterative_integer_rounding_and_cap() -> None:
    assert geometric_duration_grid(4, 1.5, 12) == [4, 6, 9, 12]
    assert geometric_duration_grid(5, 1.25, 9) == [5, 6, 8, 9]


def test_reduced_design_is_deterministic_and_retains_cell_statistics() -> None:
    config = _tiny_config()
    first = run_synthetic_duration(config, require_default_config=False)
    second = run_synthetic_duration(config, require_default_config=False)
    assert first == second
    assert first["kind"] == "deja_cue_synthetic_duration_evidence"
    assert len(first["cells"]) == 2
    assert all(
        row["methods"][method]["trials"] == 3
        for row in first["cells"]
        for method in ("sum", "mean", "sqrt_valid_count")
    )
    assert set(first["reported_by_true_duration"]) == {"4", "8"}


def test_reduced_design_requires_explicit_opt_in() -> None:
    with pytest.raises(ValueError, match="published v3 settings"):
        run_synthetic_duration(_tiny_config())


def test_packaged_config_matches_the_default_settings() -> None:
    config = json.loads(
        (ROOT / "configs" / "experiments" / "synthetic_duration_v3.json").read_text(
            encoding="utf-8"
        )
    )
    for key, value in DEFAULT_CONFIG.items():
        assert config[key] == value
