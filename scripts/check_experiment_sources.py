#!/usr/bin/env python3
"""Recalculate experiment results and compare them with reference files."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deja_cue.experiments.diagnostics import summarize_recurrence_conditions
from deja_cue.experiments.perturbations import perturbation_cells
from deja_cue.experiments.synthetic import run_synthetic_duration


def read_json(path: Path) -> dict:
    """Read one package JSON object."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path.name}")
    return value


def main() -> int:
    """Check synthetic, recurrence, and tracking experiment settings."""

    synthetic_config = read_json(
        ROOT / "configs" / "experiments" / "synthetic_duration_v3.json"
    )
    synthetic_reference = read_json(
        ROOT / "data" / "reference" / "robustness" / "synthetic_duration.json"
    )
    if run_synthetic_duration(synthetic_config) != synthetic_reference:
        raise ValueError("Synthetic-duration result differs from the reference")

    recurrence = read_json(
        ROOT / "data" / "reference" / "robustness" / "dnerf_recurrence.json"
    )
    calculated = summarize_recurrence_conditions(
        recurrence["queries"], recurrence["references"], recurrence["predictions"]
    )
    if calculated != recurrence["reported_metrics"]:
        raise ValueError("Recurrence result differs from the reference")

    cells = perturbation_cells()
    if len(cells) != 19 or any(
        row["replicate_seeds"] != [3407, 3408, 3409] for row in cells
    ):
        raise ValueError("Configured tracking perturbation set differs")

    print(
        json.dumps(
            {
                "passed": True,
                "recurrence_conditions": len(calculated),
                "synthetic_cells": len(synthetic_reference["cells"]),
                "tracking_cells": len(cells),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
