"""Reproduce the seven-history coordinate-origin evaluation.

The script evaluates the same frozen visual arrays and 33-duration schedule
under absolute, trajectory-centered, vocabulary-centered, and dual-centered
coordinates.  The result is compared with the compact reference summary; the
comparison is over the three metrics used in the paper appendix and the complete
history-level prediction roster.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deja_cue.data import load_protocol
from deja_cue.evaluation import evaluate_condition
from deja_cue.io import write_json
from deja_cue.seven_history import (
    load_seven_history_records,
    prediction_window_map,
)


COORDINATES = {
    "absolute_coordinates": (0.0, 0.0),
    "visual_centered_coordinates": (1.0, 0.0),
    "vocabulary_centered_coordinates": (0.0, 1.0),
    "dual_centered_coordinates": (1.0, 1.0),
}
METRICS = (
    "state_macro_target_r1_tiou_0.3",
    "state_macro_target_r1_tiou_0.5",
    "state_macro_target_top1_tiou",
)


def build_parser() -> argparse.ArgumentParser:
    """Build command-line options for the seven-history coordinate study."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "seven_history_reproduction.json",
    )
    return parser


def main() -> int:
    """Reproduce all four coordinate origins and compare 128 selected windows."""

    args = build_parser().parse_args()
    protocol = load_protocol(ROOT)
    records = load_seven_history_records(ROOT)
    histories = tuple(record.history for record in records)
    history_aliases = {
        record.history.history_id: record.scene for record in records
    }
    schedule = tuple(int(value) for value in protocol["window_schedule"])
    observed = {}
    for name, (visual_centering, query_centering) in COORDINATES.items():
        observed[name] = evaluate_condition(
            histories,
            schedule,
            visual_centering=visual_centering,
            query_centering=query_centering,
            device=args.device,
            bootstrap_resamples=int(protocol["bootstrap_resamples"]),
            seed=int(protocol["seed"]),
        )

    reference = json.loads(
        (ROOT / "data" / "reference" / "seven_history_summary.json").read_text(
            encoding="utf-8"
        )
    )
    errors: list[str] = []
    for name in COORDINATES:
        expected = reference["coordinate_conditions"][name]
        actual = observed[name]
        for metric in METRICS:
            value = float(actual["aggregate"]["source_component_macro"][metric])
            target = float(expected[metric])
            if abs(value - target) > 1e-8:
                errors.append(f"{name}/{metric}: {value} != {target}")
        expected_windows = reference.get("coordinate_prediction_windows", {}).get(name)
        if expected_windows is not None:
            if prediction_window_map(actual, history_aliases) != {
                tuple(key.split("|")): tuple(value)
                for key, value in expected_windows.items()
            }:
                errors.append(f"{name}/prediction_windows")

    result = {
        "schema_version": 1,
        "kind": "deja_cue_seven_history_reproduction",
        "device": args.device,
        "window_schedule": list(schedule),
        "conditions": observed,
        "reference_check": {"passed": not errors, "errors": errors},
    }
    write_json(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "passed": not errors,
                "histories": len(histories),
                "queries": sum(len(history.queries) for history in histories),
            },
            sort_keys=True,
        )
    )
    if errors:
        raise SystemExit("Seven-history reproduction differs: " + "; ".join(errors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
