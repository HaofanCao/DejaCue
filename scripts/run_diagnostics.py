#!/usr/bin/env python3
"""Run one diagnostic from a JSON input file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deja_cue.experiments.diagnostics import (
    hard_negative_margins,
    paired_margin_summary,
    paraphrase_window_consistency,
    permute_state_assignments,
    reference_boundary_sensitivity,
    run_circular_shift_control,
    summarize_annotation_agreement,
    summarize_recurrence_conditions,
)


DIAGNOSTICS = (
    "annotation-agreement",
    "boundary-sensitivity",
    "circular-shift",
    "state-permutation",
    "hard-negative",
    "paired-margins",
    "paraphrase-consistency",
    "recurrence",
)


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def run_diagnostic(name: str, payload: Mapping[str, Any]) -> Any:
    """Run one diagnostic using a validated JSON object."""

    if name == "annotation-agreement":
        return summarize_annotation_agreement(payload.get("histories", ()))
    if name == "boundary-sensitivity":
        return reference_boundary_sensitivity(
            _mapping(payload.get("predictions_by_condition"), label="predictions_by_condition"),
            _mapping(payload.get("references_by_set"), label="references_by_set"),
            baseline_condition=str(payload.get("baseline_condition", "")),
        )
    if name == "circular-shift":
        return run_circular_shift_control(
            payload.get("values", ()),
            payload.get("frame_indices", ()),
            seed=int(payload.get("seed")),
        )
    if name == "state-permutation":
        return {
            "schema_version": 1,
            "kind": "deja_cue_state_assignment_permutation",
            "query_rows": permute_state_assignments(
                payload.get("query_rows", ()),
                _mapping(payload.get("source_to_target"), label="source_to_target"),
                require_derangement=bool(payload.get("require_derangement", True)),
            ),
        }
    if name == "hard-negative":
        rows = []
        for index, query in enumerate(payload.get("queries", ())):
            query = _mapping(query, label=f"queries[{index}]")
            history_id = str(query.get("history_id", ""))
            query_id = str(query.get("query_id", ""))
            if not history_id or not query_id:
                raise ValueError(f"queries[{index}] needs history_id and query_id")
            result = hard_negative_margins(
                query.get("target_candidates", ()),
                query.get("queried_references", ()),
                query.get("sibling_references", ()),
                query.get("auxiliary_tracks", ()),
                positive_tiou=float(payload.get("positive_tiou", 0.5)),
            )
            rows.append(
                {
                    "history_id": history_id,
                    "query_id": query_id,
                    **result,
                }
            )
        if not rows:
            raise ValueError("Hard-negative input contains no queries")
        return {
            "schema_version": 1,
            "kind": "deja_cue_hard_negative_margins",
            "queries": rows,
        }
    if name == "paired-margins":
        return paired_margin_summary(
            payload.get("baseline_rows", ()),
            payload.get("treatment_rows", ()),
            margin_field=str(payload.get("margin_field", "")),
            seed=int(payload.get("seed", 3407)),
            bootstrap_resamples=int(payload.get("bootstrap_resamples", 10_000)),
        )
    if name == "paraphrase-consistency":
        return paraphrase_window_consistency(
            payload.get("prediction_rows", ()),
            component_key=str(payload.get("component_key", "source_component_id")),
        )
    if name == "recurrence":
        metrics = summarize_recurrence_conditions(
            payload.get("queries", ()),
            _mapping(payload.get("references"), label="references"),
            _mapping(payload.get("predictions_by_condition"), label="predictions_by_condition"),
        )
        return {
            "schema_version": 1,
            "kind": "deja_cue_recurrence_metrics",
            "reported_metrics": metrics,
        }
    raise ValueError(f"Unsupported diagnostic: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", choices=DIAGNOSTICS, required=True)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="JSON file containing the inputs required by the diagnostic.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    result = run_diagnostic(args.diagnostic, _mapping(payload, label="input"))
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")
    print(
        json.dumps(
            {
                "diagnostic": args.diagnostic,
                "output": args.output.as_posix(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
