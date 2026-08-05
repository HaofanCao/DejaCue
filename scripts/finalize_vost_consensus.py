"""Validate complete VOST consensus and reproduce the fixed cohort intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from deja_cue.preprocessing import sha256_file
from deja_cue.vost_annotation import (
    derive_consensus_cohort,
    validate_consensus,
    validate_derived_cohort_against_reference,
)


ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pass-a", type=Path, required=True)
    parser.add_argument("--pass-b", type=Path, required=True)
    parser.add_argument("--consensus", type=Path, required=True)
    parser.add_argument(
        "--reference-cohort",
        type=Path,
        default=ROOT / "data" / "reference" / "vost_cohort.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")

    histories = validate_consensus(
        _load(args.pass_a), _load(args.pass_b), _load(args.consensus)
    )
    derived = derive_consensus_cohort(histories)
    reference = _load(args.reference_cohort)
    validate_derived_cohort_against_reference(derived, reference)
    summary = {
        "schema_version": 1,
        "kind": "deja_cue_vost_consensus_summary",
        "status": "complete",
        "inputs": {
            "pass_A_sha256": sha256_file(args.pass_a),
            "pass_B_sha256": sha256_file(args.pass_b),
            "consensus_sha256": sha256_file(args.consensus),
            "reference_cohort_sha256": sha256_file(args.reference_cohort),
        },
        "counts": derived["counts"],
        "cohort": derived,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
