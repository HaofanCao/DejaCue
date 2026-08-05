#!/usr/bin/env python3
"""Generate results for the five default tracking perturbations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deja_cue.data import load_protocol
from deja_cue.experiments.synthetic import canonical_json_bytes
from deja_cue.experiments.tracking import run_tracking_perturbations
from deja_cue.seven_history import load_seven_history_records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--check-reference",
        type=Path,
        help="Fail unless the generated JSON object equals this reference file.",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    protocol = load_protocol(ROOT)
    payload = run_tracking_perturbations(
        load_seven_history_records(ROOT),
        protocol["window_schedule"],
        device=args.device,
        bootstrap_resamples=int(protocol["bootstrap_resamples"]),
        seed=int(protocol["seed"]),
    )
    reference_match = None
    if args.check_reference is not None:
        expected = json.loads(args.check_reference.read_text(encoding="utf-8"))
        reference_match = payload == expected
        if not reference_match:
            raise SystemExit("generated tracking evidence differs from the reference")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(payload))
    print(
        json.dumps(
            {
                "conditions": len(payload["conditions"]),
                "reference_match": reference_match,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
