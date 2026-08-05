#!/usr/bin/env python3
"""Generate synthetic-duration experiment results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deja_cue.experiments.synthetic import run_synthetic_duration, write_synthetic_result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "experiments" / "synthetic_duration_v3.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--check-reference",
        type=Path,
        help="Fail unless the generated JSON object equals this reference file.",
    )
    parser.add_argument(
        "--reduced-design",
        action="store_true",
        help="Permit a smaller custom design instead of the published settings.",
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    payload = run_synthetic_duration(
        config, require_default_config=not args.reduced_design
    )
    if args.check_reference is not None:
        expected = json.loads(args.check_reference.read_text(encoding="utf-8"))
        if payload != expected:
            raise SystemExit("generated synthetic evidence differs from the reference")
    write_synthetic_result(payload, args.output)
    print(
        json.dumps(
            {
                "design_cells": len(payload["cells"]),
                "output": args.output.as_posix(),
                "reference_match": args.check_reference is not None,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
