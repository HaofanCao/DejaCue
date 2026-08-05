"""Reproduce the primary 78-history VOST table and exact query windows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deja_cue.data import load_histories, load_protocol
from deja_cue.evaluation import evaluate_main
from deja_cue.io import write_json
from deja_cue.reference import compare_main, load_reference, require_reference_match


def build_parser() -> argparse.ArgumentParser:
    """Build command-line options for the primary fixed-feature evaluation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results" / "main_reproduction.json"
    )
    return parser


def main() -> int:
    """Reproduce the primary methods and compare every output with its reference."""

    args = build_parser().parse_args()
    protocol = load_protocol(ROOT)
    result = evaluate_main(load_histories(ROOT), protocol, device=args.device)
    check = compare_main(result, load_reference("main", ROOT))
    result["reference_check"] = check
    write_json(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "passed": check["passed"],
                "query_windows_checked": check["total_query_windows_checked"],
            },
            sort_keys=True,
        )
    )
    require_reference_match(check, label="Primary reproduction")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
