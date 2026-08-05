"""Reproduce the VOST prompt, readout, ranking, and duration analyses."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deja_cue.data import load_histories, load_protocol
from deja_cue.evaluation import evaluate_extended
from deja_cue.io import write_json
from deja_cue.reference import compare_extended, load_reference, require_reference_match


def build_parser() -> argparse.ArgumentParser:
    """Build command-line options for the matched extended evaluation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "extended_reproduction.json",
    )
    return parser


def main() -> int:
    """Recompute prompt, normalization, peak, ranking, and duration controls."""

    args = build_parser().parse_args()
    protocol = load_protocol(ROOT)
    prompt_histories = {
        variant: load_histories(ROOT, prompt_variant=variant)
        for variant in protocol["prompt_variants"]
    }
    result = evaluate_extended(
        load_histories(ROOT),
        prompt_histories,
        protocol,
        device=args.device,
    )
    check = compare_extended(result, load_reference("extended", ROOT))
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
    require_reference_match(check, label="Extended reproduction")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
