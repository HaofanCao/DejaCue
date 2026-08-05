"""Select the fixed VOST 100-history source-only round-robin roster."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from deja_cue.vost_sampling import (
    DEFAULT_HISTORY_COUNT,
    DEFAULT_SEED,
    build_cohort_selection,
    validate_selection_against_cohort,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-split",
        type=Path,
        default=ROOT / "data" / "protocol" / "vost" / "train.txt",
    )
    parser.add_argument(
        "--validation-split",
        type=Path,
        default=ROOT / "data" / "protocol" / "vost" / "validation.txt",
    )
    parser.add_argument(
        "--reference-cohort",
        type=Path,
        default=ROOT / "data" / "reference" / "vost_cohort.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--history-count", type=int, default=DEFAULT_HISTORY_COUNT)
    args = parser.parse_args()

    selection = build_cohort_selection(
        args.train_split,
        args.validation_split,
        seed=args.seed,
        history_count=args.history_count,
    )
    reference = json.loads(args.reference_cohort.read_text(encoding="utf-8"))
    validate_selection_against_cohort(selection, reference)
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(selection["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
