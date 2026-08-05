#!/usr/bin/env python3
"""Benchmark the Deja Cue scan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deja_cue.data import load_histories, load_protocol
from deja_cue.experiments.benchmark import benchmark_scan
from deja_cue.experiments.synthetic import canonical_json_bytes
from deja_cue.seven_history import load_seven_histories


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--cohort", choices=("vost", "seven_history"), default="vost"
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument(
        "--limit-histories",
        type=int,
        help="Use only the first N histories for a quick smoke benchmark.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    histories = (
        load_histories(ROOT)
        if args.cohort == "vost"
        else load_seven_histories(ROOT)
    )
    if args.limit_histories is not None:
        if args.limit_histories <= 0:
            raise SystemExit("--limit-histories must be positive")
        histories = histories[: args.limit_histories]
    protocol = load_protocol(ROOT)
    payload = benchmark_scan(
        histories,
        protocol["window_schedule"],
        device=args.device,
        warmup_repetitions=args.warmup,
        timed_repetitions=args.repetitions,
    )
    encoded = canonical_json_bytes(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(encoded)
    print(json.dumps(payload, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
