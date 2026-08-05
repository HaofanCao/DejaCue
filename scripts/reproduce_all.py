"""Run every integrity, statistical, numerical, and regression check."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    """Build options shared by CPU/CUDA full-workflow runs."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="omit the unit-test step while retaining all result comparisons",
    )
    return parser


def main() -> int:
    """Run each integrity, statistics, evaluation, and test stage in order."""

    args = build_parser().parse_args()
    commands = [
        ("package", [sys.executable, "scripts/verify_package.py"]),
        ("cohort", [sys.executable, "scripts/inspect_vost_cohort.py"]),
        ("reference", [sys.executable, "scripts/check_reference_results.py"]),
        ("robustness", [sys.executable, "scripts/check_robustness_results.py"]),
        (
            "experiment_sources",
            [sys.executable, "scripts/check_experiment_sources.py"],
        ),
        (
            "primary",
            [sys.executable, "scripts/reproduce_main.py", "--device", args.device],
        ),
        (
            "extended",
            [sys.executable, "scripts/reproduce_extended.py", "--device", args.device],
        ),
        (
            "seven_history",
            [
                sys.executable,
                "scripts/reproduce_seven_history.py",
                "--device",
                args.device,
            ],
        ),
    ]
    if not args.skip_tests:
        commands.append(("tests", [sys.executable, "-m", "pytest", "-q"]))

    elapsed = {}
    total_start = time.perf_counter()
    for label, command in commands:
        start = time.perf_counter()
        subprocess.run(command, cwd=ROOT, check=True)
        elapsed[label] = round(time.perf_counter() - start, 3)
    print(
        json.dumps(
            {
                "device": args.device,
                "passed": True,
                "seconds": elapsed,
                "total_seconds": round(time.perf_counter() - total_start, 3),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
