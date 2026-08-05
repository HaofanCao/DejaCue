"""Installed console entry point for the repository workflows."""

from __future__ import annotations

import argparse
import subprocess
import sys

from .data import package_root


def main() -> int:
    """Dispatch the selected workflow and return its subprocess exit status."""

    parser = argparse.ArgumentParser(prog="deja-cue")
    parser.add_argument(
        "workflow",
        choices=(
            "all",
            "main",
            "extended",
            "seven-history",
            "reference",
            "robustness",
            "cohort",
            "verify",
        ),
    )
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    script = {
        "all": "reproduce_all.py",
        "main": "reproduce_main.py",
        "extended": "reproduce_extended.py",
        "seven-history": "reproduce_seven_history.py",
        "reference": "check_reference_results.py",
        "robustness": "check_robustness_results.py",
        "cohort": "inspect_vost_cohort.py",
        "verify": "verify_package.py",
    }[args.workflow]
    command = [
        sys.executable,
        str(package_root() / "scripts" / script),
        *args.arguments,
    ]
    return int(subprocess.run(command, check=False).returncode)


if __name__ == "__main__":
    raise SystemExit(main())
