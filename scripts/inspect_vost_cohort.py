"""Inspect and validate the bundled VOST cohort roster."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deja_cue.vost_protocol import load_cohort_asset, validate_cohort_asset  # noqa: E402


def main() -> int:
    """Validate the fixed 100-history roster and report retained exclusions."""

    parser = argparse.ArgumentParser(
        description="Validate the sampled, retained, and excluded VOST histories."
    )
    parser.add_argument(
        "--asset",
        type=Path,
        default=ROOT / "data" / "reference" / "vost_cohort.json",
    )
    parser.add_argument(
        "--show-roster",
        action="store_true",
        help="Print one compact line for each sampled history.",
    )
    args = parser.parse_args()

    payload = load_cohort_asset(args.asset)
    counts = validate_cohort_asset(payload)
    report = {
        "passed": True,
        **counts,
        "minimum_stable_frames_per_side": payload["protocol"]["eligibility"][
            "minimum_stable_frames_per_side"
        ],
        "excluded_history_ids": payload["excluded_history_ids"],
    }
    print(json.dumps(report, sort_keys=True))
    if args.show_roster:
        for row in payload["roster"]:
            print(
                "\t".join(
                    (
                        row["history_id"],
                        row["status"],
                        row["sequence_id"],
                        row["action"],
                        row["target_noun"],
                        str(row["qualifying_event_count"]),
                    )
                )
            )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"VOST cohort check failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
