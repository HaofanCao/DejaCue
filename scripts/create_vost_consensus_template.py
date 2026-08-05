"""Create a human-review template from two complete independent VOST passes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from deja_cue.vost_annotation import build_consensus_template


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pass-a", type=Path, required=True)
    parser.add_argument("--pass-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    template = build_consensus_template(_load(args.pass_a), _load(args.pass_b))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(template, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"disagreements": template["disagreement_count"]}))


if __name__ == "__main__":
    main()
