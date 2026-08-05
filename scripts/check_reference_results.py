"""Recompute statistics contained in the compact experiment summaries."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deja_cue.reference_results import validate_reference_results  # noqa: E402


def main() -> int:
    """Validate learned, seven-history, and vocabulary reference summaries."""

    checks = validate_reference_results(ROOT)
    print(json.dumps({"passed": True, **checks}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"reference result check failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
