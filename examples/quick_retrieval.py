"""Run vocabulary-relative retrieval for one bundled object history."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deja_cue.data import load_histories, load_protocol  # noqa: E402
from deja_cue.scan import run_scan  # noqa: E402


def main() -> None:
    """Print one prediction row per state description in the first history."""

    protocol = load_protocol(ROOT)
    history = load_histories(ROOT)[0]
    output = run_scan(
        history,
        protocol["window_schedule"],
        visual_centering=0.0,
        query_centering=1.0,
        normalization=protocol["primary_window_normalization"],
        device="cpu",
        mad_factor=protocol["mad_consistency_factor"],
        mad_floor=protocol["mad_floor"],
        minimum_query_residual_norm=protocol["minimum_query_residual_norm"],
    )
    rows = [
        {
            "state_id": query.state_id,
            "description": query.text,
            "window": list(window),
            "score": score,
        }
        for query, window, score in zip(
            history.queries, output.windows, output.selected_scores
        )
    ]
    print(
        json.dumps(
            {"history_id": history.history_id, "predictions": rows},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
