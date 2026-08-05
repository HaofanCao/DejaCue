"""Evaluate one native learned-decoder checkpoint on all seven unseen histories."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Keep checkpoint evaluation on the training process's deterministic cuBLAS
# settings before any CUDA operation is created.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from deja_cue.learned.checkpoint import load_checkpoint
from deja_cue.learned.native import native_training_adapter, predict_native_records
from deja_cue.learned.protocol import (
    DECODER_REGISTRY,
    TRAINING_SEEDS,
    make_training_config,
    set_deterministic_seed,
)
from deja_cue.learned.records import build_evaluation_run_records, select_best_proposal
from deja_cue.seven_history import load_seven_histories


def _temporal_iou(
    window: tuple[int, int], references: Sequence[tuple[int, int]]
) -> float:
    start, end = window
    best = 0.0
    for target_start, target_end in references:
        intersection = max(0, min(end, target_end) - max(start, target_start) + 1)
        union = (
            end
            - start
            + 1
            + target_end
            - target_start
            + 1
            - intersection
        )
        best = max(best, intersection / union)
    return float(best)


def build_parser() -> argparse.ArgumentParser:
    """Build the fixed-protocol checkpoint evaluation options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", choices=tuple(DECODER_REGISTRY), required=True)
    parser.add_argument("--seed", choices=TRAINING_SEEDS, type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    """Restore one final checkpoint and aggregate description/state/history metrics."""

    args = build_parser().parse_args()
    if args.output.exists():
        raise FileExistsError("Evaluation output already exists")
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for learned-decoder evaluation") from exc
    checkpoint = load_checkpoint(
        args.checkpoint,
        expected_model_id=args.model_id,
        expected_seed=args.seed,
        map_location=args.device,
    )
    architecture = checkpoint["metadata"].get("architecture")
    if not isinstance(architecture, dict):
        raise ValueError("Native checkpoint lacks architecture/source binding")
    maximum_video_tokens = int(architecture["maximum_video_tokens"])
    set_deterministic_seed(args.seed, torch_module=torch)
    adapter = native_training_adapter(
        args.model_id,
        maximum_video_tokens=(
            maximum_video_tokens if args.model_id == "sim_detr" else None
        ),
    )
    config = make_training_config(args.model_id, args.seed)
    device = torch.device(args.device)
    components = adapter.build(config, device)
    if dict(components.architecture_metadata or {}) != architecture:
        raise ValueError("Checkpoint source/config binding differs from this package")
    components.model.load_state_dict(checkpoint["state_dict"], strict=True)

    histories = load_seven_histories(ROOT)
    records = build_evaluation_run_records(histories)
    by_query: dict[tuple[str, int], list[object]] = defaultdict(list)
    for record in records:
        by_query[(record.history_id, record.query_index)].append(record)
    rows = []
    history_metrics = []
    for history in histories:
        description_rows = []
        for query_index, query in enumerate(history.queries):
            proposals = predict_native_records(
                components.model,
                by_query[(history.history_id, query_index)],
                model_id=args.model_id,
                device=device,
            )
            selected = select_best_proposal(proposals)
            tiou = _temporal_iou(
                (selected.start, selected.end), history.references[query.state_id]
            )
            description_rows.append(
                {
                    "query_index": query_index,
                    "state_id": query.state_id,
                    "prediction": [selected.start, selected.end],
                    "score": selected.score,
                    "top1_tiou": tiou,
                    "r1_tiou_0.3": float(tiou >= 0.3),
                    "r1_tiou_0.5": float(tiou >= 0.5),
                }
            )
        state_rows = []
        for state_id in history.references:
            selected_rows = [
                row for row in description_rows if row["state_id"] == state_id
            ]
            state_rows.append(
                {
                    "state_id": state_id,
                    **{
                        name: float(np.mean([row[name] for row in selected_rows]))
                        for name in ("top1_tiou", "r1_tiou_0.3", "r1_tiou_0.5")
                    },
                }
            )
        metrics = {
            name: float(np.mean([row[name] for row in state_rows]))
            for name in ("top1_tiou", "r1_tiou_0.3", "r1_tiou_0.5")
        }
        history_metrics.append(metrics)
        rows.append(
            {
                "history_id": history.history_id,
                "history_key": history.sequence_id,
                "metrics": metrics,
                "states": state_rows,
                "descriptions": description_rows,
            }
        )
    aggregate = {
        name: float(np.mean([row[name] for row in history_metrics]))
        for name in ("top1_tiou", "r1_tiou_0.3", "r1_tiou_0.5")
    }
    payload = {
        "schema_version": 1,
        "kind": "deja_cue_learned_decoder_seed_evaluation",
        "model_id": args.model_id,
        "seed": args.seed,
        "parameter_sha256": checkpoint["parameter_sha256"],
        "aggregation": "description_to_state_to_history",
        "aggregate": aggregate,
        "histories": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), **aggregate}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
