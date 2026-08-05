"""Train one native learned decoder on the frozen five-history development set."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# This must be set before the first CUDA operation.  The protocol module checks
# the same value again before enabling PyTorch deterministic algorithms.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from deja_cue.learned.checkpoint import save_checkpoint
from deja_cue.learned.development import load_development_histories
from deja_cue.learned.native import native_training_adapter
from deja_cue.learned.protocol import DECODER_REGISTRY, TRAINING_SEEDS, make_training_config
from deja_cue.learned.records import build_positive_run_records
from deja_cue.learned.training import train_decoder


def build_parser() -> argparse.ArgumentParser:
    """Expose identities and paths while keeping all scientific settings fixed."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", choices=tuple(DECODER_REGISTRY), required=True)
    parser.add_argument("--seed", choices=TRAINING_SEEDS, type=int, required=True)
    parser.add_argument("--device", required=True, help="PyTorch device, normally cuda:0")
    parser.add_argument(
        "--development-manifest",
        type=Path,
        default=ROOT / "data" / "learned" / "development" / "manifest.json",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser


def main() -> int:
    """Validate the full roster, train for 200 epochs, and bind final weights."""

    args = build_parser().parse_args()
    if args.checkpoint.exists() or args.summary.exists():
        raise FileExistsError("Checkpoint and summary outputs must not already exist")
    histories = load_development_histories(ROOT, manifest=args.development_manifest)
    records = build_positive_run_records(histories)
    maximum_video_tokens = max(record.num_frames for record in records)
    adapter = native_training_adapter(
        args.model_id,
        maximum_video_tokens=(
            maximum_video_tokens if args.model_id == "sim_detr" else None
        ),
    )
    config = make_training_config(args.model_id, args.seed)
    result = train_decoder(records, adapter, config, device=args.device)
    checkpoint_info = save_checkpoint(
        args.checkpoint,
        result.model.state_dict(),
        config,
        architecture=result.architecture_metadata,
    )
    payload = {
        "schema_version": 1,
        "kind": "deja_cue_learned_decoder_training_summary",
        "development_history_keys": [history.sequence_id for history in histories],
        "maximum_development_run_tokens": maximum_video_tokens,
        "training": result.summary.to_dict(),
        "checkpoint": checkpoint_info,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {
                "model_id": args.model_id,
                "seed": args.seed,
                "checkpoint": str(args.checkpoint),
                "parameter_sha256": checkpoint_info["parameter_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
