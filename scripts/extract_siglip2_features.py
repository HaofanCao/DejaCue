"""Extract pinned SigLIP2 features from a state-reference-blind manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from deja_cue.preprocessing import extract_feature_manifest, load_pinned_encoder


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Feature manifest must be a JSON object")
    model, processor, model_info = load_pinned_encoder(
        args.model_directory, device=device
    )
    summary = extract_feature_manifest(
        manifest,
        raw_root=args.raw_root,
        output_root=args.output_root,
        model=model,
        processor=processor,
        model_info=model_info,
        batch_size=args.batch_size,
        device=device,
    )
    print(
        json.dumps(
            {
                "history_count": summary["history_count"],
                "feature_dimension": model_info["feature_dimension"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
