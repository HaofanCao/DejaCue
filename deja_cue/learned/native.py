"""Native model factories, objectives, and inference for all eight decoders.

The shared trainer implements the common data and optimization settings. This
module verifies the bundled third-party source, constructs each original
architecture, converts the shared batch to its native arguments, applies the
original loss, and decodes native inference outputs.
It implements the paper appendix, "Low-Data Decoder Adaptation," especially the
"Shared Adaptation Protocol" subsection.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import random
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from .protocol import (
    BATCH_SIZE,
    EPOCHS,
    GRADIENT_CLIP,
    LEARNING_RATE,
    TEXT_TOKEN_DIM,
    TRAINING_SEEDS,
    VIDEO_TOKEN_DIM,
    WEIGHT_DECAY,
    TrainingConfig,
    decoder_spec,
)
from .records import (
    LearnedRunRecord,
    PreparedBatch,
    build_target_masks,
    collate_records,
    decode_proposals,
    reorder_sim_detr_text,
)
from .training import (
    DecoderTrainingAdapter,
    DecoderTrainingComponents,
    LossOutput,
    TrainingStepContext,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
CONFIG_LOCK_PATH = PACKAGE_ROOT / "configs" / "learned" / "native_decoders.json"
THIRD_PARTY_ROOT = PACKAGE_ROOT / "third_party"
LIGHTHOUSE_ROOT = THIRD_PARTY_ROOT / "lighthouse"
SIM_DETR_ROOT = THIRD_PARTY_ROOT / "sim_detr"

LIGHTHOUSE_COMMIT = "d095eaa552cecef240897a8b750306b3b2a08740"
SIM_DETR_COMMIT = "1965e994bd4ef486c2a9b137ef4b0b57837330c3"
TASKWEAVE_VIDEO_TOKENS = 75


class _AttributeDict(dict[str, Any]):
    """Mapping with the attribute access expected by Lighthouse factories."""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_vendored_source(project: str) -> dict[str, Any]:
    """Verify every bundled third-party file against its immutable lock."""

    projects = {
        "lighthouse": (LIGHTHOUSE_ROOT, "Lighthouse", LIGHTHOUSE_COMMIT),
        "sim_detr": (SIM_DETR_ROOT, "Sim-DETR", SIM_DETR_COMMIT),
    }
    try:
        root, display_name, expected_commit = projects[project]
    except KeyError as exc:
        raise ValueError(f"Unknown vendored project: {project!r}") from exc
    root = root.resolve(strict=True)
    lock_path = root / "SOURCE_LOCK.json"
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    rows = payload.get("files")
    if (
        payload.get("schema_version") != 1
        or payload.get("kind") != "deja_cue_vendored_source_lock"
        or payload.get("project") != display_name
        or payload.get("upstream_commit") != expected_commit
        or not isinstance(rows, list)
    ):
        raise ValueError(f"{display_name} source lock differs")

    expected_paths: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"{display_name} source lock contains a non-file row")
        relative = str(row.get("path", ""))
        candidate = (root / Path(relative)).resolve(strict=True)
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"{display_name} source lock escapes its root") from exc
        if (
            not relative
            or "\\" in relative
            or relative in expected_paths
            or not candidate.is_file()
            or candidate.is_symlink()
            or candidate.stat().st_size != int(row.get("size_bytes", -1))
            or _sha256_file(candidate) != row.get("sha256")
        ):
            raise ValueError(f"{display_name} vendored file differs: {relative}")
        expected_paths.add(relative)

    observed_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.name != "SOURCE_LOCK.json"
        and path.suffix.lower() not in {".pyc", ".pyo"}
        and "__pycache__" not in path.parts
    }
    if observed_paths != expected_paths:
        raise ValueError(
            f"{display_name} vendored file list differs: "
            f"missing={sorted(expected_paths - observed_paths)}, "
            f"unexpected={sorted(observed_paths - expected_paths)}"
        )
    return {
        "project": project,
        "upstream_commit": expected_commit,
        "file_count": len(expected_paths),
        "source_lock_sha256": _sha256_file(lock_path),
    }


def _config_lock() -> dict[str, Any]:
    payload = json.loads(CONFIG_LOCK_PATH.read_text(encoding="utf-8"))
    expected_shared = {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip": GRADIENT_CLIP,
        "seeds": list(TRAINING_SEEDS),
        "checkpoint_selection": "final_checkpoint_only",
    }
    if (
        payload.get("schema_version") != 1
        or payload.get("kind") != "deja_cue_native_decoder_config_lock"
        or payload.get("shared_training") != expected_shared
        or payload.get("lighthouse", {}).get("upstream_commit")
        != LIGHTHOUSE_COMMIT
        or payload.get("sim_detr", {}).get("upstream_commit") != SIM_DETR_COMMIT
    ):
        raise ValueError("Native decoder configuration lock differs")
    return payload


def resolve_native_config(
    model_id: str,
    *,
    device: str,
    maximum_video_tokens: int | None = None,
) -> dict[str, Any]:
    """Resolve the exact architecture configuration used by one decoder."""

    spec = decoder_spec(model_id)
    lock = _config_lock()
    if spec.backend == "lighthouse":
        models = lock["lighthouse"]["models"]
        if model_id not in models:
            raise ValueError(f"Missing Lighthouse config: {model_id}")
        config = dict(lock["lighthouse"]["base"])
        config.update(models[model_id])
        config.update(
            model_name="taskweave" if model_id == "taskweave_mr2hd" else model_id,
            stable_model_name=model_id,
            dset_name="state_moment_retrieval",
            device=str(device),
            v_feat_dim=VIDEO_TOKEN_DIM,
            t_feat_dim=TEXT_TOKEN_DIM,
            a_feat_dim=0,
            max_v_l=(
                TASKWEAVE_VIDEO_TOKENS
                if model_id == "taskweave_mr2hd"
                else 4096
            ),
            max_windows=10,
            span_loss_type="l1",
        )
        if model_id == "tr_detr" and (
            float(config.get("CTC_loss_coef", math.nan)) != 0.5
            or float(config.get("VTC_loss_coef", math.nan)) != 0.3
        ):
            raise ValueError("TR-DETR loss coefficients differ")
        if model_id == "taskweave_mr2hd" and (
            config.get("mr2hd") is not True
            or config.get("model_ema") is not True
            or float(config.get("ema_decay", math.nan)) != 0.9
        ):
            raise ValueError("TaskWeave MR-to-HD/EMA configuration differs")
        return config

    if maximum_video_tokens is None or maximum_video_tokens <= 0:
        raise ValueError("Sim-DETR requires the development-set maximum run length")
    if maximum_video_tokens > 4096:
        raise ValueError("Sim-DETR maximum run length exceeds the shared limit")
    config = dict(lock["sim_detr"]["config"])
    config.update(
        device=str(device),
        v_feat_dim=VIDEO_TOKEN_DIM,
        t_feat_dim=TEXT_TOKEN_DIM,
        max_v_l=int(maximum_video_tokens),
    )
    return config


def _activate_vendor(root: Path, top_level_package: str) -> None:
    """Give the verified local source precedence over ambient installations."""

    resolved = root.resolve(strict=True)
    sys.dont_write_bytecode = True
    for name, module in tuple(sys.modules.items()):
        if name != top_level_package and not name.startswith(f"{top_level_package}."):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            del sys.modules[name]
            continue
        try:
            Path(module_file).resolve().relative_to(resolved)
        except ValueError:
            del sys.modules[name]
    source = str(resolved)
    if source in sys.path:
        sys.path.remove(source)
    sys.path.insert(0, source)
    importlib.invalidate_caches()


def _architecture_metadata(
    model_id: str,
    config: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "model_family": decoder_spec(model_id).backend,
        "encoder_layers": int(config["enc_layers"]),
        "decoder_layers": int(config["dec_layers"]),
        "feedforward_dimension": int(config["dim_feedforward"]),
        "hidden_dimension": int(config["hidden_dim"]),
        "attention_heads": int(config["nheads"]),
        "proposal_count": int(config["num_queries"]),
        "input_dropout": float(config["input_dropout"]),
        "dropout": float(config["dropout"]),
        "position_embedding": str(config["position_embedding"]),
        "span_loss_type": str(config["span_loss_type"]),
        "auxiliary_loss": bool(config["aux_loss"]),
        "maximum_video_tokens": int(config["max_v_l"]),
        "native_source_commit": str(source["upstream_commit"]),
        "native_source_lock_sha256": str(source["source_lock_sha256"]),
        "native_config_lock_sha256": _sha256_file(CONFIG_LOCK_PATH),
    }


def _taskweave_horizon(
    model_inputs: dict[str, Any],
    records: Sequence[LearnedRunRecord],
    token_union: Any | None,
) -> tuple[dict[str, Any], Any | None]:
    import torch

    source = model_inputs["src_vid"]
    batch_size, _length, dimension = source.shape
    fixed = torch.zeros((batch_size, TASKWEAVE_VIDEO_TOKENS, dimension), dtype=source.dtype)
    fixed_mask = torch.zeros(
        (batch_size, TASKWEAVE_VIDEO_TOKENS), dtype=model_inputs["src_vid_mask"].dtype
    )
    fixed_union = (
        torch.zeros((batch_size, TASKWEAVE_VIDEO_TOKENS), dtype=token_union.dtype)
        if token_union is not None
        else None
    )
    for batch_index, record in enumerate(records):
        length = record.num_frames
        if length <= TASKWEAVE_VIDEO_TOKENS:
            fixed[batch_index, :length] = source[batch_index, :length]
            fixed_mask[batch_index, :length] = 1.0
            if fixed_union is not None:
                fixed_union[batch_index, :length] = token_union[batch_index, :length]
            continue
        indices = torch.round(
            torch.linspace(0, length - 1, TASKWEAVE_VIDEO_TOKENS)
        ).to(dtype=torch.long)
        fixed[batch_index] = source[batch_index, indices]
        fixed_mask[batch_index] = 1.0
        if fixed_union is not None:
            for token_index in range(TASKWEAVE_VIDEO_TOKENS):
                left = math.floor(token_index * length / TASKWEAVE_VIDEO_TOKENS)
                right = math.ceil((token_index + 1) * length / TASKWEAVE_VIDEO_TOKENS)
                fixed_union[batch_index, token_index] = token_union[
                    batch_index, left:right
                ].max()
    output = dict(model_inputs)
    output["src_vid"] = fixed
    output["src_vid_mask"] = fixed_mask
    return output, fixed_union


def _lighthouse_saliency_pairs(
    token_union: Any,
    valid_mask: Any,
    *,
    base_seed: int,
) -> tuple[Any, Any]:
    import torch

    positive_pairs: list[tuple[int, int]] = []
    negative_pairs: list[tuple[int, int]] = []
    for position, labels in enumerate(token_union):
        rng = random.Random(base_seed + position)
        valid = valid_mask[position] > 0
        positive = [
            int(value)
            for value in torch.nonzero((labels > 0) & valid, as_tuple=False).flatten()
        ]
        negative = [
            int(value)
            for value in torch.nonzero((labels <= 0) & valid, as_tuple=False).flatten()
        ]
        if not positive:
            raise ValueError("Lighthouse training requires a positive record")
        positive_pair = (
            tuple(rng.sample(positive, k=2))
            if len(positive) >= 2
            else (positive[0], positive[0])
        )
        negative_pair = (
            tuple(rng.sample(negative, k=2))
            if len(negative) >= 2
            else positive_pair
        )
        positive_pairs.append(positive_pair)
        negative_pairs.append(negative_pair)
    return (
        torch.tensor(positive_pairs, dtype=torch.long),
        torch.tensor(negative_pairs, dtype=torch.long),
    )


def _sim_saliency_pairs(
    records: Sequence[LearnedRunRecord], context: TrainingStepContext
) -> tuple[np.ndarray, np.ndarray]:
    positive_rows: list[tuple[int, int]] = []
    negative_rows: list[tuple[int, int]] = []
    for record in records:
        identity = "::".join(
            (
                str(context.config.seed),
                str(context.zero_based_epoch),
                str(context.batch_index),
                str(record.native_history_id or record.history_id),
                str(record.state_id),
                str(record.query_index),
                str(record.run_index),
            )
        )
        seed = int.from_bytes(
            hashlib.sha256(identity.encode("utf-8")).digest()[:8],
            "big",
            signed=False,
        )
        union = np.zeros(record.num_frames, dtype=bool)
        for left, right in record.target_token_spans:
            union[left:right] = True
        positive = np.flatnonzero(union)
        negative = np.flatnonzero(~union)
        if not len(positive):
            raise ValueError("Sim-DETR training requires a positive record")
        rng = np.random.default_rng(seed)

        def sample(values: np.ndarray) -> tuple[int, int]:
            if len(values) >= 2:
                chosen = rng.choice(values, size=2, replace=False)
                return int(chosen[0]), int(chosen[1])
            return int(values[0]), int(values[0])

        positive_pair = sample(positive)
        positive_rows.append(positive_pair)
        negative_rows.append(sample(negative) if len(negative) else positive_pair)
    return (
        np.asarray(positive_rows, dtype=np.int64),
        np.asarray(negative_rows, dtype=np.int64),
    )


def _native_training_tensors(
    batch: PreparedBatch,
    context: TrainingStepContext,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    model_id = context.decoder.model_id
    model_inputs, targets = batch.as_torch()
    token_union = targets["saliency_all_labels"]
    if model_id == "taskweave_mr2hd":
        model_inputs, token_union = _taskweave_horizon(
            model_inputs, batch.records, token_union
        )
    if model_id == "sim_detr":
        positive, negative = _sim_saliency_pairs(batch.records, context)
        targets["saliency_pos_labels"] = torch.from_numpy(positive)
        targets["saliency_neg_labels"] = torch.from_numpy(negative)
        union, masks = build_target_masks(
            batch.records, maximum_video_tokens=int(model_inputs["src_vid"].shape[1])
        )
        targets["src_pos_mask"] = torch.from_numpy(union)
        targets["mask_labels"] = [torch.from_numpy(value) for value in masks]
    else:
        base_seed = (
            context.config.seed * 1_000_003
            + context.zero_based_epoch * 10_007
            + context.batch_index * 101
        )
        positive, negative = _lighthouse_saliency_pairs(
            token_union, model_inputs["src_vid_mask"], base_seed=base_seed
        )
        targets["saliency_pos_labels"] = positive
        targets["saliency_neg_labels"] = negative
    targets["saliency_all_labels"] = token_union

    model_inputs = {
        name: value.to(device=context.device) for name, value in model_inputs.items()
    }
    native_targets = {
        "span_labels": [
            {"spans": row["spans"].to(device=context.device)}
            for row in targets["span_labels"]
        ],
        "saliency_pos_labels": targets["saliency_pos_labels"].to(
            device=context.device
        ),
        "saliency_neg_labels": targets["saliency_neg_labels"].to(
            device=context.device
        ),
        "saliency_all_labels": targets["saliency_all_labels"].to(
            device=context.device
        ),
    }
    if model_id == "cg_detr":
        model_inputs["vid"] = [
            f"{record.native_history_id or record.history_id}::run_{record.run_index}"
            for record in batch.records
        ]
        model_inputs["qid"] = [
            f"{record.native_history_id or record.history_id}::query_{record.query_index}"
            for record in batch.records
        ]
        native_targets["relevant_clips"] = native_targets["saliency_all_labels"]
    elif model_id == "tr_detr":
        native_targets["src_pos_mask"] = native_targets["saliency_all_labels"]
    elif model_id == "sim_detr":
        native_targets["src_pos_mask"] = targets["src_pos_mask"].to(
            device=context.device
        )
        native_targets["mask_labels"] = [
            value.to(device=context.device) for value in targets["mask_labels"]
        ]
    return model_inputs, native_targets


def _weighted_loss(loss_dict: Mapping[str, Any], criterion: Any) -> Any:
    terms = [
        value * criterion.weight_dict[name]
        for name, value in loss_dict.items()
        if name in criterion.weight_dict
    ]
    if not terms:
        raise ValueError("Native criterion returned no weighted loss terms")
    return sum(terms)


def _lighthouse_objective(
    criterion: Any,
    model_id: str,
) -> Any:
    def objective(
        model: Any, batch: PreparedBatch, context: TrainingStepContext
    ) -> LossOutput:
        criterion.train()
        model_inputs, targets = _native_training_tensors(batch, context)
        if model_id == "cg_detr":
            outputs = model(**model_inputs, targets=targets)
        elif model_id == "uvcom":
            outputs = model(**model_inputs, epoch=context.zero_based_epoch)
        elif model_id == "taskweave_mr2hd":
            outputs, (hd_log_var, mr_log_var) = model(
                **model_inputs, epoch_i=context.zero_based_epoch
            )
        else:
            outputs = model(**model_inputs)
        loss_dict = criterion(outputs, targets)
        base = _weighted_loss(loss_dict, criterion)
        components: dict[str, Any] = {"base_loss": base}
        total = base
        if model_id == "tr_detr":
            from lighthouse.common.loss_func import CTC_Loss, VTCLoss

            ctc = CTC_Loss()(
                outputs["src_vid_ed"],
                outputs["src_txt_ed"],
                targets["src_pos_mask"],
                model_inputs["src_vid_mask"],
                model_inputs["src_txt_mask"],
            )
            vtc = VTCLoss()(outputs["src_txt_cls_ed"], outputs["src_vid_cls_ed"])
            total = base + 0.5 * ctc + 0.3 * vtc
            components.update(
                ctc_loss=ctc,
                vtc_loss=vtc,
                weighted_ctc_loss=0.5 * ctc,
                weighted_vtc_loss=0.3 * vtc,
            )
        elif model_id == "taskweave_mr2hd":
            mr_terms = [
                value
                for name, value in loss_dict.items()
                if name in criterion.weight_dict
                and any(
                    keyword in name
                    for keyword in ("giou", "span", "label", "class_error")
                )
            ]
            hd_terms = [
                value
                for name, value in loss_dict.items()
                if name in criterion.weight_dict and "saliency" in name
            ]
            if not mr_terms or not hd_terms:
                raise ValueError("TaskWeave criterion lacks an MR or HD loss group")
            loss_mr = sum(mr_terms)
            loss_hd = sum(hd_terms)
            total = (
                2 * loss_hd * (-hd_log_var).exp()
                + loss_mr * (-mr_log_var).exp()
                + hd_log_var
                + mr_log_var
            ).sum()
            components = {
                "raw_mr_loss": loss_mr,
                "raw_hd_loss": loss_hd,
                "hd_log_var": hd_log_var.mean(),
                "mr_log_var": mr_log_var.mean(),
            }
        return LossOutput(total_loss=total, components=components)

    return objective


def _sim_objective(criterion: Any) -> Any:
    def objective(
        model: Any, batch: PreparedBatch, context: TrainingStepContext
    ) -> LossOutput:
        criterion.train()
        model_inputs, targets = _native_training_tensors(batch, context)
        outputs = model(**model_inputs)
        loss_dict = criterion(outputs, targets)
        base = _weighted_loss(loss_dict, criterion)
        from sim_detr.loss_fun.CTCLoss import CTC_Loss
        from sim_detr.loss_fun.VTCLoss import VTCLoss

        ctc = CTC_Loss()(
            outputs["src_vid_ed"],
            outputs["src_txt_ed"],
            targets["src_pos_mask"],
            model_inputs["src_vid_mask"],
            model_inputs["src_txt_mask"],
        )
        vtc = VTCLoss()(outputs["src_txt_cls_ed"], outputs["src_vid_cls_ed"])
        total = base + 0.5 * ctc + 0.3 * vtc
        return LossOutput(
            total_loss=total,
            components={
                "base_loss": base,
                "ctc_loss": ctc,
                "vtc_loss": vtc,
                "weighted_ctc_loss": 0.5 * ctc,
                "weighted_vtc_loss": 0.3 * vtc,
            },
        )

    return objective


def _build_lighthouse_components(
    model_id: str,
    config: TrainingConfig,
    device: Any,
) -> DecoderTrainingComponents:
    del config
    source = verify_vendored_source("lighthouse")
    _activate_vendor(LIGHTHOUSE_ROOT, "lighthouse")
    native_config = resolve_native_config(model_id, device=str(device))
    module_name = {
        "moment_detr": "moment_detr",
        "qd_detr": "qd_detr",
        "eatr": "eatr",
        "cg_detr": "cg_detr",
        "uvcom": "uvcom",
        "tr_detr": "tr_detr",
        "taskweave_mr2hd": "taskweave",
    }[model_id]
    module = importlib.import_module(f"lighthouse.common.{module_name}")
    model, criterion = module.build_model(_AttributeDict(native_config))
    model = model.to(device)
    criterion = criterion.to(device)
    update_ema = None
    load_ema = None
    if model_id == "taskweave_mr2hd":
        from lighthouse.common.utils.model_utils import ModelEMA

        ema = ModelEMA(model, decay=0.9)

        def update_ema(current_model: Any, _epoch: int) -> None:
            ema.update(current_model)

        def load_ema(current_model: Any) -> None:
            current_model.load_state_dict(ema.module.state_dict())

    return DecoderTrainingComponents(
        model=model,
        objective=_lighthouse_objective(criterion, model_id),
        architecture_metadata=_architecture_metadata(model_id, native_config, source),
        update_ema=update_ema,
        load_ema_weights=load_ema,
    )


def _build_sim_components(
    config: TrainingConfig,
    device: Any,
    *,
    maximum_video_tokens: int,
) -> DecoderTrainingComponents:
    del config
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required to construct Sim-DETR") from exc
    if device.type != "cuda" or device.index not in (None, 0):
        raise ValueError("The pinned Sim-DETR source must run on isolated cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("Sim-DETR native training requires an available CUDA device")
    torch.cuda.set_device(0)
    source = verify_vendored_source("sim_detr")
    _activate_vendor(SIM_DETR_ROOT, "sim_detr")
    native_config = resolve_native_config(
        "sim_detr",
        device=str(device),
        maximum_video_tokens=maximum_video_tokens,
    )
    model_args = {
        name: value
        for name, value in native_config.items()
        if name
        not in {
            "learning_rate",
            "weight_decay",
            "lr_drop_epoch",
            "lr_drop_gamma",
            "gradient_clip",
            "official_epochs",
            "ctc_loss_coef",
            "vtc_loss_coef",
            "checkpoint_selection",
        }
    }
    module = importlib.import_module("sim_detr.model")
    official_model, criterion = module.build_model(SimpleNamespace(**model_args))

    class TargetLastSimDETR(torch.nn.Module):
        """Retain the original checkpoint namespace around the native model."""

        def __init__(self, wrapped: Any) -> None:
            super().__init__()
            self.official_model = wrapped

        def forward(self, **model_inputs: Any) -> Mapping[str, Any]:
            text = model_inputs["src_txt"]
            mask = model_inputs["src_txt_mask"]
            for row in range(text.shape[0]):
                length = int(mask[row].sum().item())
                roles = text[row, :length, -1]
                if length <= 0 or not (
                    bool((roles[:-1] == 0).all()) and bool((roles[-1] == 1).item())
                ):
                    raise ValueError("Sim-DETR requires one target-last text token")
            return self.official_model(**model_inputs)

    model = TargetLastSimDETR(official_model).to(device)
    criterion = criterion.to(device)
    return DecoderTrainingComponents(
        model=model,
        objective=_sim_objective(criterion),
        architecture_metadata=_architecture_metadata(
            "sim_detr", native_config, source
        ),
    )


def native_training_adapter(
    model_id: str,
    *,
    maximum_video_tokens: int | None = None,
) -> DecoderTrainingAdapter:
    """Return the verified native training adapter for one supported decoder."""

    spec = decoder_spec(model_id)
    if spec.backend == "sim_detr":
        if maximum_video_tokens is None:
            raise ValueError("Sim-DETR adapter requires the maximum development run length")

        def build(config: TrainingConfig, device: Any) -> DecoderTrainingComponents:
            return _build_sim_components(
                config, device, maximum_video_tokens=int(maximum_video_tokens)
            )

    else:

        def build(config: TrainingConfig, device: Any) -> DecoderTrainingComponents:
            return _build_lighthouse_components(model_id, config, device)

    return DecoderTrainingAdapter(model_id=model_id, build=build)


def predict_native_records(
    model: Any,
    records: Sequence[LearnedRunRecord],
    *,
    model_id: str,
    device: Any,
) -> tuple[Any, ...]:
    """Run one history-description's visible runs and decode all proposals."""

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for learned-decoder inference") from exc
    selected = tuple(records)
    if not selected:
        raise ValueError("Native inference requires at least one run record")
    identities = {
        (record.history_id, record.state_id, record.query_index) for record in selected
    }
    if len(identities) != 1:
        raise ValueError("Native inference records must represent one description")
    batch = collate_records(selected, require_positive=False)
    if model_id == "sim_detr":
        batch = replace(batch, text_tokens=reorder_sim_detr_text(batch))
    model_inputs, _targets = batch.as_torch()
    if model_id == "taskweave_mr2hd":
        model_inputs, _unused = _taskweave_horizon(model_inputs, selected, None)
    model_inputs = {
        name: value.to(device=device) for name, value in model_inputs.items()
    }
    model.eval()
    with torch.no_grad():
        if model_id == "cg_detr":
            outputs = model(**model_inputs, vid=None, qid=None, targets=None)
        elif model_id == "taskweave_mr2hd":
            outputs, _uncertainty = model(**model_inputs, epoch_i=None)
        else:
            outputs = model(**model_inputs)
    iou_scores = outputs.get("iou_scores") if model_id == "sim_detr" else None
    if model_id == "sim_detr" and iou_scores is None:
        raise ValueError("Sim-DETR inference output lacks IoU scores")
    proposals = []
    for index, record in enumerate(selected):
        proposals.extend(
            decode_proposals(
                record,
                outputs["pred_spans"][index].detach().cpu().numpy(),
                outputs["pred_logits"][index].detach().cpu().numpy(),
                sim_detr_iou_scores=(
                    iou_scores[index].detach().cpu().numpy()
                    if iou_scores is not None
                    else None
                ),
            )
        )
    return tuple(proposals)
