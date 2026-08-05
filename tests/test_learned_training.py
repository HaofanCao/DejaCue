"""Execution tests for the common PyTorch learned-decoder trainer."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from deja_cue.learned.protocol import make_training_config
from deja_cue.learned.records import LearnedRunRecord
from deja_cue.learned.training import (
    DecoderTrainingAdapter,
    DecoderTrainingComponents,
    LossOutput,
    prepare_training_batch,
    train_decoder,
    validate_training_records,
)


def _unit_rows(seed: int, count: int) -> np.ndarray:
    rows = np.random.default_rng(seed).normal(size=(count, 768)).astype(np.float32)
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def _training_records() -> tuple[LearnedRunRecord, ...]:
    records: list[LearnedRunRecord] = []
    state_counts = (3, 3, 3, 2, 2)
    record_counts = (18, 18, 18, 16, 16)
    global_index = 0
    for history_index, (state_count, record_count) in enumerate(
        zip(state_counts, record_counts)
    ):
        history_id = f"development_{history_index + 1:02d}"
        query_count = state_count * 2
        embeddings = _unit_rows(100 + history_index, query_count)
        for local_index in range(record_count):
            query_index = local_index % query_count
            run_index = local_index // query_count
            context_indices = (
                query_index,
                *(index for index in range(query_count) if index != query_index),
            )
            frames = np.arange(
                global_index * 4, global_index * 4 + 3, dtype=np.int64
            )
            records.append(
                LearnedRunRecord(
                    history_id=history_id,
                    state_id=f"state_{query_index // 2}",
                    text=f"description_{query_index}",
                    query_index=query_index,
                    run_index=run_index,
                    absolute_frames=frames,
                    visual_features=np.tile(
                        embeddings[query_index], (len(frames), 1)
                    ).astype(np.float32),
                    text_embedding=embeddings[query_index].copy(),
                    text_context_embeddings=embeddings[
                        np.asarray(context_indices, dtype=np.int64)
                    ].copy(),
                    text_context_query_indices=tuple(context_indices),
                    target_token_spans=((0, 2),),
                    normalized_spans=np.asarray(
                        [[1.0 / 3.0, 2.0 / 3.0]], dtype=np.float32
                    ),
                )
            )
            global_index += 1
    return tuple(records)


class _ToyDecoder(torch.nn.Module):
    """Small differentiable stand-in that exercises the shared trainer."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(770, 1)

    def forward(self, video_tokens: torch.Tensor) -> torch.Tensor:
        """Return one scalar score per padded video token."""

        return self.projection(video_tokens).squeeze(-1)


def _build_toy_components(config, device) -> DecoderTrainingComponents:
    del config, device

    def objective(model, batch, context) -> LossOutput:
        model_inputs, targets = batch.as_torch()
        video = model_inputs["src_vid"].to(context.device)
        mask = model_inputs["src_vid_mask"].to(context.device)
        labels = targets["saliency_all_labels"].to(context.device)
        residual = (model(video) - labels) * mask
        loss = residual.square().sum() / mask.sum()
        return LossOutput(total_loss=loss, components={"saliency_fit": loss})

    return DecoderTrainingComponents(model=_ToyDecoder(), objective=objective)


def test_record_counts_and_sim_detr_target_order_are_fixed() -> None:
    records = _training_records()
    assert validate_training_records(records, model_id="moment_detr") == {
        "records": 86,
        "histories": 5,
        "states": 13,
        "descriptions": 26,
    }
    config = make_training_config("sim_detr", 3407)
    first = prepare_training_batch(
        records[:5], config=config, zero_based_epoch=2, batch_index=1
    )
    second = prepare_training_batch(
        records[:5], config=config, zero_based_epoch=2, batch_index=1
    )
    np.testing.assert_array_equal(first.saliency_positive, second.saliency_positive)
    np.testing.assert_array_equal(first.saliency_negative, second.saliency_negative)
    assert np.all(first.text_tokens[:, -1, -1] == 1.0)
    assert np.all(first.text_tokens[:, :-1, -1] == 0.0)


def test_common_trainer_executes_all_200_epochs_and_updates_parameters() -> None:
    records = _training_records()
    config = make_training_config("moment_detr", 3407)
    result = train_decoder(
        records,
        DecoderTrainingAdapter(
            model_id="moment_detr", build=_build_toy_components
        ),
        config,
        device="cpu",
    )
    summary = result.summary
    assert summary.epochs == 200
    assert len(summary.epoch_summaries) == 200
    assert summary.optimizer_steps == sum(
        epoch.batches for epoch in summary.epoch_summaries
    )
    assert summary.initial_parameter_sha256 != summary.final_parameter_sha256
    assert summary.epoch_summaries[0].learning_rate == pytest.approx(1e-4)
    assert summary.epoch_summaries[-1].epoch == 200
    assert set(summary.epoch_summaries[-1].mean_components) == {"saliency_fit"}
