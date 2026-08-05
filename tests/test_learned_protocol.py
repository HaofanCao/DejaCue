"""Focused tests for the low-data learned-decoder protocol."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from deja_cue.learned.batching import (
    build_epoch_batches,
    cross_history_cycle_batches,
    summarize_batches,
    validate_cross_history_cycle,
)
from deja_cue.learned.checkpoint import build_checkpoint, validate_checkpoint
from deja_cue.learned.development import load_development_histories
from deja_cue.learned.protocol import (
    CUBLAS_WORKSPACE_CONFIG,
    DECODER_REGISTRY,
    DEVELOPMENT_POSITIVE_RECORD_COUNT,
    TRAINING_SEEDS,
    TrainingConfig,
    configure_deterministic_cuda_workspace,
    make_training_config,
    protocol_summary,
)
from deja_cue.learned.records import (
    LearnedRunRecord,
    build_positive_run_records,
    build_target_masks,
    collate_records,
    decode_proposals,
    reorder_sim_detr_text,
    state_balanced_centroid,
    state_balanced_coordinates,
)


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class _Query:
    state_id: str
    text: str
    embedding: np.ndarray


@dataclass(frozen=True)
class _History:
    history_id: str
    frame_indices: np.ndarray
    visual_features: np.ndarray
    queries: tuple[_Query, ...]
    references: dict[str, tuple[tuple[int, int], ...]]


def _unit_rows(seed: int, count: int) -> np.ndarray:
    rows = np.random.default_rng(seed).normal(size=(count, 768)).astype(np.float32)
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def _record(history_id: str, index: int) -> LearnedRunRecord:
    text = _unit_rows(1000 + index, 1)[0]
    frames = np.arange(index * 3, index * 3 + 3, dtype=np.int64)
    return LearnedRunRecord(
        history_id=history_id,
        state_id=f"state_{index}",
        text=f"description_{index}",
        query_index=index,
        run_index=0,
        absolute_frames=frames,
        visual_features=np.tile(text, (3, 1)).astype(np.float32),
        text_embedding=text.copy(),
        text_context_embeddings=text[None, :].copy(),
        text_context_query_indices=(index,),
        target_token_spans=((0, 2),),
        normalized_spans=np.asarray([[1.0 / 3.0, 2.0 / 3.0]], dtype=np.float32),
    )


def _batch_roster() -> tuple[LearnedRunRecord, ...]:
    histories = ("history_a",) * 4 + ("history_b",) * 3 + ("history_c",) * 3
    return tuple(_record(history, index) for index, history in enumerate(histories))


def test_registry_and_frozen_training_config_match_reported_protocol() -> None:
    assert tuple(DECODER_REGISTRY) == (
        "moment_detr",
        "qd_detr",
        "eatr",
        "cg_detr",
        "uvcom",
        "tr_detr",
        "taskweave_mr2hd",
        "sim_detr",
    )
    assert TRAINING_SEEDS == (3407, 3408, 3409)
    assert DEVELOPMENT_POSITIVE_RECORD_COUNT == 86
    assert DECODER_REGISTRY["taskweave_mr2hd"].ema_decay == 0.9
    assert DECODER_REGISTRY["sim_detr"].native_target_position == "last"
    assert DECODER_REGISTRY["sim_detr"].lr_drop_epoch == 100
    assert DECODER_REGISTRY["tr_detr"].ctc_loss_coefficient == 0.5

    for model_id in DECODER_REGISTRY:
        for seed in TRAINING_SEEDS:
            config = make_training_config(model_id, seed)
            assert config.epochs == 200
            assert config.batch_size == 20
            assert config.learning_rate == pytest.approx(1e-4)
            assert config.weight_decay == pytest.approx(1e-4)
            assert config.gradient_clip == pytest.approx(0.1)
            assert config.text_coordinates == "query_centered"
            assert config.text_context_mode == "target_first_siblings"
            assert config.training_run_policy == "positive_only"
            assert config.checkpoint_selection == "final_checkpoint_only"

    summary = protocol_summary()
    assert summary["development"] == {
        "histories": 5,
        "states": 13,
        "descriptions": 26,
        "episodes": 46,
        "positive_records": 86,
    }
    assert summary["evaluation_histories"] == 7


def test_deterministic_cuda_workspace_has_one_fixed_setting(monkeypatch) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    assert configure_deterministic_cuda_workspace() == ":4096:8"
    assert CUBLAS_WORKSPACE_CONFIG == ":4096:8"

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    with pytest.raises(RuntimeError, match="must be"):
        configure_deterministic_cuda_workspace()


def test_training_config_rejects_unsupported_selection_or_seed() -> None:
    with pytest.raises(ValueError, match="Unexpected training seed"):
        make_training_config("moment_detr", 7)
    with pytest.raises(ValueError, match="differs from the protocol"):
        TrainingConfig(model_id="moment_detr", seed=3407, epochs=199)
    with pytest.raises(ValueError, match="Unsupported decoder"):
        make_training_config("unreported_decoder", 3407)


def test_state_balanced_coordinates_do_not_overweight_extra_descriptions() -> None:
    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    states = ("state_a", "state_a", "state_b")
    centroid = state_balanced_centroid(embeddings, states)
    assert centroid == pytest.approx([0.5, 0.5, 0.0])
    transformed = state_balanced_coordinates(embeddings, states)
    assert np.linalg.norm(transformed, axis=1) == pytest.approx([1.0, 1.0, 1.0])
    assert transformed[0] == pytest.approx(transformed[1])
    assert transformed[0] == pytest.approx(-transformed[2])


def test_record_construction_collation_and_target_last_reordering() -> None:
    text = _unit_rows(5, 4)
    visual = _unit_rows(9, 6)
    history = _History(
        history_id="development_01",
        frame_indices=np.asarray([10, 11, 12, 20, 21, 22], dtype=np.int64),
        visual_features=visual,
        queries=(
            _Query("state_a", "a one", text[0]),
            _Query("state_a", "a two", text[1]),
            _Query("state_b", "b one", text[2]),
            _Query("state_b", "b two", text[3]),
        ),
        references={"state_a": ((10, 11),), "state_b": ((20, 22),)},
    )
    records = build_positive_run_records((history,))
    assert len(records) == 4
    assert [record.run_index for record in records] == [0, 0, 1, 1]
    assert records[2].text_context_query_indices == (2, 0, 1, 3)
    assert records[0].target_token_spans == ((0, 2),)
    np.testing.assert_allclose(
        records[0].normalized_spans,
        np.asarray([[1.0 / 3.0, 2.0 / 3.0]], dtype=np.float32),
    )

    batch = collate_records((records[0], records[2]))
    assert batch.video_tokens.shape == (2, 3, 770)
    assert batch.text_tokens.shape == (2, 4, 769)
    np.testing.assert_allclose(
        batch.video_tokens[0, :, -2:],
        np.asarray(
            [[0.0, 1.0 / 3.0], [1.0 / 3.0, 2.0 / 3.0], [2.0 / 3.0, 1.0]],
            dtype=np.float32,
        ),
    )
    assert batch.text_tokens[:, 0, -1] == pytest.approx([1.0, 1.0])
    assert np.count_nonzero(batch.text_tokens[:, :, -1]) == 2
    assert batch.saliency_labels[0] == pytest.approx([1.0, 1.0, 0.0])

    reordered = reorder_sim_detr_text(batch)
    assert reordered[:, -1, -1] == pytest.approx([1.0, 1.0])
    assert np.all(reordered[:, :-1, -1] == 0.0)
    np.testing.assert_allclose(
        np.sort(reordered[0, :, :-1], axis=0),
        np.sort(batch.text_tokens[0, :, :-1], axis=0),
    )

    union, masks = build_target_masks((records[0], records[2]))
    np.testing.assert_allclose(
        union,
        np.asarray([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32),
    )
    assert masks[0].shape == (1, 3)
    np.testing.assert_allclose(
        masks[1], np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32)
    )


def test_proposal_decoding_uses_inclusive_absolute_frames_and_native_scores() -> None:
    record = _record("history", 0)
    spans = np.asarray([[0.5, 1.0], [0.5, 1.0 / 3.0]], dtype=np.float32)
    logits = np.asarray([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    proposals = decode_proposals(record, spans, logits)
    assert (proposals[0].start, proposals[0].end) == (0, 2)
    assert (proposals[1].start, proposals[1].end) == (1, 1)
    assert proposals[0].score > proposals[1].score

    calibrated = decode_proposals(
        record,
        spans,
        logits,
        sim_detr_iou_scores=np.asarray([[-10.0], [10.0]], dtype=np.float32),
    )
    assert calibrated[0].score < calibrated[1].score


def test_all_decoder_batch_families_are_deterministic_and_cover_records() -> None:
    records = _batch_roster()
    for model_id, spec in DECODER_REGISTRY.items():
        first = build_epoch_batches(
            records,
            model_id=model_id,
            training_seed=3407,
            zero_based_epoch=0,
            batch_size=4,
        )
        second = build_epoch_batches(
            records,
            model_id=model_id,
            training_seed=3407,
            zero_based_epoch=0,
            batch_size=4,
        )
        assert first == second
        batch_summary = summarize_batches(records, first)
        assert batch_summary["all_records_covered"] is True
        assert batch_summary["all_exposures_positive"] is True
        assert batch_summary["all_batches_have_two_records"] is True
        if spec.batching_family == "cross_history_cycle":
            for batch in first:
                validate_cross_history_cycle(tuple(records[index] for index in batch))
        if spec.batching_family in {"history_unique", "unique_history_vtc"}:
            assert all(
                len({records[index].history_id for index in batch}) == len(batch)
                for batch in first
            )


def test_bundled_development_batch_schedules_match_documented_protocol() -> None:
    records = build_positive_run_records(load_development_histories(ROOT))
    expected = {
        "moment_detr": (5, {17, 18}, 86, 0),
        "qd_detr": (5, {18, 20}, 92, 6),
        "eatr": (5, {17, 18}, 86, 0),
        "cg_detr": (5, {18, 20}, 92, 6),
        "uvcom": (5, {18, 20}, 92, 6),
        "tr_detr": (46, {2}, 92, 6),
        "taskweave_mr2hd": (46, {2}, 92, 6),
        "sim_detr": (46, {2}, 92, 6),
    }
    for model_id, protocol in expected.items():
        batches = build_epoch_batches(
            records,
            model_id=model_id,
            training_seed=3407,
            zero_based_epoch=0,
            batch_size=20,
        )
        batch_summary = summarize_batches(records, batches)
        observed = (
            batch_summary["num_batches"],
            set(batch_summary["batch_sizes"]),
            batch_summary["num_exposures"],
            batch_summary["num_duplicate_exposures"],
        )
        assert observed == protocol


def test_cyclic_batching_adds_only_the_mathematically_required_support() -> None:
    histories = ("dominant",) * 6 + ("other_a",) * 2 + ("other_b",) * 2
    records = tuple(_record(history, index) for index, history in enumerate(histories))
    batches = cross_history_cycle_batches(records, batch_size=4, seed=3407)
    batch_summary = summarize_batches(records, batches)
    # A cycle cannot place adjacent records from the dominant history.  Six
    # dominant versus four other records therefore requires exactly two support
    # exposures from non-dominant positive records.
    assert batch_summary["num_duplicate_exposures"] == 2
    for batch in batches:
        validate_cross_history_cycle(tuple(records[index] for index in batch))


def test_checkpoint_schema_binds_config_and_parameter_content() -> None:
    config = make_training_config("sim_detr", 3408)
    state = {
        "head.bias": np.asarray([0.0, 1.0], dtype=np.float32),
        "head.weight": np.arange(6, dtype=np.float32).reshape(2, 3),
    }
    payload = build_checkpoint(state, config, architecture={"hidden_dimension": 256})
    validated = validate_checkpoint(
        payload, expected_model_id="sim_detr", expected_seed=3408
    )
    assert validated["parameter_sha256"] == payload["parameter_sha256"]
    assert payload["metadata"]["evaluation_histories_used_for_training"] == 0
    assert payload["metadata"]["native_target_position"] == "last"

    tampered = dict(payload)
    tampered["state_dict"] = dict(state)
    tampered["state_dict"]["head.bias"] = np.asarray([0.0, 2.0], dtype=np.float32)
    with pytest.raises(ValueError, match="parameter digest"):
        validate_checkpoint(tampered)
    with pytest.raises(ValueError, match="unsupported fields"):
        build_checkpoint(state, config, architecture={"unexpected_field": "value"})

    task_config = make_training_config("taskweave_mr2hd", 3407)
    assert task_config.inference_weight_source == "final_ema"
    with pytest.raises(ValueError, match="differs from the protocol"):
        replace(task_config, inference_weight_source="final_model")
