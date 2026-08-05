"""Strict loader for the five development histories used by all decoders.

The bundled manifest defines the state vocabulary, zero-based inclusive
development intervals, and relative paths to frozen SigLIP 2 arrays.

The locked 5-history, 13-state, 26-description, 46-episode counts are those in
the paper appendix, "Seven-History Evaluation Design and Coordinate Results."
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping

import numpy as np

from ..data import History, Query, package_root
from .protocol import (
    DEVELOPMENT_DESCRIPTION_COUNT,
    DEVELOPMENT_EPISODE_COUNT,
    DEVELOPMENT_HISTORY_COUNT,
    DEVELOPMENT_POSITIVE_RECORD_COUNT,
    DEVELOPMENT_STATE_COUNT,
    VISUAL_FEATURE_DIM,
)


DEVELOPMENT_MANIFEST_KIND = "deja_cue_learned_development"
DEVELOPMENT_HISTORY_KEYS = (
    "hand",
    "banana",
    "lemon",
    "cookie",
    "toy_container",
)
DEVELOPMENT_OBSERVED_FRAME_COUNT = 3559
_FEATURE_SPECIFICATION = {
    "encoder": "google/siglip2-base-patch16-224",
    "dimension": VISUAL_FEATURE_DIM,
    "crop": "union_lineage_mask",
    "padding_fraction": 0.2,
    "outside_mask_rgb_value": 127,
    "text_prompts": ["{text}", "a photo of {text}", "the {text}"],
    "text_aggregation": "mean_then_l2_normalize",
}


def _relative_file(root: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty relative path")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise ValueError(f"{label} must be relative")
    if ".." in posix.parts or ".." in windows.parts:
        raise ValueError(f"{label} may not contain '..'")
    candidate = (root / Path(value)).resolve(strict=True)
    try:
        candidate.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise ValueError(f"{label} escapes the package root") from exc
    if not candidate.is_file() or candidate.is_symlink():
        raise ValueError(f"{label} is not a regular package file")
    return candidate


def _manifest_relative_path(value: Any, *, label: str) -> str:
    """Return one validated manifest path in its portable POSIX form."""

    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty relative path")
    return PurePosixPath(value.replace("\\", "/")).as_posix()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_source_lock(
    archive_root: Path,
    payload: Mapping[str, Any],
    referenced_paths: set[str],
) -> None:
    """Verify every development array against the manifest's immutable lock."""

    source_lock = payload.get("source_lock")
    if not isinstance(source_lock, Mapping):
        raise ValueError("Development manifest lacks a source lock")
    for field in ("executed_snapshot_sha256", "staged_payload_index_sha256"):
        value = source_lock.get(field)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"Invalid development source-lock field: {field}")

    rows = source_lock.get("files")
    if not isinstance(rows, list):
        raise ValueError("Development source lock has no file list")
    locked_paths: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Development source-lock rows must be mappings")
        relative_path = _manifest_relative_path(
            row.get("path"), label="source-lock path"
        )
        if relative_path in locked_paths:
            raise ValueError(f"Duplicate development source-lock path: {relative_path}")
        locked_paths.add(relative_path)
        path = _relative_file(
            archive_root, relative_path, label=f"source-lock file {relative_path}"
        )
        expected_bytes = row.get("bytes")
        expected_sha256 = row.get("sha256")
        if (
            not isinstance(expected_bytes, int)
            or expected_bytes < 0
            or not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
        ):
            raise ValueError(f"Invalid development source lock: {relative_path}")
        if path.stat().st_size != expected_bytes or _sha256_file(path) != expected_sha256:
            raise ValueError(f"Development source-lock mismatch: {relative_path}")
    if locked_paths != referenced_paths:
        raise ValueError(
            "Development source lock and history feature references differ"
        )


def _visual_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        expected = {"frame_indices", "visual_features", "visibility_count"}
        if set(archive.files) != expected:
            raise ValueError(f"Development visual fields differ: {path.name}")
        frames = np.asarray(archive["frame_indices"], dtype=np.int64)
        features = np.asarray(archive["visual_features"], dtype=np.float32)
        visibility = np.asarray(archive["visibility_count"], dtype=np.int64)
    if (
        frames.ndim != 1
        or not len(frames)
        or features.shape != (len(frames), VISUAL_FEATURE_DIM)
        or visibility.shape != frames.shape
        or np.any(np.diff(frames) <= 0)
        or np.any(visibility <= 0)
        or not np.isfinite(features).all()
        or not np.allclose(np.linalg.norm(features, axis=1), 1.0, atol=2e-3)
    ):
        raise ValueError(f"Development visual arrays are invalid: {path.name}")
    return frames, features, visibility


def _text_queries(path: Path) -> tuple[Query, ...]:
    with np.load(path, allow_pickle=False) as archive:
        expected = {"state_ids", "state_texts", "text_features"}
        if set(archive.files) != expected:
            raise ValueError(f"Development text fields differ: {path.name}")
        state_ids = np.asarray(archive["state_ids"], dtype=str)
        texts = np.asarray(archive["state_texts"], dtype=str)
        features = np.asarray(archive["text_features"], dtype=np.float32)
    if (
        state_ids.ndim != 1
        or texts.shape != state_ids.shape
        or features.shape != (len(state_ids), VISUAL_FEATURE_DIM)
        or not len(state_ids)
        or not np.isfinite(features).all()
        or not np.allclose(np.linalg.norm(features, axis=1), 1.0, atol=2e-3)
    ):
        raise ValueError(f"Development text arrays are invalid: {path.name}")
    return tuple(
        Query(str(state_id), str(text), feature.copy())
        for state_id, text, feature in zip(state_ids, texts, features)
    )


def _states(
    row: Mapping[str, Any], queries: tuple[Query, ...]
) -> dict[str, tuple[tuple[int, int], ...]]:
    values = row.get("states")
    if not isinstance(values, list) or not values:
        raise ValueError(f"{row.get('history_id')} has no development states")
    references: dict[str, tuple[tuple[int, int], ...]] = {}
    expected_queries: list[tuple[str, str]] = []
    for state in values:
        if not isinstance(state, Mapping):
            raise ValueError("Development state rows must be mappings")
        state_id = str(state.get("state_id", ""))
        descriptions = state.get("descriptions")
        raw_intervals = state.get("references")
        if (
            not state_id
            or state_id in references
            or not isinstance(descriptions, list)
            or len(descriptions) != 2
            or not all(isinstance(value, str) and value for value in descriptions)
            or not isinstance(raw_intervals, list)
            or not raw_intervals
        ):
            raise ValueError(f"Invalid development state: {state_id!r}")
        intervals = tuple((int(value[0]), int(value[1])) for value in raw_intervals)
        if any(start < 0 or end < start for start, end in intervals):
            raise ValueError(f"Invalid inclusive interval in {state_id}")
        references[state_id] = intervals
        expected_queries.extend((state_id, str(text)) for text in descriptions)
    observed_queries = [(query.state_id, query.text) for query in queries]
    if observed_queries != expected_queries:
        raise ValueError(f"Text feature order differs for {row.get('history_id')}")
    return references


def load_development_histories(
    root: Path | None = None,
    *,
    manifest: Path | None = None,
) -> tuple[History, ...]:
    """Load and validate the 5/13/26/46 development data."""

    archive_root = package_root() if root is None else Path(root).resolve(strict=True)
    manifest_path = (
        archive_root / "data" / "learned" / "development" / "manifest.json"
        if manifest is None
        else Path(manifest).resolve(strict=True)
    )
    try:
        manifest_path.relative_to(archive_root)
    except ValueError as exc:
        raise ValueError("Development manifest must remain inside the package") from exc
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = payload.get("histories")
    if (
        payload.get("schema_version") != 1
        or payload.get("kind") != DEVELOPMENT_MANIFEST_KIND
        or payload.get("task") != "identity_conditioned_state_moment_retrieval"
        or payload.get("role") != "development_only"
        or payload.get("interval_convention") != "zero_based_inclusive"
        or payload.get("feature_specification") != _FEATURE_SPECIFICATION
        or not isinstance(rows, list)
    ):
        raise ValueError("Unexpected learned-development manifest")

    histories: list[History] = []
    referenced_paths: set[str] = set()
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise ValueError("Development history rows must be mappings")
        history_id = str(row.get("history_id", ""))
        if history_id != f"D{index:02d}":
            raise ValueError("Development history IDs are incomplete or out of order")
        key = str(row.get("label", ""))
        source_component_id = str(row.get("native_history_id", ""))
        if not source_component_id:
            raise ValueError(f"{key} lacks a stable source-component identity")
        visual_reference = _manifest_relative_path(
            row.get("visual_features"), label=f"{key} visual features"
        )
        text_reference = _manifest_relative_path(
            row.get("text_features"), label=f"{key} text features"
        )
        referenced_paths.update((visual_reference, text_reference))
        visual_path = _relative_file(
            archive_root, visual_reference, label=f"{key} visual features"
        )
        text_path = _relative_file(
            archive_root, text_reference, label=f"{key} text features"
        )
        frames, features, visibility = _visual_arrays(visual_path)
        queries = _text_queries(text_path)
        references = _states(row, queries)
        histories.append(
            History(
                history_id=history_id,
                sequence_id=key,
                source_component_id=source_component_id,
                frame_indices=frames,
                visual_features=features,
                visibility_count=visibility,
                queries=queries,
                references=references,
            )
        )

    _verify_source_lock(archive_root, payload, referenced_paths)

    counts = {
        "histories": len(histories),
        "states": sum(len(history.references) for history in histories),
        "descriptions": sum(len(history.queries) for history in histories),
        "episodes": sum(
            len(intervals)
            for history in histories
            for intervals in history.references.values()
        ),
        "observed_frames": sum(len(history.frame_indices) for history in histories),
    }
    expected = {
        "histories": DEVELOPMENT_HISTORY_COUNT,
        "states": DEVELOPMENT_STATE_COUNT,
        "descriptions": DEVELOPMENT_DESCRIPTION_COUNT,
        "episodes": DEVELOPMENT_EPISODE_COUNT,
        "observed_frames": DEVELOPMENT_OBSERVED_FRAME_COUNT,
    }
    if counts != expected:
        raise ValueError(
            f"Development counts differ: expected={expected}, observed={counts}"
        )
    expected_manifest_counts = {
        "histories": DEVELOPMENT_HISTORY_COUNT,
        "states": DEVELOPMENT_STATE_COUNT,
        "descriptions": DEVELOPMENT_DESCRIPTION_COUNT,
        "positive_episodes": DEVELOPMENT_EPISODE_COUNT,
        "observed_frames": DEVELOPMENT_OBSERVED_FRAME_COUNT,
        "positive_training_records": DEVELOPMENT_POSITIVE_RECORD_COUNT,
    }
    if payload.get("counts") != expected_manifest_counts:
        raise ValueError("Development manifest counts differ from the reported data")
    if tuple(history.sequence_id for history in histories) != DEVELOPMENT_HISTORY_KEYS:
        raise ValueError("Development histories differ from the paper roster")
    if len({history.source_component_id for history in histories}) != len(histories):
        raise ValueError("Development source-component identities must be unique")
    return tuple(histories)
