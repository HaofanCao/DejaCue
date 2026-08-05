"""Validated loaders for the bundled benchmark and frozen feature arrays.

Only documented relative paths are accepted. NPZ archives are always opened
with pickle disabled, and every numerical assumption used by the evaluator is
validated before a result can be produced.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class Query:
    """One closed-vocabulary state description and its frozen text feature."""

    state_id: str
    text: str
    embedding: np.ndarray


@dataclass(frozen=True)
class History:
    """All immutable inputs needed to score one tracked object history."""

    history_id: str
    sequence_id: str
    source_component_id: str
    frame_indices: np.ndarray
    visual_features: np.ndarray
    visibility_count: np.ndarray
    queries: tuple[Query, ...]
    references: Mapping[str, tuple[tuple[int, int], ...]]

    def dense(self) -> tuple[int, np.ndarray, np.ndarray]:
        """Return a first-to-last-frame tensor and its visibility mask.

        Zero rows represent missing observations only. The scan receives the
        mask separately and therefore never treats a zero row as evidence.
        """

        first = int(self.frame_indices[0])
        last = int(self.frame_indices[-1])
        features = np.zeros(
            (last - first + 1, self.visual_features.shape[1]), dtype=np.float32
        )
        valid = np.zeros(last - first + 1, dtype=bool)
        offsets = self.frame_indices.astype(np.int64) - first
        features[offsets] = self.visual_features
        valid[offsets] = True
        return first, features, valid

    def with_queries(self, queries: tuple[Query, ...]) -> "History":
        """Return the same history with a hash-aligned text feature variant."""

        if [(q.state_id, q.text) for q in queries] != [
            (q.state_id, q.text) for q in self.queries
        ]:
            raise ValueError(f"Query variant differs from {self.history_id}")
        return replace(self, queries=queries)


def package_root() -> Path:
    """Resolve the archive root without depending on the caller's directory."""

    return Path(__file__).resolve().parents[1]


def load_protocol(root: Path | None = None) -> dict[str, Any]:
    """Load the fixed evaluation schedule, seeds, and statistical constants."""

    path = (
        (package_root() if root is None else Path(root)) / "configs" / "protocol.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") != "deja_cue_fixed_protocol":
        raise ValueError("Unexpected protocol kind")
    return payload


def _relative_file(root: Path, value: object, *, label: str) -> Path:
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
    if not candidate.is_file():
        raise ValueError(f"{label} is not a regular file")
    return candidate


def _load_visual(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"frame_indices", "visual_features", "visibility_count"}
        if not required.issubset(archive.files):
            raise ValueError(f"Visual archive lacks required arrays: {path.name}")
        frames = np.asarray(archive["frame_indices"], dtype=np.int64)
        features = np.asarray(archive["visual_features"], dtype=np.float32)
        visibility = np.asarray(archive["visibility_count"], dtype=np.int64)
    if (
        frames.ndim != 1
        or features.ndim != 2
        or visibility.ndim != 1
        or not len(frames)
        or len(frames) != len(features)
        or len(frames) != len(visibility)
    ):
        raise ValueError(f"Visual array shapes are inconsistent: {path.name}")
    if np.any(np.diff(frames) <= 0) or np.any(visibility <= 0):
        raise ValueError(f"Frames or visibility are invalid: {path.name}")
    if not np.isfinite(features).all():
        raise ValueError(f"Visual features are non-finite: {path.name}")
    if not np.allclose(np.linalg.norm(features, axis=1), 1.0, atol=2e-3):
        raise ValueError(f"Visual features are not unit normalized: {path.name}")
    return frames, features, visibility


def _load_text(path: Path, *, variant: str | None = None) -> tuple[Query, ...]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"state_ids", "state_texts", "text_features"}
        if not required.issubset(archive.files):
            raise ValueError(f"Text archive lacks required arrays: {path.name}")
        state_ids = np.asarray(archive["state_ids"], dtype=str)
        texts = np.asarray(archive["state_texts"], dtype=str)
        features = np.asarray(archive["text_features"], dtype=np.float32)
        if variant is not None:
            if "variant_names" not in archive.files or features.ndim != 3:
                raise ValueError(f"Text archive has no prompt variants: {path.name}")
            names = [str(value) for value in archive["variant_names"].tolist()]
            if variant not in names:
                raise ValueError(f"Unknown prompt variant {variant!r}")
            features = features[names.index(variant)]
    if (
        state_ids.ndim != 1
        or texts.ndim != 1
        or features.ndim != 2
        or not len(state_ids)
        or len(state_ids) != len(texts)
        or len(state_ids) != len(features)
    ):
        raise ValueError(f"Text array shapes are inconsistent: {path.name}")
    if not np.isfinite(features).all():
        raise ValueError(f"Text features are non-finite: {path.name}")
    if not np.allclose(np.linalg.norm(features, axis=1), 1.0, atol=2e-3):
        raise ValueError(f"Text features are not unit normalized: {path.name}")
    return tuple(
        Query(str(state_id), str(text), feature.copy())
        for state_id, text, feature in zip(state_ids, texts, features)
    )


def _reference_map(row: Mapping[str, Any]) -> dict[str, tuple[tuple[int, int], ...]]:
    states = row.get("states")
    if not isinstance(states, list) or len(states) != 2:
        raise ValueError(f"{row.get('history_id')} must define exactly two states")
    output: dict[str, tuple[tuple[int, int], ...]] = {}
    for state in states:
        if not isinstance(state, Mapping):
            raise ValueError("State rows must be objects")
        state_id = str(state.get("state_id", ""))
        intervals = state.get("references")
        if not state_id or not isinstance(intervals, list) or not intervals:
            raise ValueError(f"Invalid state reference in {row.get('history_id')}")
        parsed = tuple((int(value[0]), int(value[1])) for value in intervals)
        if any(start < 0 or end < start for start, end in parsed):
            raise ValueError(f"Invalid inclusive interval in {row.get('history_id')}")
        output[state_id] = parsed
    return output


def load_histories(
    root: Path | None = None,
    *,
    prompt_variant: str | None = None,
) -> tuple[History, ...]:
    """Load the fixed primary roster with the selected text-prompt variant."""

    archive_root = package_root() if root is None else Path(root).resolve(strict=True)
    manifest_path = archive_root / "data" / "benchmark.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = payload.get("histories")
    if payload.get("kind") != "deja_cue_compact_benchmark" or not isinstance(
        rows, list
    ):
        raise ValueError("Unexpected benchmark manifest")

    histories: list[History] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("History rows must be objects")
        history_id = str(row.get("history_id", ""))
        visual_path = _relative_file(
            archive_root, row.get("siglip2_visual"), label=f"{history_id} visual"
        )
        frames, visual, visibility = _load_visual(visual_path)
        if prompt_variant is None or prompt_variant == "ensemble":
            text_path = _relative_file(
                archive_root,
                row.get("siglip2_text"),
                label=f"{history_id} text",
            )
            # The ensemble is the primary normalized text feature. It is not
            # duplicated in every prompt archive.
            queries = _load_text(text_path)
        else:
            prompt_path = _relative_file(
                archive_root,
                row.get("siglip2_prompt_text"),
                label=f"{history_id} prompt text",
            )
            queries = _load_text(prompt_path, variant=prompt_variant)

        references = _reference_map(row)
        if set(references) != {query.state_id for query in queries}:
            raise ValueError(f"Feature states differ from references for {history_id}")
        histories.append(
            History(
                history_id=history_id,
                sequence_id=str(row.get("sequence_id", "")),
                source_component_id=str(row.get("source_component_id", "")),
                frame_indices=frames,
                visual_features=visual,
                visibility_count=visibility,
                queries=queries,
                references=references,
            )
        )

    expected_ids = [f"H{index:03d}" for index in range(1, 79)]
    if [history.history_id for history in histories] != expected_ids:
        raise ValueError("Benchmark history roster is incomplete or out of order")
    if len({history.source_component_id for history in histories}) != len(histories):
        raise ValueError("Primary source components must be singleton")
    return tuple(histories)
