"""Load the seven-history SigLIP 2 evaluation cohort.

The bundled manifest contains the positive state vocabulary, inclusive
reference intervals, and relative paths to the frozen arrays used by the
evaluation and hard-negative diagnostics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .data import (
    History,
    Query,
    _load_visual,
    _relative_file,
    package_root,
)


@dataclass(frozen=True)
class SevenHistoryDistractor:
    """One auxiliary track retained for identity and margin diagnostics."""

    track_id: int
    object_label: str
    frame_indices: np.ndarray
    visual_features: np.ndarray
    visibility_count: np.ndarray
    co_visible_moments: Mapping[str, tuple[tuple[int, int], ...]]


@dataclass(frozen=True)
class SevenHistoryRecord:
    """A scored history plus its optional auxiliary tracks."""

    history: History
    stratum: str
    dataset: str
    scene: str
    object_group_id: str
    distractors: tuple[SevenHistoryDistractor, ...]


def prediction_window_map(
    condition: Mapping[str, Any],
    history_aliases: Mapping[str, str],
) -> dict[tuple[str, str, str], tuple[int, int]]:
    """Index predicted windows with the scene identifiers in reference results."""

    return {
        (
            history_aliases[str(history["history_id"])],
            str(query["state_id"]),
            str(query["text"]),
        ): tuple(int(value) for value in query["window"])
        for history in condition["histories"]
        for query in history["queries"]
    }


def _read_manifest(root: Path) -> dict[str, Any]:
    path = root / "data" / "seven_history" / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") != "deja_cue_seven_history_feature_manifest":
        raise ValueError("Unexpected seven-history manifest kind")
    rows = payload.get("histories")
    if not isinstance(rows, list) or len(rows) != 7:
        raise ValueError("The seven-history manifest must contain seven rows")
    if payload.get("feature_dimension") != 768:
        raise ValueError("The seven-history feature dimension must be 768")
    return payload


def _load_text_queries(
    path: Path,
    *,
    state_field: str,
    state_rows: list[Mapping[str, Any]],
) -> tuple[Query, ...]:
    with np.load(path, allow_pickle=False) as archive:
        required = {state_field, "state_texts", "text_features"}
        if not required.issubset(archive.files):
            raise ValueError(f"Text archive lacks {sorted(required)}: {path.name}")
        feature_state_ids = np.asarray(archive[state_field], dtype=str)
        texts = np.asarray(archive["state_texts"], dtype=str)
        features = np.asarray(archive["text_features"], dtype=np.float32)
    if (
        feature_state_ids.ndim != 1
        or texts.ndim != 1
        or features.ndim != 2
        or len(feature_state_ids) != len(texts)
        or len(texts) != len(features)
        or not len(texts)
    ):
        raise ValueError(f"Text archive arrays are inconsistent: {path.name}")
    if not np.isfinite(features).all():
        raise ValueError(f"Text archive contains non-finite values: {path.name}")
    if not np.allclose(np.linalg.norm(features, axis=1), 1.0, atol=2e-3):
        raise ValueError(f"Text archive rows are not unit normalized: {path.name}")

    indexed = {
        (str(state_id), str(text)): feature
        for state_id, text, feature in zip(
            feature_state_ids.tolist(), texts.tolist(), features
        )
    }
    queries: list[Query] = []
    for state in state_rows:
        state_id = str(state["state_id"])
        feature_state_id = str(state["feature_state_id"])
        for description in state["descriptions"]:
            key = (feature_state_id, str(description))
            if key not in indexed:
                raise ValueError(f"Missing seven-history query {state_id}: {description}")
            queries.append(
                Query(state_id, str(description), np.asarray(indexed[key]).copy())
            )
    if len(queries) != sum(len(state["descriptions"]) for state in state_rows):
        raise ValueError("Seven-history query roster is incomplete")
    return tuple(queries)


def _load_distractors(
    root: Path, row: Mapping[str, Any]
) -> tuple[SevenHistoryDistractor, ...]:
    output: list[SevenHistoryDistractor] = []
    for item in row.get("distractors", []):
        path = _relative_file(
            root,
            item.get("visual_features"),
            label=f"{row['history_id']} distractor feature",
        )
        frames, features, visibility = _load_visual(path)
        moments: dict[str, list[tuple[int, int]]] = {}
        for moment in item.get("co_visible_moments", []):
            moments.setdefault(str(moment["state_id"]), []).append(
                (int(moment["start"]), int(moment["end"]))
            )
        output.append(
            SevenHistoryDistractor(
                track_id=int(item["track_id"]),
                object_label=str(item["object_label"]),
                frame_indices=frames,
                visual_features=features,
                visibility_count=visibility,
                co_visible_moments={
                    key: tuple(value) for key, value in moments.items()
                },
            )
        )
    return tuple(output)


def load_seven_history_records(
    root: Path | None = None,
) -> tuple[SevenHistoryRecord, ...]:
    """Load and validate all seven histories and their frozen feature arrays."""

    archive_root = package_root() if root is None else Path(root).resolve(strict=True)
    payload = _read_manifest(archive_root)
    records: list[SevenHistoryRecord] = []
    for row in payload["histories"]:
        history_id = str(row["history_id"])
        visual_path = _relative_file(
            archive_root,
            row.get("target_visual_features"),
            label=f"{history_id} target features",
        )
        text_path = _relative_file(
            archive_root,
            row.get("text_features"),
            label=f"{history_id} text features",
        )
        frames, visual, visibility = _load_visual(visual_path)
        states = row.get("states")
        if not isinstance(states, list) or not states:
            raise ValueError(f"{history_id} has no positive states")
        references: dict[str, tuple[tuple[int, int], ...]] = {}
        for state in states:
            state_id = str(state["state_id"])
            intervals = tuple(
                (int(value[0]), int(value[1])) for value in state["references"]
            )
            if not intervals or any(start < 0 or end < start for start, end in intervals):
                raise ValueError(f"Invalid reference interval in {history_id}/{state_id}")
            references[state_id] = intervals
        queries = _load_text_queries(
            text_path,
            state_field=str(row["text_state_field"]),
            state_rows=states,
        )
        if {query.state_id for query in queries} != set(references):
            raise ValueError(f"State/query roster differs for {history_id}")
        history = History(
            history_id=history_id,
            sequence_id=str(row["scene"]),
            source_component_id=str(row["object_group_id"]),
            frame_indices=frames,
            visual_features=visual,
            visibility_count=visibility,
            queries=queries,
            references=references,
        )
        records.append(
            SevenHistoryRecord(
                history=history,
                stratum=str(row["stratum"]),
                dataset=str(row["dataset"]),
                scene=str(row["scene"]),
                object_group_id=str(row["object_group_id"]),
                distractors=_load_distractors(archive_root, row),
            )
        )
    expected_ids = [f"S{index:02d}" for index in range(1, 8)]
    if [record.history.history_id for record in records] != expected_ids:
        raise ValueError("Seven-history rows are incomplete or out of order")
    return tuple(records)


def load_seven_histories(root: Path | None = None) -> tuple[History, ...]:
    """Return the seven histories in the same interface as the VOST loader."""

    return tuple(record.history for record in load_seven_history_records(root))
