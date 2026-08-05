"""Deterministic VOST roster construction used by the finalized protocol.

The sampler sees only source split identities. It never reads frame labels,
reference intervals, model scores, or extracted features. With seed 3407 it
round-robins over actions and then target nouns, and admits at most one source
sequence for each action-object pair across the official train/validation
splits.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_SEED = 3407
DEFAULT_HISTORY_COUNT = 100


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of one regular file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_key(seed: int, *parts: str) -> str:
    """Hash a ranking key without depending on platform or Python hash state."""

    material = "\0".join((str(int(seed)), *(str(part) for part in parts)))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def parse_sequence_id(sequence_id: str) -> tuple[str, str]:
    """Extract the action and target-noun token from a VOST sequence ID."""

    parts = str(sequence_id).strip().split("_", 2)
    if len(parts) != 3 or not parts[0].isdigit() or not parts[1] or not parts[2]:
        raise ValueError(f"Malformed VOST sequence ID: {sequence_id!r}")
    return parts[1], parts[2]


def read_split(path: Path, source_partition: str) -> list[dict[str, str]]:
    """Read one official split list into source-only candidate records."""

    partition = str(source_partition).strip().lower()
    if partition not in {"train", "validation"}:
        raise ValueError("source_partition must be 'train' or 'validation'")
    rows: list[dict[str, str]] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        sequence_id = raw.strip()
        if not sequence_id:
            continue
        action, target_noun = parse_sequence_id(sequence_id)
        rows.append(
            {
                "sequence_id": sequence_id,
                "source_partition": partition,
                "action": action,
                "target_noun_token": target_noun,
            }
        )
    if not rows:
        raise ValueError(f"VOST split is empty: {path}")
    return rows


def rank_candidates(
    rows: Sequence[Mapping[str, str]], *, seed: int = DEFAULT_SEED
) -> list[dict[str, Any]]:
    """Apply the fixed action/noun round-robin ranking.

    Multiple source sequences can share an action-object pair. They remain
    adjacent in the ranking, in a stable hash order, so the first one can be
    selected and the rest retained as deterministic reserves.
    """

    by_action_noun: defaultdict[
        str, defaultdict[str, list[Mapping[str, str]]]
    ] = defaultdict(lambda: defaultdict(list))
    seen_sequences: set[str] = set()
    for row in rows:
        sequence_id = str(row.get("sequence_id", "")).strip()
        action = str(row.get("action", "")).strip()
        noun = str(row.get("target_noun_token", "")).strip()
        partition = str(row.get("source_partition", "")).strip()
        parsed_action, parsed_noun = parse_sequence_id(sequence_id)
        if (action, noun) != (parsed_action, parsed_noun):
            raise ValueError(f"Candidate fields disagree with {sequence_id}")
        if partition not in {"train", "validation"}:
            raise ValueError(f"Invalid source partition for {sequence_id}")
        if sequence_id in seen_sequences:
            raise ValueError(f"Duplicate VOST sequence across splits: {sequence_id}")
        seen_sequences.add(sequence_id)
        by_action_noun[action][noun].append(dict(row))

    action_order = sorted(
        by_action_noun, key=lambda value: stable_key(seed, "action", value)
    )
    noun_orders = {
        action: sorted(
            by_action_noun[action],
            key=lambda noun: stable_key(seed, action, noun),
        )
        for action in action_order
    }

    ranked: list[dict[str, Any]] = []
    noun_round = 0
    while any(noun_round < len(noun_orders[action]) for action in action_order):
        for action in action_order:
            nouns = noun_orders[action]
            if noun_round >= len(nouns):
                continue
            noun = nouns[noun_round]
            pair_rows = sorted(
                by_action_noun[action][noun],
                key=lambda row: stable_key(
                    seed, action, noun, str(row["sequence_id"])
                ),
            )
            for within_pair_rank, row in enumerate(pair_rows, start=1):
                ranked.append(
                    {
                        **dict(row),
                        "action_noun_round": noun_round,
                        "within_pair_rank": within_pair_rank,
                    }
                )
        noun_round += 1
    return ranked


def select_roster(
    candidates: Iterable[Mapping[str, Any]],
    *,
    history_count: int = DEFAULT_HISTORY_COUNT,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select the primary roster and deterministic reserve list."""

    if isinstance(history_count, bool) or int(history_count) <= 0:
        raise ValueError("history_count must be a positive integer")
    target = int(history_count)
    primary: list[dict[str, Any]] = []
    reserves: list[dict[str, Any]] = []
    used_pairs: set[tuple[str, str]] = set()
    used_sequences: set[str] = set()
    for candidate in candidates:
        row = dict(candidate)
        sequence_id = str(row["sequence_id"])
        pair = (str(row["action"]), str(row["target_noun_token"]))
        if sequence_id in used_sequences:
            raise ValueError(f"Duplicate ranked VOST sequence: {sequence_id}")
        used_sequences.add(sequence_id)
        if len(primary) < target and pair not in used_pairs:
            used_pairs.add(pair)
            primary.append(
                {
                    **row,
                    "history_id": f"H{len(primary) + 1:03d}",
                    "roster_phase": "primary",
                }
            )
        else:
            reserves.append({**row, "roster_phase": "reserve"})
    if len(primary) != target:
        raise ValueError(
            f"Only {len(primary)} unique action-object pairs are available"
        )
    return primary, reserves


def build_cohort_selection(
    train_split: Path,
    validation_split: Path,
    *,
    seed: int = DEFAULT_SEED,
    history_count: int = DEFAULT_HISTORY_COUNT,
) -> dict[str, Any]:
    """Select the VOST cohort from split identities without using labels or scores."""

    train_path = Path(train_split)
    validation_path = Path(validation_split)
    rows = read_split(train_path, "train") + read_split(
        validation_path, "validation"
    )
    ranked = rank_candidates(rows, seed=seed)
    primary, reserves = select_roster(ranked, history_count=history_count)
    return {
        "schema_version": 1,
        "kind": "deja_cue_vost_round_robin_selection",
        "seed": int(seed),
        "policy": (
            "round_robin_over_actions_then_target_nouns_with_at_most_one_"
            "source_sequence_per_action_object_pair"
        ),
        "state_or_reference_data_accessed": False,
        "model_scores_accessed": False,
        "source_splits": {
            "train": {
                "path": "train.txt",
                "sha256": sha256_file(train_path),
                "sequence_count": len(read_split(train_path, "train")),
            },
            "validation": {
                "path": "validation.txt",
                "sha256": sha256_file(validation_path),
                "sequence_count": len(
                    read_split(validation_path, "validation")
                ),
            },
        },
        "counts": {
            "candidate_sequences": len(ranked),
            "primary_histories": len(primary),
            "primary_action_object_pairs": len(
                {(row["action"], row["target_noun_token"]) for row in primary}
            ),
            "reserve_sequences": len(reserves),
        },
        "primary_histories": primary,
        "reserve_histories": reserves,
    }


def validate_selection_against_cohort(
    selection: Mapping[str, Any], cohort: Mapping[str, Any]
) -> None:
    """Check that a selection reproduces the bundled 100-history roster."""

    primary = selection.get("primary_histories")
    roster = cohort.get("roster")
    if not isinstance(primary, list) or not isinstance(roster, list):
        raise ValueError("Selection or cohort is missing its roster")
    observed = [str(row.get("sequence_id", "")) for row in primary]
    expected = [str(row.get("sequence_id", "")) for row in roster]
    if observed != expected:
        raise ValueError("Round-robin selection differs from the fixed cohort")
    if len(observed) != DEFAULT_HISTORY_COUNT or len(set(observed)) != len(observed):
        raise ValueError("The fixed VOST cohort must contain 100 unique sequences")
