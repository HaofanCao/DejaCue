"""Deterministic epoch batching for the eight adapted decoders.

The model families have different native contrastive assumptions.  The shared
dispatcher below preserves those assumptions while ensuring that every one of
the 86 positive development records is exposed in each epoch.  Duplicates are
introduced only when a batch-family constraint makes them mathematically
necessary.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Mapping, Sequence

import numpy as np

from .protocol import BATCH_SIZE, decoder_spec, epoch_seed
from .records import LearnedRunRecord


def _history_identity(record: LearnedRunRecord) -> str:
    """Return the exact source-component identity used by native batching."""

    return str(record.native_history_id or record.history_id)


def _balanced_sizes(num_records: int, batch_size: int) -> tuple[int, ...]:
    if num_records <= 0 or batch_size < 2:
        raise ValueError("Balanced batching requires records and batch_size >= 2")
    count = math.ceil(num_records / batch_size)
    base, extra = divmod(num_records, count)
    sizes = tuple(base + int(index < extra) for index in range(count))
    if sum(sizes) != num_records or min(sizes) <= 0 or max(sizes) > batch_size:
        raise RuntimeError("Balanced batch partition is inconsistent")
    return sizes


def stratified_exact_once_batches(
    records: Sequence[LearnedRunRecord], *, batch_size: int, seed: int
) -> tuple[tuple[int, ...], ...]:
    """Balanced exact-once batches with a positive anchor in every batch."""

    selected = tuple(records)
    sizes = _balanced_sizes(len(selected), batch_size)
    positives = [index for index, record in enumerate(selected) if record.has_target]
    if len(positives) < len(sizes):
        raise ValueError("Every stratified batch requires a distinct positive anchor")
    rng = np.random.default_rng(seed)
    shuffled_positive = [int(index) for index in rng.permutation(positives)]
    positive_set = set(positives)
    empty = [index for index in range(len(selected)) if index not in positive_set]
    shuffled_empty = [int(index) for index in rng.permutation(empty)]
    anchors = shuffled_positive[: len(sizes)]
    residual = shuffled_positive[len(sizes) :] + shuffled_empty
    residual = [int(index) for index in rng.permutation(residual)]

    batches: list[tuple[int, ...]] = []
    offset = 0
    for anchor, size in zip(anchors, sizes):
        take = size - 1
        batches.append((anchor, *residual[offset : offset + take]))
        offset += take
    flattened = [index for batch in batches for index in batch]
    if sorted(flattened) != list(range(len(selected))):
        raise RuntimeError("Stratified schedule is not exact once")
    if any(not any(selected[index].has_target for index in batch) for batch in batches):
        raise RuntimeError("A stratified batch lacks a positive anchor")
    return tuple(batches)


def qd_records_compatible(left: LearnedRunRecord, right: LearnedRunRecord) -> bool:
    """Whether cyclic query-dependent negatives are valid for two records."""

    return bool(
        _history_identity(left) != _history_identity(right)
        and left.text != right.text
        and not np.allclose(
            left.text_embedding, right.text_embedding, atol=1e-7, rtol=1e-7
        )
    )


def validate_cross_history_cycle(records: Sequence[LearnedRunRecord]) -> None:
    """Require every cyclic neighbor to be a valid cross-history negative."""

    selected = tuple(records)
    if len(selected) < 2 or not any(record.has_target for record in selected):
        raise ValueError("A cyclic batch needs at least two records and one positive")
    for index, record in enumerate(selected):
        if not qd_records_compatible(record, selected[(index + 1) % len(selected)]):
            raise ValueError("Adjacent cyclic negatives must come from compatible histories")


def _find_cycle(
    indices: Sequence[int],
    neighbors: Mapping[int, frozenset[int]],
    ranks: Mapping[int, float],
) -> tuple[int, ...] | None:
    candidates = tuple(int(index) for index in indices)
    if len(candidates) < 2:
        return None
    if len(candidates) == 2:
        return candidates if candidates[1] in neighbors[candidates[0]] else None
    local = {
        index: neighbors[index].intersection(candidates) for index in candidates
    }
    if any(len(values) < 2 for values in local.values()):
        return None
    starts = sorted(candidates, key=lambda value: (len(local[value]), ranks[value], value))
    for start in starts:
        path = [start]
        remaining = set(candidates) - {start}
        while remaining:
            options = [value for value in remaining if value in local[path[-1]]]
            if len(remaining) == 1:
                options = [value for value in options if start in local[value]]
            else:
                options = [
                    value
                    for value in options
                    if local[value].intersection(remaining - {value})
                ]
            if not options:
                break
            chosen = min(
                options,
                key=lambda value: (
                    len(local[value].intersection(remaining)),
                    ranks[value],
                    value,
                ),
            )
            path.append(chosen)
            remaining.remove(chosen)
        if not remaining and start in local[path[-1]]:
            return tuple(path)
    return None


def _minimum_cycle_support(records: Sequence[LearnedRunRecord]) -> int:
    counts = Counter(_history_identity(record) for record in records)
    if not counts:
        raise ValueError("Cycle support requires records")
    return max(0, 2 * max(counts.values()) - len(records))


def _even_batch_sizes(num_exposures: int, batch_size: int) -> tuple[int, ...]:
    if num_exposures < 2 or num_exposures % 2 or batch_size < 2:
        raise ValueError("Cyclic support needs an even exposure count and batch_size >= 2")
    limit = batch_size - batch_size % 2
    count = math.ceil(num_exposures / limit)
    if num_exposures < 2 * count:
        raise ValueError("Cyclic exposures cannot form batches of at least two")
    sizes = [2] * count
    for _ in range((num_exposures - 2 * count) // 2):
        available = [index for index, size in enumerate(sizes) if size < limit]
        if not available:
            raise ValueError("Cyclic exposures exceed batch capacity")
        chosen = min(available, key=lambda index: (sizes[index], index))
        sizes[chosen] += 2
    return tuple(sorted(sizes, reverse=True))


def _support_rotation(records: Sequence[LearnedRunRecord]) -> tuple[int, ...]:
    counts = Counter(_history_identity(record) for record in records)
    dominant = min(counts, key=lambda value: (-counts[value], value))
    return tuple(
        sorted(
            (
                index
                for index, record in enumerate(records)
                if _history_identity(record) != dominant
            ),
            key=lambda index: (
                _history_identity(records[index]),
                records[index].query_index,
                records[index].run_index,
                records[index].text,
                index,
            ),
        )
    )


def _minimum_oversampled_cycles(
    records: Sequence[LearnedRunRecord],
    *,
    batch_size: int,
    seed: int,
    max_attempts: int,
) -> tuple[tuple[int, ...], ...]:
    selected = tuple(records)
    support_count = _minimum_cycle_support(selected)
    if support_count <= 0 or not all(record.has_target for record in selected):
        raise ValueError("Positive support oversampling is not applicable")
    counts = Counter(_history_identity(record) for record in selected)
    dominant = min(counts, key=lambda value: (-counts[value], value))
    dominant_count = counts[dominant]
    rotation = _support_rotation(selected)
    if not rotation:
        raise ValueError("Support oversampling requires another history")
    rotation_start = (support_count * seed) % len(rotation)
    support = [
        rotation[(rotation_start + offset) % len(rotation)]
        for offset in range(support_count)
    ]
    exposures = list(range(len(selected))) + support
    sizes = _even_batch_sizes(len(exposures), batch_size)
    dominant_tokens = [
        token
        for token, index in enumerate(exposures)
        if _history_identity(selected[index]) == dominant
    ]
    support_tokens = [
        token
        for token, index in enumerate(exposures)
        if _history_identity(selected[index]) != dominant
    ]
    if len(dominant_tokens) != dominant_count or len(support_tokens) != dominant_count:
        raise RuntimeError("Minimum cyclic support is not balanced")

    rng = np.random.default_rng(seed)
    for _ in range(max_attempts):
        left = rng.permutation(dominant_tokens).tolist()
        right = rng.permutation(support_tokens).tolist()
        left_offset = right_offset = 0
        cycles: list[tuple[int, ...]] = []
        feasible = True
        for size in sizes:
            half = size // 2
            left_part = left[left_offset : left_offset + half]
            right_part = right[right_offset : right_offset + half]
            left_offset += half
            right_offset += half
            tokens = tuple(token for pair in zip(left_part, right_part) for token in pair)
            cycle = tuple(exposures[token] for token in tokens)
            try:
                validate_cross_history_cycle(tuple(selected[index] for index in cycle))
            except ValueError:
                feasible = False
                break
            cycles.append(cycle)
        if feasible:
            flattened = [index for cycle in cycles for index in cycle]
            if set(flattened) != set(range(len(selected))):
                raise RuntimeError("Oversampled cycles omitted an original record")
            if len(flattened) - len(selected) != support_count:
                raise RuntimeError("Oversampled cycles used non-minimal support")
            return tuple(cycles)
    raise ValueError("Could not construct minimum-support cyclic batches")


def cross_history_cycle_batches(
    records: Sequence[LearnedRunRecord],
    *,
    batch_size: int,
    seed: int,
    max_attempts: int = 256,
) -> tuple[tuple[int, ...], ...]:
    """Create deterministic cyclic negatives with minimum support duplication."""

    selected = tuple(records)
    if len(selected) < 2 or batch_size < 2 or max_attempts <= 0:
        raise ValueError("Invalid cyclic batching inputs")
    if len({_history_identity(record) for record in selected}) < 2:
        raise ValueError("Cyclic batching requires at least two histories")
    positives = [index for index, record in enumerate(selected) if record.has_target]
    if not positives:
        raise ValueError("Cyclic batching requires positive records")
    if _minimum_cycle_support(selected) > 0:
        return _minimum_oversampled_cycles(
            selected, batch_size=batch_size, seed=seed, max_attempts=max_attempts
        )

    num_batches = math.ceil(len(selected) / batch_size)
    if len(selected) < 2 * num_batches or len(positives) < num_batches:
        raise ValueError("Batch size cannot support valid positive cyclic batches")
    base, extra = divmod(len(selected), num_batches)
    target_sizes = tuple(base + int(index < extra) for index in range(num_batches))
    mutable_neighbors = {index: set() for index in range(len(selected))}
    for left, record in enumerate(selected):
        for right in range(left + 1, len(selected)):
            if qd_records_compatible(record, selected[right]):
                mutable_neighbors[left].add(right)
                mutable_neighbors[right].add(left)
    neighbors = {
        index: frozenset(values) for index, values in mutable_neighbors.items()
    }
    degrees = {index: len(values) for index, values in neighbors.items()}
    if any(value == 0 for value in degrees.values()):
        raise ValueError("A cyclic record has no compatible neighbor")

    rng = np.random.default_rng(seed)
    all_indices = set(range(len(selected)))
    for _ in range(max_attempts):
        ranks = {index: float(rng.random()) for index in all_indices}
        shuffled_positive = rng.permutation(positives).tolist()
        bins = [[int(shuffled_positive[index])] for index in range(num_batches)]
        assigned = {values[0] for values in bins}
        remaining = sorted(
            all_indices - assigned,
            key=lambda index: (degrees[index], ranks[index], index),
        )
        feasible = True
        for index in remaining:
            available = [
                bin_index
                for bin_index, values in enumerate(bins)
                if len(values) < target_sizes[bin_index]
            ]
            if not available:
                feasible = False
                break
            chosen = min(
                available,
                key=lambda bin_index: (
                    sum(other not in neighbors[index] for other in bins[bin_index]),
                    sum(
                        _history_identity(selected[index])
                        == _history_identity(selected[other])
                        for other in bins[bin_index]
                    ),
                    len(bins[bin_index]) / target_sizes[bin_index],
                    float(rng.random()),
                    bin_index,
                ),
            )
            bins[chosen].append(index)
        if not feasible:
            continue
        cycles = []
        for values in bins:
            cycle = _find_cycle(values, neighbors, ranks)
            if cycle is None:
                break
            cycles.append(cycle)
        if len(cycles) != num_batches:
            continue
        flattened = [index for cycle in cycles for index in cycle]
        if len(flattened) != len(set(flattened)) or set(flattened) != all_indices:
            raise RuntimeError("Cyclic schedule is not exact once")
        for cycle in cycles:
            validate_cross_history_cycle(tuple(selected[index] for index in cycle))
        return tuple(cycles)
    raise ValueError("Could not construct balanced cyclic batches")


def _history_unique_lower_bounds(
    records: Sequence[LearnedRunRecord], batch_size: int
) -> dict[str, int]:
    selected = tuple(records)
    if not selected or batch_size < 2 or not all(record.has_target for record in selected):
        raise ValueError("History-unique training requires positive records")
    counts = Counter(_history_identity(record) for record in selected)
    if len(counts) < 2:
        raise ValueError("History-unique training requires two histories")
    effective = min(batch_size, len(counts))
    history_bound = max(counts.values())
    capacity_bound = math.ceil(len(selected) / effective)
    num_batches = max(history_bound, capacity_bound)
    minimum_duplicates = max(0, num_batches - len(selected), 2 * num_batches - len(selected))
    if len(selected) + minimum_duplicates > num_batches * effective:
        raise ValueError("History-unique lower bounds are infeasible")
    return {
        "effective": effective,
        "num_batches": num_batches,
        "minimum_duplicates": minimum_duplicates,
    }


def history_unique_batches(
    records: Sequence[LearnedRunRecord], *, batch_size: int, seed: int
) -> tuple[tuple[int, ...], ...]:
    """Place at most one record from each history in a native batch."""

    selected = tuple(records)
    bounds = _history_unique_lower_bounds(selected, batch_size)
    by_history: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(selected):
        by_history[_history_identity(record)].append(index)
    rng = np.random.default_rng(seed)
    batches: list[list[int]] = [[] for _ in range(bounds["num_batches"])]
    batch_histories: list[set[str]] = [set() for _ in batches]
    ordered = sorted(by_history.items(), key=lambda item: (-len(item[1]), item[0]))
    for history_index, (history, raw_indices) in enumerate(ordered):
        indices = [int(index) for index in rng.permutation(raw_indices)]
        tie_breaks = rng.random(len(batches))
        candidates = sorted(
            range(len(batches)),
            key=lambda index: (len(batches[index]), tie_breaks[index], index),
        )
        candidates = [
            index for index in candidates if len(batches[index]) < bounds["effective"]
        ]
        if len(candidates) < len(indices):
            raise RuntimeError(f"Could not place history {history_index}")
        for record_index, batch_index in zip(indices, candidates):
            batches[batch_index].append(record_index)
            batch_histories[batch_index].add(history)

    singletons = [index for index, batch in enumerate(batches) if len(batch) < 2]
    if len(singletons) != bounds["minimum_duplicates"]:
        raise RuntimeError("History-unique construction misses its support lower bound")
    positive_indices = [index for index, record in enumerate(selected) if record.has_target]
    support_offset = seed % len(positive_indices)
    for rank, batch_index in enumerate(singletons):
        compatible = [
            index
            for index in positive_indices
            if _history_identity(selected[index]) not in batch_histories[batch_index]
        ]
        if not compatible:
            raise ValueError("No positive support can complete a singleton batch")
        support = compatible[(support_offset + rank) % len(compatible)]
        batches[batch_index].append(support)
        batch_histories[batch_index].add(_history_identity(selected[support]))

    flattened = [index for batch in batches for index in batch]
    counts = Counter(flattened)
    if set(counts) != set(range(len(selected))):
        raise RuntimeError("History-unique batching omitted a record")
    if len(flattened) - len(selected) != bounds["minimum_duplicates"]:
        raise RuntimeError("History-unique batching used non-minimal support")
    if any(
        len(batch) < 2
        or len(batch) > bounds["effective"]
        or len({_history_identity(selected[index]) for index in batch}) != len(batch)
        for batch in batches
    ):
        raise RuntimeError("History-unique batch constraint failed")
    return tuple(tuple(batch) for batch in batches)


def unique_history_vtc_batches(
    records: Sequence[LearnedRunRecord], *, batch_size: int, seed: int
) -> tuple[tuple[int, ...], ...]:
    """Build Sim-DETR VTC batches with unique history identities."""

    selected = tuple(records)
    if len(selected) < 2 or batch_size < 2 or not all(record.has_target for record in selected):
        raise ValueError("VTC batching requires at least two positive records")
    by_history: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(selected):
        by_history[_history_identity(record)].append(index)
    if len(by_history) < 2:
        raise ValueError("VTC batching requires at least two histories")
    capacity = min(batch_size, len(by_history))
    num_batches = max(
        max(len(indices) for indices in by_history.values()),
        math.ceil(len(selected) / capacity),
    )
    rng = np.random.default_rng(seed)
    batches: list[list[int]] = [[] for _ in range(num_batches)]
    for history in sorted(by_history, key=lambda value: (-len(by_history[value]), value)):
        indices = [int(index) for index in rng.permutation(by_history[history])]
        tie_break = rng.permutation(num_batches).tolist()
        order = sorted(
            range(num_batches),
            key=lambda index: (len(batches[index]), tie_break.index(index)),
        )
        if len(indices) > len(order):
            raise RuntimeError("A history cannot be unique across VTC batches")
        for record_index, batch_index in zip(indices, order):
            batches[batch_index].append(record_index)
    if sum(map(len, batches)) != len(selected):
        raise RuntimeError("VTC assignment did not cover every record once")

    minimum_duplicates = sum(max(0, 2 - len(batch)) for batch in batches)
    support_order = [int(index) for index in rng.permutation(len(selected))]
    support_offset = 0
    for batch in batches:
        while len(batch) < 2:
            present = {_history_identity(selected[index]) for index in batch}
            support = None
            for shift in range(len(support_order)):
                candidate = support_order[(support_offset + shift) % len(support_order)]
                if _history_identity(selected[candidate]) not in present:
                    support = candidate
                    support_offset = (support_offset + shift + 1) % len(support_order)
                    break
            if support is None:
                raise RuntimeError("Could not complete a VTC singleton batch")
            batch.append(support)
    output = tuple(tuple(batch) for batch in batches)
    flattened = [index for batch in output for index in batch]
    if set(flattened) != set(range(len(selected))):
        raise RuntimeError("VTC batching omitted a record")
    if len(flattened) - len(selected) != minimum_duplicates:
        raise RuntimeError("VTC batching used non-minimal support")
    if any(
        len(batch) < 2
        or len(batch) > capacity
        or len({_history_identity(selected[index]) for index in batch}) != len(batch)
        for batch in output
    ):
        raise RuntimeError("VTC history-identity constraint failed")
    return output


def build_epoch_batches(
    records: Sequence[LearnedRunRecord],
    *,
    model_id: str,
    training_seed: int,
    zero_based_epoch: int,
    batch_size: int = BATCH_SIZE,
) -> tuple[tuple[int, ...], ...]:
    """Dispatch to the decoder's deterministic batch family."""

    spec = decoder_spec(model_id)
    seed = epoch_seed(training_seed, zero_based_epoch)
    builders = {
        "stratified_exact_once": stratified_exact_once_batches,
        "cross_history_cycle": cross_history_cycle_batches,
        "history_unique": history_unique_batches,
        "unique_history_vtc": unique_history_vtc_batches,
    }
    return builders[spec.batching_family](records, batch_size=batch_size, seed=seed)


def summarize_batches(
    records: Sequence[LearnedRunRecord], batches: Sequence[Sequence[int]]
) -> dict[str, object]:
    """Return coverage statistics for a generated epoch schedule."""

    selected = tuple(records)
    flattened = [int(index) for batch in batches for index in batch]
    if not flattened or any(index < 0 or index >= len(selected) for index in flattened):
        raise ValueError("Batch schedule is empty or contains an invalid index")
    counts = Counter(flattened)
    return {
        "num_batches": len(batches),
        "batch_sizes": [len(batch) for batch in batches],
        "num_unique_records": len(counts),
        "num_exposures": len(flattened),
        "num_duplicate_exposures": len(flattened) - len(counts),
        "all_records_covered": set(counts) == set(range(len(selected))),
        "all_exposures_positive": all(selected[index].has_target for index in flattened),
        "all_batches_have_two_records": all(len(batch) >= 2 for batch in batches),
    }
