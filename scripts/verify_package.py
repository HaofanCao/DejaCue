"""Validate the file list and every bundled evaluation input."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
IGNORED_RUNTIME_DIRECTORIES = {
    ".git",
    ".venv",
    ".pytest_cache",
    "__pycache__",
    "results",
}
DISALLOWED_PARTS = {"build", "dist"}
DISALLOWED_SUFFIXES = {".log", ".partial", ".pyc", ".pyo", ".tar", ".tmp", ".zip"}
README_HTML_ATTRIBUTES = {
    "a": {"href"},
    "div": {"align"},
    "em": set(),
    "h1": set(),
    "img": {"alt", "src", "width"},
    "p": {"align"},
    "strong": set(),
}
SHELL_COMMAND_PATTERN = re.compile(
    r"(?:git\s+clone|hf\s+download|python(?:\s+-m|\s+scripts/)|deja-cue\s+)",
    re.IGNORECASE,
)


class ReadmeHTMLValidator(HTMLParser):
    """Restrict README HTML to the visible subset verified on GitHub."""

    def __init__(self, source: Path) -> None:
        super().__init__(convert_charrefs=True)
        self.source = source
        self.stack: list[str] = []

    def validate_tag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Reject unsupported tags, attributes, and event handlers."""

        if tag not in README_HTML_ATTRIBUTES:
            raise ValueError(f"Unsupported README HTML tag <{tag}> in {self.source}")
        allowed = README_HTML_ATTRIBUTES[tag]
        unexpected = sorted(name for name, _ in attrs if name not in allowed)
        if unexpected:
            raise ValueError(
                f"Unsupported attributes on <{tag}> in {self.source}: {unexpected}"
            )

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.validate_tag(tag, attrs)
        if tag != "img":
            self.stack.append(tag)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.validate_tag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if tag not in README_HTML_ATTRIBUTES:
            raise ValueError(
                f"Unsupported README HTML closing tag </{tag}> in {self.source}"
            )
        if tag == "img" or not self.stack or self.stack[-1] != tag:
            raise ValueError(
                f"Unbalanced README HTML closing tag </{tag}> in {self.source}"
            )
        self.stack.pop()

    def handle_comment(self, data: str) -> None:
        raise ValueError(f"Hidden HTML comments are not allowed in {self.source}")

    def handle_data(self, data: str) -> None:
        if self.stack and SHELL_COMMAND_PATTERN.search(data):
            raise ValueError(
                f"Shell commands must not be hidden inside HTML in {self.source}"
            )


def markdown_without_fenced_code(text: str, source: Path) -> str:
    """Remove fenced code before validating embedded HTML."""

    visible_lines: list[str] = []
    fence: str | None = None
    for line in text.splitlines():
        stripped = line.lstrip()
        if fence is None and stripped.startswith(("```", "~~~")):
            fence = stripped[:3]
            continue
        if fence is not None and stripped.startswith(fence):
            fence = None
            continue
        if fence is None:
            visible_lines.append(line)
    if fence is not None:
        raise ValueError(f"Unclosed Markdown code fence in {source}")
    return "\n".join(visible_lines)


def validate_readme_html(source: Path) -> None:
    """Validate the small, visible HTML subset used by the public README."""

    text = source.read_text(encoding="utf-8")
    parser = ReadmeHTMLValidator(source)
    parser.feed(markdown_without_fenced_code(text, source))
    parser.close()
    if parser.stack:
        raise ValueError(f"Unclosed README HTML tags in {source}: {parser.stack}")


def sha256_file(path: Path) -> str:
    """Hash a file in bounded blocks so feature arrays do not enter memory twice."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_files() -> list[Path]:
    """Return tracked files while ignoring outputs created by documented commands."""

    files = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if IGNORED_RUNTIME_DIRECTORIES.intersection(relative.parts) or any(
            part.endswith(".egg-info") for part in relative.parts
        ):
            continue
        if path.is_file():
            files.append(path)
    return sorted(files, key=lambda path: path.relative_to(ROOT).as_posix())


def validate_file_set(*, skip_manifest: bool) -> tuple[int, int]:
    """Check generated-file exclusions and, when present, the SHA-256 file list."""

    files = package_files()
    for path in files:
        relative = path.relative_to(ROOT)
        if DISALLOWED_PARTS.intersection(relative.parts):
            raise ValueError(f"Generated directory is present: {relative.as_posix()}")
        if path.suffix.lower() in DISALLOWED_SUFFIXES:
            raise ValueError(
                f"Temporary or packed file is present: {relative.as_posix()}"
            )

    tracked = [path for path in files if path.name != "MANIFEST.json"]
    total_bytes = sum(path.stat().st_size for path in tracked)
    if skip_manifest:
        return len(tracked), total_bytes

    manifest_path = ROOT / "MANIFEST.json"
    if not manifest_path.is_file():
        raise ValueError("MANIFEST.json is missing")
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    entries = manifest.get("files")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "deja_cue_file_manifest"
        or manifest.get("algorithm") != "sha256"
        or not isinstance(entries, list)
    ):
        raise ValueError("MANIFEST.json has an unexpected schema")

    expected = {str(entry["path"]): entry for entry in entries}
    if len(expected) != len(entries) or "MANIFEST.json" in expected:
        raise ValueError("Manifest entries repeat or include the manifest itself")
    observed = {path.relative_to(ROOT).as_posix(): path for path in tracked}
    if set(expected) != set(observed):
        missing = sorted(set(expected).difference(observed))
        extra = sorted(set(observed).difference(expected))
        raise ValueError(f"Manifest file set differs: missing={missing}, extra={extra}")
    for relative, path in observed.items():
        entry = expected[relative]
        size = path.stat().st_size
        if int(entry["bytes"]) != size:
            raise ValueError(f"Size differs for {relative}")
        if str(entry["sha256"]) != sha256_file(path):
            raise ValueError(f"SHA-256 differs for {relative}")
    if int(manifest.get("file_count", -1)) != len(tracked):
        raise ValueError("Manifest file_count differs")
    if int(manifest.get("total_bytes", -1)) != total_bytes:
        raise ValueError("Manifest total_bytes differs")
    return len(tracked), total_bytes


def relative_input(value: object, *, label: str) -> Path:
    """Resolve one manifest path and reject absolute or parent-traversing forms."""

    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty path")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise ValueError(f"{label} must be relative")
    if ".." in posix.parts or ".." in windows.parts:
        raise ValueError(f"{label} may not traverse parent directories")
    path = (ROOT / Path(value)).resolve(strict=True)
    try:
        path.relative_to(ROOT.resolve(strict=True))
    except ValueError as exc:
        raise ValueError(f"{label} escapes the archive") from exc
    if not path.is_file():
        raise ValueError(f"{label} is not a file")
    return path


def require_arrays(
    path: Path, names: Iterable[str], *, exact: bool = True
) -> dict[str, np.ndarray]:
    """Load a non-pickled NPZ archive and check its declared fields."""

    required = set(names)
    with np.load(path, allow_pickle=False) as archive:
        available = set(archive.files)
        if not required.issubset(available) or (exact and available != required):
            raise ValueError(
                f"Array fields differ for {path.relative_to(ROOT).as_posix()}: "
                f"expected={sorted(required)}, observed={sorted(available)}"
            )
        return {name: np.asarray(archive[name]) for name in required}


def require_unit_rows(values: np.ndarray, *, label: str) -> None:
    """Check finite rank-two feature rows at the export precision tolerance."""

    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError(f"{label} must be a non-empty finite matrix")
    if not np.allclose(np.linalg.norm(values, axis=1), 1.0, atol=2e-3, rtol=0.0):
        raise ValueError(f"{label} rows are not unit normalized")


def validate_visual(
    arrays: dict[str, np.ndarray], *, label: str
) -> tuple[np.ndarray, int]:
    """Validate aligned frame, visibility, and visual-feature arrays."""

    frames = arrays["frame_indices"]
    visibility = arrays["visibility_count"]
    features = arrays["visual_features"]
    if frames.ndim != 1 or visibility.ndim != 1 or len(frames) != len(visibility):
        raise ValueError(f"{label} frame and visibility vectors differ")
    if not len(frames) or len(features) != len(frames) or np.any(np.diff(frames) <= 0):
        raise ValueError(f"{label} frames are empty, repeated, or unordered")
    if not np.issubdtype(frames.dtype, np.integer) or np.any(visibility <= 0):
        raise ValueError(f"{label} frame or visibility values are invalid")
    require_unit_rows(features, label=f"{label} visual_features")
    return frames.astype(np.int64, copy=False), int(features.shape[1])


def validate_text(
    arrays: dict[str, np.ndarray],
    *,
    label: str,
    expected_states: set[str],
    expected_pairs: list[tuple[str, str]] | None = None,
) -> tuple[list[tuple[str, str]], int]:
    """Validate four descriptions, state balance, text alignment, and row norms."""

    state_ids = arrays["state_ids"]
    texts = arrays["state_texts"]
    features = arrays["text_features"]
    if state_ids.ndim != 1 or texts.ndim != 1 or len(state_ids) != 4 or len(texts) != 4:
        raise ValueError(f"{label} must contain four text rows")
    pairs = [(str(state_id), str(text)) for state_id, text in zip(state_ids, texts)]
    if {state_id for state_id, _ in pairs} != expected_states:
        raise ValueError(f"{label} states differ from the annotations")
    if any(
        sum(state_id == member[0] for member in pairs) != 2
        for state_id in expected_states
    ):
        raise ValueError(f"{label} must contain two descriptions per state")
    if expected_pairs is not None and pairs != expected_pairs:
        raise ValueError(f"{label} descriptions differ across feature conditions")
    require_unit_rows(features, label=f"{label} text_features")
    return pairs, int(features.shape[1])


def validate_benchmark() -> dict[str, int]:
    """Validate the full 78-history roster and all three feature conditions."""

    benchmark = json.loads(
        (ROOT / "data" / "benchmark.json").read_text(encoding="utf-8")
    )
    rows = benchmark.get("histories")
    if benchmark.get("kind") != "deja_cue_compact_benchmark" or not isinstance(
        rows, list
    ):
        raise ValueError("benchmark.json has an unexpected schema")
    expected_ids = [f"H{index:03d}" for index in range(1, 79)]
    if [row.get("history_id") for row in rows] != expected_ids:
        raise ValueError("The history roster must be ordered H001 through H078")
    components = [str(row.get("source_component_id", "")) for row in rows]
    if any(not value for value in components) or len(set(components)) != 78:
        raise ValueError("Each history must have one distinct source component")

    state_count = 0
    description_count = 0
    for row in rows:
        history_id = str(row["history_id"])
        states = row.get("states")
        if not isinstance(states, list) or len(states) != 2:
            raise ValueError(f"{history_id} must have exactly two states")
        expected_states = {str(state.get("state_id", "")) for state in states}
        if len(expected_states) != 2 or "" in expected_states:
            raise ValueError(f"{history_id} state identifiers are invalid")
        for state in states:
            references = state.get("references")
            if not isinstance(references, list) or not references:
                raise ValueError(f"{history_id} has no reference interval")
            for interval in references:
                if (
                    not isinstance(interval, list)
                    or len(interval) != 2
                    or not all(isinstance(value, int) for value in interval)
                    or interval[0] < 0
                    or interval[1] < interval[0]
                ):
                    raise ValueError(f"{history_id} has an invalid inclusive interval")

        primary_visual = require_arrays(
            relative_input(
                row.get("siglip2_visual"), label=f"{history_id} primary visual"
            ),
            ("frame_indices", "visual_features", "visibility_count"),
        )
        primary_frames, primary_dimension = validate_visual(
            primary_visual, label=f"{history_id} primary"
        )
        primary_text = require_arrays(
            relative_input(row.get("siglip2_text"), label=f"{history_id} primary text"),
            ("state_ids", "state_texts", "text_features"),
        )
        pairs, text_dimension = validate_text(
            primary_text, label=f"{history_id} primary", expected_states=expected_states
        )
        if primary_dimension != text_dimension:
            raise ValueError(f"{history_id} primary visual/text dimensions differ")

        prompt = require_arrays(
            relative_input(
                row.get("siglip2_prompt_text"), label=f"{history_id} prompt text"
            ),
            ("state_ids", "state_texts", "variant_names", "text_features"),
        )
        names = [str(value) for value in prompt["variant_names"].tolist()]
        if names != ["raw", "photo", "definite"]:
            raise ValueError(f"{history_id} prompt variants differ")
        prompt_features = prompt["text_features"]
        if prompt_features.ndim != 3 or prompt_features.shape[:2] != (3, 4):
            raise ValueError(f"{history_id} prompt feature dimensions differ")
        prompt_pairs = [
            (str(state_id), str(text))
            for state_id, text in zip(prompt["state_ids"], prompt["state_texts"])
        ]
        if prompt_pairs != pairs:
            raise ValueError(f"{history_id} prompt descriptions differ")
        for index, name in enumerate(names):
            require_unit_rows(
                prompt_features[index], label=f"{history_id} prompt {name}"
            )
        if prompt_features.shape[2] != primary_dimension:
            raise ValueError(f"{history_id} prompt feature dimension differs")
        # The normalized three-prompt average is stored once as primary text.
        require_unit_rows(
            primary_text["text_features"], label=f"{history_id} prompt ensemble"
        )

        state_count += len(states)
        description_count += len(pairs)

    protocol = json.loads(
        (ROOT / "configs" / "protocol.json").read_text(encoding="utf-8")
    )
    if protocol.get("kind") != "deja_cue_fixed_protocol":
        raise ValueError("protocol.json has an unexpected schema")
    expected_counts = {
        "history_count": 78,
        "state_count": state_count,
        "description_count": description_count,
        "source_component_count": 78,
    }
    if any(
        int(protocol.get(key, -1)) != value for key, value in expected_counts.items()
    ):
        raise ValueError("Protocol entries differ from the benchmark")
    schedule = protocol.get("window_schedule")
    if (
        not isinstance(schedule, list)
        or len(schedule) != 33
        or any(not isinstance(value, int) or value <= 0 for value in schedule)
    ):
        raise ValueError("The fixed window schedule is invalid")
    for name in (
        "main",
        "extended",
        "runtime",
        "seven_history_summary",
        "seven_history_learned",
        "vocabulary_stress",
        "vost_cohort",
    ):
        if not (ROOT / "data" / "reference" / f"{name}.json").is_file():
            raise ValueError(f"Reference file is missing: {name}.json")
    return {
        "histories": len(rows),
        "states": state_count,
        "descriptions": description_count,
    }


def validate_supporting_assets() -> dict[str, int]:
    """Validate the seven-history arrays and all compact evidence families."""

    sys.path.insert(0, str(ROOT))
    from deja_cue.learned.development import load_development_histories
    from deja_cue.learned.native import verify_vendored_source
    from deja_cue.learned.protocol import protocol_summary
    from deja_cue.learned.records import build_positive_run_records
    from deja_cue.preprocessing import load_encoder_lock
    from deja_cue.robustness import validate_robustness_results
    from deja_cue.seven_history import load_seven_history_records
    from deja_cue.vost_protocol import load_cohort_asset, validate_cohort_asset
    from deja_cue.vost_sampling import (
        build_cohort_selection,
        validate_selection_against_cohort,
    )

    records = load_seven_history_records(ROOT)
    if len(records) != 7:
        raise ValueError("Seven-history feature roster differs")
    query_count = sum(len(record.history.queries) for record in records)
    distractor_count = sum(len(record.distractors) for record in records)
    if query_count != 32 or distractor_count != 8:
        raise ValueError("Seven-history query or distractor entries differ")

    cohort_payload = load_cohort_asset()
    cohort = validate_cohort_asset(cohort_payload)
    if cohort != {
        "sampled_histories": 100,
        "retained_histories": 78,
        "excluded_histories": 22,
    }:
        raise ValueError("VOST cohort counts differ")
    selection = build_cohort_selection(
        ROOT / "data" / "protocol" / "vost" / "train.txt",
        ROOT / "data" / "protocol" / "vost" / "validation.txt",
    )
    validate_selection_against_cohort(selection, cohort_payload)

    robustness = validate_robustness_results(ROOT)
    if len(robustness) != 8:
        raise ValueError("Robustness evidence family count differs")
    learned = protocol_summary()
    if (
        len(learned["decoders"]) != 8
        or learned["development"]["positive_records"] != 86
        or learned["seeds"] != [3407, 3408, 3409]
    ):
        raise ValueError("Learned-decoder protocol differs")
    development = load_development_histories(ROOT)
    positive_records = build_positive_run_records(development)
    if (
        len(development) != 5
        or sum(len(row.references) for row in development) != 13
        or sum(len(row.queries) for row in development) != 26
        or len(positive_records) != 86
    ):
        raise ValueError("Learned-development source files differ")
    for project in ("lighthouse", "sim_detr"):
        verify_vendored_source(project)

    encoder = load_encoder_lock()
    if (
        encoder.get("revision") != "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
        or encoder.get("feature_dimension") != 768
        or encoder.get("transformers_version") != "4.57.6"
        or len(encoder.get("required_files", [])) != 7
    ):
        raise ValueError("SigLIP2 encoder lock differs")

    runtime = json.loads(
        (ROOT / "data" / "reference" / "runtime.json").read_text(encoding="utf-8")
    )
    measurement = runtime.get("measurement", {})
    runtime_query_count = int(measurement.get("query_count", -1))
    wall = float(measurement.get("retrieval_wall_seconds", math.nan))
    per_query = float(measurement.get("seconds_per_query", math.nan))
    if (
        runtime.get("kind") != "deja_cue_runtime_reference"
        or runtime_query_count != 16
        or not math.isclose(
            wall, runtime_query_count * per_query, rel_tol=0.0, abs_tol=1e-12
        )
        or round(1000.0 * per_query, 1) != float(runtime.get("paper_value_ms_per_query"))
    ):
        raise ValueError("Runtime reference arithmetic differs")
    return {
        "development_records": len(positive_records),
        "seven_history_queries": query_count,
        "seven_history_distractors": distractor_count,
        "robustness_families": len(robustness),
    }


def build_parser() -> argparse.ArgumentParser:
    """Build options for full file validation or pre-manifest checks."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-manifest",
        action="store_true",
        help="validate inputs before the final manifest has been generated",
    )
    return parser


def main() -> int:
    """Validate the package file set, arrays, cohorts, protocols, and evidence."""

    args = build_parser().parse_args()
    file_count, total_bytes = validate_file_set(skip_manifest=args.skip_manifest)
    validate_readme_html(ROOT / "README.md")
    inventory = validate_benchmark()
    supporting = validate_supporting_assets()
    if not math.isfinite(float(total_bytes)):
        raise ValueError("Package byte count is invalid")
    print(
        json.dumps(
            {
                "bytes": total_bytes,
                "descriptions": inventory["descriptions"],
                "files": file_count,
                "histories": inventory["histories"],
                "passed": True,
                **supporting,
                "states": inventory["states"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"package verification failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
