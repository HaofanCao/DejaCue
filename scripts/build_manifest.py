"""Rebuild the repository release manifest from the current tracked files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST.json"
IGNORED_PARTS = {".git", ".venv", ".pytest_cache", "__pycache__", "results"}
DISALLOWED_PARTS = {"build", "dist"}
DISALLOWED_SUFFIXES = {".log", ".partial", ".pyc", ".pyo", ".tar", ".tmp", ".zip"}


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def release_files() -> list[Path]:
    """Return deterministic release files and reject generated artifacts."""

    files: list[Path] = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if IGNORED_PARTS.intersection(relative.parts) or any(
            part.endswith(".egg-info") for part in relative.parts
        ):
            continue
        if not path.is_file() or path == MANIFEST:
            continue
        if DISALLOWED_PARTS.intersection(relative.parts):
            raise ValueError(f"Generated directory is present: {relative.as_posix()}")
        if path.suffix.lower() in DISALLOWED_SUFFIXES:
            raise ValueError(f"Packed or temporary file is present: {relative.as_posix()}")
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(ROOT).as_posix())


def main() -> None:
    """Write MANIFEST.json and print a short summary."""

    files = release_files()
    entries = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    payload = {
        "schema_version": 1,
        "kind": "deja_cue_file_manifest",
        "algorithm": "sha256",
        "file_count": len(entries),
        "total_bytes": sum(entry["bytes"] for entry in entries),
        "files": entries,
    }
    MANIFEST.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="ascii",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "manifest": str(MANIFEST),
                "file_count": payload["file_count"],
                "total_bytes": payload["total_bytes"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
