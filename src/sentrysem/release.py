"""Deterministic release-manifest construction and validation."""

from __future__ import annotations

import string
from pathlib import Path

from .hashing import sha256_file

MANIFEST_NAME = "RELEASE_MANIFEST.sha256"
EXCLUDED_DIRECTORIES = {
    ".git",
    ".idea",
    ".conda",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".vscode",
    "__pycache__",
    "build",
    "dist",
    "htmlcov",
    "venv",
}
EXCLUDED_PREFIXES = (
    "artifacts/downloads/",
    "artifacts/generated_inputs/",
    "data/cache/",
    "data/raw/",
    "figures/generated/",
    "results/generated/",
    "results/smoke/",
)


def release_files(repository_root: Path) -> list[Path]:
    """Return the files that constitute a clean GitHub release tree."""

    repository_root = Path(repository_root).resolve()
    files: list[Path] = []
    for path in repository_root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(repository_root)
        relative_text = relative.as_posix()
        if path.name in {
            MANIFEST_NAME,
            ".DS_Store",
            ".coverage",
            "Thumbs.db",
            "PKG-INFO",
            "setup.cfg",
        } or path.suffix in {
            ".pyc",
            ".log",
            ".tmp",
            ".temp",
        }:
            continue
        if any(
            part in EXCLUDED_DIRECTORIES or part.endswith(".egg-info") for part in relative.parts
        ):
            continue
        if relative_text.startswith(EXCLUDED_PREFIXES):
            continue
        files.append(relative)
    return sorted(files, key=lambda item: item.as_posix())


def build_release_manifest(repository_root: Path) -> dict[str, str]:
    repository_root = Path(repository_root).resolve()
    return {
        relative.as_posix(): sha256_file(repository_root / relative)
        for relative in release_files(repository_root)
    }


def write_release_manifest(repository_root: Path) -> Path:
    repository_root = Path(repository_root).resolve()
    destination = repository_root / MANIFEST_NAME
    records = build_release_manifest(repository_root)
    content = "".join(f"{digest}  {path}\n" for path, digest in records.items())
    destination.write_text(content, encoding="utf-8")
    return destination


def verify_release_manifest(repository_root: Path) -> int:
    repository_root = Path(repository_root).resolve()
    manifest_path = repository_root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    declared: dict[str, str] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        relative_path = Path(relative)
        if (
            not separator
            or len(digest) != 64
            or any(character not in string.hexdigits for character in digest)
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or not relative
        ):
            raise AssertionError("Malformed release-manifest entry")
        if relative in declared:
            raise AssertionError(f"Duplicate release-manifest entry: {relative}")
        declared[relative] = digest.lower()

    observed = build_release_manifest(repository_root)
    if set(declared) != set(observed):
        missing = sorted(set(declared) - set(observed))
        untracked = sorted(set(observed) - set(declared))
        raise AssertionError(
            f"Release-manifest file set differs; missing={missing}, untracked={untracked}"
        )
    for relative, expected_digest in declared.items():
        if observed[relative] != expected_digest:
            raise AssertionError(f"Release-manifest digest mismatch: {relative}")
    return len(declared)
