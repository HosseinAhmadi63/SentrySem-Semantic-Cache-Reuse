"""Cryptographic digests used to identify SentrySem artifacts."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

READ_BLOCK_BYTES = 1 << 20


def sha256_bytes(data: bytes | bytearray | memoryview) -> str:
    """Return the hexadecimal SHA-256 digest of an in-memory byte sequence."""

    return hashlib.sha256(bytes(data)).hexdigest()


def sha256_file(path: str | Path, block_bytes: int = READ_BLOCK_BYTES) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""

    source = Path(path)
    if block_bytes <= 0:
        raise ValueError("block_bytes must be positive")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(block_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    """Hash an array together with its dtype and shape."""

    value = np.asarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def sha256_arrays(arrays: Mapping[str, np.ndarray]) -> str:
    """Hash a named collection of arrays in lexicographic name order."""

    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.asarray(arrays[name])
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(4, "big"))
        digest.update(encoded_name)
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def sha256_model_state(state: Mapping[str, Any]) -> str:
    """Reproduce the paper's digest of the frozen feature-extractor state."""

    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def file_manifest(
    directory: str | Path, recursive: bool = False
) -> dict[str, dict[str, int | str]]:
    """Describe regular files in a directory by size and SHA-256 digest."""

    root = Path(directory)
    iterator = root.rglob("*") if recursive else root.iterdir()
    manifest: dict[str, dict[str, int | str]] = {}
    for path in sorted(item for item in iterator if item.is_file()):
        name = path.relative_to(root).as_posix()
        manifest[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return manifest
