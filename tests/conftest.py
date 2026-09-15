from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def frozen_inputs(repository_root: Path) -> Path:
    return repository_root / "artifacts" / "frozen_inputs"
