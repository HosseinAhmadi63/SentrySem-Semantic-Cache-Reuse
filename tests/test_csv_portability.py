from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

import pytest

from sentrysem.experiments import allocation, comparators, stress
from sentrysem.results import write_csv as write_standard_csv

CsvWriter = Callable[[Path, Iterable[Mapping[str, object]]], None]


@pytest.mark.parametrize(
    "writer",
    [write_standard_csv, allocation.write_csv, comparators.write_csv, stress.csv_write],
)
def test_csv_writers_use_repository_lf_endings(tmp_path: Path, writer: CsvWriter) -> None:
    destination = tmp_path / "portable.csv"
    writer(destination, [{"value": 1}])
    assert destination.read_bytes() == b"value\n1\n"
