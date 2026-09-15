from __future__ import annotations

import json

import pytest

from sentrysem.config import load_config, load_experiment_config, validate_experiment_config


def test_paper_and_smoke_configs(repository_root) -> None:
    paper = load_experiment_config(repository_root / "configs" / "paper.json")
    smoke = load_experiment_config(repository_root / "configs" / "smoke.json")
    assert paper["allocation"]["candidate_splits"][1] == [256, 768]
    assert smoke["smoke"]["queries"] == 20
    assert load_config(repository_root / "configs" / "paper.json").protocol.audit_bits == 768


def test_config_rejects_search_audit_seed_overlap(repository_root) -> None:
    values = json.loads((repository_root / "configs" / "paper.json").read_text())
    values["allocation"]["evaluation_protocol_seeds"][0] = values["allocation"][
        "tune_protocol_seeds"
    ][0]
    with pytest.raises(ValueError, match="disjoint"):
        validate_experiment_config(values)
