"""Read-only validation of the frozen and regenerated SentrySem artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from pypdf import PdfReader

from .config import load_experiment_config
from .data import load_frozen_embeddings
from .hashing import sha256_file
from .release import verify_release_manifest
from .statistics import binomial_boundary, binomial_cdf, observed_mismatch_probability
from .traffic import REFRESH_BITS, fixed_audit_traffic


def _close(observed: float, expected: float, tolerance: float = 1e-11) -> None:
    if not np.isclose(float(observed), float(expected), rtol=0.0, atol=tolerance):
        raise AssertionError(f"Observed {observed!r}; expected {expected!r}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _record(checks: list[dict[str, Any]], name: str, details: Any) -> None:
    checks.append({"check": name, "status": "PASS", "details": details})


def _group_key(row: dict[str, str], fields: tuple[str, ...]) -> tuple[Any, ...]:
    values: list[Any] = []
    for field in fields:
        value = row[field]
        if field == "bsc":
            values.append(float(value))
        elif field in {"search_bits", "audit_bits"}:
            values.append(int(value))
        else:
            values.append(value)
    return tuple(values)


def _validate_seed_aggregates(
    seed_path: Path,
    aggregate_path: Path,
    key_fields: tuple[str, ...],
    metrics: tuple[str, ...],
) -> int:
    seed_rows = _read_csv(seed_path)
    aggregate_rows = _read_csv(aggregate_path)
    groups: dict[tuple[Any, ...], list[dict[str, str]]] = {}
    for row in seed_rows:
        groups.setdefault(_group_key(row, key_fields), []).append(row)
    aggregate = {_group_key(row, key_fields): row for row in aggregate_rows}
    if set(groups) != set(aggregate):
        raise AssertionError(f"Seed and aggregate keys differ: {aggregate_path}")

    checked = 0
    for key, rows in groups.items():
        summary = aggregate[key]
        if int(summary["protocol_seeds"]) != len(rows):
            raise AssertionError(f"Protocol-seed count changed: {aggregate_path}, {key}")
        for metric in metrics:
            values = np.asarray(
                [float(row[metric]) for row in rows if row.get(metric, "") != ""], dtype=float
            )
            if not len(values):
                continue
            statistics = {
                "mean": float(values.mean()),
                "seed_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "seed_min": float(values.min()),
                "seed_max": float(values.max()),
            }
            for suffix, observed in statistics.items():
                _close(observed, float(summary[f"{metric}_{suffix}"]), 1e-9)
                checked += 1
    return checked


def _resolve_validation_dir(results_root: Path, inputs: Path) -> Path:
    candidate = results_root / "validation"
    return candidate if (candidate / "selection_stress.json").is_file() else inputs


def _validate_primary(results_root: Path, expected: dict, checks: list[dict[str, Any]]) -> None:
    aggregate = _read_csv(results_root / "stress" / "sentrysem_fixed_aggregate.csv")
    primary = [
        row for row in aggregate if row["scenario"] == "full" and row["condition"] == "paired_iid"
    ]
    if len(primary) != 2:
        raise AssertionError("Primary aggregate must contain two channel conditions")
    for row in primary:
        key = str(float(row["bsc"]))
        contract = expected["primary"][key]
        for field in (
            "reuse_rate",
            "identity_precision_among_reuses",
            "mean_completion_bits",
            "saving_vs_always_refresh",
            "bad_reuse_rate",
        ):
            _close(float(row[f"{field}_mean"]), float(contract[field]))
    _record(checks, "primary headline values", {"channels": [0.0, 0.05], "rows": 2})

    sessions = _read_csv(results_root / "stress" / "sentrysem_fixed_session_results.csv")
    if len(sessions) != 22_000:
        raise AssertionError(f"Expected 22,000 stress sessions, found {len(sessions)}")
    for row in sessions:
        if int(row["completion_bits"]) != int(row["protocol_bits"]) + int(row["fallback_bits"]):
            raise AssertionError("Stress completion traffic does not equal protocol plus refresh")
    primary_sessions = [
        row for row in sessions if row["scenario"] == "full" and row["condition"] == "paired_iid"
    ]
    if len(primary_sessions) != 5_000:
        raise AssertionError("Primary evaluation must contain 2,500 sessions per channel")
    if any(int(row["bad_reuse"]) for row in primary_sessions):
        raise AssertionError("Primary records contain a below-threshold reuse")
    identity_repeats: dict[tuple[float, int], int] = {}
    for row in primary_sessions:
        key = (float(row["bsc"]), int(row["target_id"]))
        identity_repeats[key] = identity_repeats.get(key, 0) + 1
    if set(identity_repeats.values()) != {5} or len(identity_repeats) != 1_000:
        raise AssertionError("Primary source identities must retain all five protocol seeds")
    _record(
        checks,
        "stress session ledger",
        {"sessions": len(sessions), "primary_sessions": 5000, "identity_seed_groups": 1000},
    )


def _validate_allocation(results_root: Path, config: dict, checks: list[dict[str, Any]]) -> None:
    directory = results_root / "allocation"
    selection = json.loads((directory / "selection.json").read_text())
    if (selection["selected_search_bits"], selection["selected_audit_bits"]) != (256, 768):
        raise AssertionError("The selected proposal/audit allocation changed")
    tune = tuple(map(int, selection["cal_tune_protocol_seeds"]))
    evaluation = tuple(map(int, selection["evaluation_protocol_seeds"]))
    if tune != tuple(config["allocation"]["tune_protocol_seeds"]):
        raise AssertionError("Allocation tuning seeds changed")
    if evaluation != tuple(config["allocation"]["evaluation_protocol_seeds"]):
        raise AssertionError("Primary evaluation seeds changed")
    if set(tune) & set(evaluation):
        raise AssertionError("Tuning and evaluation protocol seeds overlap")
    tune_rows = _read_csv(directory / "cal_tune_ratio_seed_results.csv")
    final_rows = _read_csv(directory / "selected_ratio_final_seed_results.csv")
    if len(tune_rows) != 70 or len(final_rows) != 10:
        raise AssertionError("Allocation seed-result row counts changed")
    _record(
        checks,
        "allocation selection",
        {"selected": [256, 768], "tune_rows": 70, "evaluation_rows": 10},
    )


def _validate_comparators(results_root: Path, checks: list[dict[str, Any]]) -> None:
    directory = results_root / "comparators"
    rows = _read_csv(directory / "publication_session_results.csv")
    if len(rows) != 50_000:
        raise AssertionError(f"Expected 50,000 comparator sessions, found {len(rows)}")
    methods = sorted({row["method"] for row in rows})
    if len(methods) != 10:
        raise AssertionError("Comparator experiment must contain ten methods")
    for row in rows:
        if int(row["completion_bits"]) != int(row["protocol_bits"]) + int(row["fallback_bits"]):
            raise AssertionError("Comparator completion traffic does not add")
    _record(checks, "comparator session ledger", {"sessions": len(rows), "methods": methods})


def _validate_figures(
    repository_root: Path,
    results_root: Path,
    config: dict,
    checks: list[dict[str, Any]],
) -> None:
    reference_results = (repository_root / config["paths"]["reference_results"]).resolve()
    if results_root == reference_results:
        directory = repository_root / "figures" / "paper"
    else:
        directory = repository_root / config["paths"]["generated_figures"]
    expected = [
        "Figure_1_System_Model.pdf",
        "Figure_2_Statistical_Validation.pdf",
        "Figure_3_Search_Audit_Allocation.pdf",
        "Figure_4_Primary_Performance.pdf",
        "Figure_5_Protected_Method_Tradeoff.pdf",
        "Figure_6_Evaluation_Inputs.pdf",
        "Figure_7_Cache_Query_Stress.pdf",
    ]
    for filename in expected:
        path = directory / filename
        if not path.is_file() or path.stat().st_size <= 1_000:
            raise AssertionError(f"Missing or empty manuscript figure: {filename}")
        if len(PdfReader(str(path)).pages) != 1:
            raise AssertionError(f"Manuscript figure must be a single-page PDF: {filename}")
    _record(
        checks,
        "manuscript figures",
        {
            "directory": directory.relative_to(repository_root).as_posix(),
            "pdf_files": len(expected),
        },
    )


def _validate_statistical_records(
    directory: Path,
    expected: dict,
    checks: list[dict[str, Any]],
) -> None:
    selection = json.loads((directory / "selection_stress.json").read_text())
    counts = expected["selection_bias_counts"]
    if len(selection) != 3:
        raise AssertionError("Selection-bias experiment must contain three search widths")
    for row in selection:
        candidates = str(int(row["candidates"]))
        trials = int(row["trials"])
        same = int(round(float(row["naive_reuse_false_accept_rate"]) * trials))
        fresh = int(round(float(row["fresh_false_accept_rate"]) * trials))
        if same != int(counts["same_evidence"][candidates]):
            raise AssertionError("Same-evidence boundary count changed")
        if fresh != int(counts["fresh_audit"][candidates]):
            raise AssertionError("Fresh-audit boundary count changed")
    _record(checks, "selection-bias experiment", {"counts": counts})

    sequential = json.loads((directory / "null_validation.json").read_text())
    for row in sequential:
        key = str(float(row["bsc"]))
        if int(row["familywise_false_accepts"]) != int(expected["sequential_null_counts"][key]):
            raise AssertionError("Sequential-null false-accept count changed")
    _record(checks, "sequential-null experiment", {"rows": len(sequential)})


def verify_repository(
    repository_root: Path,
    config_path: Path,
    results_root: Path | None = None,
    inputs: Path | None = None,
) -> dict:
    """Validate repository integrity, experiment ledgers, and paper-level values."""

    repository_root = Path(repository_root).resolve()
    config = load_experiment_config(config_path)
    inputs = Path(inputs or repository_root / config["paths"]["frozen_inputs"]).resolve()
    results_root = Path(
        results_root or repository_root / config["paths"]["reference_results"]
    ).resolve()
    checks: list[dict[str, Any]] = []

    embedding_path = inputs / "frozen_embeddings.npz"
    calibration_path = inputs / "calibration.json"
    reference_inputs = (repository_root / config["paths"]["frozen_inputs"]).resolve()
    if inputs == reference_inputs:
        if sha256_file(embedding_path) != config["expected"]["frozen_embeddings_sha256"]:
            raise AssertionError("Frozen embedding archive hash changed")
        if sha256_file(calibration_path) != config["expected"]["calibration_sha256"]:
            raise AssertionError("Frozen calibration record hash changed")
    arrays = load_frozen_embeddings(embedding_path)
    _record(
        checks, "frozen embedding schema", {"arrays": len(arrays), "archive": str(embedding_path)}
    )

    tune_ids = set(arrays["cal_tune_target_ids"].astype(int).tolist())
    validation_ids = set(arrays["cal_validate_target_ids"].astype(int).tolist())
    evaluation_ids = set(arrays["final_bank_ids"].astype(int).tolist())
    if tune_ids & validation_ids or (tune_ids | validation_ids) & evaluation_ids:
        raise AssertionError("Calibration and evaluation source identities are not disjoint")
    _record(
        checks,
        "identity separation",
        {
            "cal_tune": len(tune_ids),
            "cal_validate": len(validation_ids),
            "evaluation": len(evaluation_ids),
        },
    )

    calibration = json.loads(calibration_path.read_text())
    threshold = float(calibration["threshold_cosine"])
    _close(threshold, 0.7717004776000975)
    if (
        calibration["cal_tune"]["selected_pairs"] != 305
        or calibration["cal_validate"]["selected_pairs"] != 288
    ):
        raise AssertionError("Calibration selected-pair counts changed")
    _record(
        checks, "threshold calibration", {"threshold": threshold, "tune": 305, "validation": 288}
    )

    split_indices = json.loads((inputs / "split_indices.json").read_text())
    split_map = {
        "development_cache": "development_bank_ids",
        "calibration_tune": "cal_tune_target_ids",
        "calibration_validate": "cal_validate_target_ids",
        "evaluation_cache": "final_bank_ids",
        "standard_queries": "paired_iid_target_ids",
        "strong_queries": "paired_hard_ood_target_ids",
        "novel_queries": "novel_unpaired_target_ids",
    }
    for name, array_name in split_map.items():
        if split_indices["splits"][name]["source_ids"] != arrays[array_name].astype(int).tolist():
            raise AssertionError(f"Split index mismatch: {name}")
    _record(checks, "source identity manifest", {"splits": len(split_map)})

    traffic = fixed_audit_traffic(256, 768)
    if traffic.protocol_bits != 1944 or REFRESH_BITS != 4272:
        raise AssertionError("Traffic constants changed")
    _record(checks, "traffic accounting", {"protocol_bits": 1944, "refresh_bits": 4272})

    for probability in (0.0, 0.05):
        q0 = float(observed_mismatch_probability(threshold, probability))
        boundary = binomial_boundary(768, q0, 0.01)
        key = str(probability)
        if boundary != int(config["expected"]["audit_boundaries"][key]):
            raise AssertionError("Fixed-audit boundary changed")
        _close(
            binomial_cdf(boundary, 768, q0),
            config["expected"]["audit_boundary_null_probabilities"][key],
        )
    _record(checks, "fixed-audit boundaries", config["expected"]["audit_boundaries"])

    _validate_statistical_records(
        _resolve_validation_dir(results_root, inputs), config["expected"], checks
    )
    _validate_allocation(results_root, config, checks)
    _validate_primary(results_root, config["expected"], checks)
    _validate_comparators(results_root, checks)

    aggregate_cells = 0
    aggregate_cells += _validate_seed_aggregates(
        results_root / "allocation" / "cal_tune_ratio_seed_results.csv",
        results_root / "allocation" / "cal_tune_ratio_aggregate.csv",
        ("phase", "bsc", "search_bits", "audit_bits"),
        ("reuse_rate", "bad_reuse_rate", "identity_precision_among_reuses", "mean_completion_bits"),
    )
    aggregate_cells += _validate_seed_aggregates(
        results_root / "comparators" / "publication_seed_summary.csv",
        results_root / "comparators" / "publication_aggregate_summary.csv",
        ("bsc", "method"),
        (
            "reuse_rate",
            "bad_reuse_rate",
            "candidate_identity_rate_among_reuses",
            "mean_completion_bits",
        ),
    )
    aggregate_cells += _validate_seed_aggregates(
        results_root / "stress" / "sentrysem_fixed_seed_summary.csv",
        results_root / "stress" / "sentrysem_fixed_aggregate.csv",
        ("scenario", "condition", "bsc"),
        ("reuse_rate", "bad_reuse_rate", "identity_precision_among_reuses", "mean_completion_bits"),
    )
    _record(checks, "seed aggregate consistency", {"statistics_recomputed": aggregate_cells})

    with np.load(inputs / "qualitative_examples.npz", allow_pickle=False) as qualitative:
        if qualitative["class_ids"].astype(int).tolist() != [0, 3, 9]:
            raise AssertionError("Qualitative source selection changed")
        _close(float(qualitative["standard_similarity"][0]), 0.9327563047409058, 1e-9)
    _record(checks, "qualitative figure inputs", {"classes": [0, 3, 9]})

    _validate_figures(repository_root, results_root, config, checks)

    release_files = verify_release_manifest(repository_root)
    _record(checks, "release manifest", {"files_verified": release_files})

    return {
        "status": "PASS",
        "checks_passed": len(checks),
        "results_root": str(results_root),
        "inputs": str(inputs),
        "checks": checks,
    }
