"""Result serialization, aggregation, and identity-cluster uncertainty."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .hashing import file_manifest
from .traffic import REFRESH_BITS

PUBLICATION_METRICS = (
    "certification_rate",
    "direct_output_rate",
    "reuse_rate",
    "fallback_rate",
    "false_certification_rate",
    "bad_reuse_rate",
    "candidate_identity_rate_among_certified",
    "candidate_identity_rate_among_direct_outputs",
    "candidate_identity_rate_among_reuses",
    "candidate_good_rate_among_certified",
    "completion_verifier_good_rate",
    "service_completion_rate",
    "mean_protocol_bits",
    "mean_protocol_forward_bits",
    "mean_protocol_reverse_bits",
    "mean_fallback_bits",
    "mean_completion_forward_bits",
    "mean_completion_bits",
    "saving_vs_always_refresh",
    "mean_audit_bits",
    "mean_attempts",
    "mean_logical_round_trips",
)

PUBLICATION_METHOD_ORDER = (
    "proposal64_unverified",
    "screened512_heuristic",
    "bonferroni512",
    "split256_256",
    "screened1024_heuristic",
    "bonferroni1024",
    "split512_512",
    "sentrysem1",
    "sentrysem3",
    "int8_fallback",
)


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    return value


def write_json(path: str | Path, value: Any) -> None:
    """Write deterministic, UTF-8 JSON with NumPy values converted safely."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_json_value(value), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: str | Path) -> Any:
    """Read one UTF-8 JSON artifact."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write non-empty dictionaries to a UTF-8 CSV with a stable column order."""

    materialized = [dict(row) for row in rows]
    if not materialized:
        raise ValueError("cannot write an empty CSV")
    columns = list(materialized[0])
    if any(set(row) != set(columns) for row in materialized):
        raise ValueError("all CSV rows must have the same fields")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(materialized)


def _parse_csv_value(value: str) -> str | int | float | None:
    if value == "":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def read_csv(path: str | Path) -> list[dict[str, str | int | float | None]]:
    """Read a CSV artifact and restore empty, integer, and floating-point fields."""

    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return [
            {name: _parse_csv_value(value) for name, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def prepare_output_directory(path: str | Path) -> Path:
    """Create an output directory while refusing to overwrite prior results."""

    destination = Path(path)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def write_artifact_manifest(directory: str | Path, recursive: bool = False) -> Path:
    """Write file sizes and SHA-256 values for a completed result directory."""

    root = Path(directory)
    path = root / "artifact_manifest.json"
    manifest = file_manifest(root, recursive)
    manifest.pop(path.relative_to(root).as_posix(), None)
    write_json(path, manifest)
    return path


def summarize_group(rows: Sequence[Mapping[str, Any]]) -> dict[str, int | float | None]:
    """Summarize publication comparator sessions for one method and seed."""

    if not rows:
        raise ValueError("cannot summarize an empty group")
    certified = np.asarray([row["certified"] for row in rows], dtype=bool)
    direct = np.asarray([row["direct_output"] for row in rows], dtype=bool)
    reuse = np.asarray([row["cache_reuse"] for row in rows], dtype=bool)
    candidate_identity = np.asarray([row["candidate_identity_match"] for row in rows], dtype=bool)
    candidate_good = np.asarray([row["candidate_verifier_good"] for row in rows], dtype=bool)
    completion_bits = np.asarray([row["completion_bits"] for row in rows], dtype=np.float64)
    return {
        "sessions": len(rows),
        "certification_rate": float(certified.mean()),
        "direct_output_rate": float(direct.mean()),
        "reuse_rate": float(reuse.mean()),
        "fallback_rate": float(np.mean([row["fallback_used"] for row in rows])),
        "false_certification_rate": float(np.mean([row["false_certification"] for row in rows])),
        "bad_reuse_rate": float(np.mean([row["bad_reuse"] for row in rows])),
        "candidate_identity_rate_among_certified": (
            float(candidate_identity[certified].mean()) if certified.any() else None
        ),
        "candidate_identity_rate_among_direct_outputs": (
            float(candidate_identity[direct].mean()) if direct.any() else None
        ),
        "candidate_identity_rate_among_reuses": (
            float(candidate_identity[reuse].mean()) if reuse.any() else None
        ),
        "candidate_good_rate_among_certified": (
            float(candidate_good[certified].mean()) if certified.any() else None
        ),
        "completion_verifier_good_rate": float(
            np.mean([row["completion_verifier_good"] for row in rows])
        ),
        "service_completion_rate": 1.0,
        "mean_protocol_bits": float(np.mean([row["protocol_bits"] for row in rows])),
        "mean_protocol_forward_bits": float(
            np.mean([row["protocol_forward_bits"] for row in rows])
        ),
        "mean_protocol_reverse_bits": float(
            np.mean([row["protocol_reverse_bits"] for row in rows])
        ),
        "mean_fallback_bits": float(np.mean([row["fallback_bits"] for row in rows])),
        "mean_completion_forward_bits": float(
            np.mean([row["completion_forward_bits"] for row in rows])
        ),
        "mean_completion_bits": float(completion_bits.mean()),
        "saving_vs_always_refresh": float(1.0 - completion_bits.mean() / REFRESH_BITS),
        "mean_audit_bits": float(np.mean([row["audit_bits"] for row in rows])),
        "mean_attempts": float(np.mean([row["attempts"] for row in rows])),
        "mean_logical_round_trips": float(np.mean([row["logical_round_trips"] for row in rows])),
    }


def per_seed_summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Group comparator sessions by protocol seed, channel, and method."""

    groups: dict[tuple[int, float, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (int(row["protocol_seed"]), float(row["bsc"]), str(row["method"]))
        groups.setdefault(key, []).append(row)
    return [
        {
            "protocol_seed": seed,
            "bsc": probability,
            "method": method,
            **summarize_group(group),
        }
        for (seed, probability, method), group in sorted(groups.items())
    ]


def aggregate_summaries(seed_summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate seed-level comparator metrics with SD and observed range."""

    groups: dict[tuple[float, str], list[Mapping[str, Any]]] = {}
    for row in seed_summaries:
        groups.setdefault((float(row["bsc"]), str(row["method"])), []).append(row)
    aggregates: list[dict[str, Any]] = []
    for (probability, method), group in sorted(groups.items()):
        item: dict[str, Any] = {
            "bsc": probability,
            "method": method,
            "protocol_seeds": len(group),
        }
        for metric in PUBLICATION_METRICS:
            values = np.asarray(
                [row[metric] for row in group if row.get(metric) is not None], dtype=np.float64
            )
            item[f"{metric}_mean"] = float(values.mean()) if len(values) else None
            item[f"{metric}_seed_sd"] = (
                float(values.std(ddof=1)) if len(values) > 1 else (0.0 if len(values) else None)
            )
            item[f"{metric}_seed_min"] = float(values.min()) if len(values) else None
            item[f"{metric}_seed_max"] = float(values.max()) if len(values) else None
        aggregates.append(item)
    return aggregates


def identity_cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    trials: int = 2000,
    seed: int = 777311,
) -> list[dict[str, Any]]:
    """Resample query identities while retaining every paired protocol seed."""

    if trials <= 0:
        raise ValueError("trials must be positive")
    rng = np.random.default_rng(seed)
    reports: list[dict[str, Any]] = []
    present_keys = {(float(row["bsc"]), str(row["method"])) for row in rows}
    probabilities = sorted({key[0] for key in present_keys})
    listed_methods = [
        method
        for method in PUBLICATION_METHOD_ORDER
        if any(key[1] == method for key in present_keys)
    ]
    listed_methods.extend(sorted({key[1] for key in present_keys} - set(listed_methods)))
    keys = [
        (probability, method)
        for probability in probabilities
        for method in listed_methods
        if (probability, method) in present_keys
    ]
    for probability, method in keys:
        group = [
            row for row in rows if float(row["bsc"]) == probability and str(row["method"]) == method
        ]
        protocol_seeds = sorted({int(row["protocol_seed"]) for row in group})
        query_indices = sorted({int(row["query_index"]) for row in group})
        lookup = {(int(row["protocol_seed"]), int(row["query_index"])): row for row in group}
        expected = len(protocol_seeds) * len(query_indices)
        if len(lookup) != expected:
            raise ValueError(f"incomplete seed-by-identity grid for {method}, BSC={probability}")
        reuse = np.asarray(
            [
                [lookup[(value, query)]["cache_reuse"] for query in query_indices]
                for value in protocol_seeds
            ],
            dtype=bool,
        )
        bad = np.asarray(
            [
                [lookup[(value, query)]["bad_reuse"] for query in query_indices]
                for value in protocol_seeds
            ],
            dtype=bool,
        )
        identity = np.asarray(
            [
                [lookup[(value, query)]["candidate_identity_match"] for query in query_indices]
                for value in protocol_seeds
            ],
            dtype=bool,
        )
        bits = np.asarray(
            [
                [lookup[(value, query)]["completion_bits"] for query in query_indices]
                for value in protocol_seeds
            ],
            dtype=np.float64,
        )
        draws: dict[str, list[float]] = {
            "reuse_rate": [],
            "bad_reuse_rate": [],
            "identity_precision_among_reuses": [],
            "mean_completion_bits": [],
            "saving_vs_always_refresh": [],
        }
        for _ in range(trials):
            chosen = rng.integers(0, len(query_indices), len(query_indices))
            sampled_reuse = reuse[:, chosen]
            sampled_identity = identity[:, chosen]
            mean_bits = float(bits[:, chosen].mean())
            draws["reuse_rate"].append(float(sampled_reuse.mean()))
            draws["bad_reuse_rate"].append(float(bad[:, chosen].mean()))
            draws["identity_precision_among_reuses"].append(
                float(sampled_identity[sampled_reuse].mean()) if sampled_reuse.any() else np.nan
            )
            draws["mean_completion_bits"].append(mean_bits)
            draws["saving_vs_always_refresh"].append(1.0 - mean_bits / REFRESH_BITS)
        report: dict[str, Any] = {
            "bsc": probability,
            "method": method,
            "protocol_seeds_retained_per_identity": len(protocol_seeds),
            "query_identity_clusters": len(query_indices),
            "bootstrap_trials": trials,
        }
        for metric, samples in draws.items():
            values = np.asarray(samples, dtype=np.float64)
            values = values[np.isfinite(values)]
            report[f"{metric}_cluster_bootstrap95_low"] = (
                float(np.quantile(values, 0.025)) if len(values) else None
            )
            report[f"{metric}_cluster_bootstrap95_high"] = (
                float(np.quantile(values, 0.975)) if len(values) else None
            )
        reports.append(report)
    return reports


def stress_cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    trials: int = 2000,
    seed: int = 929331,
) -> list[dict[str, Any]]:
    """Compute cluster intervals for the cache/source stress evaluation."""

    if trials <= 0:
        raise ValueError("trials must be positive")
    rng = np.random.default_rng(seed)
    reports: list[dict[str, Any]] = []
    keys = sorted(
        {(str(row["scenario"]), str(row["condition"]), float(row["bsc"])) for row in rows}
    )
    for scenario, condition, probability in keys:
        group = [
            row
            for row in rows
            if (
                str(row["scenario"]),
                str(row["condition"]),
                float(row["bsc"]),
            )
            == (scenario, condition, probability)
        ]
        protocol_seeds = sorted({int(row["protocol_seed"]) for row in group})
        query_indices = sorted({int(row["query_index"]) for row in group})
        lookup = {(int(row["protocol_seed"]), int(row["query_index"])): row for row in group}
        shape = (len(protocol_seeds), len(query_indices))
        arrays: dict[str, np.ndarray] = {}
        for field in (
            "target_present",
            "cache_reuse",
            "identity_match",
            "bad_reuse",
            "completion_bits",
        ):
            dtype: Any = np.float64 if field == "completion_bits" else bool
            arrays[field] = np.asarray(
                [
                    [lookup[(value, query)][field] for query in query_indices]
                    for value in protocol_seeds
                ],
                dtype=dtype,
            )
            if arrays[field].shape != shape:
                raise ValueError(f"incomplete stress grid for {scenario}/{condition}")
        draws: dict[str, list[float]] = {
            "reuse_rate": [],
            "bad_reuse_rate": [],
            "identity_precision_among_reuses": [],
            "identity_recovery_rate_when_present": [],
            "reuse_rate_when_target_absent": [],
            "mean_completion_bits": [],
        }
        for _ in range(trials):
            chosen = rng.integers(0, len(query_indices), len(query_indices))
            present = arrays["target_present"][:, chosen]
            reuse = arrays["cache_reuse"][:, chosen]
            identity = arrays["identity_match"][:, chosen]
            draws["reuse_rate"].append(float(reuse.mean()))
            draws["bad_reuse_rate"].append(float(arrays["bad_reuse"][:, chosen].mean()))
            draws["identity_precision_among_reuses"].append(
                float(identity[reuse].mean()) if reuse.any() else np.nan
            )
            draws["identity_recovery_rate_when_present"].append(
                float((reuse & identity)[present].mean()) if present.any() else np.nan
            )
            draws["reuse_rate_when_target_absent"].append(
                float(reuse[~present].mean()) if (~present).any() else np.nan
            )
            draws["mean_completion_bits"].append(float(arrays["completion_bits"][:, chosen].mean()))
        report: dict[str, Any] = {
            "scenario": scenario,
            "condition": condition,
            "bsc": probability,
            "protocol_seeds_retained_per_identity": len(protocol_seeds),
            "query_identity_clusters": len(query_indices),
            "bootstrap_trials": trials,
        }
        for metric, samples in draws.items():
            values = np.asarray(samples, dtype=np.float64)
            values = values[np.isfinite(values)]
            report[f"{metric}_cluster_bootstrap95_low"] = (
                float(np.quantile(values, 0.025)) if len(values) else None
            )
            report[f"{metric}_cluster_bootstrap95_high"] = (
                float(np.quantile(values, 0.975)) if len(values) else None
            )
        reports.append(report)
    return reports
