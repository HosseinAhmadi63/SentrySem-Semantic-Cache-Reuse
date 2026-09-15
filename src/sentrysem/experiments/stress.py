#!/usr/bin/env python3
"""Evaluate SentrySem-Fixed (256/768) under cache and query shift."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sentrysem-stress-mpl"))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from sentrysem.experiments import comparators as core

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=core.DEFAULT_INPUT_DIR)
    parser.add_argument(
        "--selection",
        type=Path,
        default=REPOSITORY_ROOT / "results" / "paper" / "allocation" / "selection.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def stratified_half(labels, seed):
    rng = np.random.default_rng(seed)
    selected = []
    for label in sorted(np.unique(labels)):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        selected.extend(indices[: len(indices) // 2].tolist())
    return np.asarray(sorted(selected), dtype=np.int32)


def csv_write(path, rows):
    rows = list(rows)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    groups = {}
    for row in rows:
        key = (row["scenario"], row["condition"], row["bsc"], row["protocol_seed"])
        groups.setdefault(key, []).append(row)
    seed_rows = []
    for (scenario, condition, probability, seed), group in sorted(groups.items()):
        present = np.asarray([r["target_present"] for r in group], bool)
        reuse = np.asarray([r["cache_reuse"] for r in group], bool)
        identity = np.asarray([r["identity_match"] for r in group], bool)
        bad = np.asarray([r["bad_reuse"] for r in group], bool)
        seed_rows.append(
            {
                "scenario": scenario,
                "condition": condition,
                "bsc": probability,
                "protocol_seed": seed,
                "sessions": len(group),
                "target_present_fraction": float(present.mean()),
                "reuse_rate": float(reuse.mean()),
                "bad_reuse_rate": float(bad.mean()),
                "identity_precision_among_reuses": float(identity[reuse].mean())
                if reuse.any()
                else None,
                "reuse_rate_when_target_present": float(reuse[present].mean())
                if present.any()
                else None,
                "identity_recovery_rate_when_present": float((reuse & identity)[present].mean())
                if present.any()
                else None,
                "reuse_rate_when_target_absent": float(reuse[~present].mean())
                if (~present).any()
                else None,
                "mean_completion_bits": float(np.mean([r["completion_bits"] for r in group])),
                "saving_vs_always_refresh": float(
                    1.0 - np.mean([r["completion_bits"] for r in group]) / core.FALLBACK_BITS
                ),
            }
        )
    metrics = [
        key
        for key in seed_rows[0]
        if key not in {"scenario", "condition", "bsc", "protocol_seed", "sessions"}
    ]
    aggregate = []
    keys = sorted({(r["scenario"], r["condition"], r["bsc"]) for r in seed_rows})
    for scenario, condition, probability in keys:
        group = [
            r
            for r in seed_rows
            if (r["scenario"], r["condition"], r["bsc"]) == (scenario, condition, probability)
        ]
        item = {
            "scenario": scenario,
            "condition": condition,
            "bsc": probability,
            "protocol_seeds": len(group),
            "independent_query_identities": group[0]["sessions"],
        }
        for metric in metrics:
            values = np.asarray([r[metric] for r in group if r[metric] is not None], float)
            item[f"{metric}_mean"] = float(values.mean()) if len(values) else None
            item[f"{metric}_seed_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            item[f"{metric}_seed_min"] = float(values.min()) if len(values) else None
            item[f"{metric}_seed_max"] = float(values.max()) if len(values) else None
        aggregate.append(item)
    return seed_rows, aggregate


def cluster_bootstrap(rows, trials=2000):
    rng = np.random.default_rng(929331)
    reports = []
    keys = sorted({(r["scenario"], r["condition"], r["bsc"]) for r in rows})
    for scenario, condition, probability in keys:
        group = [
            r
            for r in rows
            if (r["scenario"], r["condition"], r["bsc"]) == (scenario, condition, probability)
        ]
        seeds = sorted({r["protocol_seed"] for r in group})
        query_indices = sorted({r["query_index"] for r in group})
        lookup = {(r["protocol_seed"], r["query_index"]): r for r in group}
        shape = (len(seeds), len(query_indices))
        arrays = {}
        for field in (
            "target_present",
            "cache_reuse",
            "identity_match",
            "bad_reuse",
            "completion_bits",
        ):
            dtype = float if field == "completion_bits" else bool
            arrays[field] = np.asarray(
                [[lookup[(seed, q)][field] for q in query_indices] for seed in seeds], dtype=dtype
            )
            if arrays[field].shape != shape:
                raise AssertionError(field)
        draws = {
            name: []
            for name in (
                "reuse_rate",
                "bad_reuse_rate",
                "identity_precision_among_reuses",
                "identity_recovery_rate_when_present",
                "reuse_rate_when_target_absent",
                "mean_completion_bits",
            )
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
        report = {
            "scenario": scenario,
            "condition": condition,
            "bsc": probability,
            "protocol_seeds_retained_per_identity": len(seeds),
            "query_identity_clusters": len(query_indices),
            "bootstrap_trials": trials,
        }
        for metric, values in draws.items():
            values = np.asarray(values, float)
            values = values[np.isfinite(values)]
            report[f"{metric}_cluster_bootstrap95_low"] = (
                float(np.quantile(values, 0.025)) if len(values) else None
            )
            report[f"{metric}_cluster_bootstrap95_high"] = (
                float(np.quantile(values, 0.975)) if len(values) else None
            )
        reports.append(report)
    return reports


def make_plot(aggregate, output):
    labels_present = [
        ("full", "paired_iid", "Full IID reuse", "reuse_rate"),
        (
            "evict50",
            "paired_iid",
            "Evict50 identity recovery",
            "identity_recovery_rate_when_present",
        ),
        (
            "missing3classes",
            "paired_iid",
            "Missing-class identity recovery",
            "identity_recovery_rate_when_present",
        ),
        ("full", "paired_hard_ood", "Hard-shift reuse", "reuse_rate"),
    ]
    labels_absent = [
        ("evict50", "paired_iid", "Evict50 absent reuse", "reuse_rate_when_target_absent"),
        (
            "missing3classes",
            "paired_iid",
            "Missing-class absent reuse",
            "reuse_rate_when_target_absent",
        ),
        ("full", "novel_unpaired", "Novel-image reuse", "reuse_rate"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, definitions, title in [
        (axes[0], labels_present, "Target-present recovery / reuse"),
        (axes[1], labels_absent, "Reuse when target is absent"),
    ]:
        x = np.arange(len(definitions))
        width = 0.36
        for offset, probability, color in [
            (-width / 2, 0.0, "#2574a9"),
            (width / 2, 0.05, "#e07b2d"),
        ]:
            values = []
            for scenario, condition, _, metric in definitions:
                row = next(
                    r
                    for r in aggregate
                    if r["scenario"] == scenario
                    and r["condition"] == condition
                    and r["bsc"] == probability
                )
                values.append(row[f"{metric}_mean"])
            ax.bar(x + offset, values, width, color=color, label=f"BSC={probability:.2f}")
        ax.set_xticks(x, [d[2] for d in definitions], rotation=18, ha="right")
        ax.set_ylabel("rate")
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "sentrysem_fixed_stress.png", dpi=220)
    fig.savefig(output / "sentrysem_fixed_stress.pdf")
    plt.close(fig)


def main(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    selection = json.loads(args.selection.read_text())
    if not selection["protocol_seed_sets_disjoint"]:
        raise AssertionError("selection/evaluation seed sets are not disjoint")
    seeds = list(map(int, selection["evaluation_protocol_seeds"]))
    search_bits = int(selection["selected_search_bits"])
    audit_bits = int(selection["selected_audit_bits"])
    if (search_bits, audit_bits) != (256, 768):
        raise AssertionError("unexpected frozen selected ratio")

    embedding_path = args.input_dir / "frozen_embeddings.npz"
    arrays = np.load(embedding_path)
    calibration_path = args.input_dir / "calibration.json"
    core.THRESHOLD = float(json.loads(calibration_path.read_text())["threshold_cosine"])
    bank = np.asarray(arrays["final_bank"], np.float32)
    bank_labels = np.asarray(arrays["final_bank_labels"], np.int64)
    bank_ids = np.asarray(arrays["final_bank_ids"], np.int64)
    cache_indices = {
        "full": np.arange(len(bank), dtype=np.int32),
        "evict50": stratified_half(bank_labels, 20260908),
        "missing3classes": np.flatnonzero(~np.isin(bank_labels, [0, 1, 2])).astype(np.int32),
    }
    conditions = {
        "paired_iid": (
            np.asarray(arrays["paired_iid_vectors"], np.float32),
            np.asarray(arrays["paired_iid_target_ids"], np.int64),
        ),
        "paired_hard_ood": (
            np.asarray(arrays["paired_hard_ood_vectors"], np.float32),
            np.asarray(arrays["paired_hard_ood_target_ids"], np.int64),
        ),
        "novel_unpaired": (
            np.asarray(arrays["novel_unpaired_vectors"], np.float32),
            np.asarray(arrays["novel_unpaired_target_ids"], np.int64),
        ),
    }
    tasks = [
        ("full", "paired_iid"),
        ("evict50", "paired_iid"),
        ("missing3classes", "paired_iid"),
        ("full", "paired_hard_ood"),
        ("full", "novel_unpaired"),
    ]
    protocol_bits = (
        core.proposal_frame_bits(search_bits)
        + core.COMMIT_BITS
        + core.SEED_BITS
        + core.proposal_frame_bits(audit_bits)
        + core.FEEDBACK_BITS
    )
    if protocol_bits != 1944:
        raise AssertionError(protocol_bits)
    q0_by_channel = {p: float(core.observed_q(core.THRESHOLD, p)) for p in (0.0, 0.05)}
    boundary_by_channel = {
        p: core.binomial_boundary(audit_bits, q0_by_channel[p], core.TOTAL_ALPHA)
        for p in q0_by_channel
    }
    rows = []
    for seed in seeds:
        plane_rng = np.random.default_rng(np.random.SeedSequence([seed, 100]))
        # The frozen evaluation call draws only its selected 256 proposal rows;
        # unlike CAL-TUNE's seven-ratio sweep it does not allocate a 896-row master.
        planes = plane_rng.standard_normal((search_bits, 512)).astype(np.float32)
        for scenario, condition in tasks:
            query, target_ids = conditions[condition]
            ids = cache_indices[scenario]
            cache, original_ids = bank[ids], bank_ids[ids]
            bank_signs = cache @ planes.T >= 0.0
            query_signs = query @ planes.T >= 0.0
            for probability in (0.0, 0.05):
                channel_rng = np.random.default_rng(
                    np.random.SeedSequence([seed, int(round(1000 * probability)), 5200])
                )
                received = np.bitwise_xor(
                    query_signs, channel_rng.random(query_signs.shape) < probability
                )
                distances = core.hamming_matrix(
                    bank_signs[:, :search_bits], received[:, :search_bits]
                )
                candidate = np.argmin(distances, axis=1).astype(np.int32)
                cosine = core.cosine_for(query, cache, candidate)
                audit_rng = np.random.default_rng(
                    np.random.SeedSequence([seed, int(round(1000 * probability)), 5300])
                )
                uniform = audit_rng.random((len(query), audit_bits))
                mismatch = np.count_nonzero(
                    uniform < core.observed_q(cosine, probability)[:, None], axis=1
                )
                reuse = mismatch <= boundary_by_channel[probability]
                target_present = np.isin(target_ids, original_ids)
                identity = original_ids[candidate] == target_ids
                good = cosine > core.THRESHOLD
                for index in range(len(query)):
                    rows.append(
                        {
                            "scenario": scenario,
                            "condition": condition,
                            "protocol_seed": seed,
                            "bsc": probability,
                            "query_index": index,
                            "target_id": int(target_ids[index]),
                            "target_present": int(target_present[index]),
                            "candidate_original_id": int(original_ids[candidate[index]]),
                            "candidate_cosine": float(cosine[index]),
                            "cache_reuse": int(reuse[index]),
                            "verifier_good": int(good[index]),
                            "bad_reuse": int(reuse[index] and not good[index]),
                            "identity_match": int(identity[index]),
                            "audit_mismatches": int(mismatch[index]),
                            "audit_boundary": int(boundary_by_channel[probability]),
                            "protocol_bits": protocol_bits,
                            "fallback_bits": int(0 if reuse[index] else core.FALLBACK_BITS),
                            "completion_bits": int(
                                protocol_bits + (0 if reuse[index] else core.FALLBACK_BITS)
                            ),
                        }
                    )
                print(f"seed={seed} {scenario}/{condition} p={probability:.2f} PASS")

    seed_summary, aggregate = summarize(rows)
    bootstrap = cluster_bootstrap(rows)
    csv_write(args.output / "sentrysem_fixed_session_results.csv", rows)
    csv_write(args.output / "sentrysem_fixed_seed_summary.csv", seed_summary)
    csv_write(args.output / "sentrysem_fixed_aggregate.csv", aggregate)
    csv_write(args.output / "sentrysem_fixed_cluster_bootstrap.csv", bootstrap)
    (args.output / "sentrysem_fixed_aggregate.json").write_text(json.dumps(aggregate, indent=2))
    (args.output / "sentrysem_fixed_cluster_bootstrap.json").write_text(
        json.dumps(bootstrap, indent=2)
    )
    config = {
        "input_artifact_directory": args.input_dir.name,
        "input_embeddings_sha256": core.sha256(embedding_path),
        "input_calibration_sha256": core.sha256(calibration_path),
        "selection_sha256": core.sha256(args.selection),
        "script_sha256": core.sha256(Path(__file__)),
        "selected_ratio": [search_bits, audit_bits],
        "protocol_bits": protocol_bits,
        "evaluation_protocol_seeds": seeds,
        "conditions": tasks,
        "cache_sizes": {key: len(value) for key, value in cache_indices.items()},
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "traffic_scope": "bidirectional logical framed bits with reliable control and refresh",
        "semantic_endpoint": "frozen normalized ResNet-18 feature",
    }
    (args.output / "configuration.json").write_text(json.dumps(config, indent=2))
    reference_path = args.selection.parent / "selected_ratio_final_aggregate.json"
    reference = json.loads(reference_path.read_text())
    exact_match = []
    for probability in (0.0, 0.05):
        observed = next(
            r
            for r in aggregate
            if r["scenario"] == "full"
            and r["condition"] == "paired_iid"
            and r["bsc"] == probability
        )
        expected = next(r for r in reference if r["bsc"] == probability)
        pairs = {
            "reuse_rate": (observed["reuse_rate_mean"], expected["reuse_rate_mean"]),
            "bad_reuse_rate": (observed["bad_reuse_rate_mean"], expected["bad_reuse_rate_mean"]),
            "identity_precision_among_reuses": (
                observed["identity_precision_among_reuses_mean"],
                expected["identity_precision_among_reuses_mean"],
            ),
            "mean_completion_bits": (
                observed["mean_completion_bits_mean"],
                expected["mean_completion_bits_mean"],
            ),
        }
        if not all(np.isclose(a, b, rtol=0, atol=1e-12) for a, b in pairs.values()):
            raise AssertionError({"bsc": probability, "differences": pairs})
        exact_match.append({"bsc": probability, "metrics": pairs, "pass": True})
    (args.output / "headline_exact_match.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "reference": reference_path.name,
                "same_seed_realizations": True,
                "note": (
                    "Both evaluation paths draw exactly 256 proposal/channel values "
                    "and 768 audit uniforms per query for the frozen selected ratio."
                ),
                "checks": exact_match,
            },
            indent=2,
        )
    )
    make_plot(aggregate, args.output)
    manifest = {
        path.name: {"bytes": path.stat().st_size, "sha256": core.sha256(path)}
        for path in sorted(args.output.iterdir())
        if path.is_file()
    }
    (args.output / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("PASS", args.output)


if __name__ == "__main__":
    main(arguments())
