#!/usr/bin/env python3
"""Select the fixed 1,024-sign SentrySem proposal/audit allocation.

Seven search/audit allocations are compared on CAL-TUNE against the development
cache.  A deterministic rule freezes the allocation with the smallest average
fallback-inclusive cost across BSC p=0 and p=.05.  Only that one frozen ratio is
then evaluated on the calibration-disjoint paired-IID evaluation pool.
"""

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
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sentrysem-allocation-mpl"))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from sentrysem.experiments import comparators as core

RATIOS = [(search, 1024 - search) for search in range(128, 897, 128)]
PROTOCOL_BITS = (
    1024 + 2 * core.FRAME_OVERHEAD_BITS + core.COMMIT_BITS + core.SEED_BITS + core.FEEDBACK_BITS
)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=core.DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=20260907)
    return parser.parse_args()


def write_csv(path: Path, rows):
    rows = list(rows)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def evaluate_phase(
    phase: str,
    query: np.ndarray,
    target_ids: np.ndarray,
    bank: np.ndarray,
    bank_ids: np.ndarray,
    ratios,
    seeds,
):
    rows = []
    for seed in seeds:
        plane_rng = np.random.default_rng(np.random.SeedSequence([seed, 100]))
        planes = plane_rng.standard_normal((max(s for s, _ in ratios), 512)).astype(np.float32)
        bank_signs = bank @ planes.T >= 0.0
        query_signs = query @ planes.T >= 0.0
        for bsc_probability in (0.0, 0.05):
            channel_rng = np.random.default_rng(
                np.random.SeedSequence([seed, int(round(1000 * bsc_probability)), 5200])
            )
            received = np.bitwise_xor(
                query_signs, channel_rng.random(query_signs.shape) < bsc_probability
            )
            audit_rng = np.random.default_rng(
                np.random.SeedSequence([seed, int(round(1000 * bsc_probability)), 5300])
            )
            shared_uniform = audit_rng.random((len(query), max(a for _, a in ratios)))
            q0 = float(core.observed_q(core.THRESHOLD, bsc_probability))
            for search_bits, audit_bits in ratios:
                distances = core.hamming_matrix(
                    bank_signs[:, :search_bits], received[:, :search_bits]
                )
                candidate = np.argmin(distances, axis=1).astype(np.int32)
                cosine = core.cosine_for(query, bank, candidate)
                mismatch = np.count_nonzero(
                    shared_uniform[:, :audit_bits]
                    < core.observed_q(cosine, bsc_probability)[:, None],
                    axis=1,
                )
                boundary = core.binomial_boundary(audit_bits, q0, core.TOTAL_ALPHA)
                reuse = mismatch <= boundary
                identity = bank_ids[candidate] == target_ids
                good = cosine > core.THRESHOLD
                row = {
                    "phase": phase,
                    "protocol_seed": int(seed),
                    "bsc": bsc_probability,
                    "search_bits": search_bits,
                    "audit_bits": audit_bits,
                    "sessions": len(query),
                    "audit_boundary": boundary,
                    "exact_boundary_null_probability": core.binomial_cdf(boundary, audit_bits, q0),
                    "reuse_rate": float(reuse.mean()),
                    "bad_reuse_rate": float(np.mean(reuse & ~good)),
                    "identity_precision_among_reuses": (
                        float(identity[reuse].mean()) if reuse.any() else None
                    ),
                    "mean_protocol_bits": float(PROTOCOL_BITS),
                    "mean_completion_bits": float(
                        PROTOCOL_BITS + (~reuse).mean() * core.FALLBACK_BITS
                    ),
                    "saving_vs_always_refresh": float(
                        1.0
                        - (PROTOCOL_BITS + (~reuse).mean() * core.FALLBACK_BITS)
                        / core.FALLBACK_BITS
                    ),
                }
                if row["exact_boundary_null_probability"] > core.TOTAL_ALPHA + 1e-12:
                    raise AssertionError(row)
                rows.append(row)
            print(f"{phase}: seed={seed}, BSC={bsc_probability:.2f} PASS")
    return rows


def aggregate(rows):
    result = []
    keys = sorted({(r["phase"], r["bsc"], r["search_bits"], r["audit_bits"]) for r in rows})
    for phase, bsc_probability, search_bits, audit_bits in keys:
        group = [
            r
            for r in rows
            if (r["phase"], r["bsc"], r["search_bits"], r["audit_bits"])
            == (phase, bsc_probability, search_bits, audit_bits)
        ]
        item = {
            "phase": phase,
            "bsc": bsc_probability,
            "search_bits": search_bits,
            "audit_bits": audit_bits,
            "protocol_seeds": len(group),
        }
        for metric in (
            "reuse_rate",
            "bad_reuse_rate",
            "identity_precision_among_reuses",
            "mean_protocol_bits",
            "mean_completion_bits",
            "saving_vs_always_refresh",
        ):
            values = np.asarray([r[metric] for r in group if r[metric] is not None], float)
            item[f"{metric}_mean"] = float(values.mean()) if len(values) else None
            item[f"{metric}_seed_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            item[f"{metric}_seed_min"] = float(values.min()) if len(values) else None
            item[f"{metric}_seed_max"] = float(values.max()) if len(values) else None
        result.append(item)
    return result


def cluster_bootstrap_final(session_rows, trials=2000):
    seeds = sorted({r["protocol_seed"] for r in session_rows})
    # This compact selection experiment stores summaries, so recover a conservative
    # identity-cluster interval for mean completion cost from the previously generated
    # publication session file in the caller package is intentionally avoided.  Seed
    # min/max and SD remain the primary uncertainty report for this tuning check.
    return {
        "trials": 0,
        "protocol_seeds": len(seeds),
        "note": "Selected-ratio uncertainty is reported as seed SD/min/max; no identities are pooled.",
    }


def plot_tuning(tune_aggregate, selected, output):
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    for probability, color in [(0.0, "#2070b4"), (0.05, "#e27a24")]:
        rows = sorted(
            [r for r in tune_aggregate if r["bsc"] == probability], key=lambda r: r["search_bits"]
        )
        x = [r["search_bits"] for r in rows]
        axes[0].plot(
            x,
            [r["mean_completion_bits_mean"] for r in rows],
            "o-",
            color=color,
            label=f"BSC={probability:.2f}",
        )
        axes[1].plot(
            x,
            [r["reuse_rate_mean"] for r in rows],
            "o-",
            color=color,
            label=f"BSC={probability:.2f}",
        )
    for ax in axes:
        ax.axvline(
            selected[0],
            color="black",
            linestyle="--",
            linewidth=1,
            label=f"selected {selected[0]}+{selected[1]}",
        )
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        ax.set_xlabel("proposal/search bits (total source signs = 1024)")
    axes[0].set_ylabel("mean completion bits on CAL-TUNE")
    axes[1].set_ylabel("reuse rate on CAL-TUNE")
    fig.tight_layout()
    fig.savefig(output / "sentrysem_allocation_tuning.png", dpi=220)
    fig.savefig(output / "sentrysem_allocation_tuning.pdf")
    plt.close(fig)


def main(args):
    if args.seeds < 2:
        raise ValueError("at least two seeds required")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    arrays = np.load(args.input_dir / "frozen_embeddings.npz")
    calibration = json.loads((args.input_dir / "calibration.json").read_text())
    core.THRESHOLD = float(calibration["threshold_cosine"])
    children = np.random.SeedSequence(args.base_seed).spawn(2 * args.seeds)
    tune_seeds = [
        int(child.generate_state(1, dtype=np.uint64)[0]) for child in children[: args.seeds]
    ]
    final_seeds = [
        int(child.generate_state(1, dtype=np.uint64)[0]) for child in children[args.seeds :]
    ]
    if set(tune_seeds) & set(final_seeds):
        raise AssertionError("tuning and evaluation protocol seeds must be disjoint")

    development_bank = np.asarray(arrays["development_bank"], np.float32)
    development_ids = np.asarray(arrays["development_bank_ids"], np.int64)
    tune_query = np.asarray(arrays["cal_tune"], np.float32)
    tune_ids = np.asarray(arrays["cal_tune_target_ids"], np.int64)
    if not np.isin(tune_ids, development_ids).all():
        raise AssertionError("CAL-TUNE targets missing from development cache")
    tune_rows = evaluate_phase(
        "cal_tune", tune_query, tune_ids, development_bank, development_ids, RATIOS, tune_seeds
    )
    tune_aggregate = aggregate(tune_rows)

    # Predeclared deterministic rule: minimize average cost over the two channel
    # values; break exact ties by proximity to balanced and then fewer search bits.
    candidates = []
    for ratio in RATIOS:
        matching = [r for r in tune_aggregate if (r["search_bits"], r["audit_bits"]) == ratio]
        mean_cost = float(np.mean([r["mean_completion_bits_mean"] for r in matching]))
        candidates.append((mean_cost, abs(ratio[0] - ratio[1]), ratio[0], ratio))
    selected_cost, _, _, selected = min(candidates)

    final_bank = np.asarray(arrays["final_bank"], np.float32)
    final_ids = np.asarray(arrays["final_bank_ids"], np.int64)
    final_query = np.asarray(arrays["paired_iid_vectors"], np.float32)
    final_target_ids = np.asarray(arrays["paired_iid_target_ids"], np.int64)
    final_rows = evaluate_phase(
        "calibration_disjoint_evaluation",
        final_query,
        final_target_ids,
        final_bank,
        final_ids,
        [selected],
        final_seeds,
    )
    final_aggregate = aggregate(final_rows)

    selection = {
        "selection_data": "CAL-TUNE against development cache only",
        "evaluation_pool_used_for_selection": False,
        "candidate_ratios": [{"search_bits": s, "audit_bits": a} for s, a in RATIOS],
        "selection_rule": (
            "minimum mean fallback-inclusive logical bits averaged equally "
            "over BSC p=0 and p=.05; ties favor the most balanced split"
        ),
        "selected_search_bits": selected[0],
        "selected_audit_bits": selected[1],
        "selected_cal_tune_mean_cost_across_channels": selected_cost,
        "fixed_prefallback_protocol_bits": PROTOCOL_BITS,
        "cal_tune_protocol_seeds": tune_seeds,
        "evaluation_protocol_seeds": final_seeds,
        "protocol_seed_sets_disjoint": True,
        "formal_alpha": core.TOTAL_ALPHA,
        "common_semantic_refresh_bits": core.FALLBACK_BITS,
        "input_artifact_directory": args.input_dir.name,
        "input_embeddings_sha256": core.sha256(args.input_dir / "frozen_embeddings.npz"),
        "input_calibration_sha256": core.sha256(args.input_dir / "calibration.json"),
        "script_sha256": core.sha256(Path(__file__)),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    (args.output / "selection.json").write_text(json.dumps(selection, indent=2))
    write_csv(args.output / "cal_tune_ratio_seed_results.csv", tune_rows)
    write_csv(args.output / "cal_tune_ratio_aggregate.csv", tune_aggregate)
    write_csv(args.output / "selected_ratio_final_seed_results.csv", final_rows)
    write_csv(args.output / "selected_ratio_final_aggregate.csv", final_aggregate)
    (args.output / "cal_tune_ratio_aggregate.json").write_text(json.dumps(tune_aggregate, indent=2))
    (args.output / "selected_ratio_final_aggregate.json").write_text(
        json.dumps(final_aggregate, indent=2)
    )
    plot_tuning(tune_aggregate, selected, args.output)
    manifest = {
        path.name: {"bytes": path.stat().st_size, "sha256": core.sha256(path)}
        for path in sorted(args.output.iterdir())
        if path.is_file()
    }
    (args.output / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("SELECTED", selection)
    print("FINAL", json.dumps(final_aggregate, indent=2))


if __name__ == "__main__":
    main(arguments())
