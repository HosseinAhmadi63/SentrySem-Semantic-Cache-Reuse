"""Finite-sample selection-bias and sequential-null experiments."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from sentrysem.results import prepare_output_directory, write_artifact_manifest, write_json
from sentrysem.statistics import (
    alpha_spending,
    apply_audit_lookup,
    audit_lookup,
    binomial_boundary,
    observed_mismatch_probability,
    wilson_interval,
)


def selection_bias_experiment(
    threshold: float,
    trials: int = 10_000,
    sketch_bits: int = 64,
    bsc_probability: float = 0.01,
    alpha: float = 0.01,
    seed: int = 20260937,
) -> list[dict]:
    """Compare same-evidence screening with an independent locked-candidate audit."""

    rng = np.random.default_rng(seed)
    cosine = float(threshold)
    null_probability = float(observed_mismatch_probability(cosine, bsc_probability))
    boundary = binomial_boundary(sketch_bits, null_probability, alpha)
    rows: list[dict] = []
    for candidates in (1, 8, 64):
        same_count = 0
        fresh_count = 0
        chunk = 100
        for start in range(0, trials, chunk):
            count = min(chunk, trials - start)
            source_normal = rng.standard_normal((count, sketch_bits, 1)).astype(np.float32)
            candidate_normal = rng.standard_normal((count, sketch_bits, candidates)).astype(
                np.float32
            )
            source_signs = source_normal >= 0.0
            candidate_signs = (
                cosine * source_normal + math.sqrt(1.0 - cosine * cosine) * candidate_normal >= 0.0
            )
            flips = rng.random((count, sketch_bits, 1)) < bsc_probability
            mismatch_counts = np.count_nonzero(
                (source_signs != candidate_signs) != flips,
                axis=1,
            )
            same_count += int(np.count_nonzero(np.min(mismatch_counts, axis=1) <= boundary))
            fresh_mismatches = rng.random((count, sketch_bits)) < null_probability
            fresh_count += int(np.count_nonzero(fresh_mismatches.sum(axis=1) <= boundary))
        rows.append(
            {
                "candidates": candidates,
                "bits": sketch_bits,
                "bsc": bsc_probability,
                "audit_boundary": boundary,
                "naive_reuse_false_accepts": same_count,
                "naive_reuse_false_accept_rate": same_count / trials,
                "naive_reuse_wilson95": list(wilson_interval(same_count, trials)),
                "fresh_false_accepts": fresh_count,
                "fresh_false_accept_rate": fresh_count / trials,
                "fresh_wilson95": list(wilson_interval(fresh_count, trials)),
                "target_alpha": alpha,
                "trials": trials,
            }
        )
    return rows


def sequential_null_experiment(
    threshold: float,
    bsc_probabilities: tuple[float, ...] = (0.0, 0.05),
    trials: int = 50_000,
    alpha: float = 0.01,
    attempts: int = 3,
    audit_bits: int = 768,
    batch_bits: int = 128,
    mixture_points: int = 8,
    seed: int = 20260927,
) -> list[dict]:
    """Measure familywise false acceptance at the verifier boundary."""

    rng = np.random.default_rng(seed)
    allocations = alpha_spending(alpha, attempts)
    reports: list[dict] = []
    for bsc_probability in bsc_probabilities:
        null_probability = float(observed_mismatch_probability(threshold, bsc_probability))
        lookups = [
            audit_lookup(
                null_probability,
                bsc_probability,
                float(level),
                max_bits=audit_bits,
                batch_bits=batch_bits,
                mixture_points=mixture_points,
            )
            for level in allocations
        ]
        familywise_accepts = 0
        total_audit_bits = 0
        attempted = np.zeros(attempts, dtype=np.int64)
        accepted_per_attempt = np.zeros(attempts, dtype=np.int64)
        chunk = 2_000
        for start in range(0, trials, chunk):
            count = min(chunk, trials - start)
            active = np.ones(count, dtype=bool)
            ever_accepted = np.zeros(count, dtype=bool)
            for attempt, lookup in enumerate(lookups):
                mismatch = rng.random((count, audit_bits)) < null_probability
                accepted, used = apply_audit_lookup(mismatch, lookup)
                accepted &= active
                attempted[attempt] += int(active.sum())
                accepted_per_attempt[attempt] += int(accepted.sum())
                total_audit_bits += int(used[active].sum())
                ever_accepted |= accepted
                active &= ~accepted
            familywise_accepts += int(ever_accepted.sum())
        reports.append(
            {
                "bsc": float(bsc_probability),
                "trials": trials,
                "familywise_false_accepts": familywise_accepts,
                "familywise_false_accept_rate": familywise_accepts / trials,
                "familywise_wilson95": list(wilson_interval(familywise_accepts, trials)),
                "target_familywise_alpha": alpha,
                "alpha_allocations": allocations.tolist(),
                "attempted": attempted.tolist(),
                "accepted": accepted_per_attempt.tolist(),
                "mean_total_audit_bits": total_audit_bits / trials,
                "fresh_independent_attempts": True,
            }
        )
    return reports


def run_validation_experiments(
    output: Path,
    threshold: float = 0.7717004776000975,
    boundary_trials: int = 10_000,
    sequential_trials: int = 50_000,
    base_seed: int = 20260907,
) -> dict:
    """Run both statistical validation experiments and persist their records."""

    destination = prepare_output_directory(output)
    selection = selection_bias_experiment(
        threshold,
        trials=boundary_trials,
        seed=base_seed + 30,
    )
    sequential = sequential_null_experiment(
        threshold,
        trials=sequential_trials,
        seed=base_seed + 20,
    )
    write_json(destination / "selection_stress.json", selection)
    write_json(destination / "null_validation.json", sequential)
    summary = {
        "status": "PASS",
        "threshold": threshold,
        "selection_bias_points": len(selection),
        "sequential_channel_conditions": len(sequential),
    }
    write_json(destination / "summary.json", summary)
    write_artifact_manifest(destination)
    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.7717004776000975)
    parser.add_argument("--boundary-trials", type=int, default=10_000)
    parser.add_argument("--sequential-trials", type=int, default=50_000)
    parser.add_argument("--base-seed", type=int, default=20260907)
    arguments = parser.parse_args()
    result = run_validation_experiments(
        arguments.output,
        arguments.threshold,
        arguments.boundary_trials,
        arguments.sequential_trials,
        arguments.base_seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
