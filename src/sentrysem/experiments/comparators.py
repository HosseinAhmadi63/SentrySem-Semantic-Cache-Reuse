#!/usr/bin/env python3
"""Run the SentrySem comparator experiment from frozen semantic embeddings.

The workflow evaluates independent fixed audits, sequential fresh audits,
Bonferroni-protected search, same-evidence diagnostics, and the always-refresh
reference under common protocol seeds.  Every abstaining protocol completes by
transmitting the same framed int8 semantic feature.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sentrysem-comparison-mpl"))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_DIR = REPOSITORY_ROOT / "artifacts" / "frozen_inputs"
FRAME_OVERHEAD_BITS = 18 * 8
COMMIT_BITS = FRAME_OVERHEAD_BITS + 16 * 8
SEED_BITS = FRAME_OVERHEAD_BITS + 8 * 8
FEEDBACK_BITS = FRAME_OVERHEAD_BITS + 1 * 8
FALLBACK_BITS = FRAME_OVERHEAD_BITS + 4 * 8 + 512 * 8
TOTAL_ALPHA = 0.01
AUDIT_BATCH_BITS = 128
AUDIT_MAX_BITS = 768
SHORTLIST = 32
POPCOUNT = np.asarray([int(i).bit_count() for i in range(256)], dtype=np.uint8)

METHOD_ORDER = [
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
]

DISPLAY = {
    "proposal64_unverified": "64-bit unverified",
    "screened512_heuristic": "Screened-512 (heuristic)",
    "bonferroni512": "Bonferroni-512",
    "split256_256": "SentrySem-Fixed (256/256)",
    "screened1024_heuristic": "Screened-1024 (heuristic)",
    "bonferroni1024": "Bonferroni-1024",
    "split512_512": "SentrySem-Fixed (512/512)",
    "sentrysem1": "SentrySem-Sequential-1",
    "sentrysem3": "SentrySem-Sequential-3",
    "int8_fallback": "Always refresh",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=20260907)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def observed_q(cosine, bsc_probability):
    cosine = np.asarray(cosine, dtype=np.float64)
    angle_rate = np.arccos(np.clip(cosine, -1.0, 1.0)) / np.pi
    return bsc_probability + (1.0 - 2.0 * bsc_probability) * angle_rate


def binomial_boundary(n: int, q0: float, alpha: float) -> int:
    """Largest k with P[Binomial(n,q0) <= k] <= alpha."""
    if not (0.0 < q0 < 1.0 and 0.0 < alpha < 1.0):
        raise ValueError((n, q0, alpha))
    probability = (1.0 - q0) ** n
    cumulative = 0.0
    boundary = -1
    odds = q0 / (1.0 - q0)
    for k in range(n + 1):
        cumulative += probability
        if cumulative <= alpha:
            boundary = k
        if k < n:
            probability *= (n - k) / (k + 1) * odds
    return boundary


def binomial_cdf(k: int, n: int, q: float) -> float:
    if k < 0:
        return 0.0
    probability = (1.0 - q) ** n
    cumulative = 0.0
    odds = q / (1.0 - q)
    for value in range(min(k, n) + 1):
        cumulative += probability
        if value < n:
            probability *= (n - value) / (value + 1) * odds
    return float(cumulative)


def logsumexp(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    maximum = np.max(values)
    return float(maximum + np.log(np.exp(values - maximum).sum()))


def mixture_state(q0: float, bsc_probability: float, points: int = 8):
    lower = max(1e-6, bsc_probability)
    alternatives = np.linspace(lower, q0, points + 1, dtype=np.float64)[:-1]
    with np.errstate(divide="ignore"):
        log_bad = np.log(alternatives / q0)
    log_good = np.log((1.0 - alternatives) / (1.0 - q0))
    return log_bad, log_good


def audit_lookup(q0: float, bsc_probability: float, alpha: float):
    log_bad, log_good = mixture_state(q0, bsc_probability)
    threshold = math.log(1.0 / alpha)
    rows = []
    for n in range(AUDIT_BATCH_BITS, AUDIT_MAX_BITS + 1, AUDIT_BATCH_BITS):
        accept = np.zeros(n + 1, dtype=bool)
        futile = np.zeros(n + 1, dtype=bool)
        remaining = AUDIT_MAX_BITS - n
        for k in range(n + 1):
            log_lr = k * log_bad + (n - k) * log_good
            log_e = logsumexp(log_lr) - math.log(len(log_lr))
            accept[k] = log_e >= threshold
            best = logsumexp(log_lr + remaining * log_good) - math.log(len(log_lr))
            futile[k] = best < threshold
        if n == AUDIT_MAX_BITS:
            futile = ~accept
        rows.append((n, accept, futile))
    return rows


def apply_lookup(bit_matrix: np.ndarray, lookup):
    cumulative = np.cumsum(bit_matrix, axis=1, dtype=np.int16)
    active = np.ones(len(bit_matrix), dtype=bool)
    accepted = np.zeros(len(bit_matrix), dtype=bool)
    used = np.full(len(bit_matrix), AUDIT_MAX_BITS, dtype=np.int16)
    for n, accept_table, futile_table in lookup:
        mismatch = cumulative[:, n - 1]
        yes = active & accept_table[mismatch]
        no = active & futile_table[mismatch] & ~yes
        accepted[yes] = True
        used[yes | no] = n
        active[yes | no] = False
    if active.any():
        raise AssertionError("anytime audit failed to terminate")
    return accepted, used


def proposal_frame_bits(payload_bits: int) -> int:
    if payload_bits % 8:
        raise ValueError("all evaluated sketches are byte aligned")
    return FRAME_OVERHEAD_BITS + payload_bits


def hamming_matrix(bank_bits: np.ndarray, received_bits: np.ndarray, chunk: int = 25):
    bank_packed = np.packbits(bank_bits, axis=1)
    query_packed = np.packbits(received_bits, axis=1)
    distances = np.empty((len(received_bits), len(bank_bits)), dtype=np.uint16)
    for start in range(0, len(received_bits), chunk):
        stop = min(start + chunk, len(received_bits))
        xor = np.bitwise_xor(query_packed[start:stop, None, :], bank_packed[None, :, :])
        distances[start:stop] = POPCOUNT[xor].sum(axis=2, dtype=np.uint16)
    return distances


def cosine_for(query: np.ndarray, bank: np.ndarray, candidates: np.ndarray):
    return np.einsum("ij,ij->i", query, bank[candidates], dtype=np.float64)


def int8_semantic_refresh(query: np.ndarray, bank: np.ndarray):
    """Reliable source-embedding refresh and a diagnostic nearest-cache index."""
    scales = np.maximum(np.max(np.abs(query), axis=1, keepdims=True) / 127.0, 1e-8)
    quantized = np.clip(np.rint(query / scales), -127, 127).astype(np.int8)
    reconstructed = quantized.astype(np.float32) * scales.astype(np.float32)
    reconstructed /= np.linalg.norm(reconstructed, axis=1, keepdims=True).clip(1e-12)
    nearest = np.argmax(reconstructed @ bank.T, axis=1).astype(np.int32)
    refresh_cosine = np.einsum("ij,ij->i", query, reconstructed, dtype=np.float64)
    return nearest, refresh_cosine


def sentrysem_one(
    candidate_cosines: np.ndarray,
    bsc_probability: float,
    rng: np.random.Generator,
    alpha: float = TOTAL_ALPHA,
):
    q0 = float(observed_q(THRESHOLD, bsc_probability))
    lookup = audit_lookup(q0, bsc_probability, alpha)
    probabilities = observed_q(candidate_cosines, bsc_probability)
    mismatches = rng.random((len(candidate_cosines), AUDIT_MAX_BITS)) < probabilities[:, None]
    accepted, used = apply_lookup(mismatches, lookup)
    batches = used // AUDIT_BATCH_BITS
    protocol_bits = (
        proposal_frame_bits(64)
        + COMMIT_BITS
        + SEED_BITS
        + batches * (proposal_frame_bits(AUDIT_BATCH_BITS) + FEEDBACK_BITS)
    )
    return accepted, used.astype(np.int32), protocol_bits.astype(np.int32)


def joint_hyperplane_signs(
    source: np.ndarray, candidates: np.ndarray, bits: int, rng: np.random.Generator
):
    vectors = np.vstack([source[None, :], candidates]).astype(np.float64)
    gram = np.clip(vectors @ vectors.T, -1.0, 1.0)
    gram = (gram + gram.T) / 2.0
    # A general eigendecomposition is not bit-reproducible when eigenvalues are
    # repeated: valid solvers may choose different signs/bases.  Use a fixed-order,
    # unpivoted Cholesky recurrence with a documented diagonal fallback instead.
    # math.fsum fixes the factor accumulation order; an unoptimized einsum then
    # uses a fixed contraction instead of a solver-dependent eigenbasis.
    factor = None
    for jitter in (1e-12, 1e-10, 1e-8, 1e-6):
        trial = gram.copy()
        trial.flat[:: len(trial) + 1] += jitter
        lower = np.zeros_like(trial)
        valid = True
        for i in range(len(trial)):
            for j in range(i + 1):
                residual = trial[i, j] - math.fsum(lower[i, k] * lower[j, k] for k in range(j))
                if i == j:
                    if residual <= 0.0:
                        valid = False
                        break
                    lower[i, j] = math.sqrt(residual)
                else:
                    lower[i, j] = residual / lower[j, j]
            if not valid:
                break
        if valid:
            factor = lower
            break
    if factor is None:
        raise AssertionError("deterministic Cholesky failed after fixed jitter schedule")
    draws = rng.standard_normal((bits, len(vectors)))
    projections = np.einsum("bk,ik->bi", draws, factor, optimize=False)
    return projections >= 0.0


def sentrysem_three(
    query: np.ndarray, bank: np.ndarray, shortlists: np.ndarray, bsc_probability: float, seed: int
):
    n_queries = len(query)
    selected = np.empty(n_queries, dtype=np.int32)
    accepted = np.zeros(n_queries, dtype=bool)
    used_total = np.zeros(n_queries, dtype=np.int32)
    attempts_total = np.zeros(n_queries, dtype=np.int8)
    protocol_bits = np.full(n_queries, proposal_frame_bits(64), dtype=np.int32)
    alpha_spend = TOTAL_ALPHA * np.asarray([4.0, 2.0, 1.0]) / 7.0
    q0 = float(observed_q(THRESHOLD, bsc_probability))
    lookups = [audit_lookup(q0, bsc_probability, float(alpha)) for alpha in alpha_spend]

    for query_index in range(n_queries):
        local_ids = shortlists[query_index]
        current_position = 0
        untried = list(range(1, len(local_ids)))
        retired_received = []
        retired_candidates = []
        local_seed = np.random.SeedSequence(
            [int(seed), int(round(bsc_probability * 1000)), int(query_index), 3003]
        )
        rng = np.random.default_rng(local_seed)
        candidate_vectors = bank[local_ids]
        for attempt in range(3):
            attempts_total[query_index] = attempt + 1
            protocol_bits[query_index] += COMMIT_BITS + SEED_BITS
            joint = joint_hyperplane_signs(
                query[query_index], candidate_vectors, AUDIT_MAX_BITS, rng
            )
            received_source = np.bitwise_xor(
                joint[:, 0], rng.random(AUDIT_MAX_BITS) < bsc_probability
            )
            candidate_signs = joint[:, 1:]
            mismatch = received_source != candidate_signs[:, current_position]
            yes, used = apply_lookup(mismatch[None, :], lookups[attempt])
            used_here = int(used[0])
            used_total[query_index] += used_here
            batches = used_here // AUDIT_BATCH_BITS
            protocol_bits[query_index] += batches * (
                proposal_frame_bits(AUDIT_BATCH_BITS) + FEEDBACK_BITS
            )
            if bool(yes[0]):
                accepted[query_index] = True
                break
            retired_received.append(received_source[:used_here])
            retired_candidates.append(candidate_signs[:used_here])
            if attempt == 2 or not untried:
                break
            scores = np.zeros(len(local_ids), dtype=np.int32)
            for prior_received, prior_candidates in zip(
                retired_received, retired_candidates, strict=False
            ):
                scores += np.count_nonzero(prior_candidates != prior_received[:, None], axis=0)
            remaining = np.asarray(untried, dtype=np.int32)
            # Stable because remaining follows original proposal rank.
            winner = int(remaining[np.argmin(scores[remaining])])
            untried.remove(winner)
            current_position = winner
        selected[query_index] = int(local_ids[current_position])
    return selected, accepted, used_total, attempts_total, protocol_bits


def method_rows(
    method: str,
    seed: int,
    bsc_probability: float,
    query: np.ndarray,
    bank: np.ndarray,
    bank_ids: np.ndarray,
    target_ids: np.ndarray,
    candidates: np.ndarray,
    accepted: np.ndarray,
    protocol_bits: np.ndarray,
    audit_bits: np.ndarray,
    attempts: np.ndarray,
    fallback_candidates: np.ndarray,
    fallback_cosines: np.ndarray,
    certified_method: bool,
    heuristic_method: bool,
    direct_output: bool,
):
    rows = []
    pre_cosine = cosine_for(query, bank, candidates)
    pre_good = pre_cosine > THRESHOLD
    pre_identity = bank_ids[candidates] == target_ids
    fallback_used = ~accepted
    final_candidates = np.where(fallback_used, fallback_candidates, candidates)
    reuse_or_direct_cosine = cosine_for(query, bank, candidates)
    final_cosine = np.where(fallback_used, fallback_cosines, reuse_or_direct_cosine)
    for index in range(len(query)):
        is_certified = bool(certified_method and accepted[index])
        is_direct = bool(direct_output and accepted[index])
        is_heuristic = bool(heuristic_method and accepted[index])
        is_reuse = bool(accepted[index])
        batches = int(audit_bits[index]) // AUDIT_BATCH_BITS
        if method == "proposal64_unverified":
            forward, reverse, round_trips = proposal_frame_bits(64), 0, 0
        elif method.startswith("screened") or method.startswith("bonferroni"):
            sketch_bits = 1024 if "1024" in method else 512
            forward = proposal_frame_bits(sketch_bits)
            reverse, round_trips = FEEDBACK_BITS, 1
        elif method.startswith("split"):
            split_bits = int(method.split("_")[0].replace("split", ""))
            forward = proposal_frame_bits(split_bits) + SEED_BITS + proposal_frame_bits(split_bits)
            reverse, round_trips = COMMIT_BITS + FEEDBACK_BITS, 2
        elif method.startswith("sentrysem"):
            forward = proposal_frame_bits(64) + int(attempts[index]) * SEED_BITS
            forward += batches * proposal_frame_bits(AUDIT_BATCH_BITS)
            reverse = int(attempts[index]) * COMMIT_BITS + batches * FEEDBACK_BITS
            round_trips = int(attempts[index]) + batches
        elif method == "int8_fallback":
            forward, reverse, round_trips = 0, 0, 0
        else:
            raise ValueError(method)
        if forward + reverse != int(protocol_bits[index]):
            raise AssertionError((method, forward, reverse, int(protocol_bits[index])))
        rows.append(
            {
                "protocol_seed": int(seed),
                "bsc": float(bsc_probability),
                "query_index": int(index),
                "target_id": int(target_ids[index]),
                "method": method,
                "decision": (
                    "unverified_reuse"
                    if is_direct
                    else (
                        "heuristic_reuse"
                        if is_heuristic
                        else ("certify" if is_certified else "semantic_refresh")
                    )
                ),
                "certified": int(is_certified),
                "direct_output": int(is_direct),
                "heuristic_reuse": int(is_heuristic),
                "cache_reuse": int(is_reuse),
                "candidate_bank_index": int(candidates[index]),
                "candidate_original_id": int(bank_ids[candidates[index]]),
                "candidate_cosine": float(pre_cosine[index]),
                "candidate_verifier_good": int(pre_good[index]),
                "candidate_identity_match": int(pre_identity[index]),
                "false_certification": int(is_certified and not pre_good[index]),
                "bad_reuse": int(is_reuse and not pre_good[index]),
                "fallback_used": int(fallback_used[index]),
                "completion_bank_index": int(final_candidates[index]),
                "completion_original_id": (
                    None if fallback_used[index] else int(bank_ids[final_candidates[index]])
                ),
                "completion_cosine": float(final_cosine[index]),
                "completion_verifier_good": int(final_cosine[index] > THRESHOLD),
                "completion_identity_match": (
                    None if fallback_used[index] else int(pre_identity[index])
                ),
                "protocol_bits": int(protocol_bits[index]),
                "protocol_forward_bits": int(forward),
                "protocol_reverse_bits": int(reverse),
                "fallback_bits": int(FALLBACK_BITS if fallback_used[index] else 0),
                "completion_forward_bits": int(
                    forward + (FALLBACK_BITS if fallback_used[index] else 0)
                ),
                "completion_bits": int(
                    protocol_bits[index] + (FALLBACK_BITS if fallback_used[index] else 0)
                ),
                "audit_bits": int(audit_bits[index]),
                "attempts": int(attempts[index]),
                "logical_round_trips": int(round_trips),
            }
        )
    return rows


def summarize_group(rows: list[dict]):
    count = len(rows)
    certified = np.asarray([r["certified"] for r in rows], dtype=bool)
    direct = np.asarray([r["direct_output"] for r in rows], dtype=bool)
    reuse = np.asarray([r["cache_reuse"] for r in rows], dtype=bool)
    candidate_identity = np.asarray([r["candidate_identity_match"] for r in rows], dtype=bool)
    candidate_good = np.asarray([r["candidate_verifier_good"] for r in rows], dtype=bool)
    result = {
        "sessions": count,
        "certification_rate": float(certified.mean()),
        "direct_output_rate": float(direct.mean()),
        "reuse_rate": float(reuse.mean()),
        "fallback_rate": float(np.mean([r["fallback_used"] for r in rows])),
        "false_certification_rate": float(np.mean([r["false_certification"] for r in rows])),
        "bad_reuse_rate": float(np.mean([r["bad_reuse"] for r in rows])),
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
            np.mean([r["completion_verifier_good"] for r in rows])
        ),
        "service_completion_rate": 1.0,
        "mean_protocol_bits": float(np.mean([r["protocol_bits"] for r in rows])),
        "mean_protocol_forward_bits": float(np.mean([r["protocol_forward_bits"] for r in rows])),
        "mean_protocol_reverse_bits": float(np.mean([r["protocol_reverse_bits"] for r in rows])),
        "mean_fallback_bits": float(np.mean([r["fallback_bits"] for r in rows])),
        "mean_completion_forward_bits": float(
            np.mean([r["completion_forward_bits"] for r in rows])
        ),
        "mean_completion_bits": float(np.mean([r["completion_bits"] for r in rows])),
        "saving_vs_always_refresh": float(
            1.0 - np.mean([r["completion_bits"] for r in rows]) / FALLBACK_BITS
        ),
        "mean_audit_bits": float(np.mean([r["audit_bits"] for r in rows])),
        "mean_attempts": float(np.mean([r["attempts"] for r in rows])),
        "mean_logical_round_trips": float(np.mean([r["logical_round_trips"] for r in rows])),
    }
    return result


def per_seed_summaries(rows: list[dict]):
    groups = {}
    for row in rows:
        key = (row["protocol_seed"], row["bsc"], row["method"])
        groups.setdefault(key, []).append(row)
    summaries = []
    for (seed, bsc_probability, method), group in sorted(groups.items()):
        summary = {
            "protocol_seed": seed,
            "bsc": bsc_probability,
            "method": method,
            **summarize_group(group),
        }
        summaries.append(summary)
    return summaries


def aggregate_summaries(seed_summaries: list[dict]):
    groups = {}
    for row in seed_summaries:
        groups.setdefault((row["bsc"], row["method"]), []).append(row)
    metrics = [
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
    ]
    aggregates = []
    for (bsc_probability, method), group in sorted(groups.items()):
        item = {"bsc": bsc_probability, "method": method, "protocol_seeds": len(group)}
        for metric in metrics:
            values = np.asarray([r[metric] for r in group if r[metric] is not None], dtype=float)
            if not len(values):
                item[f"{metric}_mean"] = None
                item[f"{metric}_seed_sd"] = None
                item[f"{metric}_seed_min"] = None
                item[f"{metric}_seed_max"] = None
                continue
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_seed_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            item[f"{metric}_seed_min"] = float(values.min())
            item[f"{metric}_seed_max"] = float(values.max())
        aggregates.append(item)
    return aggregates


def identity_cluster_bootstrap(rows: list[dict], trials: int = 2000):
    """Resample query identities while retaining every paired protocol seed."""
    reports = []
    rng = np.random.default_rng(777311)
    for bsc_probability in (0.0, 0.05):
        for method in METHOD_ORDER:
            group = [r for r in rows if r["bsc"] == bsc_probability and r["method"] == method]
            seeds = sorted({r["protocol_seed"] for r in group})
            query_indices = sorted({r["query_index"] for r in group})
            lookup = {(r["protocol_seed"], r["query_index"]): r for r in group}
            reuse = np.asarray(
                [[lookup[(seed, q)]["cache_reuse"] for q in query_indices] for seed in seeds],
                dtype=bool,
            )
            bad = np.asarray(
                [[lookup[(seed, q)]["bad_reuse"] for q in query_indices] for seed in seeds],
                dtype=bool,
            )
            identity = np.asarray(
                [
                    [lookup[(seed, q)]["candidate_identity_match"] for q in query_indices]
                    for seed in seeds
                ],
                dtype=bool,
            )
            bits = np.asarray(
                [[lookup[(seed, q)]["completion_bits"] for q in query_indices] for seed in seeds],
                dtype=np.float64,
            )
            samples = {
                "reuse_rate": [],
                "bad_reuse_rate": [],
                "identity_precision_among_reuses": [],
                "mean_completion_bits": [],
                "saving_vs_always_refresh": [],
            }
            for _ in range(trials):
                chosen = rng.integers(0, len(query_indices), len(query_indices))
                reuse_sample = reuse[:, chosen]
                identity_sample = identity[:, chosen]
                mean_bits = float(bits[:, chosen].mean())
                samples["reuse_rate"].append(float(reuse_sample.mean()))
                samples["bad_reuse_rate"].append(float(bad[:, chosen].mean()))
                samples["identity_precision_among_reuses"].append(
                    float(identity_sample[reuse_sample].mean()) if reuse_sample.any() else np.nan
                )
                samples["mean_completion_bits"].append(mean_bits)
                samples["saving_vs_always_refresh"].append(1.0 - mean_bits / FALLBACK_BITS)
            report = {
                "bsc": bsc_probability,
                "method": method,
                "protocol_seeds_retained_per_identity": len(seeds),
                "query_identity_clusters": len(query_indices),
                "bootstrap_trials": trials,
            }
            for metric, draws in samples.items():
                values = np.asarray(draws, dtype=float)
                values = values[np.isfinite(values)]
                report[f"{metric}_cluster_bootstrap95_low"] = (
                    float(np.quantile(values, 0.025)) if len(values) else None
                )
                report[f"{metric}_cluster_bootstrap95_high"] = (
                    float(np.quantile(values, 0.975)) if len(values) else None
                )
            reports.append(report)
    return reports


def write_csv(path: Path, rows: Iterable[dict]):
    rows = list(rows)
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def validate_protocol_math(output: Path, bsc_values=(0.0, 0.05)):
    rng = np.random.default_rng(819271)
    reports = []
    for bsc_probability in bsc_values:
        q0 = float(observed_q(THRESHOLD, bsc_probability))
        for bits in (256, 512):
            boundary = binomial_boundary(bits, q0, TOTAL_ALPHA)
            cdf = binomial_cdf(boundary, bits, q0)
            reports.append(
                {
                    "test": f"split{bits}_{bits}",
                    "bsc": bsc_probability,
                    "boundary": boundary,
                    "exact_boundary_null_probability": cdf,
                    "claimed_upper_bound": TOTAL_ALPHA,
                    "pass": bool(cdf <= TOTAL_ALPHA + 1e-12),
                }
            )
        for bits in (512, 1024):
            screened_boundary = binomial_boundary(bits, q0, TOTAL_ALPHA)
            reports.append(
                {
                    "test": f"screened{bits}_heuristic",
                    "bsc": bsc_probability,
                    "boundary": screened_boundary,
                    "single_fixed_candidate_null_probability": binomial_cdf(
                        screened_boundary, bits, q0
                    ),
                    "formal_post_selection_certificate": False,
                    "pass": True,
                    "note": (
                        "Search and screening reuse the same sketch. The single-candidate "
                        "p-value is selection-biased and is reported only as a heuristic baseline."
                    ),
                }
            )
            per_candidate_alpha = TOTAL_ALPHA / BANK_SIZE
            boundary = binomial_boundary(bits, q0, per_candidate_alpha)
            cdf = binomial_cdf(boundary, bits, q0)
            reports.append(
                {
                    "test": f"bonferroni{bits}",
                    "bsc": bsc_probability,
                    "cache_candidates": BANK_SIZE,
                    "boundary": boundary,
                    "exact_per_candidate_null_probability": cdf,
                    "union_bound": BANK_SIZE * cdf,
                    "claimed_upper_bound": TOTAL_ALPHA,
                    "pass": bool(BANK_SIZE * cdf <= TOTAL_ALPHA + 1e-12),
                }
            )

        trials = 50_000
        for name, allocations in [
            ("sentrysem1", [TOTAL_ALPHA]),
            ("sentrysem3", list(TOTAL_ALPHA * np.asarray([4.0, 2.0, 1.0]) / 7.0)),
        ]:
            false_accepts = 0
            lookups = [audit_lookup(q0, bsc_probability, float(alpha)) for alpha in allocations]
            for start in range(0, trials, 2500):
                count = min(2500, trials - start)
                alive = np.ones(count, dtype=bool)
                ever = np.zeros(count, dtype=bool)
                for lookup in lookups:
                    mismatch = rng.random((count, AUDIT_MAX_BITS)) < q0
                    yes, _ = apply_lookup(mismatch, lookup)
                    yes &= alive
                    ever |= yes
                    alive &= ~yes
                false_accepts += int(ever.sum())
            observed = false_accepts / trials
            reports.append(
                {
                    "test": name,
                    "bsc": bsc_probability,
                    "boundary_trials": trials,
                    "boundary_false_acceptance": observed,
                    "claimed_upper_bound": TOTAL_ALPHA,
                    "pass": bool(observed <= TOTAL_ALPHA + 0.0025),
                    "note": "Monte Carlo screen; the formal guarantee follows from the e-process and alpha spending.",
                }
            )
    if not all(row["pass"] for row in reports):
        raise AssertionError("at least one protocol-math validation failed")
    (output / "protocol_validation.json").write_text(json.dumps(reports, indent=2))
    return reports


def plot_results(aggregate: list[dict], output: Path, number_of_seeds: int):
    methods = [m for m in METHOD_ORDER if m != "proposal64_unverified"]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))
    colors = {0.0: "#2070b4", 0.05: "#e27a24"}
    markers = {0.0: "o", 0.05: "s"}
    for bsc_probability in (0.0, 0.05):
        rows = {r["method"]: r for r in aggregate if r["bsc"] == bsc_probability}
        for method in methods:
            row = rows[method]
            axes[0].scatter(
                row["mean_completion_bits_mean"],
                row["reuse_rate_mean"],
                color=colors[bsc_probability],
                marker=markers[bsc_probability],
                s=48,
            )
            axes[0].annotate(
                DISPLAY[method],
                (row["mean_completion_bits_mean"], row["reuse_rate_mean"]),
                fontsize=7,
                xytext=(3, 3),
                textcoords="offset points",
            )
            axes[1].scatter(
                row["mean_completion_bits_mean"],
                row["candidate_identity_rate_among_reuses_mean"] or 0.0,
                color=colors[bsc_probability],
                marker=markers[bsc_probability],
                s=48,
            )
        sentry = rows["sentrysem1"]
        axes[2].errorbar(
            bsc_probability,
            sentry["mean_completion_bits_mean"],
            yerr=sentry["mean_completion_bits_seed_sd"],
            fmt=markers[bsc_probability],
            color=colors[bsc_probability],
            capsize=4,
            label=f"p={bsc_probability:.2f}",
        )
    axes[0].set(
        xlabel="mean logical completion bits",
        ylabel="cache-reuse rate",
        title="Cache reuse with common semantic refresh",
    )
    axes[1].set(
        xlabel="mean logical completion bits",
        ylabel="identity precision among reuses",
        title="Reuse precision versus completion cost",
    )
    axes[2].set(
        xlabel="BSC crossover probability",
        ylabel="SentrySem-1 completion bits",
        title="Protocol-seed mean ± SD",
    )
    axes[1].set_ylim(0.94, 1.002)
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle(
        f"SentrySem comparators: frozen CIFAR-10/ResNet-18 embeddings, {number_of_seeds} protocol seeds"
    )
    fig.tight_layout()
    fig.savefig(output / "publication_completion_tradeoffs.png", dpi=220)
    fig.savefig(output / "publication_completion_tradeoffs.pdf")
    plt.close(fig)


def run(args: argparse.Namespace):
    if args.seeds < 2:
        raise ValueError("publication run requires at least two protocol seeds")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    embedding_path = args.input_dir / "frozen_embeddings.npz"
    calibration_path = args.input_dir / "calibration.json"
    if not embedding_path.exists() or not calibration_path.exists():
        raise FileNotFoundError("validated stage-7 embeddings/calibration are missing")
    calibration = json.loads(calibration_path.read_text())
    global THRESHOLD, BANK_SIZE
    THRESHOLD = float(calibration["threshold_cosine"])
    arrays = np.load(embedding_path)
    bank = np.asarray(arrays["final_bank"], dtype=np.float32)
    bank_ids = np.asarray(arrays["final_bank_ids"], dtype=np.int64)
    query = np.asarray(arrays["paired_iid_vectors"], dtype=np.float32)
    target_ids = np.asarray(arrays["paired_iid_target_ids"], dtype=np.int64)
    BANK_SIZE = len(bank)
    if bank.shape != (1000, 512) or query.shape != (500, 512):
        raise AssertionError((bank.shape, query.shape))
    if not np.allclose(np.linalg.norm(bank, axis=1), 1.0, atol=2e-5):
        raise AssertionError("bank embeddings are not unit normalized")
    if not np.allclose(np.linalg.norm(query, axis=1), 1.0, atol=2e-5):
        raise AssertionError("query embeddings are not unit normalized")
    if not np.isin(target_ids, bank_ids).all():
        raise AssertionError("the common-fallback evaluation requires every target in cache")

    fallback_candidates, fallback_cosines = int8_semantic_refresh(query, bank)
    fallback_identity = bank_ids[fallback_candidates] == target_ids
    if float(fallback_identity.mean()) < 0.99:
        raise AssertionError("validated int8 fallback unexpectedly changed")

    seeds = [
        int(child.generate_state(1, dtype=np.uint64)[0])
        for child in np.random.SeedSequence(args.base_seed).spawn(args.seeds)
    ]
    config = {
        "input_artifact_directory": args.input_dir.name,
        "input_embeddings": embedding_path.name,
        "input_embeddings_sha256": sha256(embedding_path),
        "input_calibration": calibration_path.name,
        "input_calibration_sha256": sha256(calibration_path),
        "experiment_script_sha256": sha256(Path(__file__)),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "threshold_cosine_frozen": THRESHOLD,
        "protocol_seeds": seeds,
        "bsc_values": [0.0, 0.05],
        "bank_size": BANK_SIZE,
        "paired_iid_queries": len(query),
        "common_fallback": "reliable framed int8 512-D source-embedding semantic refresh",
        "common_fallback_bits": FALLBACK_BITS,
        "logical_frame_overhead_bits": FRAME_OVERHEAD_BITS,
        "reliable_control_and_fallback_assumption": True,
        "physical_fec_cost_included": False,
        "embedding_or_model_training": False,
        "fallback_is_cache_identity_output": False,
        "endpoint_note": (
            "A rejected cache reuse is completed by delivering the source semantic "
            "representation itself; cache-identity precision is reported only for reuse."
        ),
        "method_order": METHOD_ORDER,
    }
    (args.output / "experiment_configuration.json").write_text(json.dumps(config, indent=2))
    validation = validate_protocol_math(args.output)
    print(f"PASS protocol math ({len(validation)} checks)")
    print(f"Reliable int8 fallback identity accuracy: {fallback_identity.mean():.4f}")

    all_rows = []
    for seed_index, seed in enumerate(seeds, start=1):
        plane_rng = np.random.default_rng(np.random.SeedSequence([seed, 100]))
        planes = plane_rng.standard_normal((1024, 512)).astype(np.float32)
        bank_master = bank @ planes.T >= 0.0
        query_master = query @ planes.T >= 0.0
        for bsc_probability in (0.0, 0.05):
            channel_rng = np.random.default_rng(
                np.random.SeedSequence([seed, int(round(bsc_probability * 1000)), 200])
            )
            received_master = np.bitwise_xor(
                query_master, channel_rng.random(query_master.shape) < bsc_probability
            )
            distance = {}
            order = {}
            for bits in (64, 256, 512, 1024):
                distance[bits] = hamming_matrix(bank_master[:, :bits], received_master[:, :bits])
                keep = SHORTLIST if bits == 64 else 1
                order[bits] = np.argsort(distance[bits], axis=1, kind="stable")[:, :keep]

            zeros = np.zeros(len(query), dtype=np.int32)
            ones = np.ones(len(query), dtype=np.int8)

            # Unverified diagnostic, included only to expose the reliability cost.
            candidate = order[64][:, 0].astype(np.int32)
            all_rows.extend(
                method_rows(
                    "proposal64_unverified",
                    seed,
                    bsc_probability,
                    query,
                    bank,
                    bank_ids,
                    target_ids,
                    candidate,
                    np.ones(len(query), dtype=bool),
                    np.full(len(query), proposal_frame_bits(64), dtype=np.int32),
                    zeros,
                    ones,
                    fallback_candidates,
                    fallback_cosines,
                    certified_method=False,
                    heuristic_method=False,
                    direct_output=True,
                )
            )

            for bits in (512, 1024):
                candidate = order[bits][:, 0].astype(np.int32)
                minimum = distance[bits][np.arange(len(query)), candidate]
                q0 = float(observed_q(THRESHOLD, bsc_probability))
                # Practical same-evidence screen.  It is deliberately labelled a
                # heuristic because cache search and acceptance reuse the same bits.
                screened_boundary = binomial_boundary(bits, q0, TOTAL_ALPHA)
                screened = minimum <= screened_boundary
                protocol = np.full(
                    len(query), proposal_frame_bits(bits) + FEEDBACK_BITS, dtype=np.int32
                )
                all_rows.extend(
                    method_rows(
                        f"screened{bits}_heuristic",
                        seed,
                        bsc_probability,
                        query,
                        bank,
                        bank_ids,
                        target_ids,
                        candidate,
                        screened,
                        protocol,
                        np.full(len(query), bits),
                        ones,
                        fallback_candidates,
                        fallback_cosines,
                        certified_method=False,
                        heuristic_method=True,
                        direct_output=False,
                    )
                )

                boundary = binomial_boundary(bits, q0, TOTAL_ALPHA / BANK_SIZE)
                accepted = minimum <= boundary
                all_rows.extend(
                    method_rows(
                        f"bonferroni{bits}",
                        seed,
                        bsc_probability,
                        query,
                        bank,
                        bank_ids,
                        target_ids,
                        candidate,
                        accepted,
                        protocol,
                        np.full(len(query), bits),
                        ones,
                        fallback_candidates,
                        fallback_cosines,
                        certified_method=True,
                        heuristic_method=False,
                        direct_output=False,
                    )
                )

            for bits in (256, 512):
                candidate = order[bits][:, 0].astype(np.int32)
                cosine = cosine_for(query, bank, candidate)
                audit_rng = np.random.default_rng(
                    np.random.SeedSequence([seed, int(round(bsc_probability * 1000)), bits, 400])
                )
                counts = audit_rng.binomial(bits, observed_q(cosine, bsc_probability))
                boundary = binomial_boundary(
                    bits, float(observed_q(THRESHOLD, bsc_probability)), TOTAL_ALPHA
                )
                accepted = counts <= boundary
                protocol = np.full(
                    len(query),
                    proposal_frame_bits(bits)
                    + COMMIT_BITS
                    + SEED_BITS
                    + proposal_frame_bits(bits)
                    + FEEDBACK_BITS,
                    dtype=np.int32,
                )
                all_rows.extend(
                    method_rows(
                        f"split{bits}_{bits}",
                        seed,
                        bsc_probability,
                        query,
                        bank,
                        bank_ids,
                        target_ids,
                        candidate,
                        accepted,
                        protocol,
                        np.full(len(query), bits),
                        ones,
                        fallback_candidates,
                        fallback_cosines,
                        certified_method=True,
                        heuristic_method=False,
                        direct_output=False,
                    )
                )

            candidate = order[64][:, 0].astype(np.int32)
            cosine = cosine_for(query, bank, candidate)
            audit_rng = np.random.default_rng(
                np.random.SeedSequence([seed, int(round(bsc_probability * 1000)), 1001])
            )
            accepted, used, protocol = sentrysem_one(cosine, bsc_probability, audit_rng)
            all_rows.extend(
                method_rows(
                    "sentrysem1",
                    seed,
                    bsc_probability,
                    query,
                    bank,
                    bank_ids,
                    target_ids,
                    candidate,
                    accepted,
                    protocol,
                    used,
                    ones,
                    fallback_candidates,
                    fallback_cosines,
                    certified_method=True,
                    heuristic_method=False,
                    direct_output=False,
                )
            )

            selected, accepted, used, attempts, protocol = sentrysem_three(
                query, bank, order[64], bsc_probability, seed
            )
            all_rows.extend(
                method_rows(
                    "sentrysem3",
                    seed,
                    bsc_probability,
                    query,
                    bank,
                    bank_ids,
                    target_ids,
                    selected,
                    accepted,
                    protocol,
                    used,
                    attempts,
                    fallback_candidates,
                    fallback_cosines,
                    certified_method=True,
                    heuristic_method=False,
                    direct_output=False,
                )
            )

            all_rows.extend(
                method_rows(
                    "int8_fallback",
                    seed,
                    bsc_probability,
                    query,
                    bank,
                    bank_ids,
                    target_ids,
                    fallback_candidates,
                    np.zeros(len(query), dtype=bool),
                    np.zeros(len(query), dtype=np.int32),
                    zeros,
                    zeros,
                    fallback_candidates,
                    fallback_cosines,
                    certified_method=False,
                    heuristic_method=False,
                    direct_output=False,
                )
            )
            # The reference always sends its fallback; method_rows interprets rejection
            # as a fallback and therefore adds the same 4272-bit frame.
            print(f"seed {seed_index}/{len(seeds)} ({seed}), BSC={bsc_probability:.2f}: PASS")

    expected_rows = len(seeds) * 2 * len(query) * len(METHOD_ORDER)
    if len(all_rows) != expected_rows:
        raise AssertionError((len(all_rows), expected_rows))
    seed_summary = per_seed_summaries(all_rows)
    aggregate = aggregate_summaries(seed_summary)
    bootstrap = identity_cluster_bootstrap(all_rows)

    # Invariants used by all manuscript tables.
    for row in all_rows:
        if row["completion_bits"] != row["protocol_bits"] + row["fallback_bits"]:
            raise AssertionError("completion ledger does not add")
        if row["false_certification"] and not row["certified"]:
            raise AssertionError("false certificate recorded without certification")
    for item in aggregate:
        if item["method"] not in ("proposal64_unverified", "int8_fallback"):
            if item["false_certification_rate_mean"] > TOTAL_ALPHA + 0.005:
                raise AssertionError(f"empirical risk screen failed: {item}")

    write_csv(args.output / "publication_session_results.csv", all_rows)
    write_csv(args.output / "publication_seed_summary.csv", seed_summary)
    write_csv(args.output / "publication_aggregate_summary.csv", aggregate)
    (args.output / "publication_aggregate_summary.json").write_text(json.dumps(aggregate, indent=2))
    write_csv(args.output / "identity_cluster_bootstrap.csv", bootstrap)
    (args.output / "identity_cluster_bootstrap.json").write_text(json.dumps(bootstrap, indent=2))
    plot_results(aggregate, args.output, len(seeds))

    manifest = {}
    for path in sorted(args.output.iterdir()):
        if path.is_file():
            manifest[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    (args.output / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"PASS publication package: {args.output.resolve()}")


if __name__ == "__main__":
    run(parse_args())
