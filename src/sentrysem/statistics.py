"""Finite-sample tests and confidence intervals for semantic audits."""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

NORMAL_95 = 1.959963984540054


class AuditCheckpoint(NamedTuple):
    """Precomputed decisions for one sequential-audit inspection time."""

    bits: int
    accept: NDArray[np.bool_]
    futile: NDArray[np.bool_]


def _probability(value: float, name: str, inclusive: bool = True) -> float:
    result = float(value)
    valid = 0.0 <= result <= 1.0 if inclusive else 0.0 < result < 1.0
    if not valid or not math.isfinite(result):
        interval = "[0, 1]" if inclusive else "(0, 1)"
        raise ValueError(f"{name} must lie in {interval}")
    return result


def observed_mismatch_probability(cosine: ArrayLike, bsc_probability: float) -> NDArray[np.float64]:
    """Map cosine similarity to the received random-hyperplane mismatch rate.

    For unit vectors with similarity ``s``, a random hyperplane separates the
    vectors with probability ``arccos(s) / pi``.  Passing the source sign
    through a binary symmetric channel with crossover probability ``p`` gives
    ``p + (1 - 2p) arccos(s) / pi``.
    """

    p = _probability(bsc_probability, "bsc_probability")
    if p > 0.5:
        raise ValueError("bsc_probability must not exceed 0.5")
    similarity = np.asarray(cosine, dtype=np.float64)
    angle_rate = np.arccos(np.clip(similarity, -1.0, 1.0)) / np.pi
    return np.asarray(p + (1.0 - 2.0 * p) * angle_rate, dtype=np.float64)


observed_q = observed_mismatch_probability


def binomial_cdf(k: int, n: int, probability: float) -> float:
    """Evaluate ``P[Binomial(n, probability) <= k]`` by a fixed recurrence."""

    if n < 0:
        raise ValueError("n must be non-negative")
    q = _probability(probability, "probability")
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if q == 0.0:
        return 1.0
    if q == 1.0:
        return 0.0
    mass = (1.0 - q) ** n
    cumulative = 0.0
    odds = q / (1.0 - q)
    for value in range(k + 1):
        cumulative += mass
        if value < n:
            mass *= (n - value) / (value + 1) * odds
    return float(min(1.0, max(0.0, cumulative)))


def binomial_boundary(n: int, null_probability: float, alpha: float) -> int:
    """Return the largest lower-tail count whose exact null mass is at most alpha."""

    if n <= 0:
        raise ValueError("n must be positive")
    q0 = _probability(null_probability, "null_probability", inclusive=False)
    level = _probability(alpha, "alpha", inclusive=False)
    mass = (1.0 - q0) ** n
    cumulative = 0.0
    boundary = -1
    odds = q0 / (1.0 - q0)
    for count in range(n + 1):
        cumulative += mass
        if cumulative <= level:
            boundary = count
        if count < n:
            mass *= (n - count) / (count + 1) * odds
    return boundary


def wilson_interval(successes: int, total: int, z: float = NORMAL_95) -> tuple[float, float]:
    """Return a two-sided Wilson score interval for a binomial proportion."""

    if total < 0 or successes < 0 or successes > total:
        raise ValueError("successes and total must satisfy 0 <= successes <= total")
    if z <= 0.0 or not math.isfinite(z):
        raise ValueError("z must be positive and finite")
    if total == 0:
        return 0.0, 1.0
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def wilson_lower(successes: int, total: int, z: float = NORMAL_95) -> float:
    """Return the lower endpoint of a Wilson score interval."""

    if total < 0 or successes < 0 or successes > total:
        raise ValueError("successes and total must satisfy 0 <= successes <= total")
    if z <= 0.0 or not math.isfinite(z):
        raise ValueError("z must be positive and finite")
    if total == 0:
        return 0.0
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = proportion + z * z / (2.0 * total)
    radius = z * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total))
    return max(0.0, (center - radius) / denominator)


def logsumexp(values: ArrayLike) -> float:
    """Compute the logarithm of an exponential sum without avoidable overflow."""

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("values must be non-empty")
    maximum = float(np.max(array))
    if math.isinf(maximum) and maximum < 0.0:
        return maximum
    return float(maximum + np.log(np.exp(array - maximum).sum()))


def mixture_state(
    null_probability: float,
    bsc_probability: float,
    points: int = 8,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Build the predeclared alternative grid and Bernoulli log factors."""

    q0 = _probability(null_probability, "null_probability", inclusive=False)
    p = _probability(bsc_probability, "bsc_probability")
    if p >= q0:
        raise ValueError("bsc_probability must be below the boundary mismatch probability")
    if points <= 0:
        raise ValueError("points must be positive")
    lower = max(1e-6, p)
    alternatives = np.linspace(lower, q0, points + 1, dtype=np.float64)[:-1]
    with np.errstate(divide="ignore"):
        log_mismatch = np.log(alternatives / q0)
    log_agreement = np.log((1.0 - alternatives) / (1.0 - q0))
    return alternatives, log_mismatch, log_agreement


def audit_lookup(
    null_probability: float,
    bsc_probability: float,
    alpha: float,
    max_bits: int = 768,
    batch_bits: int = 128,
    mixture_points: int = 8,
) -> list[AuditCheckpoint]:
    """Precompute accept and futility decisions for the anytime-valid audit."""

    level = _probability(alpha, "alpha", inclusive=False)
    if max_bits <= 0 or batch_bits <= 0 or max_bits % batch_bits:
        raise ValueError("max_bits must be a positive multiple of batch_bits")
    alternatives, log_mismatch, log_agreement = mixture_state(
        null_probability, bsc_probability, mixture_points
    )
    log_threshold = math.log(1.0 / level)
    checkpoints: list[AuditCheckpoint] = []
    for bits in range(batch_bits, max_bits + 1, batch_bits):
        accept = np.zeros(bits + 1, dtype=bool)
        futile = np.zeros(bits + 1, dtype=bool)
        remaining = max_bits - bits
        for mismatches in range(bits + 1):
            log_ratios = mismatches * log_mismatch + (bits - mismatches) * log_agreement
            log_e_value = logsumexp(log_ratios) - math.log(len(alternatives))
            accept[mismatches] = log_e_value >= log_threshold
            best_possible = logsumexp(log_ratios + remaining * log_agreement) - math.log(
                len(alternatives)
            )
            futile[mismatches] = best_possible < log_threshold
        if bits == max_bits:
            futile = ~accept
        checkpoints.append(AuditCheckpoint(bits, accept, futile))
    return checkpoints


def apply_audit_lookup(
    mismatch_bits: ArrayLike,
    lookup: list[AuditCheckpoint],
) -> tuple[NDArray[np.bool_], NDArray[np.int32]]:
    """Apply a precomputed anytime-valid audit to one or more mismatch streams."""

    matrix = np.asarray(mismatch_bits, dtype=bool)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    if matrix.ndim != 2 or not lookup:
        raise ValueError("mismatch_bits must be a non-empty one- or two-dimensional array")
    maximum = lookup[-1].bits
    if matrix.shape[1] < maximum:
        raise ValueError(f"mismatch streams must contain at least {maximum} bits")
    cumulative = np.cumsum(matrix[:, :maximum], axis=1, dtype=np.int32)
    active = np.ones(matrix.shape[0], dtype=bool)
    accepted = np.zeros(matrix.shape[0], dtype=bool)
    used = np.full(matrix.shape[0], maximum, dtype=np.int32)
    for checkpoint in lookup:
        counts = cumulative[:, checkpoint.bits - 1]
        yes = active & checkpoint.accept[counts]
        no = active & checkpoint.futile[counts] & ~yes
        accepted[yes] = True
        used[yes | no] = checkpoint.bits
        active[yes | no] = False
    if active.any():
        raise AssertionError("sequential audit did not terminate at its maximum sample size")
    return accepted, used


apply_lookup = apply_audit_lookup


def alpha_spending(total_alpha: float = 0.01, attempts: int = 3) -> NDArray[np.float64]:
    """Return the predeclared 4:2:1 familywise alpha allocation."""

    level = _probability(total_alpha, "total_alpha", inclusive=False)
    if attempts < 1 or attempts > 3:
        raise ValueError("attempts must be between one and three")
    weights = np.asarray([4.0, 2.0, 1.0], dtype=np.float64)[:attempts]
    allocation = level * weights / 7.0
    if allocation.sum() > level + 1e-15:
        raise AssertionError("alpha spending exceeds the familywise budget")
    return allocation
