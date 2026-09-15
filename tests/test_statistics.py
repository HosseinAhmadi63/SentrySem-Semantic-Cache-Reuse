from __future__ import annotations

import math

import numpy as np

from sentrysem.statistics import (
    alpha_spending,
    apply_audit_lookup,
    audit_lookup,
    binomial_boundary,
    binomial_cdf,
    observed_mismatch_probability,
    wilson_interval,
)


def test_random_hyperplane_channel_formula() -> None:
    cosine = np.asarray([-1.0, 0.0, 1.0])
    clean = observed_mismatch_probability(cosine, 0.0)
    assert np.allclose(clean, [1.0, 0.5, 0.0])
    assert np.allclose(observed_mismatch_probability(cosine, 0.05), 0.05 + 0.9 * clean)


def test_paper_audit_boundaries() -> None:
    threshold = 0.7717004776000975
    expected = {0.0: (141, 0.008307522884450153), 0.05: (162, 0.009719658202633178)}
    for probability, (boundary, mass) in expected.items():
        q0 = float(observed_mismatch_probability(threshold, probability))
        assert binomial_boundary(768, q0, 0.01) == boundary
        assert math.isclose(binomial_cdf(boundary, 768, q0), mass, abs_tol=1e-14)
        assert binomial_cdf(boundary + 1, 768, q0) > 0.01


def test_anytime_lookup_terminates() -> None:
    q0 = float(observed_mismatch_probability(0.7717004776000975, 0.05))
    lookup = audit_lookup(q0, 0.05, 0.01, 768, 128, 8)
    mismatch = np.zeros((3, 768), dtype=bool)
    accepted, used = apply_audit_lookup(mismatch, lookup)
    assert accepted.all()
    assert np.isin(used, [128, 256, 384, 512, 640, 768]).all()
    assert math.isclose(alpha_spending().sum(), 0.01)


def test_wilson_interval_contains_estimate() -> None:
    low, high = wilson_interval(45, 10_000)
    assert low < 45 / 10_000 < high
