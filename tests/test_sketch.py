from __future__ import annotations

import numpy as np

from sentrysem.sketch import hamming_matrix, ranked_candidates


def test_hamming_matrix_matches_direct_count() -> None:
    rng = np.random.default_rng(19)
    bank = rng.integers(0, 2, (13, 64), dtype=np.uint8).astype(bool)
    query = rng.integers(0, 2, (7, 64), dtype=np.uint8).astype(bool)
    packed = hamming_matrix(bank, query)
    direct = np.count_nonzero(query[:, None, :] != bank[None, :, :], axis=2)
    assert np.array_equal(packed, direct)


def test_candidate_ties_use_lowest_cache_index() -> None:
    distances = np.asarray([[4, 2, 2, 3], [1, 1, 1, 1]])
    candidates = ranked_candidates(distances, shortlist=3)
    assert candidates.tolist() == [[1, 2, 3], [0, 1, 2]]
