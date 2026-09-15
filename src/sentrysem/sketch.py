"""Random-hyperplane sketches, binary channels, and cache search."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

POPCOUNT = np.asarray([int(value).bit_count() for value in range(256)], dtype=np.uint8)


@dataclass(frozen=True)
class Proposal:
    """Received proposal signs and the deterministically ranked cache shortlist."""

    received_signs: NDArray[np.bool_]
    candidate_indices: NDArray[np.int32]
    hamming_distances: NDArray[np.int32]


def random_hyperplanes(
    bits: int,
    dimension: int,
    rng: np.random.Generator,
) -> NDArray[np.float32]:
    """Draw the float32 Gaussian projection matrix used by the experiments."""

    if bits <= 0 or dimension <= 0:
        raise ValueError("bits and dimension must be positive")
    return rng.standard_normal((bits, dimension)).astype(np.float32)


def hyperplane_signs(vectors: ArrayLike, planes: ArrayLike) -> NDArray[np.bool_]:
    """Return one bit for the sign of every vector--hyperplane projection."""

    values = np.asarray(vectors)
    directions = np.asarray(planes)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or directions.ndim != 2 or values.shape[1] != directions.shape[1]:
        raise ValueError("vectors and planes must be compatible two-dimensional arrays")
    return np.asarray(values @ directions.T >= 0.0, dtype=bool)


signs = hyperplane_signs


def binary_symmetric_channel(
    bits: ArrayLike,
    probability: float,
    rng: np.random.Generator,
) -> NDArray[np.bool_]:
    """Flip each sign independently with the declared BSC crossover probability."""

    if not 0.0 <= probability <= 0.5:
        raise ValueError("probability must lie in [0, 0.5]")
    source = np.asarray(bits, dtype=bool)
    return np.bitwise_xor(source, rng.random(source.shape) < probability)


bsc = binary_symmetric_channel


def hamming_matrix(
    bank_bits: ArrayLike,
    query_bits: ArrayLike,
    chunk: int = 25,
) -> NDArray[np.uint16]:
    """Compute all query-to-cache Hamming distances with packed-byte popcounts."""

    cache = np.asarray(bank_bits, dtype=bool)
    queries = np.asarray(query_bits, dtype=bool)
    if cache.ndim != 2 or queries.ndim != 2 or cache.shape[1] != queries.shape[1]:
        raise ValueError("bank_bits and query_bits must have matching bit dimensions")
    if cache.shape[1] == 0 or len(cache) == 0 or chunk <= 0:
        raise ValueError("the cache, bit dimension, and chunk size must be positive")
    bank_packed = np.packbits(cache, axis=1)
    query_packed = np.packbits(queries, axis=1)
    distances = np.empty((len(queries), len(cache)), dtype=np.uint16)
    for start in range(0, len(queries), chunk):
        stop = min(start + chunk, len(queries))
        xor = np.bitwise_xor(query_packed[start:stop, None, :], bank_packed[None, :, :])
        distances[start:stop] = POPCOUNT[xor].sum(axis=2, dtype=np.uint16)
    return distances


def ranked_candidates(distances: ArrayLike, shortlist: int = 1) -> NDArray[np.int32]:
    """Rank candidates by distance, resolving every tie by lower cache index."""

    values = np.asarray(distances)
    one_query = values.ndim == 1
    if one_query:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("distances must describe at least one cache candidate")
    if shortlist <= 0:
        raise ValueError("shortlist must be positive")
    keep = min(shortlist, values.shape[1])
    order = np.argsort(values, axis=1, kind="stable")[:, :keep].astype(np.int32)
    return order[0] if one_query else order


def proposal_from_received(
    received_signs: ArrayLike,
    bank_signs: ArrayLike,
    shortlist: int = 32,
) -> Proposal:
    """Search one receiver cache using a noisy proposal sketch."""

    received = np.asarray(received_signs, dtype=bool)
    cache = np.asarray(bank_signs, dtype=bool)
    if received.ndim != 1 or cache.ndim != 2 or cache.shape[1] != len(received):
        raise ValueError("received_signs must match the cache sketch width")
    distances = np.count_nonzero(cache != received[None, :], axis=1).astype(np.int32)
    indices = ranked_candidates(distances, shortlist).astype(np.int32)
    selected_distances = distances[indices].astype(np.int32)
    received = received.copy()
    indices = indices.copy()
    selected_distances = selected_distances.copy()
    received.flags.writeable = False
    indices.flags.writeable = False
    selected_distances.flags.writeable = False
    return Proposal(received, indices, selected_distances)


def sample_proposal(
    source: ArrayLike,
    bank_signs: ArrayLike,
    planes: ArrayLike,
    bsc_probability: float,
    rng: np.random.Generator,
    shortlist: int = 32,
) -> Proposal:
    """Generate, transmit, and search one proposal sketch."""

    source_signs = hyperplane_signs(np.asarray(source)[None, :], planes)[0]
    received = binary_symmetric_channel(source_signs, bsc_probability, rng)
    return proposal_from_received(received, bank_signs, shortlist)


def cosine_for(
    query: ArrayLike,
    bank: ArrayLike,
    candidates: ArrayLike,
) -> NDArray[np.float64]:
    """Return row-wise query similarity to selected cache candidates."""

    queries = np.asarray(query)
    cache = np.asarray(bank)
    indices = np.asarray(candidates, dtype=np.int64)
    if queries.ndim != 2 or cache.ndim != 2 or queries.shape[1] != cache.shape[1]:
        raise ValueError("query and bank must have compatible feature dimensions")
    if indices.shape != (len(queries),):
        raise ValueError("candidates must contain one cache index per query")
    if np.any(indices < 0) or np.any(indices >= len(cache)):
        raise IndexError("candidate index lies outside the cache")
    return np.einsum("ij,ij->i", queries, cache[indices], dtype=np.float64)


def joint_hyperplane_signs(
    source: ArrayLike,
    candidates: ArrayLike,
    bits: int,
    rng: np.random.Generator,
) -> NDArray[np.bool_]:
    """Sample joint signs from a cosine Gram matrix in deterministic order."""

    source_vector = np.asarray(source, dtype=np.float64)
    candidate_matrix = np.asarray(candidates, dtype=np.float64)
    if source_vector.ndim != 1 or candidate_matrix.ndim != 2:
        raise ValueError("source must be a vector and candidates must be a matrix")
    if candidate_matrix.shape[1] != len(source_vector) or bits <= 0:
        raise ValueError("feature dimensions must match and bits must be positive")
    vectors = np.vstack([source_vector[None, :], candidate_matrix])
    gram = np.clip(vectors @ vectors.T, -1.0, 1.0)
    gram = (gram + gram.T) / 2.0
    factor: NDArray[np.float64] | None = None
    for jitter in (1e-12, 1e-10, 1e-8, 1e-6):
        trial = gram.copy()
        trial.flat[:: len(trial) + 1] += jitter
        lower = np.zeros_like(trial)
        valid = True
        for row in range(len(trial)):
            for column in range(row + 1):
                residual = trial[row, column] - math.fsum(
                    lower[row, prior] * lower[column, prior] for prior in range(column)
                )
                if row == column:
                    if residual <= 0.0:
                        valid = False
                        break
                    lower[row, column] = math.sqrt(residual)
                else:
                    lower[row, column] = residual / lower[column, column]
            if not valid:
                break
        if valid:
            factor = lower
            break
    if factor is None:
        raise ArithmeticError("deterministic Cholesky factorization failed")
    draws = rng.standard_normal((bits, len(vectors)))
    projections = np.einsum("bk,ik->bi", draws, factor, optimize=False)
    return np.asarray(projections >= 0.0, dtype=bool)
