"""SentrySem proposal, independent audit, calibration, and retry protocols."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .sketch import (
    binary_symmetric_channel,
    cosine_for,
    hamming_matrix,
    hyperplane_signs,
    joint_hyperplane_signs,
    random_hyperplanes,
    ranked_candidates,
    sample_proposal,
)
from .statistics import (
    alpha_spending,
    apply_audit_lookup,
    audit_lookup,
    binomial_boundary,
    binomial_cdf,
    observed_mismatch_probability,
    wilson_lower,
)
from .traffic import (
    AUDIT_SEED_BITS,
    CANDIDATE_LOCK_BITS,
    FEEDBACK_BITS,
    REFRESH_BITS,
    fixed_audit_traffic,
    frame_bits,
)

SELECTED_PROPOSAL_BITS = 256
SELECTED_AUDIT_BITS = 768
TOTAL_ALPHA = 0.01
SEQUENTIAL_PROPOSAL_BITS = 64
AUDIT_BATCH_BITS = 128
AUDIT_MAX_BITS = 768
SHORTLIST = 32


@dataclass(frozen=True)
class FixedAuditResult:
    """Vectorized result of the frozen 256/768 proposal--audit protocol."""

    protocol_seed: int
    bsc_probability: float
    proposal_bits: int
    audit_bits: int
    candidate_indices: NDArray[np.int32]
    proposal_distances: NDArray[np.int32]
    candidate_cosines: NDArray[np.float64]
    audit_mismatches: NDArray[np.int32]
    audit_boundary: int
    exact_boundary_null_probability: float
    reused: NDArray[np.bool_]
    protocol_bits: int

    @property
    def fallback_used(self) -> NDArray[np.bool_]:
        return ~self.reused

    @property
    def completion_bits(self) -> NDArray[np.int32]:
        return np.asarray(
            self.protocol_bits + self.fallback_used.astype(np.int32) * REFRESH_BITS,
            dtype=np.int32,
        )


@dataclass(frozen=True)
class SequentialAuditResult:
    """Outputs of the one- or three-candidate anytime-valid retry protocol."""

    candidate_indices: NDArray[np.int32]
    reused: NDArray[np.bool_]
    audit_bits: NDArray[np.int32]
    attempts: NDArray[np.int8]
    protocol_bits: NDArray[np.int32]


@dataclass(frozen=True)
class ThresholdCalibration:
    """Frozen similarity threshold and its disjoint tuning/validation reports."""

    status: str
    tune_status: str
    threshold_cosine: float
    threshold_angle_over_pi: float
    tune_empirical_precision_target: float
    independent_validation_lower95_target: float
    minimum_selected_pairs: int
    tune: dict[str, int | float]
    independent_validation: dict[str, int | float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _feature_matrices(query: ArrayLike, bank: ArrayLike) -> tuple[NDArray[Any], NDArray[Any]]:
    queries = np.asarray(query)
    cache = np.asarray(bank)
    if queries.ndim != 2 or cache.ndim != 2 or queries.shape[1] != cache.shape[1]:
        raise ValueError("query and bank must be compatible two-dimensional feature arrays")
    if len(queries) == 0 or len(cache) == 0:
        raise ValueError("query and bank must be non-empty")
    return queries, cache


def audit_candidate_cosines(
    candidate_cosines: ArrayLike,
    threshold_cosine: float,
    bsc_probability: float,
    audit_bits: int,
    rng: np.random.Generator,
    alpha: float = TOTAL_ALPHA,
    uniforms: ArrayLike | None = None,
) -> tuple[NDArray[np.bool_], NDArray[np.int32], int, float]:
    """Test locked candidates with an independent exact fixed-length audit."""

    cosine = np.asarray(candidate_cosines, dtype=np.float64)
    if cosine.ndim != 1 or audit_bits <= 0:
        raise ValueError("candidate_cosines must be a vector and audit_bits must be positive")
    mismatch_probability = observed_mismatch_probability(cosine, bsc_probability)
    if uniforms is None:
        random_values = rng.random((len(cosine), audit_bits))
    else:
        random_values = np.asarray(uniforms, dtype=np.float64)
        if random_values.shape[0] != len(cosine) or random_values.shape[1] < audit_bits:
            raise ValueError("uniforms do not cover every candidate and audit sign")
        random_values = random_values[:, :audit_bits]
    mismatches = np.count_nonzero(random_values < mismatch_probability[:, None], axis=1).astype(
        np.int32
    )
    null_probability = float(observed_mismatch_probability(threshold_cosine, bsc_probability))
    boundary = binomial_boundary(audit_bits, null_probability, alpha)
    null_mass = binomial_cdf(boundary, audit_bits, null_probability)
    return mismatches <= boundary, mismatches, boundary, null_mass


def run_fixed_audit(
    query: ArrayLike,
    bank: ArrayLike,
    threshold_cosine: float,
    bsc_probability: float,
    protocol_seed: int,
    proposal_bits: int = SELECTED_PROPOSAL_BITS,
    audit_bits: int = SELECTED_AUDIT_BITS,
    alpha: float = TOTAL_ALPHA,
    plane_purpose: int = 100,
    channel_purpose: int = 5200,
    audit_purpose: int = 5300,
) -> FixedAuditResult:
    """Run the selected fixed protocol using the paper's independent RNG streams."""

    queries, cache = _feature_matrices(query, bank)
    if proposal_bits <= 0 or audit_bits <= 0:
        raise ValueError("proposal_bits and audit_bits must be positive")
    probability_code = int(round(1000.0 * bsc_probability))
    plane_rng = np.random.default_rng(np.random.SeedSequence([int(protocol_seed), plane_purpose]))
    planes = random_hyperplanes(proposal_bits, queries.shape[1], plane_rng)
    bank_signs = hyperplane_signs(cache, planes)
    query_signs = hyperplane_signs(queries, planes)
    channel_rng = np.random.default_rng(
        np.random.SeedSequence([int(protocol_seed), probability_code, channel_purpose])
    )
    received = binary_symmetric_channel(query_signs, bsc_probability, channel_rng)
    distances = hamming_matrix(bank_signs, received)
    candidates = ranked_candidates(distances, 1).reshape(-1).astype(np.int32)
    proposal_distances = distances[np.arange(len(queries)), candidates].astype(np.int32)
    candidate_cosines = cosine_for(queries, cache, candidates)
    audit_rng = np.random.default_rng(
        np.random.SeedSequence([int(protocol_seed), probability_code, audit_purpose])
    )
    reused, mismatches, boundary, null_mass = audit_candidate_cosines(
        candidate_cosines,
        threshold_cosine,
        bsc_probability,
        audit_bits,
        audit_rng,
        alpha,
    )
    traffic = fixed_audit_traffic(proposal_bits, audit_bits)
    return FixedAuditResult(
        protocol_seed=int(protocol_seed),
        bsc_probability=float(bsc_probability),
        proposal_bits=proposal_bits,
        audit_bits=audit_bits,
        candidate_indices=candidates,
        proposal_distances=proposal_distances,
        candidate_cosines=candidate_cosines,
        audit_mismatches=mismatches,
        audit_boundary=boundary,
        exact_boundary_null_probability=null_mass,
        reused=np.asarray(reused, dtype=bool),
        protocol_bits=traffic.protocol_bits,
    )


def same_evidence_screen(
    minimum_distances: ArrayLike,
    sketch_bits: int,
    threshold_cosine: float,
    bsc_probability: float,
    alpha: float = TOTAL_ALPHA,
    cache_candidates: int | None = None,
) -> tuple[NDArray[np.bool_], int, float]:
    """Apply the uncorrected or Bonferroni search-and-screen comparator."""

    distances = np.asarray(minimum_distances)
    if distances.ndim != 1 or sketch_bits <= 0:
        raise ValueError("minimum_distances must be a vector and sketch_bits must be positive")
    effective_alpha = alpha
    if cache_candidates is not None:
        if cache_candidates <= 0:
            raise ValueError("cache_candidates must be positive")
        effective_alpha /= cache_candidates
    q0 = float(observed_mismatch_probability(threshold_cosine, bsc_probability))
    boundary = binomial_boundary(sketch_bits, q0, effective_alpha)
    null_mass = binomial_cdf(boundary, sketch_bits, q0)
    return np.asarray(distances <= boundary, dtype=bool), boundary, null_mass


def run_anytime_single_candidate(
    candidate_cosines: ArrayLike,
    threshold_cosine: float,
    bsc_probability: float,
    rng: np.random.Generator,
    alpha: float = TOTAL_ALPHA,
    max_bits: int = AUDIT_MAX_BITS,
    batch_bits: int = AUDIT_BATCH_BITS,
    mixture_points: int = 8,
) -> tuple[NDArray[np.bool_], NDArray[np.int32], NDArray[np.int32]]:
    """Run the finite anytime verifier for one locked candidate per query."""

    cosines = np.asarray(candidate_cosines, dtype=np.float64)
    if cosines.ndim != 1:
        raise ValueError("candidate_cosines must be a vector")
    q0 = float(observed_mismatch_probability(threshold_cosine, bsc_probability))
    lookup = audit_lookup(q0, bsc_probability, alpha, max_bits, batch_bits, mixture_points)
    probabilities = observed_mismatch_probability(cosines, bsc_probability)
    mismatch = rng.random((len(cosines), max_bits)) < probabilities[:, None]
    accepted, used = apply_audit_lookup(mismatch, lookup)
    batches = used // batch_bits
    protocol_bits = (
        frame_bits(SEQUENTIAL_PROPOSAL_BITS)
        + CANDIDATE_LOCK_BITS
        + AUDIT_SEED_BITS
        + batches * (frame_bits(batch_bits) + FEEDBACK_BITS)
    )
    return accepted, used.astype(np.int32), protocol_bits.astype(np.int32)


def run_sequential_cache_protocol(
    query: ArrayLike,
    bank: ArrayLike,
    shortlists: ArrayLike,
    threshold_cosine: float,
    bsc_probability: float,
    protocol_seed: int,
    attempts: int = 3,
    alpha: float = TOTAL_ALPHA,
    max_bits: int = AUDIT_MAX_BITS,
    batch_bits: int = AUDIT_BATCH_BITS,
    mixture_points: int = 8,
) -> SequentialAuditResult:
    """Run the predeclared one-to-three candidate sequential retry extension."""

    queries, cache = _feature_matrices(query, bank)
    candidate_lists = np.asarray(shortlists, dtype=np.int32)
    if candidate_lists.ndim != 2 or candidate_lists.shape[0] != len(queries):
        raise ValueError("shortlists must contain one ranked list per query")
    if candidate_lists.shape[1] < attempts or not 1 <= attempts <= 3:
        raise ValueError("shortlists do not support the requested number of attempts")
    if np.any(candidate_lists < 0) or np.any(candidate_lists >= len(cache)):
        raise IndexError("shortlist contains an invalid cache index")

    number_of_queries = len(queries)
    selected = np.empty(number_of_queries, dtype=np.int32)
    accepted = np.zeros(number_of_queries, dtype=bool)
    used_total = np.zeros(number_of_queries, dtype=np.int32)
    attempts_total = np.zeros(number_of_queries, dtype=np.int8)
    protocol_bits = np.full(number_of_queries, frame_bits(SEQUENTIAL_PROPOSAL_BITS), dtype=np.int32)
    q0 = float(observed_mismatch_probability(threshold_cosine, bsc_probability))
    allocations = (
        np.asarray([alpha], dtype=np.float64) if attempts == 1 else alpha_spending(alpha, attempts)
    )
    lookups = [
        audit_lookup(q0, bsc_probability, float(level), max_bits, batch_bits, mixture_points)
        for level in allocations
    ]
    probability_code = int(round(bsc_probability * 1000.0))

    for query_index in range(number_of_queries):
        local_ids = candidate_lists[query_index]
        current_position = 0
        untried = list(range(1, len(local_ids)))
        retired_received: list[NDArray[np.bool_]] = []
        retired_candidates: list[NDArray[np.bool_]] = []
        local_rng = np.random.default_rng(
            np.random.SeedSequence([int(protocol_seed), probability_code, int(query_index), 3003])
        )
        candidate_vectors = cache[local_ids]
        for attempt in range(attempts):
            attempts_total[query_index] = attempt + 1
            protocol_bits[query_index] += CANDIDATE_LOCK_BITS + AUDIT_SEED_BITS
            joint = joint_hyperplane_signs(
                queries[query_index], candidate_vectors, max_bits, local_rng
            )
            received_source = np.bitwise_xor(
                joint[:, 0], local_rng.random(max_bits) < bsc_probability
            )
            candidate_signs = joint[:, 1:]
            mismatch = received_source != candidate_signs[:, current_position]
            decision, used = apply_audit_lookup(mismatch, lookups[attempt])
            used_here = int(used[0])
            used_total[query_index] += used_here
            batches = used_here // batch_bits
            protocol_bits[query_index] += batches * (frame_bits(batch_bits) + FEEDBACK_BITS)
            if bool(decision[0]):
                accepted[query_index] = True
                break
            retired_received.append(received_source[:used_here])
            retired_candidates.append(candidate_signs[:used_here])
            if attempt == attempts - 1 or not untried:
                break
            scores = np.zeros(len(local_ids), dtype=np.int32)
            for prior_received, prior_candidates in zip(
                retired_received, retired_candidates, strict=True
            ):
                scores += np.count_nonzero(prior_candidates != prior_received[:, None], axis=0)
            remaining = np.asarray(untried, dtype=np.int32)
            winner = int(remaining[np.argmin(scores[remaining])])
            untried.remove(winner)
            current_position = winner
        selected[query_index] = int(local_ids[current_position])
    return SequentialAuditResult(selected, accepted, used_total, attempts_total, protocol_bits)


def calibration_pairs(
    vectors: ArrayLike,
    labels: ArrayLike,
    target_ids: ArrayLike,
    bank_vectors: ArrayLike,
    bank_labels: ArrayLike,
    bank_original_ids: ArrayLike,
    bank_signs: ArrayLike,
    proposal_planes: ArrayLike,
    rng: np.random.Generator,
    shortlist: int = SHORTLIST,
) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.bool_]]:
    """Collect candidate similarity and identity outcomes for threshold calibration."""

    source_vectors, cache = _feature_matrices(vectors, bank_vectors)
    source_labels = np.asarray(labels, dtype=np.int64)
    identities = np.asarray(target_ids, dtype=np.int64)
    cache_labels = np.asarray(bank_labels, dtype=np.int64)
    cache_ids = np.asarray(bank_original_ids, dtype=np.int64)
    if source_labels.shape != (len(source_vectors),) or identities.shape != (len(source_vectors),):
        raise ValueError("calibration labels and identities must align with the query vectors")
    cosines: list[float] = []
    exact_identity: list[bool] = []
    same_class: list[bool] = []
    for source, label, target_id in zip(source_vectors, source_labels, identities, strict=True):
        proposal = sample_proposal(source, bank_signs, proposal_planes, 0.0, rng, shortlist)
        candidate = int(proposal.candidate_indices[0])
        cosines.append(float(source @ cache[candidate]))
        exact_identity.append(bool(cache_ids[candidate] == target_id))
        same_class.append(bool(cache_labels[candidate] == label))
    return (
        np.asarray(cosines, dtype=np.float64),
        np.asarray(exact_identity, dtype=bool),
        np.asarray(same_class, dtype=bool),
    )


def calibration_report(
    cosines: ArrayLike,
    identities: ArrayLike,
    label_matches: ArrayLike,
    threshold_cosine: float,
) -> dict[str, int | float]:
    """Summarize candidate quality above a fixed similarity threshold."""

    similarity = np.asarray(cosines, dtype=np.float64)
    exact = np.asarray(identities, dtype=bool)
    same_class = np.asarray(label_matches, dtype=bool)
    if similarity.shape != exact.shape or similarity.shape != same_class.shape:
        raise ValueError("calibration arrays must have identical shapes")
    selected = similarity > threshold_cosine
    count = int(selected.sum())
    successes = int(exact[selected].sum())
    return {
        "pairs": len(similarity),
        "selected_pairs": count,
        "selected_fraction": float(selected.mean()),
        "identity_precision": float(exact[selected].mean()) if count else 0.0,
        "identity_precision_wilson_lower95": wilson_lower(successes, count),
        "same_class_precision": float(same_class[selected].mean()) if count else 0.0,
    }


def calibrate_threshold(
    tune_vectors: ArrayLike,
    tune_labels: ArrayLike,
    tune_target_ids: ArrayLike,
    validate_vectors: ArrayLike,
    validate_labels: ArrayLike,
    validate_target_ids: ArrayLike,
    bank_vectors: ArrayLike,
    bank_labels: ArrayLike,
    bank_original_ids: ArrayLike,
    base_seed: int = 20260907,
    proposal_bits: int = SEQUENTIAL_PROPOSAL_BITS,
    shortlist: int = SHORTLIST,
    minimum_pairs: int = 200,
    tune_precision_target: float = 0.95,
    validation_wilson_target: float = 0.90,
) -> ThresholdCalibration:
    """Select on CAL-TUNE and validate once on identity-disjoint CAL-VALIDATE."""

    cache = np.asarray(bank_vectors, dtype=np.float32)
    plane_rng = np.random.default_rng(base_seed + 100)
    proposal_planes = random_hyperplanes(proposal_bits, cache.shape[1], plane_rng)
    bank_signs = hyperplane_signs(cache, proposal_planes)
    tune = calibration_pairs(
        tune_vectors,
        tune_labels,
        tune_target_ids,
        cache,
        bank_labels,
        bank_original_ids,
        bank_signs,
        proposal_planes,
        np.random.default_rng(base_seed + 10),
        shortlist,
    )
    validation = calibration_pairs(
        validate_vectors,
        validate_labels,
        validate_target_ids,
        cache,
        bank_labels,
        bank_original_ids,
        bank_signs,
        proposal_planes,
        np.random.default_rng(base_seed + 11),
        shortlist,
    )
    tune_cosines, tune_identity, _ = tune
    candidates = np.unique(np.quantile(tune_cosines, np.linspace(0.05, 0.95, 91)))
    feasible: list[tuple[int, float, float]] = []
    for threshold in candidates:
        selected = tune_cosines > threshold
        count = int(selected.sum())
        precision = float(tune_identity[selected].mean()) if count else 0.0
        if count >= minimum_pairs and precision >= tune_precision_target:
            feasible.append((count, float(threshold), precision))
    if feasible:
        _, threshold, _ = max(feasible, key=lambda row: row[0])
        tune_status = "PASS"
    else:
        eligible = [
            value for value in candidates if int((tune_cosines > value).sum()) >= minimum_pairs
        ]
        pool = eligible if eligible else list(candidates)
        threshold = float(
            max(
                pool,
                key=lambda value: (
                    float(tune_identity[tune_cosines > value].mean())
                    if np.any(tune_cosines > value)
                    else -1.0
                ),
            )
        )
        tune_status = "FAIL_TUNE"
    tune_report = calibration_report(*tune, threshold)
    validation_report = calibration_report(*validation, threshold)
    validation_pass = (
        int(validation_report["selected_pairs"]) >= minimum_pairs
        and float(validation_report["identity_precision_wilson_lower95"])
        >= validation_wilson_target
    )
    status = "PASS" if tune_status == "PASS" and validation_pass else "FAIL_TASK_ALIGNMENT"
    return ThresholdCalibration(
        status=status,
        tune_status=tune_status,
        threshold_cosine=threshold,
        threshold_angle_over_pi=float(np.arccos(np.clip(threshold, -1.0, 1.0)) / np.pi),
        tune_empirical_precision_target=tune_precision_target,
        independent_validation_lower95_target=validation_wilson_target,
        minimum_selected_pairs=minimum_pairs,
        tune=tune_report,
        independent_validation=validation_report,
    )
