from __future__ import annotations

import json

import numpy as np

from sentrysem.data import load_frozen_embeddings
from sentrysem.protocol import calibrate_threshold, run_fixed_audit


def test_calibration_reproduces_frozen_threshold(frozen_inputs) -> None:
    arrays = load_frozen_embeddings(frozen_inputs / "frozen_embeddings.npz")
    result = calibrate_threshold(
        arrays["cal_tune"],
        arrays["cal_tune_labels"],
        arrays["cal_tune_target_ids"],
        arrays["cal_validate"],
        arrays["cal_validate_labels"],
        arrays["cal_validate_target_ids"],
        arrays["development_bank"],
        arrays["development_bank_labels"],
        arrays["development_bank_ids"],
    )
    assert result.status == "PASS"
    assert result.threshold_cosine == 0.7717004776000975
    assert result.tune["selected_pairs"] == 305
    assert result.independent_validation["selected_pairs"] == 288


def test_audit_stream_does_not_change_locked_candidate(frozen_inputs) -> None:
    arrays = load_frozen_embeddings(frozen_inputs / "frozen_embeddings.npz")
    query = arrays["paired_iid_vectors"][:12]
    bank = arrays["final_bank"]
    threshold = json.loads((frozen_inputs / "calibration.json").read_text())["threshold_cosine"]
    first = run_fixed_audit(query, bank, threshold, 0.05, 4108815543219190477)
    second = run_fixed_audit(
        query,
        bank,
        threshold,
        0.05,
        4108815543219190477,
        audit_purpose=5301,
    )
    assert np.array_equal(first.candidate_indices, second.candidate_indices)
    assert not np.array_equal(first.audit_mismatches, second.audit_mismatches)
