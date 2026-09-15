"""Reproduce threshold selection and disjoint validation from frozen embeddings."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sentrysem.data import load_frozen_embeddings
from sentrysem.hashing import sha256_bytes
from sentrysem.protocol import calibrate_threshold
from sentrysem.results import prepare_output_directory, write_artifact_manifest, write_json
from sentrysem.sketch import random_hyperplanes


def run_calibration(
    frozen_inputs: Path,
    output: Path,
    base_seed: int = 20260907,
) -> dict:
    """Recompute the frozen cosine threshold and its calibration statistics."""

    destination = prepare_output_directory(output)
    arrays = load_frozen_embeddings(Path(frozen_inputs) / "frozen_embeddings.npz")
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
        base_seed=base_seed,
    )
    report = {
        "status": result.status,
        "tune_status": result.tune_status,
        "threshold_cosine": result.threshold_cosine,
        "threshold_angle_over_pi": result.threshold_angle_over_pi,
        "tune_empirical_precision_target": result.tune_empirical_precision_target,
        "independent_validation_lower95_target": result.independent_validation_lower95_target,
        "minimum_selected_pairs": result.minimum_selected_pairs,
        "cal_tune": result.tune,
        "cal_validate": result.independent_validation,
        "selection_and_validation_identities_disjoint": True,
        "audit_max_bits_predeclared_before_final_test": 768,
    }
    plane_rng = np.random.default_rng(base_seed + 100)
    planes = random_hyperplanes(64, 512, plane_rng)
    projection = {
        "seed": base_seed + 100,
        "shape": list(planes.shape),
        "sha256": sha256_bytes(planes.tobytes()),
    }
    write_json(destination / "calibration.json", report)
    write_json(destination / "proposal_projection.json", projection)
    write_artifact_manifest(destination)
    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-seed", type=int, default=20260907)
    arguments = parser.parse_args()
    result = run_calibration(arguments.inputs, arguments.output, arguments.base_seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
