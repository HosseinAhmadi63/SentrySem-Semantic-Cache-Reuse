"""Regenerate the paper's frozen semantic embeddings from official CIFAR-10 images."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np

from .config import load_config
from .data import data_manifest, load_cifar10_splits
from .encoder import (
    EXPECTED_STATE_DIGEST,
    embed_image_splits,
    embedding_arrays,
    load_frozen_encoder,
    save_frozen_embeddings,
    seed_libraries,
)
from .hashing import file_manifest, sha256_file
from .protocol import calibrate_threshold
from .results import prepare_output_directory, write_json


def _split_indices(splits: dict, base_seed: int) -> dict:
    offsets = {
        "cal_tune": 10_000_000,
        "cal_validate": 20_000_000,
        "paired_iid": 30_000_000,
        "paired_hard_ood": 40_000_000,
        "novel_unpaired": 50_000_000,
    }
    names = {
        "dev_bank": "development_cache",
        "cal_tune": "calibration_tune",
        "cal_validate": "calibration_validate",
        "test_bank": "evaluation_cache",
        "paired_iid": "standard_queries",
        "paired_hard_ood": "strong_queries",
        "novel_unpaired": "novel_queries",
    }
    records = {}
    for source_name, public_name in names.items():
        split = splits[source_name]
        item = {
            "count": len(split),
            "source_ids": split.item_ids.astype(int).tolist(),
            "labels": split.labels.astype(int).tolist(),
        }
        if source_name in offsets:
            item["view_seeds"] = (
                (base_seed + offsets[source_name] + split.item_ids.astype(np.int64))
                .astype(object)
                .tolist()
            )
        records[public_name] = item
    labels = splits["test_bank"].labels
    rng = np.random.default_rng(base_seed + 1)
    half = []
    for label in sorted(np.unique(labels)):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        half.extend(indices[: len(indices) // 2].tolist())
    records["stress_cache_indices"] = {
        "full": list(range(len(labels))),
        "evict50": sorted(map(int, half)),
        "missing3classes": np.flatnonzero(~np.isin(labels, [0, 1, 2])).astype(int).tolist(),
    }
    return {
        "schema_version": 1,
        "base_seed": base_seed,
        "wire_id_offset": 1_000_000,
        "splits": records,
    }


def _qualitative_examples(splits: dict, arrays: dict, destination: Path) -> None:
    class_ids = (0, 3, 9)
    cache_ids = arrays["final_bank_ids"]
    rows = []
    for class_id in class_ids:
        query_index = int(np.flatnonzero(arrays["paired_iid_labels"] == class_id)[0])
        cache_index = int(
            np.flatnonzero(cache_ids == arrays["paired_iid_target_ids"][query_index])[0]
        )
        novel_index = int(np.flatnonzero(arrays["novel_unpaired_labels"] == class_id)[0])
        rows.append((class_id, query_index, cache_index, novel_index))
    np.savez_compressed(
        destination,
        class_ids=np.asarray(class_ids, dtype=np.int64),
        cached_source_ids=np.asarray(
            [arrays["paired_iid_target_ids"][query_index] for _, query_index, _, _ in rows],
            dtype=np.int64,
        ),
        novel_source_ids=np.asarray(
            [arrays["novel_unpaired_target_ids"][novel_index] for _, _, _, novel_index in rows],
            dtype=np.int64,
        ),
        cached_images=np.stack(
            [splits["test_bank"].images[cache_index] for _, _, cache_index, _ in rows]
        ),
        standard_images=np.stack(
            [splits["paired_iid"].images[query_index] for _, query_index, _, _ in rows]
        ),
        strong_images=np.stack(
            [splits["paired_hard_ood"].images[query_index] for _, query_index, _, _ in rows]
        ),
        novel_images=np.stack(
            [splits["novel_unpaired"].images[novel_index] for _, _, _, novel_index in rows]
        ),
        standard_similarity=np.asarray(
            [
                arrays["paired_iid_vectors"][query_index] @ arrays["final_bank"][cache_index]
                for _, query_index, cache_index, _ in rows
            ],
            dtype=np.float64,
        ),
        strong_similarity=np.asarray(
            [
                arrays["paired_hard_ood_vectors"][query_index] @ arrays["final_bank"][cache_index]
                for _, query_index, cache_index, _ in rows
            ],
            dtype=np.float64,
        ),
        novel_max_similarity=np.asarray(
            [
                (arrays["novel_unpaired_vectors"][novel_index] @ arrays["final_bank"].T).max()
                for _, _, _, novel_index in rows
            ],
            dtype=np.float64,
        ),
    )


def extract_embeddings(
    config_path: Path,
    output: Path,
    data_root: Path,
    download: bool = True,
) -> dict:
    """Download CIFAR-10, rebuild deterministic views, and encode every split on CPU."""

    destination = prepare_output_directory(output)
    config = load_config(config_path)
    seed_libraries(config.seed)
    splits, classes = load_cifar10_splits(data_root, config.seed, config.data, download)
    observed_data = data_manifest(splits, classes)
    reference_manifest_path = (
        Path(config_path).resolve().parents[1]
        / "artifacts"
        / "frozen_inputs"
        / "data_manifest.json"
    )
    if reference_manifest_path.is_file():
        reference_data = json.loads(reference_manifest_path.read_text())
        if observed_data["sha256_images_labels_ids"] != reference_data["sha256_images_labels_ids"]:
            raise AssertionError(
                "CIFAR-10 split or deterministic view hashes do not match the paper"
            )

    model, verifier_manifest = load_frozen_encoder("cpu", config.encoder)
    if verifier_manifest["state_dict_sha256"] != EXPECTED_STATE_DIGEST:
        raise AssertionError("ResNet-18 state digest does not match the paper encoder")
    batches = embed_image_splits(model, splits, "cpu", config.encoder)
    save_frozen_embeddings(batches, destination / "frozen_embeddings.npz")
    arrays = embedding_arrays(batches)

    calibration = calibrate_threshold(
        arrays["cal_tune"],
        arrays["cal_tune_labels"],
        arrays["cal_tune_target_ids"],
        arrays["cal_validate"],
        arrays["cal_validate_labels"],
        arrays["cal_validate_target_ids"],
        arrays["development_bank"],
        arrays["development_bank_labels"],
        arrays["development_bank_ids"],
        base_seed=config.seed,
    )
    calibration_record = {
        "status": calibration.status,
        "tune_status": calibration.tune_status,
        "threshold_cosine": calibration.threshold_cosine,
        "threshold_angle_over_pi": calibration.threshold_angle_over_pi,
        "tune_empirical_precision_target": calibration.tune_empirical_precision_target,
        "independent_validation_lower95_target": calibration.independent_validation_lower95_target,
        "minimum_selected_pairs": calibration.minimum_selected_pairs,
        "cal_tune": calibration.tune,
        "cal_validate": calibration.independent_validation,
        "selection_and_validation_identities_disjoint": True,
        "audit_max_bits_predeclared_before_final_test": 768,
    }
    write_json(destination / "data_manifest.json", observed_data)
    write_json(destination / "verifier_manifest.json", verifier_manifest)
    write_json(destination / "calibration.json", calibration_record)
    write_json(destination / "split_indices.json", _split_indices(splits, config.seed))
    _qualitative_examples(splits, arrays, destination / "qualitative_examples.npz")
    provenance = {
        "status": "PASS",
        "configuration": Path(config_path).name,
        "python": sys.version,
        "platform": platform.platform(),
        "device": "cpu",
        "frozen_embeddings_sha256": sha256_file(destination / "frozen_embeddings.npz"),
        "threshold": calibration.threshold_cosine,
    }
    write_json(destination / "extraction_provenance.json", provenance)
    write_json(destination / "artifact_manifest.json", file_manifest(destination))
    return provenance


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/cache"))
    parser.add_argument("--no-download", action="store_true")
    arguments = parser.parse_args()
    record = extract_embeddings(
        arguments.config,
        arguments.output,
        arguments.data_root,
        not arguments.no_download,
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
