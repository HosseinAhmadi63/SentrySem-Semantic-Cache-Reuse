"""Deterministic CIFAR-10 identity splits and repeated-view generation."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from PIL import Image, ImageEnhance

from .config import DataConfig, ViewConfig

ViewSeverity = Literal["standard", "strong"]


@dataclass(frozen=True)
class ImageSplit:
    """Images, semantic labels, and globally unambiguous source identities."""

    images: NDArray[np.uint8]
    labels: NDArray[np.int64]
    item_ids: NDArray[np.int64]

    def __post_init__(self) -> None:
        if self.images.ndim != 4 or self.images.shape[1:] != (32, 32, 3):
            raise ValueError("CIFAR image arrays must have shape (n, 32, 32, 3)")
        if self.labels.shape != (len(self.images),) or self.item_ids.shape != (len(self.images),):
            raise ValueError("images, labels, and item_ids must have equal length")

    def __len__(self) -> int:
        return len(self.images)


class ArrayImageDataset:
    """Minimal PyTorch-compatible dataset backed by a fixed image split."""

    def __init__(self, split: ImageSplit, transform: Callable[[Image.Image], Any]) -> None:
        self.images = np.asarray(split.images, dtype=np.uint8)
        self.labels = np.asarray(split.labels, dtype=np.int64)
        self.item_ids = np.asarray(split.item_ids, dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[Any, int, int]:
        image = Image.fromarray(self.images[index])
        return self.transform(image), int(self.labels[index]), int(self.item_ids[index])


def stratified_take(
    labels: ArrayLike,
    per_class_counts: list[int] | tuple[int, ...],
    seed: int,
) -> list[NDArray[np.int64]]:
    """Draw disjoint, class-balanced index groups and return each group sorted."""

    targets = np.asarray(labels)
    if targets.ndim != 1 or len(targets) == 0:
        raise ValueError("labels must be a non-empty vector")
    counts = tuple(int(value) for value in per_class_counts)
    if not counts or any(value < 0 for value in counts):
        raise ValueError("per_class_counts must be non-negative")
    rng = np.random.default_rng(seed)
    result: list[list[int]] = [[] for _ in counts]
    for label in sorted(np.unique(targets).tolist()):
        indices = np.flatnonzero(targets == label)
        rng.shuffle(indices)
        if sum(counts) > len(indices):
            raise ValueError(
                f"requested {sum(counts)} samples from class {label} with {len(indices)} available"
            )
        cursor = 0
        for group, count in enumerate(counts):
            result[group].extend(indices[cursor : cursor + count].tolist())
            cursor += count
    return [np.asarray(sorted(group), dtype=np.int64) for group in result]


def evolved_view(
    image_array: ArrayLike,
    seed: int,
    severity: ViewSeverity = "standard",
    config: ViewConfig | None = None,
) -> NDArray[np.uint8]:
    """Generate one deterministic translated and photometrically shifted view."""

    parameters = config or ViewConfig()
    parameters.validate()
    source = np.asarray(image_array, dtype=np.uint8)
    if source.shape != (32, 32, 3):
        raise ValueError("image_array must have shape (32, 32, 3)")
    if severity not in ("standard", "strong"):
        raise ValueError("severity must be 'standard' or 'strong'")
    rng = np.random.default_rng(seed)
    strong = severity == "strong"
    padding = parameters.strong_reflection_pad if strong else parameters.reflection_pad
    padded = np.pad(source, ((padding, padding), (padding, padding), (0, 0)), mode="reflect")
    left, top = rng.integers(0, 2 * padding + 1, size=2)
    image = Image.fromarray(padded[top : top + 32, left : left + 32])
    if rng.random() < parameters.horizontal_flip_probability:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if strong:
        fill = tuple(np.asarray(image).reshape(-1, 3).mean(axis=0).astype(np.uint8).tolist())
        image = image.rotate(
            float(rng.uniform(*parameters.strong_rotation_degrees)),
            Image.Resampling.BILINEAR,
            fillcolor=fill,
        )
        brightness = parameters.strong_brightness_range
        contrast = parameters.strong_contrast_range
        noise_sigma = parameters.strong_sensor_noise_sigma
    else:
        brightness = parameters.brightness_range
        contrast = parameters.contrast_range
        noise_sigma = parameters.sensor_noise_sigma
    image = ImageEnhance.Brightness(image).enhance(float(rng.uniform(*brightness)))
    image = ImageEnhance.Contrast(image).enhance(float(rng.uniform(*contrast)))
    shifted = np.asarray(image, dtype=np.float32)
    shifted += rng.normal(0.0, noise_sigma, shifted.shape)
    return np.clip(shifted, 0.0, 255.0).astype(np.uint8)


def make_views(
    images: ArrayLike,
    item_ids: ArrayLike,
    severity: ViewSeverity,
    seed: int,
    seed_offset: int,
    config: ViewConfig | None = None,
) -> NDArray[np.uint8]:
    """Generate one view per source using the published identity-derived seeds."""

    source = np.asarray(images, dtype=np.uint8)
    identities = np.asarray(item_ids, dtype=np.int64)
    if len(source) != len(identities):
        raise ValueError("images and item_ids must have equal length")
    return np.stack(
        [
            evolved_view(image, seed + seed_offset + int(identity), severity, config)
            for image, identity in zip(source, identities, strict=True)
        ]
    )


def build_cifar10_splits(
    train_images: ArrayLike,
    train_labels: ArrayLike,
    test_images: ArrayLike,
    test_labels: ArrayLike,
    seed: int = 20260907,
    config: DataConfig | None = None,
) -> dict[str, ImageSplit]:
    """Construct every development and evaluation split used in the paper."""

    parameters = config or DataConfig()
    parameters.validate()
    development_images = np.asarray(train_images, dtype=np.uint8)
    development_labels = np.asarray(train_labels, dtype=np.int64)
    evaluation_images = np.asarray(test_images, dtype=np.uint8)
    evaluation_labels = np.asarray(test_labels, dtype=np.int64)
    if development_images.shape != (50_000, 32, 32, 3) or development_labels.shape != (50_000,):
        raise ValueError("training arrays do not match the official CIFAR-10 split")
    if evaluation_images.shape != (10_000, 32, 32, 3) or evaluation_labels.shape != (10_000,):
        raise ValueError("test arrays do not match the official CIFAR-10 split")

    (development_ids,) = stratified_take(
        development_labels, [parameters.development_cache_per_class], seed + 2
    )
    development_cache_images = development_images[development_ids]
    development_cache_labels = development_labels[development_ids]
    tune_local, validate_local = stratified_take(
        development_cache_labels,
        [parameters.calibration_tune_per_class, parameters.calibration_validate_per_class],
        seed + 3,
    )
    if set(development_ids[tune_local]) & set(development_ids[validate_local]):
        raise AssertionError("CAL-TUNE and CAL-VALIDATE identities overlap")

    _, final_bank_ids, novel_ids = stratified_take(
        evaluation_labels,
        [
            parameters.final_holdout_skip_per_class,
            parameters.evaluation_cache_per_class,
            parameters.novel_queries_per_class,
        ],
        seed + 4,
    )
    final_bank_images = evaluation_images[final_bank_ids]
    final_bank_labels = evaluation_labels[final_bank_ids]
    (paired_local,) = stratified_take(
        final_bank_labels, [parameters.repeated_queries_per_class], seed + 5
    )
    final_wire_ids = parameters.test_identity_offset + final_bank_ids
    paired_wire_ids = final_wire_ids[paired_local]
    novel_wire_ids = parameters.test_identity_offset + novel_ids
    if set(paired_wire_ids) & set(novel_wire_ids):
        raise AssertionError("repeated and novel evaluation identities overlap")

    splits = {
        "dev_bank": ImageSplit(
            development_cache_images,
            development_cache_labels,
            development_ids,
        ),
        "cal_tune": ImageSplit(
            make_views(
                development_cache_images[tune_local],
                development_ids[tune_local],
                "standard",
                seed,
                parameters.tune_view_seed_offset,
                parameters.views,
            ),
            development_cache_labels[tune_local],
            development_ids[tune_local],
        ),
        "cal_validate": ImageSplit(
            make_views(
                development_cache_images[validate_local],
                development_ids[validate_local],
                "standard",
                seed,
                parameters.validate_view_seed_offset,
                parameters.views,
            ),
            development_cache_labels[validate_local],
            development_ids[validate_local],
        ),
        "test_bank": ImageSplit(final_bank_images, final_bank_labels, final_wire_ids),
        "paired_iid": ImageSplit(
            make_views(
                final_bank_images[paired_local],
                paired_wire_ids,
                "standard",
                seed,
                parameters.repeated_view_seed_offset,
                parameters.views,
            ),
            final_bank_labels[paired_local],
            paired_wire_ids,
        ),
        "paired_hard_ood": ImageSplit(
            make_views(
                final_bank_images[paired_local],
                paired_wire_ids,
                "strong",
                seed,
                parameters.strong_view_seed_offset,
                parameters.views,
            ),
            final_bank_labels[paired_local],
            paired_wire_ids,
        ),
        "novel_unpaired": ImageSplit(
            make_views(
                evaluation_images[novel_ids],
                novel_wire_ids,
                "standard",
                seed,
                parameters.novel_view_seed_offset,
                parameters.views,
            ),
            evaluation_labels[novel_ids],
            novel_wire_ids,
        ),
    }
    expected_sizes = {
        "dev_bank": 1000,
        "cal_tune": 500,
        "cal_validate": 500,
        "test_bank": 1000,
        "paired_iid": 500,
        "paired_hard_ood": 500,
        "novel_unpaired": 200,
    }
    if (
        parameters == DataConfig()
        and {name: len(split) for name, split in splits.items()} != expected_sizes
    ):
        raise AssertionError("publication split sizes changed")
    return splits


def load_cifar10_splits(
    data_root: str | Path,
    seed: int = 20260907,
    config: DataConfig | None = None,
    download: bool = True,
) -> tuple[dict[str, ImageSplit], list[str]]:
    """Load official CIFAR-10 arrays and construct the publication splits."""

    try:
        from torchvision.datasets import CIFAR10
    except ImportError as exc:
        raise RuntimeError("torchvision is required to load CIFAR-10") from exc
    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    train = CIFAR10(root, train=True, download=download)
    test = CIFAR10(root, train=False, download=download)
    splits = build_cifar10_splits(
        train.data,
        np.asarray(train.targets, dtype=np.int64),
        test.data,
        np.asarray(test.targets, dtype=np.int64),
        seed,
        config,
    )
    return splits, list(train.classes)


def split_digest(split: ImageSplit) -> str:
    """Reproduce the image-label-identity SHA-256 recorded in the data manifest."""

    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(split.images).tobytes())
    digest.update(np.ascontiguousarray(split.labels).tobytes())
    digest.update(np.ascontiguousarray(split.item_ids).tobytes())
    return digest.hexdigest()


def data_manifest(splits: dict[str, ImageSplit], classes: list[str]) -> dict[str, Any]:
    """Build the publication data provenance record."""

    return {
        "source": "CIFAR-10 paired evolving views",
        "classes": list(classes),
        "sizes": {name: len(split) for name, split in splits.items()},
        "sha256_images_labels_ids": {name: split_digest(split) for name, split in splits.items()},
        "calibration_query_identities_disjoint": True,
        "development_and_final_cifar_splits_disjoint": True,
        "final_holdout_skip_per_class": 200,
        "final_conditions": ["paired_iid", "paired_hard_ood", "novel_unpaired"],
    }


FROZEN_EMBEDDING_KEYS = (
    "development_bank",
    "development_bank_labels",
    "development_bank_ids",
    "cal_tune",
    "cal_tune_labels",
    "cal_tune_target_ids",
    "cal_validate",
    "cal_validate_labels",
    "cal_validate_target_ids",
    "final_bank",
    "final_bank_labels",
    "final_bank_ids",
    "paired_iid_vectors",
    "paired_iid_labels",
    "paired_iid_target_ids",
    "paired_hard_ood_vectors",
    "paired_hard_ood_labels",
    "paired_hard_ood_target_ids",
    "novel_unpaired_vectors",
    "novel_unpaired_labels",
    "novel_unpaired_target_ids",
)


def load_frozen_embeddings(path: str | Path, validate: bool = True) -> dict[str, NDArray[Any]]:
    """Load the frozen paper features and verify their schema and normalization."""

    with np.load(Path(path), allow_pickle=False) as archive:
        missing = set(FROZEN_EMBEDDING_KEYS) - set(archive.files)
        if missing:
            raise ValueError(f"frozen embedding archive is missing: {', '.join(sorted(missing))}")
        arrays = {name: np.asarray(archive[name]) for name in FROZEN_EMBEDDING_KEYS}
    if validate:
        expected_feature_shapes = {
            "development_bank": (1000, 512),
            "cal_tune": (500, 512),
            "cal_validate": (500, 512),
            "final_bank": (1000, 512),
            "paired_iid_vectors": (500, 512),
            "paired_hard_ood_vectors": (500, 512),
            "novel_unpaired_vectors": (200, 512),
        }
        for name, shape in expected_feature_shapes.items():
            if arrays[name].shape != shape or arrays[name].dtype != np.float32:
                raise ValueError(
                    f"{name} has {arrays[name].shape}/{arrays[name].dtype}, expected {shape}/float32"
                )
            norms = np.linalg.norm(arrays[name], axis=1)
            if not np.allclose(norms, 1.0, atol=2e-5):
                raise ValueError(f"{name} contains non-unit semantic features")
        for name in FROZEN_EMBEDDING_KEYS:
            if name not in expected_feature_shapes and arrays[name].dtype != np.int64:
                raise ValueError(f"{name} must use int64 identity or class values")
    return arrays
