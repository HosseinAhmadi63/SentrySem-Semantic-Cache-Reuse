"""Frozen ResNet-18 semantic encoder used to create the paper features."""

from __future__ import annotations

import platform
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import EncoderConfig
from .data import ArrayImageDataset, ImageSplit
from .hashing import sha256_model_state

EXPECTED_PARAMETER_COUNT = 11_176_512
EXPECTED_STATE_DIGEST = "d7ecbcb063a266bf46ab86ce4e0d141732ba0d286c3fb25a8d55783bda06e41e"


@dataclass(frozen=True)
class EmbeddingBatch:
    """Unit-normalized features with their class and source identities."""

    vectors: NDArray[np.float32]
    labels: NDArray[np.int64]
    item_ids: NDArray[np.int64]


def seed_libraries(seed: int = 20260907) -> None:
    """Seed Python, NumPy, and PyTorch before feature extraction."""

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to construct semantic features") from exc
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def preprocessing_transform(config: EncoderConfig | None = None) -> Any:
    """Create the exact resize, tensor conversion, and ImageNet normalization."""

    parameters = config or EncoderConfig()
    parameters.validate()
    try:
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode
    except ImportError as exc:
        raise RuntimeError("TorchVision is required for semantic preprocessing") from exc
    return transforms.Compose(
        [
            transforms.Resize(
                (parameters.input_pixels, parameters.input_pixels),
                interpolation=InterpolationMode.BILINEAR,
                antialias=parameters.antialias,
            ),
            transforms.ToTensor(),
            transforms.Normalize(parameters.normalization_mean, parameters.normalization_std),
        ]
    )


def load_frozen_encoder(
    device: str = "cpu",
    config: EncoderConfig | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Load ImageNet-1K V1 ResNet-18, remove its classifier, and freeze it."""

    parameters = config or EncoderConfig()
    parameters.validate()
    try:
        import torch
        from torch import nn
        from torchvision import models
    except ImportError as exc:
        raise RuntimeError("PyTorch and TorchVision are required to load the encoder") from exc

    class FrozenSemanticVerifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            backbone.fc = nn.Identity()
            self.backbone = backbone.eval()
            for parameter in self.parameters():
                parameter.requires_grad_(False)

        def forward(self, images: Any) -> Any:
            return nn.functional.normalize(self.backbone(images), dim=1)

    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access a CUDA device")
    model = FrozenSemanticVerifier().to(selected_device).eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if parameter_count != EXPECTED_PARAMETER_COUNT or trainable_count != 0:
        raise AssertionError("frozen ResNet-18 parameter inventory changed")
    state_digest = sha256_model_state(model.state_dict())
    if state_digest != EXPECTED_STATE_DIGEST:
        raise RuntimeError(
            "the loaded ResNet-18 state does not match the frozen publication encoder"
        )
    manifest = {
        "tag": "Frozen-ResNet18-ImageNetV1-cosine-v1",
        "state_dict_sha256": state_digest,
        "embedding_dim": parameters.embedding_dimension,
        "pretrained": True,
        "weights": "ResNet18_Weights.IMAGENET1K_V1",
        "architecture": "TorchVision ResNet-18 with classification layer replaced by identity",
        "parameters": parameter_count,
        "trainable_parameters": trainable_count,
        "trained_by_this_project": False,
    }
    return model, manifest


def embed_dataset(
    model: Any,
    dataset: Any,
    device: str = "cpu",
    batch_size: int | None = None,
    workers: int | None = None,
) -> EmbeddingBatch:
    """Encode a dataset in deterministic order and return float32 unit vectors."""

    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to encode datasets") from exc
    selected_device = torch.device(device)
    if batch_size is None:
        batch_size = 256 if selected_device.type == "cuda" else 64
    if workers is None:
        workers = 0 if platform.system() == "Darwin" else 2
    if batch_size <= 0 or workers < 0:
        raise ValueError("batch_size must be positive and workers must be non-negative")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=selected_device.type == "cuda",
    )
    vector_parts: list[NDArray[np.float32]] = []
    label_parts: list[NDArray[np.int64]] = []
    identity_parts: list[NDArray[np.int64]] = []
    model.eval()
    with torch.inference_mode():
        for images, labels, identities in loader:
            vector_parts.append(model(images.to(selected_device)).cpu().numpy())
            label_parts.append(labels.numpy())
            identity_parts.append(identities.numpy())
    if not vector_parts:
        raise ValueError("cannot encode an empty dataset")
    vectors = np.concatenate(vector_parts).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True).clip(1e-12)
    return EmbeddingBatch(
        vectors,
        np.concatenate(label_parts).astype(np.int64),
        np.concatenate(identity_parts).astype(np.int64),
    )


def embed_image_splits(
    model: Any,
    splits: dict[str, ImageSplit],
    device: str = "cpu",
    config: EncoderConfig | None = None,
    batch_size: int | None = None,
    workers: int | None = None,
) -> dict[str, EmbeddingBatch]:
    """Encode all seven frozen data partitions with one shared preprocessing map."""

    transform = preprocessing_transform(config)
    result: dict[str, EmbeddingBatch] = {}
    for name in (
        "dev_bank",
        "cal_tune",
        "cal_validate",
        "test_bank",
        "paired_iid",
        "paired_hard_ood",
        "novel_unpaired",
    ):
        alternative = {"dev_bank": "development_bank", "test_bank": "final_bank"}.get(name, name)
        key = name if name in splits else alternative
        if key not in splits:
            raise ValueError(f"image split is missing: {name}")
        dataset = ArrayImageDataset(splits[key], transform)
        result[name] = embed_dataset(model, dataset, device, batch_size, workers)
    return result


def embedding_arrays(batches: dict[str, EmbeddingBatch]) -> dict[str, NDArray[Any]]:
    """Convert encoded partitions to the stable NPZ schema used by all experiments."""

    required = {
        "dev_bank",
        "cal_tune",
        "cal_validate",
        "test_bank",
        "paired_iid",
        "paired_hard_ood",
        "novel_unpaired",
    }
    missing = required - set(batches)
    if missing:
        raise ValueError(f"encoded partitions are missing: {', '.join(sorted(missing))}")
    arrays: dict[str, NDArray[Any]] = {
        "development_bank": batches["dev_bank"].vectors,
        "development_bank_labels": batches["dev_bank"].labels,
        "development_bank_ids": batches["dev_bank"].item_ids,
        "cal_tune": batches["cal_tune"].vectors,
        "cal_tune_labels": batches["cal_tune"].labels,
        "cal_tune_target_ids": batches["cal_tune"].item_ids,
        "cal_validate": batches["cal_validate"].vectors,
        "cal_validate_labels": batches["cal_validate"].labels,
        "cal_validate_target_ids": batches["cal_validate"].item_ids,
        "final_bank": batches["test_bank"].vectors,
        "final_bank_labels": batches["test_bank"].labels,
        "final_bank_ids": batches["test_bank"].item_ids,
    }
    for condition in ("paired_iid", "paired_hard_ood", "novel_unpaired"):
        arrays[f"{condition}_vectors"] = batches[condition].vectors
        arrays[f"{condition}_labels"] = batches[condition].labels
        arrays[f"{condition}_target_ids"] = batches[condition].item_ids
    return arrays


def save_frozen_embeddings(batches: dict[str, EmbeddingBatch], path: str | Path) -> None:
    """Write all semantic features in the compressed publication NPZ schema."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **embedding_arrays(batches))
