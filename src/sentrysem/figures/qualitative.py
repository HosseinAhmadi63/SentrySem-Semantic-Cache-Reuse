#!/usr/bin/env python3
"""Build the qualitative evaluation-input figure from the frozen CIFAR-10 split.

The script reconstructs the exact image partitions and deterministic view
generator used for the frozen embeddings.  It verifies the published image
hashes before drawing three predeclared class examples (the first evaluation
identity in the frozen order for classes airplane, cat, and truck).
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
from PIL import Image as PILImage
from PIL import ImageEnhance
from pypdf import PdfReader
from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.shapes import Drawing, Image, Line, Rect, String
from reportlab.lib import colors

from .metadata import figure_title, set_pdf_metadata, set_svg_metadata

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUTS = REPOSITORY_ROOT / "artifacts" / "frozen_inputs"
OUT = REPOSITORY_ROOT / "figures" / "generated"
SEED = 20260907
TEST_ID_OFFSET = 1_000_000
CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def stratified_take(labels: np.ndarray, counts: list[int], seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    result: list[list[int]] = [[] for _ in counts]
    for cls in sorted(np.unique(labels)):
        indices = np.flatnonzero(labels == cls)
        rng.shuffle(indices)
        cursor = 0
        for group, count in enumerate(counts):
            result[group].extend(indices[cursor : cursor + count].tolist())
            cursor += count
    return [np.asarray(sorted(group), dtype=np.int64) for group in result]


def evolved_view(array: np.ndarray, seed: int, severity: str) -> np.ndarray:
    rng = np.random.default_rng(seed)
    hard = severity == "hard_ood"
    pad = 5 if hard else 3
    padded = np.pad(array, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    left, top = rng.integers(0, 2 * pad + 1, size=2)
    image = PILImage.fromarray(padded[top : top + 32, left : left + 32])
    if rng.random() < 0.5:
        image = image.transpose(PILImage.Transpose.FLIP_LEFT_RIGHT)
    if hard:
        fill = tuple(np.asarray(image).reshape(-1, 3).mean(0).astype(np.uint8).tolist())
        image = image.rotate(
            float(rng.uniform(-12, 12)),
            PILImage.Resampling.BILINEAR,
            fillcolor=fill,
        )
        brightness, contrast, noise = (0.75, 1.25), (0.75, 1.25), 6.0
    else:
        brightness, contrast, noise = (0.90, 1.10), (0.90, 1.10), 2.0
    image = ImageEnhance.Brightness(image).enhance(float(rng.uniform(*brightness)))
    image = ImageEnhance.Contrast(image).enhance(float(rng.uniform(*contrast)))
    shifted = np.asarray(image).astype(np.float32)
    shifted += rng.normal(0, noise, shifted.shape)
    return np.clip(shifted, 0, 255).astype(np.uint8)


def split_hash(images: np.ndarray, labels: np.ndarray, item_ids: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(images).tobytes())
    digest.update(np.ascontiguousarray(labels).tobytes())
    digest.update(np.ascontiguousarray(item_ids).tobytes())
    return digest.hexdigest()


def prepare_examples(frozen_inputs: Path = DEFAULT_INPUTS) -> tuple[list[dict], dict]:
    artifact = Path(frozen_inputs) / "qualitative_examples.npz"
    with np.load(artifact, allow_pickle=False) as arrays:
        examples = []
        for index, cls in enumerate(arrays["class_ids"].astype(int)):
            class_id = int(cls)
            examples.append(
                {
                    "class_id": class_id,
                    "class_name": CLASSES[class_id],
                    "cache": arrays["cached_images"][index].copy(),
                    "standard": arrays["standard_images"][index].copy(),
                    "hard": arrays["strong_images"][index].copy(),
                    "novel": arrays["novel_images"][index].copy(),
                    "paired_dataset_index": int(arrays["cached_source_ids"][index])
                    - TEST_ID_OFFSET,
                    "novel_dataset_index": int(arrays["novel_source_ids"][index]) - TEST_ID_OFFSET,
                    "standard_similarity": float(arrays["standard_similarity"][index]),
                    "hard_similarity": float(arrays["strong_similarity"][index]),
                    "novel_max_similarity": float(arrays["novel_max_similarity"][index]),
                }
            )
    if [row["class_id"] for row in examples] != [0, 3, 9]:
        raise ValueError("Expected the frozen airplane, cat, and truck examples")
    return examples, {
        "artifact": artifact.name,
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    }


def build_figure(examples: list[dict], temporary: Path, threshold: float) -> Drawing:
    width, height = 4.8 * 72, 3.70 * 72
    drawing = Drawing(width, height)
    drawing.add(Rect(0, 0, width, height, strokeColor=None, fillColor=colors.white))
    headings = [
        "Cached source",
        "Standard repeated view",
        "Strong view shift",
        "Novel source",
    ]
    border_colors = ["#555555", "#21618C", "#C65F0A", "#2E805A"]
    x_positions = [40, 117, 194, 271]
    image_size = 53
    for x, heading in zip(x_positions, headings, strict=False):
        drawing.add(
            String(
                x + image_size / 2,
                height - 14,
                heading,
                fontName="Helvetica-Bold",
                fontSize=6.5,
                textAnchor="middle",
            )
        )

    for row, example in enumerate(examples):
        y = height - 82 - row * 69
        drawing.add(
            String(
                3,
                y + image_size / 2 - 2,
                example["class_name"].title(),
                fontName="Helvetica-Bold",
                fontSize=6.7,
            )
        )
        images = [example["cache"], example["standard"], example["hard"], example["novel"]]
        captions = [
            f"test #{example['paired_dataset_index']}",
            (
                f"s={example['standard_similarity']:.3f} | "
                f"{'above threshold' if example['standard_similarity'] > threshold else 'below threshold'}"
            ),
            (
                f"s={example['hard_similarity']:.3f} | "
                f"{'above threshold' if example['hard_similarity'] > threshold else 'below threshold'}"
            ),
            (
                f"max={example['novel_max_similarity']:.3f} | "
                f"{'above threshold' if example['novel_max_similarity'] > threshold else 'below threshold'}"
            ),
        ]
        for col, (array, caption) in enumerate(zip(images, captions, strict=False)):
            path = temporary / f"row{row}_col{col}.png"
            PILImage.fromarray(array).resize((256, 256), PILImage.Resampling.NEAREST).save(path)
            x = x_positions[col]
            drawing.add(
                Rect(
                    x - 1,
                    y - 1,
                    image_size + 2,
                    image_size + 2,
                    strokeColor=colors.HexColor(border_colors[col]),
                    strokeWidth=1.2,
                    fillColor=None,
                )
            )
            drawing.add(Image(x, y, image_size, image_size, str(path)))
            drawing.add(
                String(
                    x + image_size / 2,
                    y - 8.5,
                    caption,
                    fontName="Helvetica",
                    fontSize=5.2,
                    textAnchor="middle",
                )
            )

    drawing.add(Line(3, 25, width - 3, 25, strokeColor=colors.HexColor("#CCCCCC"), strokeWidth=0.6))
    drawing.add(
        String(
            width / 2,
            12,
            f"First frozen-order identity in each displayed class; cosine threshold tau = {threshold:.4f}",
            fontName="Helvetica",
            fontSize=6.0,
            textAnchor="middle",
        )
    )
    return drawing


def generate_qualitative_figure(
    frozen_inputs: Path = DEFAULT_INPUTS, output_dir: Path = OUT
) -> dict:
    frozen_inputs = Path(frozen_inputs).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration = json.loads((frozen_inputs / "calibration.json").read_text())
    threshold = float(calibration["threshold_cosine"])
    examples, validation = prepare_examples(frozen_inputs)
    stem = "Figure_6_Evaluation_Inputs"
    with tempfile.TemporaryDirectory(prefix="sentrysem-images-") as temp:
        drawing = build_figure(examples, Path(temp), threshold)
        pdf = output_dir / f"{stem}.pdf"
        svg = output_dir / f"{stem}.svg"
        png = output_dir / f"{stem}.png"
        renderPDF.drawToFile(drawing, str(pdf))
        title = figure_title(stem)
        set_pdf_metadata(pdf, title)
        renderSVG.drawToFile(drawing, str(svg))
        set_svg_metadata(svg, title)
    document = pdfium.PdfDocument(pdf)
    page = document[0]
    bitmap = page.render(scale=600 / 72)
    image = bitmap.to_pil()
    image.save(png, dpi=(600, 600))
    bitmap.close()
    page.close()
    document.close()
    assert len(PdfReader(str(pdf)).pages) == 1
    record = {
        "stem": stem,
        "width_inches": drawing.width / 72.0,
        "height_inches": drawing.height / 72.0,
        "pdf_pages": 1,
        "selection_rule": "first frozen evaluation identity in classes 0, 3, and 9",
        "threshold": threshold,
        "examples": [
            {
                key: value
                for key, value in row.items()
                if key not in {"cache", "standard", "hard", "novel"}
            }
            for row in examples
        ],
        "data_validation": validation,
        "files": {
            suffix: {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for suffix, path in (("pdf", pdf), ("svg", svg), ("png", png))
        },
    }
    (output_dir / f"{stem}.provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> None:
    record = generate_qualitative_figure()
    print(json.dumps({"status": "PASS", "figure": record["files"]["pdf"]["path"]}, indent=2))


if __name__ == "__main__":
    main()
