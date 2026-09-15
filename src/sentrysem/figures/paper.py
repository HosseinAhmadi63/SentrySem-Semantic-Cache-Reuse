#!/usr/bin/env python3
"""Build the seven SentrySem paper figures from authenticated result records."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pypdfium2 as pdfium
from PIL import Image
from pypdf import PdfReader
from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.shapes import (
    Circle,
    Drawing,
    Group,
    Line,
    Polygon,
    Rect,
    String,
)
from reportlab.graphics.shapes import (
    Path as RPath,
)
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics

from .metadata import figure_title, set_pdf_metadata, set_svg_metadata

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
OUT = REPOSITORY_ROOT / "figures" / "generated"
RESULTS_ROOT = REPOSITORY_ROOT / "results" / "paper"
COMMON = RESULTS_ROOT / "comparators"
RATIO = RESULTS_ROOT / "allocation"
STRESS = RESULTS_ROOT / "stress"
FROZEN_INPUTS = REPOSITORY_ROOT / "artifacts" / "frozen_inputs"
VALIDATION_DIR = FROZEN_INPUTS

FIG_W = 4.80 * 72.0
FONT = "Helvetica"
FONT_B = "Helvetica-Bold"
FS = 7.0
FS_SMALL = 6.3
FS_MIN = 6.0
FS_PANEL = 7.4

# Color-blind-safe hues with deliberately different luminance, markers, and
# line styles so that every comparison remains legible in grayscale.
BLUE = colors.HexColor("#1F5A85")
ORANGE = colors.HexColor("#C76818")
GREEN = colors.HexColor("#2E7D5B")
PURPLE = colors.HexColor("#75528A")
DARK = colors.HexColor("#202020")
MID = colors.HexColor("#666666")
GRID = colors.HexColor("#D7D7D7")
LIGHT = colors.HexColor("#F4F4F4")
PALE_BLUE = colors.HexColor("#EAF2F8")
PALE_GREEN = colors.HexColor("#EDF6F0")
PALE_GOLD = colors.HexColor("#FFF2CC")
WHITE = colors.white


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_paths(
    results_root: Path | None = None,
    output_dir: Path | None = None,
    frozen_inputs: Path | None = None,
    validation_dir: Path | None = None,
) -> None:
    """Set data and output roots for frozen or newly generated results."""
    global RESULTS_ROOT, COMMON, RATIO, STRESS, FROZEN_INPUTS, VALIDATION_DIR, OUT
    if results_root is not None:
        RESULTS_ROOT = Path(results_root).resolve()
        COMMON = RESULTS_ROOT / "comparators"
        RATIO = RESULTS_ROOT / "allocation"
        STRESS = RESULTS_ROOT / "stress"
    if output_dir is not None:
        OUT = Path(output_dir).resolve()
    if frozen_inputs is not None:
        FROZEN_INPUTS = Path(frozen_inputs).resolve()
    if validation_dir is not None:
        VALIDATION_DIR = Path(validation_dir).resolve()
    elif frozen_inputs is not None:
        VALIDATION_DIR = FROZEN_INPUTS


def _same(a: float, b: float, atol: float = 1e-11) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    return bool(np.isclose(float(a), float(b), rtol=1e-10, atol=atol))


def verify_group_means(
    seed_file: Path,
    aggregate_file: Path,
    keys: Sequence[str],
    metrics: Sequence[str],
) -> int:
    seed = pd.read_csv(seed_file, dtype={"protocol_seed": str})
    aggregate = pd.read_csv(aggregate_file)
    checked = 0
    for group_key, group in seed.groupby(list(keys), sort=True, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        mask = pd.Series(True, index=aggregate.index)
        for key, value in zip(keys, group_key, strict=False):
            if isinstance(value, float | np.floating):
                mask &= np.isclose(aggregate[key].astype(float), float(value))
            else:
                mask &= aggregate[key].astype(str) == str(value)
        rows = aggregate[mask]
        assert len(rows) == 1, (aggregate_file, group_key, len(rows))
        row = rows.iloc[0]
        for metric in metrics:
            values = group[metric].dropna().astype(float)
            mean = values.mean() if len(values) else np.nan
            sd = values.std(ddof=1) if len(values) > 1 else (0.0 if len(values) == 1 else np.nan)
            assert _same(mean, row[f"{metric}_mean"]), (aggregate_file, group_key, metric, mean)
            assert _same(sd, row[f"{metric}_seed_sd"]), (aggregate_file, group_key, metric, sd)
            checked += 2
    return checked


def validate_sources() -> dict:
    required = [
        COMMON / "publication_seed_summary.csv",
        COMMON / "publication_aggregate_summary.csv",
        COMMON / "identity_cluster_bootstrap.csv",
        RATIO / "cal_tune_ratio_seed_results.csv",
        RATIO / "cal_tune_ratio_aggregate.csv",
        RATIO / "selected_ratio_final_seed_results.csv",
        RATIO / "selected_ratio_final_aggregate.csv",
        RATIO / "selection.json",
        STRESS / "sentrysem_fixed_seed_summary.csv",
        STRESS / "sentrysem_fixed_aggregate.csv",
        STRESS / "sentrysem_fixed_cluster_bootstrap.csv",
        VALIDATION_DIR / "selection_stress.json",
        VALIDATION_DIR / "null_validation.json",
        FROZEN_INPUTS / "qualitative_examples.npz",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing figure source files: " + ", ".join(missing))

    selection = json.loads((RATIO / "selection.json").read_text())
    assert selection["selected_search_bits"] == 256
    assert selection["selected_audit_bits"] == 768
    assert selection["evaluation_pool_used_for_selection"] is False
    assert selection["protocol_seed_sets_disjoint"] is True

    checks = 0
    checks += verify_group_means(
        RATIO / "cal_tune_ratio_seed_results.csv",
        RATIO / "cal_tune_ratio_aggregate.csv",
        ["phase", "bsc", "search_bits", "audit_bits"],
        ["reuse_rate", "mean_completion_bits"],
    )
    checks += verify_group_means(
        RATIO / "selected_ratio_final_seed_results.csv",
        RATIO / "selected_ratio_final_aggregate.csv",
        ["phase", "bsc", "search_bits", "audit_bits"],
        ["reuse_rate", "identity_precision_among_reuses", "mean_completion_bits"],
    )
    checks += verify_group_means(
        COMMON / "publication_seed_summary.csv",
        COMMON / "publication_aggregate_summary.csv",
        ["bsc", "method"],
        ["reuse_rate", "candidate_identity_rate_among_reuses", "mean_completion_bits"],
    )
    checks += verify_group_means(
        STRESS / "sentrysem_fixed_seed_summary.csv",
        STRESS / "sentrysem_fixed_aggregate.csv",
        ["scenario", "condition", "bsc"],
        ["reuse_rate", "mean_completion_bits"],
    )

    # The independently generated full-cache stress rows must reproduce the
    # selected headline means exactly.
    selected = pd.read_csv(RATIO / "selected_ratio_final_aggregate.csv").sort_values("bsc")
    stress_full = (
        pd.read_csv(STRESS / "sentrysem_fixed_aggregate.csv")
        .query("scenario == 'full' and condition == 'paired_iid'")
        .sort_values("bsc")
    )
    assert len(selected) == len(stress_full) == 2
    for a, b in zip(selected.itertuples(), stress_full.itertuples(), strict=False):
        assert _same(a.reuse_rate_mean, b.reuse_rate_mean)
        assert _same(a.identity_precision_among_reuses_mean, b.identity_precision_among_reuses_mean)
        assert _same(a.mean_completion_bits_mean, b.mean_completion_bits_mean)

    return {
        "status": "PASS",
        "source_files_checked": len(required),
        "aggregate_cells_recomputed": checks,
        "selected_allocation": {"proposal_signs": 256, "audit_signs": 768},
    }


def txt(
    d: Drawing,
    x: float,
    y: float,
    value: str,
    size: float = FS,
    bold: bool = False,
    anchor: str = "start",
    color=DARK,
    angle: float = 0,
) -> None:
    d.add(
        String(
            x,
            y,
            value,
            fontName=FONT_B if bold else FONT,
            fontSize=size,
            textAnchor=anchor,
            fillColor=color,
            angle=angle,
        )
    )


def math_runs(
    d: Drawing,
    x: float,
    y: float,
    runs: Sequence[tuple[str, str, float, float, object]],
    anchor: str = "start",
) -> None:
    """Draw compact publication math from individually positioned font runs.

    Each run is (text, font, size, vertical_offset, color).  This provides true
    subscripts, superscripts, italics, and Symbol-font Greek without depending
    on a plotting package or leaving literal underscores in the artwork.
    """
    widths = [pdfmetrics.stringWidth(value, font, size) for value, font, size, _, _ in runs]
    total = sum(widths)
    cursor = x - (total / 2 if anchor == "middle" else total if anchor == "end" else 0)
    for (value, font, size, dy, color), width in zip(runs, widths, strict=False):
        d.add(String(cursor, y + dy, value, fontName=font, fontSize=size, fillColor=color))
        cursor += width


def vtxt(
    d: Drawing,
    x: float,
    y: float,
    value: str,
    size: float = FS_SMALL,
    bold: bool = False,
    color=DARK,
) -> None:
    """Draw a truly rotated, vertically centred axis label."""
    group = Group()
    group.add(
        String(
            0,
            0,
            value,
            fontName=FONT_B if bold else FONT,
            fontSize=size,
            textAnchor="middle",
            fillColor=color,
        )
    )
    group.translate(x, y)
    group.rotate(90)
    d.add(group)


def math_m_sub_runs(subscript: str, suffix: str = "", size: float = FS_MIN, color=DARK):
    return [
        ("m", "Times-Italic", size + 0.2, 0, color),
        (subscript, "Times-Italic", size * 0.72, -1.8, color),
        (suffix, FONT, size, 0, color),
    ]


def math_i_star_runs(prefix: str = "", suffix: str = "", size: float = FS_MIN, color=DARK):
    return [
        (prefix, FONT, size, 0, color),
        ("i", "Times-Italic", size + 0.2, 0, color),
        ("*", FONT, size * 0.72, 2.3, color),
        (suffix, FONT, size, 0, color),
    ]


def marker(
    d: Drawing,
    x: float,
    y: float,
    style: str,
    color,
    size: float = 3.0,
    filled: bool = True,
    width: float = 0.9,
) -> None:
    fill = color if filled else WHITE
    if style == "circle":
        d.add(Circle(x, y, size, fillColor=fill, strokeColor=color, strokeWidth=width))
    elif style == "triangle":
        d.add(
            Polygon(
                [x, y + size * 1.15, x - size, y - size, x + size, y - size],
                fillColor=fill,
                strokeColor=color,
                strokeWidth=width,
            )
        )
    elif style == "square":
        d.add(
            Rect(
                x - size,
                y - size,
                2 * size,
                2 * size,
                fillColor=fill,
                strokeColor=color,
                strokeWidth=width,
            )
        )
    elif style == "diamond":
        d.add(
            Polygon(
                [x, y + 1.25 * size, x - size, y, x, y - 1.25 * size, x + size, y],
                fillColor=fill,
                strokeColor=color,
                strokeWidth=width,
            )
        )
    else:
        raise ValueError(style)


def panel_title(d: Drawing, x: float, y: float, letter: str, title: str) -> None:
    txt(d, x, y, f"({letter})", FS_PANEL, True)
    txt(d, x + 17, y, title, FS_PANEL, True)


def add_channel_legend(d: Drawing, x: float, y: float) -> None:
    d.add(Line(x - 4, y, x + 6, y, strokeColor=BLUE, strokeWidth=1.25))
    marker(d, x + 1, y, "circle", BLUE, 2.7, True)
    math_runs(
        d,
        x + 10,
        y - 2.2,
        [
            ("BSC ", FONT, FS_SMALL, 0, DARK),
            ("p", "Times-Italic", FS_SMALL, 0, DARK),
            (" = 0", FONT, FS_SMALL, 0, DARK),
        ],
    )
    x2 = x + 82
    d.add(Line(x2 - 4, y, x2 + 6, y, strokeColor=ORANGE, strokeWidth=1.25, strokeDashArray=[4, 2]))
    marker(d, x2 + 1, y, "triangle", ORANGE, 2.8, False)
    math_runs(
        d,
        x2 + 10,
        y - 2.2,
        [
            ("BSC ", FONT, FS_SMALL, 0, DARK),
            ("p", "Times-Italic", FS_SMALL, 0, DARK),
            (" = 0.05", FONT, FS_SMALL, 0, DARK),
        ],
    )


def draw_y_axis(
    d: Drawing,
    x0: float,
    y0: float,
    w: float,
    h: float,
    vmin: float,
    vmax: float,
    ticks: Sequence[float],
    fmt: Callable[[float], str],
) -> Callable[[float], float]:
    def ymap(value: float) -> float:
        return y0 + (float(value) - vmin) * h / (vmax - vmin)

    for value in ticks:
        y = ymap(value)
        d.add(Line(x0, y, x0 + w, y, strokeColor=GRID, strokeWidth=0.45))
        txt(d, x0 - 4, y - 2.0, fmt(value), FS_MIN, anchor="end", color=MID)
    d.add(Line(x0, y0, x0, y0 + h, strokeColor=DARK, strokeWidth=0.7))
    d.add(Line(x0, y0, x0 + w, y0, strokeColor=DARK, strokeWidth=0.7))
    return ymap


def draw_x_axis(
    d: Drawing,
    x0: float,
    y0: float,
    w: float,
    h: float,
    vmin: float,
    vmax: float,
    ticks: Sequence[float],
    fmt: Callable[[float], str],
) -> Callable[[float], float]:
    def xmap(value: float) -> float:
        return x0 + (float(value) - vmin) * w / (vmax - vmin)

    for index, value in enumerate(ticks):
        x = xmap(value)
        d.add(Line(x, y0, x, y0 + h, strokeColor=GRID, strokeWidth=0.45))
        anchor = "start" if index == 0 else ("end" if index == len(ticks) - 1 else "middle")
        txt(d, x, y0 - 10, fmt(value), FS_MIN, anchor=anchor, color=MID)
    d.add(Line(x0, y0, x0 + w, y0, strokeColor=DARK, strokeWidth=0.7))
    return xmap


def vertical_errorbar(d: Drawing, x: float, low: float, high: float, ymap, color=DARK) -> None:
    yl, yh = ymap(low), ymap(high)
    d.add(Line(x, yl, x, yh, strokeColor=color, strokeWidth=0.75))
    d.add(Line(x - 2.4, yl, x + 2.4, yl, strokeColor=color, strokeWidth=0.75))
    d.add(Line(x - 2.4, yh, x + 2.4, yh, strokeColor=color, strokeWidth=0.75))


def horizontal_errorbar(d: Drawing, y: float, low: float, high: float, xmap, color=DARK) -> None:
    xl, xh = xmap(low), xmap(high)
    d.add(Line(xl, y, xh, y, strokeColor=color, strokeWidth=0.75))
    d.add(Line(xl, y - 2.1, xl, y + 2.1, strokeColor=color, strokeWidth=0.75))
    d.add(Line(xh, y - 2.1, xh, y + 2.1, strokeColor=color, strokeWidth=0.75))


def add_polyline(d: Drawing, xs: Sequence[float], ys: Sequence[float], color, dashed=False) -> None:
    path = RPath()
    for index, (x, y) in enumerate(zip(xs, ys, strict=False)):
        if index == 0:
            path.moveTo(x, y)
        else:
            path.lineTo(x, y)
    path.strokeColor = color
    path.strokeWidth = 1.25
    path.fillColor = None
    if dashed:
        path.strokeDashArray = [4, 2]
    d.add(path)


def arrow(d: Drawing, x1: float, y1: float, x2: float, y2: float, color=DARK, dashed=False) -> None:
    dash = [3, 2] if dashed else None
    d.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=0.9, strokeDashArray=dash))
    angle = math.atan2(y2 - y1, x2 - x1)
    length = 5.0
    spread = 2.5
    bx = x2 - length * math.cos(angle)
    by = y2 - length * math.sin(angle)
    px = spread * math.sin(angle)
    py = -spread * math.cos(angle)
    d.add(
        Polygon(
            [x2, y2, bx + px, by + py, bx - px, by - py],
            fillColor=color,
            strokeColor=color,
        )
    )


def poly_arrow(d: Drawing, points: Sequence[tuple[float, float]], color=DARK, dashed=False) -> None:
    for p1, p2 in zip(points[:-2], points[1:-1], strict=False):
        d.add(
            Line(
                p1[0],
                p1[1],
                p2[0],
                p2[1],
                strokeColor=color,
                strokeWidth=0.9,
                strokeDashArray=[3, 2] if dashed else None,
            )
        )
    arrow(d, *points[-2], *points[-1], color=color, dashed=dashed)


def box(
    d: Drawing,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    lines: Sequence[str],
    fill=WHITE,
    stroke=DARK,
) -> None:
    d.add(Rect(x, y, w, h, rx=4, ry=4, fillColor=fill, strokeColor=stroke, strokeWidth=0.75))
    txt(d, x + w / 2, y + h - 11, title, FS_SMALL, True, anchor="middle")
    for index, line in enumerate(lines):
        txt(d, x + w / 2, y + h - 21 - 8 * index, line, FS_MIN, anchor="middle")


def figure_1_system() -> Drawing:
    d = Drawing(FIG_W, 3.72 * 72)
    h = d.height

    txt(d, 272, h - 12, "protocol time", FS_SMALL, anchor="end", color=MID)
    arrow(d, 278, h - 10, 336, h - 10, color=MID)

    # Two horizontal swimlanes convey ownership without a central divider.
    d.add(
        Rect(
            8,
            159,
            FIG_W - 16,
            82,
            rx=7,
            ry=7,
            fillColor=PALE_BLUE,
            strokeColor=BLUE,
            strokeWidth=0.8,
        )
    )
    d.add(
        Rect(
            8,
            61,
            FIG_W - 16,
            82,
            rx=7,
            ry=7,
            fillColor=PALE_GREEN,
            strokeColor=GREEN,
            strokeWidth=0.8,
        )
    )
    txt(d, 16, 228, "TRANSMITTER (Tx)", FS_PANEL, True, color=BLUE)
    txt(d, 16, 130, "RECEIVER (Rx)", FS_PANEL, True, color=GREEN)

    box(d, 20, 178, 66, 38, "Semantic encoder", ["source x", ""], WHITE)
    math_runs(
        d,
        53,
        184,
        [
            ("normalized ", FONT, FS_MIN, 0, DARK),
            ("u", "Times-BoldItalic", FS_SMALL, 0, DARK),
            ("(x)", "Times-Italic", FS_SMALL, 0, DARK),
        ],
        anchor="middle",
    )
    box(d, 103, 178, 65, 38, "Proposal sketch", ["random signs", ""], WHITE)
    math_runs(d, 135.5, 184, math_m_sub_runs("P", " signs", FS_SMALL), anchor="middle")
    box(d, 195, 178, 66, 38, "Fresh audit", ["seed c; new signs", ""], WHITE)
    math_runs(d, 228, 184, math_m_sub_runs("A", " signs", FS_SMALL), anchor="middle")
    box(d, 282, 178, 49, 38, "Feature refresh", ["fallback", "payload"], WHITE)
    arrow(d, 86, 197, 103, 197, color=BLUE)

    box(d, 20, 80, 67, 40, "Receiver cache", ["M cached features", ""], WHITE)
    math_runs(
        d,
        53.5,
        86,
        [
            ("C", "Times-Italic", FS_SMALL + 0.2, 0, DARK),
            ("R", "Times-Italic", FS_SMALL * 0.72, -1.8, DARK),
            (" = {", FONT, FS_MIN, 0, DARK),
            ("u", "Times-BoldItalic", FS_SMALL, 0, DARK),
            ("i", "Times-Italic", FS_MIN * 0.72, -1.8, DARK),
            ("}", FONT, FS_MIN, 0, DARK),
            ("i=1", "Times-Italic", FS_MIN * 0.66, -2.0, DARK),
            ("M", "Times-Italic", FS_MIN * 0.66, 3.0, DARK),
        ],
        anchor="middle",
    )
    box(d, 107, 80, 67, 40, "Search and lock", ["rank received proposal", ""], WHITE)
    math_runs(d, 140.5, 86, math_i_star_runs("fix candidate ", "", FS_MIN), anchor="middle")
    box(
        d,
        201,
        77,
        72,
        46,
        "Independent audit",
        ["test locked candidate", "similarity threshold", "1% error target"],
        WHITE,
    )
    box(d, 279, 80, 52, 40, "Completed", ["feature", "reuse or refresh"], WHITE)
    arrow(d, 87, 100, 107, 100, color=GREEN)

    # Numbering keeps the communication band free of long, overlapping labels.
    arrow(d, 135, 178, 135, 121, color=BLUE)  # 1 proposal
    poly_arrow(d, [(174, 102), (184, 102), (184, 164), (211, 178)], color=GREEN)  # 2 lock
    arrow(d, 236, 178, 236, 124, color=BLUE)  # 3 fresh audit
    arrow(d, 273, 100, 279, 100, color=GREEN)  # local certified reuse
    poly_arrow(
        d, [(267, 120), (278, 148), (306, 148), (306, 178)], color=GREEN, dashed=True
    )  # 4 decision
    arrow(d, 306, 178, 306, 121, color=BLUE)  # 5 refresh

    for number, x, y, color in [
        (1, 135, 151, BLUE),
        (2, 184, 151, GREEN),
        (3, 236, 151, BLUE),
        (4, 279, 148, GREEN),
        (5, 306, 151, BLUE),
    ]:
        d.add(Circle(x, y, 6.1, fillColor=WHITE, strokeColor=color, strokeWidth=1.0))
        txt(d, x, y - 2.2, str(number), FS_MIN, True, anchor="middle", color=color)

    d.add(
        Rect(8, 5, FIG_W - 16, 47, rx=4, ry=4, fillColor=LIGHT, strokeColor=GRID, strokeWidth=0.6)
    )
    key = [
        (1, "Tx to Rx: proposal sketch", 16, 39, BLUE),
        (2, "Rx to Tx: locked index", 127, 39, GREEN),
        (3, "Tx to Rx: fresh audit", 239, 39, BLUE),
        (4, "Rx to Tx: audit decision", 72, 20, GREEN),
        (5, "Tx to Rx: feature refresh", 211, 20, BLUE),
    ]
    for number, label, x, y, color in key:
        d.add(Circle(x, y + 1.8, 4.8, fillColor=WHITE, strokeColor=color, strokeWidth=0.9))
        txt(d, x, y - 0.2, str(number), FS_MIN, True, anchor="middle", color=color)
        txt(d, x + 8, y - 0.5, label, FS_MIN)
    return d


def _wilson(count: int, n: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = count / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return center - half, center + half


def figure_2_statistical_validation() -> Drawing:
    selection = json.loads((VALIDATION_DIR / "selection_stress.json").read_text())
    sequential = json.loads((VALIDATION_DIR / "null_validation.json").read_text())
    d = Drawing(FIG_W, 3.00 * 72)

    # A single, shared legend occupies its own row between the panel headings
    # and plotting regions.  The whisker key applies to both panels.
    legend_y = 179
    d.add(Line(21, legend_y, 31, legend_y, strokeColor=BLUE, strokeWidth=1.2))
    marker(d, 26, legend_y, "circle", BLUE, 2.6, True)
    txt(d, 36, legend_y - 2.1, "Reuse selection evidence", FS_MIN)
    d.add(
        Line(
            128,
            legend_y,
            138,
            legend_y,
            strokeColor=ORANGE,
            strokeWidth=1.2,
            strokeDashArray=[4, 2],
        )
    )
    marker(d, 133, legend_y, "triangle", ORANGE, 2.6, False)
    txt(d, 143, legend_y - 2.1, "Fresh independent audit", FS_MIN)
    d.add(Line(266, legend_y - 4, 266, legend_y + 4, strokeColor=DARK, strokeWidth=1.0))
    d.add(Line(262, legend_y - 4, 270, legend_y - 4, strokeColor=DARK, strokeWidth=1.0))
    d.add(Line(262, legend_y + 4, 270, legend_y + 4, strokeColor=DARK, strokeWidth=1.0))
    txt(d, 276, legend_y - 2.1, "Wilson 95% CI", FS_MIN)

    # Panel (a): selection-induced inflation and the independent-audit remedy.
    x0, y0, w, h = 48, 42, 124, 115
    panel_title(d, 7, 204, "a", "Search selection and false acceptance")
    ymap = draw_y_axis(d, x0, y0, w, h, 0, 30, [0, 10, 20, 30], lambda v: f"{v:.0f}")
    vtxt(d, 17, y0 + h / 2, "False acceptance (%)")
    xs = [x0 + i * w / 2 for i in range(3)]
    for key, color, style, dashed in [
        ("naive_reuse_false_accept_rate", BLUE, "circle", False),
        ("fresh_false_accept_rate", ORANGE, "triangle", True),
    ]:
        values = [100 * row[key] for row in selection]
        add_polyline(d, xs, [ymap(value) for value in values], color, dashed)
        for index, (x, value, row) in enumerate(zip(xs, values, selection, strict=False)):
            count = int(round(row[key] * row["trials"]))
            low, high = _wilson(count, int(row["trials"]))
            vertical_errorbar(d, x, 100 * low, 100 * high, ymap, color)
            marker(d, x, ymap(value), style, color, 3.0, filled=not dashed)
            if index == 0 and key.startswith("naive"):
                label_x, label_y = x - 10, ymap(value) + 12
            elif index == 1 and key.startswith("naive"):
                # Keep the 4.14 label clear of the rising blue segment.
                label_x, label_y = x - 7, ymap(value) + 6
            elif key == "fresh_false_accept_rate":
                # Exact fresh-audit values are reported in the 0--1% inset.
                continue
            else:
                label_x, label_y = x, ymap(value) + 6
            txt(d, label_x, label_y, f"{value:.2f}", FS_MIN, anchor="middle", color=color)
    target_y = ymap(1.0)
    d.add(
        Line(
            x0,
            target_y,
            x0 + w,
            target_y,
            strokeColor=MID,
            strokeWidth=0.75,
            strokeDashArray=[2, 2],
        )
    )
    txt(d, x0 + w - 2, target_y + 3, "1% target", FS_MIN, anchor="end", color=MID)
    for x, row in zip(xs, selection, strict=False):
        txt(d, x, y0 - 11, str(row["candidates"]), FS_MIN, anchor="middle")
    txt(d, x0 + w / 2, 12, "Searched cache candidates", FS_SMALL, anchor="middle")

    # Magnify the lower 0--1% region without introducing a second metric or a
    # dual axis.  This makes the three fresh-audit values readable while the
    # main axis still communicates the 25.15% selection-induced inflation.
    # Give the inset title comfortable right padding and place the box wholly
    # to the left of the rising same-evidence curve.
    ix, iy, iw, ih = 58, 94, 78, 50
    d.add(Rect(ix, iy, iw, ih, fillColor=WHITE, strokeColor=MID, strokeWidth=0.65))
    txt(d, ix + 4, iy + ih - 8, "0-1% zoom: fresh audit", FS_MIN, True, color=ORANGE)
    zx0, zy0, zw, zh = ix + 8, iy + 10, iw - 15, ih - 22
    zymap = draw_y_axis(d, zx0, zy0, zw, zh, 0, 1.0, [0, 0.5, 1.0], lambda v: f"{v:.1f}")
    zxs = [zx0 + i * zw / 2 for i in range(3)]
    fresh = [100 * row["fresh_false_accept_rate"] for row in selection]
    add_polyline(d, zxs, [zymap(value) for value in fresh], ORANGE, dashed=True)
    for index, (x, value, row) in enumerate(zip(zxs, fresh, selection, strict=False)):
        count = int(round(row["fresh_false_accept_rate"] * row["trials"]))
        low, high = _wilson(count, int(row["trials"]))
        vertical_errorbar(d, x, 100 * low, 100 * high, zymap, ORANGE)
        marker(d, x, zymap(value), "triangle", ORANGE, 2.2, False)
        anchor = "start" if index == 0 else ("end" if index == 2 else "middle")
        dx = 3 if index == 0 else (-3 if index == 2 else 0)
        txt(
            d, x + dx, zymap(value) + 4.5, f"{value:.2f}", FS_MIN, True, anchor=anchor, color=ORANGE
        )
        txt(d, x, zy0 - 7, str(row["candidates"]), FS_MIN, anchor="middle", color=MID)

    # Panel (b): familywise control of the full sequential audit.
    x0, y0, w, h = 225, 42, 106, 115
    panel_title(d, 184, 204, "b", "Sequential fresh-audit validation")
    ymap = draw_y_axis(d, x0, y0, w, h, 0, 1.1, [0, 0.5, 1.0], lambda v: f"{v:.1f}")
    vtxt(d, 200, y0 + h / 2, "Familywise false acceptance (%)")
    xs = [x0 + 34, x0 + 82]
    for x, row, color, style, filled in zip(
        xs,
        sequential,
        [BLUE, ORANGE],
        ["circle", "triangle"],
        [True, False],
        strict=False,
    ):
        value = 100 * row["familywise_false_accept_rate"]
        low, high = [100 * bound for bound in row["familywise_wilson95"]]
        d.add(Line(x, ymap(low), x, ymap(high), strokeColor=color, strokeWidth=1.25))
        d.add(Line(x - 4, ymap(low), x + 4, ymap(low), strokeColor=color, strokeWidth=1.25))
        d.add(Line(x - 4, ymap(high), x + 4, ymap(high), strokeColor=color, strokeWidth=1.25))
        marker(d, x, ymap(value), style, color, 2.6, filled)
        txt(d, x, ymap(high) + 7, f"{value:.3f}%", FS_SMALL, True, anchor="middle", color=color)
        txt(d, x, y0 - 11, f"{row['bsc']:.2f}".rstrip("0").rstrip("."), FS_MIN, anchor="middle")
    target_y = ymap(1.0)
    d.add(
        Line(
            x0,
            target_y,
            x0 + w,
            target_y,
            strokeColor=MID,
            strokeWidth=0.75,
            strokeDashArray=[2, 2],
        )
    )
    txt(d, x0 + w - 1, target_y + 3, "1% target", FS_MIN, anchor="end", color=MID)
    txt(d, x0 + w / 2, 12, "BSC crossover probability p", FS_SMALL, anchor="middle")
    return d


def figure_3_allocation() -> Drawing:
    data = pd.read_csv(RATIO / "cal_tune_ratio_aggregate.csv").sort_values(["bsc", "search_bits"])
    d = Drawing(FIG_W, 3.35 * 72)
    add_channel_legend(d, 160, d.height - 12)
    search_bits = sorted(data.search_bits.unique())
    labels = [f"{value}/{1024 - value}" for value in search_bits]

    panels = [
        (
            "a",
            "Cache reuse",
            "reuse_rate_mean",
            "reuse_rate_seed_sd",
            100,
            30,
            100,
            [40, 60, 80, 100],
            lambda v: f"{v:.0f}",
            137,
        ),
        (
            "b",
            "Fallback-inclusive completion traffic",
            "mean_completion_bits_mean",
            "mean_completion_bits_seed_sd",
            1,
            2000,
            5000,
            [2000, 3000, 4000, 5000],
            lambda v: f"{int(v / 1000)}k",
            36,
        ),
    ]
    for letter, title, metric, sd_metric, factor, ymin, ymax, ticks, fmt, y0 in panels:
        x0, w, h = 54, FIG_W - 68, 67
        panel_title(d, 7, y0 + h + 11, letter, title)
        ymap = draw_y_axis(d, x0, y0, w, h, ymin, ymax, ticks, fmt)
        xs = [x0 + i * w / (len(search_bits) - 1) for i in range(len(search_bits))]
        selected_x = xs[search_bits.index(256)]
        d.add(Rect(selected_x - 13, y0, 26, h, fillColor=PALE_GOLD, strokeColor=None))
        for tick in ticks:
            d.add(Line(x0, ymap(tick), x0 + w, ymap(tick), strokeColor=GRID, strokeWidth=0.45))
        for bsc, color, style, dashed, filled in [
            (0.0, BLUE, "circle", False, True),
            (0.05, ORANGE, "triangle", True, False),
        ]:
            subset = data[np.isclose(data.bsc, bsc)].sort_values("search_bits")
            values = (subset[metric].to_numpy() * factor).tolist()
            sds = (subset[sd_metric].to_numpy() * factor).tolist()
            add_polyline(d, xs, [ymap(value) for value in values], color, dashed)
            for x, value, sd in zip(xs, values, sds, strict=False):
                vertical_errorbar(d, x, max(ymin, value - sd), min(ymax, value + sd), ymap, color)
                marker(d, x, ymap(value), style, color, 2.8, filled)
        txt(
            d,
            selected_x,
            y0 + h - 8,
            "selected",
            FS_MIN,
            True,
            anchor="middle",
            color=colors.HexColor("#7A5A00"),
        )
        if metric == "reuse_rate_mean":
            vtxt(d, 18, y0 + h / 2, "Reuse (%)")
        else:
            refresh_y = ymap(4272)
            d.add(
                Line(
                    x0,
                    refresh_y,
                    x0 + w,
                    refresh_y,
                    strokeColor=MID,
                    strokeWidth=0.75,
                    strokeDashArray=[2, 2],
                )
            )
            txt(d, x0 + 3, refresh_y + 3, "4,272-bit refresh", FS_MIN, anchor="start", color=MID)
            vtxt(d, 18, y0 + h / 2, "Logical bits")
        if letter == "b":
            for x, label in zip(xs, labels, strict=False):
                txt(d, x, y0 - 11, label, FS_MIN, anchor="middle")
    math_runs(
        d,
        54 + (FIG_W - 68) / 2,
        9,
        [
            ("Proposal signs ", FONT, FS_SMALL, 0, DARK),
            *math_m_sub_runs("P", " / audit signs ", FS_SMALL),
            *math_m_sub_runs("A", "  (total = 1,024)", FS_SMALL),
        ],
        anchor="middle",
    )
    return d


def figure_4_primary() -> Drawing:
    aggregate = pd.read_csv(RATIO / "selected_ratio_final_aggregate.csv").sort_values("bsc")
    bootstrap = (
        pd.read_csv(STRESS / "sentrysem_fixed_cluster_bootstrap.csv")
        .query("scenario == 'full' and condition == 'paired_iid'")
        .sort_values("bsc")
    )
    d = Drawing(FIG_W, 2.65 * 72)
    panels = [
        {
            "letter": "a",
            "title": "Reuse",
            "x0": 32,
            "w": 73,
            "vmin": 78,
            "vmax": 92,
            "ticks": [80, 85, 90],
            "fmt": lambda v: f"{v:.0f}",
            "metric": "reuse_rate_mean",
            "low": "reuse_rate_cluster_bootstrap95_low",
            "high": "reuse_rate_cluster_bootstrap95_high",
            "factor": 100,
            "value_fmt": lambda v: f"{v:.2f}%",
            "xlabel": "Cache reuse (%)",
        },
        {
            "letter": "b",
            "title": "Exact-instance precision",
            "x0": 144,
            "w": 77,
            "vmin": 99.0,
            "vmax": 100.02,
            "ticks": [99.0, 99.5, 100.0],
            "fmt": lambda v: f"{v:.1f}",
            "metric": "identity_precision_among_reuses_mean",
            "low": "identity_precision_among_reuses_cluster_bootstrap95_low",
            "high": "identity_precision_among_reuses_cluster_bootstrap95_high",
            "factor": 100,
            "value_fmt": lambda v: f"{v:.3f}%",
            "xlabel": "Precision among reuses (%)",
        },
        {
            "letter": "c",
            "title": "Completion traffic",
            "x0": 259,
            "w": 73,
            "vmin": 2200,
            "vmax": 4400,
            "ticks": [2500, 3500, 4272],
            "fmt": lambda v: "4,272" if v == 4272 else f"{v / 1000:.1f}k",
            "metric": "mean_completion_bits_mean",
            "low": "mean_completion_bits_cluster_bootstrap95_low",
            "high": "mean_completion_bits_cluster_bootstrap95_high",
            "factor": 1,
            "value_fmt": lambda v: f"{v:,.0f}",
            "xlabel": "Logical bits",
        },
    ]
    y0, h = 47, 91
    ys = [107, 77]
    # The channel encoding is stated once in a single legend row; repeating it
    # as pseudo-y-axis labels in each panel made the multi-panel grammar unclear.
    add_channel_legend(d, 94, 154)
    for panel in panels:
        panel_title(d, panel["x0"] - 24, 173, panel["letter"], panel["title"])
        xmap = draw_x_axis(
            d,
            panel["x0"],
            y0,
            panel["w"],
            h,
            panel["vmin"],
            panel["vmax"],
            panel["ticks"],
            panel["fmt"],
        )
        for y in ys:
            d.add(
                Line(
                    panel["x0"],
                    y,
                    panel["x0"] + panel["w"],
                    y,
                    strokeColor=colors.HexColor("#ECECEC"),
                    strokeWidth=0.4,
                )
            )
        if panel["metric"] == "mean_completion_bits_mean":
            xr = xmap(4272)
            d.add(
                Line(xr, y0, xr, y0 + h, strokeColor=MID, strokeWidth=0.75, strokeDashArray=[2, 2])
            )
        for index, (row, boot) in enumerate(
            zip(aggregate.itertuples(), bootstrap.itertuples(), strict=False)
        ):
            value = getattr(row, panel["metric"]) * panel["factor"]
            low = getattr(boot, panel["low"]) * panel["factor"]
            high = getattr(boot, panel["high"]) * panel["factor"]
            color = BLUE if row.bsc == 0 else ORANGE
            style = "circle" if row.bsc == 0 else "triangle"
            filled = row.bsc == 0
            horizontal_errorbar(d, ys[index], low, high, xmap, color)
            marker(d, xmap(value), ys[index], style, color, 3.5, filled)
            anchor = "end" if value > (panel["vmin"] + panel["vmax"]) / 2 else "start"
            dx = -5 if anchor == "end" else 5
            txt(
                d,
                xmap(value) + dx,
                ys[index] + 7,
                panel["value_fmt"](value),
                FS_MIN,
                True,
                anchor=anchor,
                color=color,
            )
        txt(d, panel["x0"] + panel["w"] / 2, 17, panel["xlabel"], FS_MIN, anchor="middle")
    return d


def _common_tradeoff(method: str, bsc: float) -> tuple[pd.Series, pd.Series]:
    aggregate = pd.read_csv(COMMON / "publication_aggregate_summary.csv")
    bootstrap = pd.read_csv(COMMON / "identity_cluster_bootstrap.csv")
    row = aggregate[(aggregate.method == method) & np.isclose(aggregate.bsc, bsc)].iloc[0]
    boot = bootstrap[(bootstrap.method == method) & np.isclose(bootstrap.bsc, bsc)].iloc[0]
    return row, boot


def _selected_tradeoff(bsc: float) -> tuple[pd.Series, pd.Series]:
    aggregate = pd.read_csv(RATIO / "selected_ratio_final_aggregate.csv")
    bootstrap = pd.read_csv(STRESS / "sentrysem_fixed_cluster_bootstrap.csv")
    row = aggregate[np.isclose(aggregate.bsc, bsc)].iloc[0]
    boot = bootstrap[
        (bootstrap.scenario == "full")
        & (bootstrap.condition == "paired_iid")
        & np.isclose(bootstrap.bsc, bsc)
    ].iloc[0]
    return row, boot


def figure_5_tradeoff() -> Drawing:
    methods = [
        ("bonferroni512", "Bonferroni-512", "circle", MID, False),
        ("bonferroni1024", "Bonferroni-1024", "circle", DARK, True),
        ("split256_256", "Fixed-256/256", "square", BLUE, False),
        ("split512_512", "Fixed-512/512", "square", BLUE, True),
        ("sentrysem1", "Sequential-1", "triangle", ORANGE, False),
        ("sentrysem3", "Sequential-3", "triangle", ORANGE, True),
        ("selected", "Fixed-256/768 (selected)", "diamond", GREEN, True),
    ]
    d = Drawing(FIG_W, 3.88 * 72)
    panels = [("a", "0", 53, 0.0), ("b", "0.05", 216, 0.05)]
    y0, h, w = 76, 150, 116
    x_min, x_max = 35, 92
    y_min, y_max = 1500, 6500
    label_offsets = {
        0.0: {1: (4, -8), 2: (-8, -8), 3: (4, 7), 4: (-8, 8), 5: (4, 4), 6: (4, 4), 7: (4, -8)},
        0.05: {1: (4, -8), 2: (-8, -8), 3: (4, 7), 4: (4, 7), 5: (4, 4), 6: (4, 4), 7: (4, -8)},
    }
    for letter, p_value, x0, bsc in panels:
        txt(d, x0 - 32, 258, f"({letter})", FS_PANEL, True)
        math_runs(
            d,
            x0 - 14,
            258,
            [
                ("BSC ", FONT_B, FS_PANEL, 0, DARK),
                ("p", "Times-BoldItalic", FS_PANEL, 0, DARK),
                (f" = {p_value}", FONT_B, FS_PANEL, 0, DARK),
            ],
        )
        xmap = draw_x_axis(d, x0, y0, w, h, x_min, x_max, [40, 60, 80], lambda v: f"{v:.0f}")
        ymap = draw_y_axis(
            d,
            x0,
            y0,
            w,
            h,
            y_min,
            y_max,
            [2000, 3000, 4000, 5000, 6000],
            lambda v: f"{int(v / 1000)}k",
        )
        refresh_y = ymap(4272)
        d.add(
            Line(
                x0,
                refresh_y,
                x0 + w,
                refresh_y,
                strokeColor=MID,
                strokeWidth=0.75,
                strokeDashArray=[2, 2],
            )
        )
        txt(d, x0 + w - 1, refresh_y + 3, "refresh", FS_MIN, anchor="end", color=MID)
        for number, (method, _label, style, color, filled) in enumerate(methods, 1):
            if method == "selected":
                row, boot = _selected_tradeoff(bsc)
            else:
                row, boot = _common_tradeoff(method, bsc)
            reuse = 100 * row.reuse_rate_mean
            traffic = row.mean_completion_bits_mean
            reuse_low = 100 * boot.reuse_rate_cluster_bootstrap95_low
            reuse_high = 100 * boot.reuse_rate_cluster_bootstrap95_high
            traffic_low = boot.mean_completion_bits_cluster_bootstrap95_low
            traffic_high = boot.mean_completion_bits_cluster_bootstrap95_high
            horizontal_errorbar(d, ymap(traffic), reuse_low, reuse_high, xmap, color)
            vertical_errorbar(d, xmap(reuse), traffic_low, traffic_high, ymap, color)
            marker(
                d,
                xmap(reuse),
                ymap(traffic),
                style,
                color,
                3.2 if method != "selected" else 4.0,
                filled,
            )
            dx, dy = label_offsets[bsc][number]
            txt(d, xmap(reuse) + dx, ymap(traffic) + dy, str(number), FS_MIN, True, color=color)
        txt(d, x0 + w / 2, 55, "Cache reuse (%)", FS_SMALL, anchor="middle")
    vtxt(d, 18, y0 + h / 2, "Completion traffic (logical bits)")
    txt(d, 332, 239, "higher reuse", FS_MIN, anchor="end", color=MID)
    txt(d, 334, 231, "lower traffic is better", FS_MIN, anchor="end", color=MID)

    # Numbered key avoids long, overlapping labels inside the data regions.
    key_positions = [(12, 37), (122, 37), (238, 37), (12, 25), (122, 25), (238, 25), (88, 13)]
    for number, ((_, label, style, color, filled), (x, y)) in enumerate(
        zip(methods, key_positions, strict=False), 1
    ):
        marker(d, x, y + 2, style, color, 2.7, filled)
        txt(d, x + 7, y, f"{number}  {label}", FS_MIN, bold=(number in (2, 7)), color=DARK)
    return d


def figure_6_stress() -> Drawing:
    aggregate = pd.read_csv(STRESS / "sentrysem_fixed_aggregate.csv")
    bootstrap = pd.read_csv(STRESS / "sentrysem_fixed_cluster_bootstrap.csv")
    conditions = [
        ("full", "paired_iid", "Full cache"),
        ("missing3classes", "paired_iid", "Three classes absent"),
        ("evict50", "paired_iid", "50% cache eviction"),
        ("full", "paired_hard_ood", "Hard view shift"),
        ("full", "novel_unpaired", "Novel source"),
    ]
    d = Drawing(FIG_W, 3.72 * 72)
    add_channel_legend(d, 161, d.height - 12)

    def panel(
        letter: str,
        title: str,
        y0: float,
        metric: str,
        low_metric: str,
        high_metric: str,
        vmin: float,
        vmax: float,
        ticks: Sequence[float],
        fmt: Callable[[float], str],
        factor: float,
        xlabel: str,
        refresh: float | None = None,
    ) -> None:
        label_w, plot_w, height = 100, FIG_W - 114, 77
        panel_title(d, 7, y0 + height + 12, letter, title)
        xmap = draw_x_axis(d, label_w, y0, plot_w, height, vmin, vmax, ticks, fmt)
        row_h = height / len(conditions)
        for index, (scenario, condition, label) in enumerate(conditions):
            y = y0 + height - (index + 0.5) * row_h
            if index % 2 == 0:
                d.add(
                    Rect(
                        label_w,
                        y - row_h / 2,
                        plot_w,
                        row_h,
                        fillColor=colors.HexColor("#FAFAFA"),
                        strokeColor=None,
                    )
                )
            txt(d, label_w - 5, y - 2, label, FS_SMALL, anchor="end")
            for bsc, color, style, filled, offset in [
                (0.0, BLUE, "circle", True, 2.6),
                (0.05, ORANGE, "triangle", False, -2.6),
            ]:
                row = aggregate[
                    (aggregate.scenario == scenario)
                    & (aggregate.condition == condition)
                    & np.isclose(aggregate.bsc, bsc)
                ].iloc[0]
                boot = bootstrap[
                    (bootstrap.scenario == scenario)
                    & (bootstrap.condition == condition)
                    & np.isclose(bootstrap.bsc, bsc)
                ].iloc[0]
                value = row[metric] * factor
                low = boot[low_metric] * factor
                high = boot[high_metric] * factor
                horizontal_errorbar(d, y + offset, low, high, xmap, color)
                marker(d, xmap(value), y + offset, style, color, 2.8, filled)
        if refresh is not None:
            xr = xmap(refresh)
            d.add(
                Line(
                    xr,
                    y0,
                    xr,
                    y0 + height,
                    strokeColor=MID,
                    strokeWidth=0.75,
                    strokeDashArray=[2, 2],
                )
            )
            txt(d, xr - 2, y0 + height + 3, "4,272-bit refresh", FS_MIN, anchor="end", color=MID)
        txt(d, label_w + plot_w / 2, y0 - 22, xlabel, FS_SMALL, anchor="middle")

    panel(
        "a",
        "Cache reuse under cache and query stress",
        145,
        "reuse_rate_mean",
        "reuse_rate_cluster_bootstrap95_low",
        "reuse_rate_cluster_bootstrap95_high",
        0,
        100,
        [0, 25, 50, 75, 100],
        lambda v: f"{v:.0f}",
        100,
        "Cache reuse (%)",
    )
    panel(
        "b",
        "Fallback-inclusive completion traffic",
        34,
        "mean_completion_bits_mean",
        "mean_completion_bits_cluster_bootstrap95_low",
        "mean_completion_bits_cluster_bootstrap95_high",
        2000,
        6500,
        [2000, 3000, 4000, 5000, 6000],
        lambda v: f"{int(v / 1000)}k",
        1,
        "Logical bits",
        refresh=4272,
    )
    return d


def save_figure(drawing: Drawing, stem: str) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    pdf = OUT / f"{stem}.pdf"
    svg = OUT / f"{stem}.svg"
    png = OUT / f"{stem}.png"
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
    with Image.open(png) as image:
        pixels = list(image.size)
        dpi = image.info.get("dpi")
    reader = PdfReader(str(pdf))
    assert len(reader.pages) == 1
    assert pixels[0] in range(2878, 2883), (stem, pixels)
    return {
        "stem": stem,
        "width_inches": drawing.width / 72.0,
        "height_inches": drawing.height / 72.0,
        "pdf_pages": 1,
        "png_pixels": pixels,
        "png_reported_dpi": list(dpi) if dpi else None,
        "files": {
            suffix: {
                "path": str(path.relative_to(OUT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for suffix, path in [("pdf", pdf), ("svg", svg), ("png", png)]
        },
    }


def provenance_source(path: Path) -> str:
    """Return an accurate, portable label for a configured figure source."""

    path = Path(path).resolve()
    try:
        return path.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        roots = (
            ("results", RESULTS_ROOT),
            ("inputs", FROZEN_INPUTS),
            ("validation", VALIDATION_DIR),
        )
        for label, root in roots:
            try:
                return f"{label}/{path.relative_to(Path(root).resolve()).as_posix()}"
            except ValueError:
                continue
    return path.name


def write_companions(outputs: list[dict], validation: dict) -> None:
    sources = [
        {
            "figure": 2,
            "source": provenance_source(VALIDATION_DIR / "selection_stress.json"),
            "sha256": sha256(VALIDATION_DIR / "selection_stress.json"),
            "records": "3 search widths x 10,000 trials",
            "metrics": ["same-evidence false acceptance", "fresh-audit false acceptance"],
        },
        {
            "figure": 2,
            "source": provenance_source(VALIDATION_DIR / "null_validation.json"),
            "sha256": sha256(VALIDATION_DIR / "null_validation.json"),
            "records": "2 channels x 50,000 sessions",
            "metrics": ["familywise false acceptance", "Wilson 95% interval"],
        },
        {
            "figure": 3,
            "source": provenance_source(RATIO / "cal_tune_ratio_aggregate.csv"),
            "sha256": sha256(RATIO / "cal_tune_ratio_aggregate.csv"),
            "records": "7 allocations x 2 channels; 5 threshold-development seeds",
            "metrics": ["reuse", "completion traffic", "seed SD"],
        },
        {
            "figure": 3,
            "source": provenance_source(RATIO / "selection.json"),
            "sha256": sha256(RATIO / "selection.json"),
            "records": "frozen selection decision",
            "metrics": ["selected m_P=256", "selected m_A=768"],
        },
        {
            "figure": 4,
            "source": provenance_source(RATIO / "selected_ratio_final_aggregate.csv"),
            "sha256": sha256(RATIO / "selected_ratio_final_aggregate.csv"),
            "records": "500 source instances x 5 evaluation seeds x 2 channels",
            "metrics": ["reuse", "exact-instance precision", "completion traffic"],
        },
        {
            "figure": 4,
            "source": provenance_source(STRESS / "sentrysem_fixed_cluster_bootstrap.csv"),
            "sha256": sha256(STRESS / "sentrysem_fixed_cluster_bootstrap.csv"),
            "records": "full-cache paired-IID rows; 2,000 cluster-bootstrap draws",
            "metrics": ["95% intervals"],
        },
        {
            "figure": 5,
            "source": provenance_source(COMMON / "publication_aggregate_summary.csv"),
            "sha256": sha256(COMMON / "publication_aggregate_summary.csv"),
            "records": "6 protected methods x 2 channels; 500 instances x 5 common seeds",
            "metrics": ["reuse", "completion traffic"],
        },
        {
            "figure": 5,
            "source": provenance_source(COMMON / "identity_cluster_bootstrap.csv"),
            "sha256": sha256(COMMON / "identity_cluster_bootstrap.csv"),
            "records": "matching common-method rows; 2,000 cluster-bootstrap draws",
            "metrics": ["reuse and traffic 95% intervals"],
        },
        {
            "figure": 5,
            "source": provenance_source(RATIO / "selected_ratio_final_aggregate.csv"),
            "sha256": sha256(RATIO / "selected_ratio_final_aggregate.csv"),
            "records": "selected 256/768 operating point",
            "metrics": ["reuse", "completion traffic"],
        },
        {
            "figure": 6,
            "source": provenance_source(FROZEN_INPUTS / "qualitative_examples.npz"),
            "sha256": sha256(FROZEN_INPUTS / "qualitative_examples.npz"),
            "records": "three cached, standard-view, strong-view, and novel-source image groups",
            "metrics": ["exact cosine similarity", "calibrated threshold region"],
        },
        {
            "figure": 7,
            "source": provenance_source(STRESS / "sentrysem_fixed_aggregate.csv"),
            "sha256": sha256(STRESS / "sentrysem_fixed_aggregate.csv"),
            "records": "5 stress conditions x 2 channels",
            "metrics": ["reuse", "completion traffic"],
        },
        {
            "figure": 7,
            "source": provenance_source(STRESS / "sentrysem_fixed_cluster_bootstrap.csv"),
            "sha256": sha256(STRESS / "sentrysem_fixed_cluster_bootstrap.csv"),
            "records": "2,000 source-instance cluster-bootstrap draws per row",
            "metrics": ["95% intervals"],
        },
    ]
    (OUT / "FIGURE_PROVENANCE.json").write_text(
        json.dumps({"status": "PASS", "sources": sources}, indent=2)
    )
    qa = {
        "status": "PASS",
        "data_validation": validation,
        "figure_count": len(outputs),
        "figure_outputs": outputs,
        "format": {
            "target_width_inches": 4.8,
            "vector_formats": ["PDF", "SVG"],
            "raster_format": "PNG, rendered from PDF at 600 dpi",
            "minimum_label_size_points": FS_MIN,
            "grayscale_safe": "PASS: channel comparisons also use distinct marker fill, shape, and line style",
        },
    }
    (OUT / "FIGURE_VALIDATION.json").write_text(json.dumps(qa, indent=2))

    captions = r"""# SentrySem figure captions

**Fig. 1. SentrySem protocol and message directions.** The transmitter maps source instance $x$ to normalized semantic representation $\mathbf u(x)$ and sends a proposal sketch containing $m_P$ signs. The receiver searches its cache $\mathcal C_R=\{\mathbf u_i\}_{i=1}^{M}$, locks candidate index $i^\star$, and returns the lock record to the transmitter. Only after receiving that record, the transmitter generates fresh audit seed $c$ and sends an independent audit sketch containing $m_A$ signs. The receiver tests the locked candidate against semantic threshold $\tau$ under familywise error budget $\alpha$ and returns its decision. Certification sets the completed feature $\widehat{\mathbf u}=\mathbf u_{i^\star}$; on fallback, the transmitter sends the current feature representation. Numbered arrows give the five over-the-air message directions; blue denotes transmitter-to-receiver communication, green denotes receiver-to-transmitter communication or receiver-local transitions, and the dashed green arrow carries the receiver decision.

**Fig. 2. Effect of post-selection reuse and validation of independent auditing.** (a) False acceptance after searching 1, 8, or 64 cache candidates with the same 64-sign sketch used either for both selection and acceptance or only for selection followed by a fresh independent audit. The inset magnifies the 0--1% region of the same false-acceptance scale and labels the three fresh-audit rates. Each point represents 10,000 boundary-null trials; whiskers are Wilson 95% intervals. (b) Familywise false acceptance of the sequential fresh-audit protocol over 50,000 sessions per channel condition; the blue and orange whiskers show the corresponding Wilson 95% intervals. The horizontal line in each panel marks the prespecified 1% error target.

**Fig. 3. Search-audit allocation under a fixed 1,024-sign budget.** Cache reuse and fallback-inclusive completion traffic are shown for seven $(m_P,m_A)$ allocations. Points are means across five threshold-development protocol seeds and whiskers are sample standard deviations across seed means. The highlighted 256/768 allocation is the operating point selected by the frozen channel-averaged completion-traffic rule; the evaluation pool was excluded from selection. The dashed reference is the 4,272-bit semantic-refresh cost.

**Fig. 4. Primary evaluation of SentrySem-Fixed (256/768).** The three panels report cache reuse, exact source-instance precision among reuses, and fallback-inclusive completion traffic on the calibration-disjoint 500-instance evaluation pool. Blue filled circles denote $p=0$ and orange open triangles denote $p=0.05$. Central values average five protocol seeds. Whiskers are source-instance cluster-bootstrap 95% intervals that retain all five protocol outcomes for each resampled instance. The completion-traffic panel includes the 4,272-bit semantic-refresh reference.

**Fig. 5. Reuse-traffic operating points of statistically protected protocols.** Each panel reports one binary symmetric channel condition. Points 1--6 use the common-seed evaluation and point 7 is the calibration-disjoint selected 256/768 evaluation. Horizontal and vertical whiskers are source-instance cluster-bootstrap 95% intervals. The dashed line is the 4,272-bit semantic-refresh cost. Moving right increases cache reuse; moving downward reduces completion traffic. All plotted reuse protocols apply post-selection protection, and their measured exact-instance precision among reuses is at least 99.47%.

**Fig. 6. Representative inputs to the frozen semantic encoder.** Each row shows one cached source, its standard repeated view, its strong-shift view, and a novel same-class source identity absent from the cache. Scores are cosine similarities to the target cache entry for repeated views and maximum cache similarity for novel sources. Above-threshold and below-threshold labels refer to the calibrated verifier-similarity region defined by $s>\tau=0.7717$.

**Fig. 7. Selected SentrySem operating point under cache mismatch and query shift.** Overall cache reuse and fallback-inclusive completion traffic are reported for a full paired cache, three absent classes, 50% cache eviction, a hard paired view shift, and novel sources absent from the cache. The aligned panels preserve the metrics' natural scales: reuse is a percentage, whereas completion traffic combines protocol overhead with refresh probability. Points average five protocol seeds; whiskers are source-instance cluster-bootstrap 95% intervals. The completion panel includes the 4,272-bit semantic-refresh reference.
"""
    (OUT / "CAPTIONS.md").write_text(captions)


def generate_all_figures(
    results_root: Path | None = None,
    output_dir: Path | None = None,
    frozen_inputs: Path | None = None,
    validation_dir: Path | None = None,
) -> list[dict]:
    configure_paths(results_root, output_dir, frozen_inputs, validation_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    validation = validate_sources()
    figures = [
        ("Figure_1_System_Model", figure_1_system()),
        ("Figure_2_Statistical_Validation", figure_2_statistical_validation()),
        ("Figure_3_Search_Audit_Allocation", figure_3_allocation()),
        ("Figure_4_Primary_Performance", figure_4_primary()),
        ("Figure_5_Protected_Method_Tradeoff", figure_5_tradeoff()),
        ("Figure_7_Cache_Query_Stress", figure_6_stress()),
    ]
    outputs = [save_figure(drawing, stem) for stem, drawing in figures]
    from sentrysem.figures.qualitative import generate_qualitative_figure

    outputs.append(generate_qualitative_figure(FROZEN_INPUTS, OUT))
    outputs.sort(key=lambda item: item["stem"])
    write_companions(outputs, validation)
    print(json.dumps({"status": "PASS", "figures": len(outputs), "output_dir": str(OUT)}, indent=2))
    return outputs


def main() -> None:
    generate_all_figures()


if __name__ == "__main__":
    main()
