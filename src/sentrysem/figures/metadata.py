"""Metadata utilities for publication figure PDFs."""

from __future__ import annotations

import re
from pathlib import Path
from xml.sax.saxutils import escape

from pypdf import PdfReader, PdfWriter

AUTHORS = "Hossein Ahmadi; Zahra Ziar; Ali Kuhestani"
FIGURE_TITLES = {
    "Figure_1_System_Model": "SentrySem system model and protocol sequence",
    "Figure_2_Statistical_Validation": "Post-selection and sequential-audit validation",
    "Figure_3_Search_Audit_Allocation": "Proposal-audit sign allocation",
    "Figure_4_Primary_Performance": "Primary cache reuse, precision, and traffic results",
    "Figure_5_Protected_Method_Tradeoff": "Reliability-protected method trade-off",
    "Figure_6_Evaluation_Inputs": "Representative cache and query inputs",
    "Figure_7_Cache_Query_Stress": "Cache and query stress evaluation",
}


def figure_title(stem: str) -> str:
    return FIGURE_TITLES.get(stem, stem.replace("_", " "))


def set_pdf_metadata(path: Path, title: str) -> None:
    """Write stable descriptive metadata without changing the rendered page."""

    path = Path(path)
    reader = PdfReader(str(path))
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    writer.add_metadata(
        {
            "/Title": title,
            "/Author": AUTHORS,
            "/Subject": "SentrySem manuscript figure",
            "/Creator": "SentrySem figure generator",
        }
    )
    temporary = path.with_suffix(".metadata.pdf")
    with temporary.open("wb") as stream:
        writer.write(stream)
    temporary.replace(path)


def set_svg_metadata(path: Path, title: str) -> None:
    """Set descriptive SVG title and description elements."""

    path = Path(path)
    content = path.read_text(encoding="utf-8")
    description = f"Vector figure for the SentrySem paper: {title}."
    title_element = f"<title>{escape(title)}</title>"
    description_element = f"<desc>{escape(description)}</desc>"
    if "<title>" in content:
        content = re.sub(r"<title>.*?</title>", title_element, content, count=1)
    else:
        content = re.sub(r"(<svg\b[^>]*>)", rf"\1\n{title_element}", content, count=1)
    if "<desc>" in content:
        content = re.sub(r"<desc>.*?</desc>", description_element, content, count=1)
    else:
        content = content.replace(title_element, f"{title_element}\n{description_element}", 1)
    path.write_text(content, encoding="utf-8")
