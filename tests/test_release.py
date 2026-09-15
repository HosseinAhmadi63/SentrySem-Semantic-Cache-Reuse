from __future__ import annotations

import xml.etree.ElementTree as ET
import zipfile

from pypdf import PdfReader

from sentrysem.figures import paper
from sentrysem.validation import verify_repository


def test_reference_artifacts_validate(repository_root) -> None:
    report = verify_repository(repository_root, repository_root / "configs" / "paper.json")
    assert report["status"] == "PASS"
    assert report["checks_passed"] >= 12


def test_all_seven_paper_figures_are_single_page(repository_root) -> None:
    figures = sorted((repository_root / "figures" / "paper").glob("Figure_*.pdf"))
    assert len(figures) == 7
    for figure in figures:
        assert figure.stat().st_size > 1_000
        assert len(PdfReader(str(figure)).pages) == 1


def test_paper_figure_metadata_is_publication_ready(repository_root) -> None:
    directory = repository_root / "figures" / "paper"
    for pdf_path in sorted(directory.glob("Figure_*.pdf")):
        metadata = PdfReader(str(pdf_path)).metadata
        assert metadata.author == "Hossein Ahmadi; Zahra Ziar; Ali Kuhestani"
        assert metadata.title and "Figure" not in metadata.title
        svg = pdf_path.with_suffix(".svg").read_text(encoding="utf-8")
        assert "<title>...</title>" not in svg
        assert "SentrySem" in svg


def test_curated_system_figure_bundle_and_copy(repository_root, tmp_path, monkeypatch) -> None:
    stem = "Figure_1_System_Model"
    source = repository_root / "figures" / "paper"
    for suffix in ("pdf", "png", "svg", "pptx"):
        path = source / f"{stem}.{suffix}"
        assert path.is_file()
        assert path.stat().st_size > 1_000
    with zipfile.ZipFile(source / f"{stem}.pptx") as archive:
        assert "[Content_Types].xml" in archive.namelist()
        assert "ppt/slides/slide1.xml" in archive.namelist()

    monkeypatch.setattr(paper, "OUT", tmp_path)
    result = paper.copy_curated_figure(stem)
    assert result["source_mode"] == "curated conceptual figure"
    for suffix in ("pdf", "png", "svg"):
        assert (tmp_path / f"{stem}.{suffix}").read_bytes() == (
            source / f"{stem}.{suffix}"
        ).read_bytes()
    assert not (tmp_path / f"{stem}.pptx").exists()


def test_pycharm_run_configurations_are_complete(repository_root) -> None:
    paths = sorted((repository_root / ".run").glob("*.run.xml"))
    assert len(paths) == 6
    for path in paths:
        configuration = ET.parse(path).getroot().find("configuration")
        assert configuration is not None
        options = {
            item.attrib["name"]: item.attrib["value"] for item in configuration.findall("option")
        }
        assert options["WORKING_DIRECTORY"] == "$PROJECT_DIR$"
        assert options["SCRIPT_NAME"] == "$PROJECT_DIR$/main.py"
        assert options["PARAMETERS"]
