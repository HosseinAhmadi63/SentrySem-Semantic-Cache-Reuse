"""Regenerate all seven manuscript figures from the committed records."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentrysem.cli import main

raise SystemExit(main(["figures"]))
