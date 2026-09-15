"""Rebuild the repository-wide SHA-256 release manifest."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentrysem.release import build_release_manifest, write_release_manifest

destination = write_release_manifest(ROOT)
print(f"Wrote {destination.name} for {len(build_release_manifest(ROOT))} files")
