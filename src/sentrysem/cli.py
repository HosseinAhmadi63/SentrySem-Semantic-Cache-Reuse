"""Command-line interface for SentrySem reproduction and verification."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .config import load_experiment_config
from .data import load_frozen_embeddings
from .hashing import file_manifest, sha256_file
from .protocol import run_fixed_audit
from .results import write_json
from .validation import verify_repository


def _repository_root() -> Path:
    candidates = (Path.cwd(), Path(__file__).resolve().parents[2])
    for candidate in candidates:
        if (candidate / "configs" / "paper.json").is_file() and (
            candidate / "artifacts" / "frozen_inputs"
        ).is_dir():
            return candidate.resolve()
    return Path.cwd().resolve()


REPOSITORY_ROOT = _repository_root()
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "paper.json"


def _repository_path(value: Path) -> Path:
    return value if value.is_absolute() else (REPOSITORY_ROOT / value).resolve()


def _portable_path(value: Path) -> str:
    resolved = Path(value).resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _installed_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _empty_or_create(path: Path, overwrite: bool) -> Path:
    target = path.resolve()
    generated_root = (REPOSITORY_ROOT / "results" / "generated").resolve()
    if target.exists() and any(target.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"{target} is not empty. Use --overwrite to replace this generated result tree."
            )
        if target != generated_root:
            raise ValueError(
                "--overwrite is restricted to the repository results/generated directory"
            )
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    return target


def command_verify(arguments: argparse.Namespace) -> int:
    config_path = _repository_path(arguments.config)
    results = _repository_path(arguments.results) if arguments.results else None
    inputs = _repository_path(arguments.inputs) if arguments.inputs else None
    report = verify_repository(REPOSITORY_ROOT, config_path, results, inputs)
    print(json.dumps(report, indent=2))
    return 0


def command_smoke(arguments: argparse.Namespace) -> int:
    config_path = _repository_path(arguments.config)
    config = load_experiment_config(config_path)
    inputs = _repository_path(Path(config["paths"]["frozen_inputs"]))
    arrays = load_frozen_embeddings(inputs / "frozen_embeddings.npz")
    query_count = int(config["smoke"]["queries"])
    cache_count = int(config["smoke"]["cache_entries"])
    queries = arrays["paired_iid_vectors"][:query_count]
    query_ids = arrays["paired_iid_target_ids"][:query_count]
    bank_ids = arrays["final_bank_ids"]
    target_indices = [int(np.flatnonzero(bank_ids == identity)[0]) for identity in query_ids]
    chosen = list(dict.fromkeys(target_indices))
    chosen.extend(index for index in range(len(bank_ids)) if index not in chosen)
    bank = arrays["final_bank"][np.asarray(chosen[:cache_count], dtype=np.int32)]
    threshold = float(json.loads((inputs / "calibration.json").read_text())["threshold_cosine"])
    records = []
    for probability in map(float, config["protocol"]["bsc_probabilities"]):
        result = run_fixed_audit(
            queries,
            bank,
            threshold,
            probability,
            int(config["smoke"]["protocol_seeds"][0]),
            int(config["protocol"]["proposal_signs"]),
            int(config["protocol"]["audit_signs"]),
            float(config["protocol"]["alpha"]),
        )
        expected_boundary = int(config["expected"]["audit_boundaries"][str(probability)])
        if result.audit_boundary != expected_boundary or result.protocol_bits != 1944:
            raise AssertionError("Smoke-test protocol constants changed")
        records.append(
            {
                "bsc": probability,
                "queries": query_count,
                "cache_entries": cache_count,
                "audit_boundary": result.audit_boundary,
                "protocol_bits": result.protocol_bits,
                "reuse_rate": float(result.reused.mean()),
                "mean_completion_bits": float(result.completion_bits.mean()),
            }
        )
    output = _repository_path(Path(config["paths"]["generated_results"]))
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "PASS",
        "config": _portable_path(config_path),
        "frozen_embeddings_sha256": sha256_file(inputs / "frozen_embeddings.npz"),
        "results": records,
    }
    write_json(output / "smoke_results.json", report)
    print(json.dumps(report, indent=2))
    return 0


def command_extract(arguments: argparse.Namespace) -> int:
    from .extraction import extract_embeddings

    config_path = _repository_path(arguments.config)
    output = _repository_path(arguments.output)
    data_root = _repository_path(arguments.data_root)
    report = extract_embeddings(config_path, output, data_root, not arguments.no_download)
    print(json.dumps(report, indent=2))
    return 0


def command_figures(arguments: argparse.Namespace) -> int:
    from .figures.paper import generate_all_figures

    config_path = _repository_path(arguments.config)
    config = load_experiment_config(config_path)
    results = _repository_path(arguments.results or Path(config["paths"]["reference_results"]))
    inputs = _repository_path(arguments.inputs or Path(config["paths"]["frozen_inputs"]))
    output = _repository_path(arguments.output or Path(config["paths"]["generated_figures"]))
    validation_dir = results / "validation"
    if not (validation_dir / "selection_stress.json").is_file():
        validation_dir = inputs
    generated = generate_all_figures(results, output, inputs, validation_dir)
    print(
        json.dumps({"status": "PASS", "figures": len(generated), "output": str(output)}, indent=2)
    )
    return 0


def command_reproduce(arguments: argparse.Namespace) -> int:
    from .experiments import allocation, calibration, comparators, stress, validation
    from .figures.paper import generate_all_figures

    started = time.time()
    config_path = _repository_path(arguments.config)
    config = load_experiment_config(config_path)
    inputs = _repository_path(arguments.inputs or Path(config["paths"]["frozen_inputs"]))
    output = _empty_or_create(
        _repository_path(arguments.output or Path(config["paths"]["generated_results"])),
        arguments.overwrite,
    )
    base_seed = int(config["base_seed"])
    seed_count = len(config["allocation"]["tune_protocol_seeds"])

    calibration.run_calibration(inputs, output / "calibration", base_seed)
    validation.run_validation_experiments(
        output / "validation",
        float(config["calibration"]["threshold"]),
        int(config["expected"]["selection_bias_counts"]["trials_per_point"]),
        int(config["expected"]["sequential_null_counts"]["trials_per_channel"]),
        base_seed,
    )
    allocation.main(
        SimpleNamespace(
            input_dir=inputs,
            output=output / "allocation",
            seeds=seed_count,
            base_seed=base_seed,
        )
    )
    comparators.run(
        SimpleNamespace(
            input_dir=inputs,
            output=output / "comparators",
            seeds=seed_count,
            base_seed=base_seed,
        )
    )
    stress.main(
        SimpleNamespace(
            input_dir=inputs,
            selection=output / "allocation" / "selection.json",
            output=output / "stress",
        )
    )
    figures_output = _repository_path(Path(config["paths"]["generated_figures"]))
    generate_all_figures(output, figures_output, inputs, output / "validation")
    verification = verify_repository(REPOSITORY_ROOT, config_path, output, inputs)
    manifest = {
        "status": "PASS",
        "elapsed_seconds": time.time() - started,
        "package_version": _installed_version("sentrysem"),
        "configuration": _portable_path(config_path),
        "configuration_sha256": sha256_file(config_path),
        "inputs": _portable_path(inputs),
        "results": _portable_path(output),
        "figures": _portable_path(figures_output),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": _installed_version("pandas"),
            "matplotlib": _installed_version("matplotlib"),
        },
        "stages": [
            "calibration",
            "post-selection validation",
            "proposal-audit allocation",
            "method comparison",
            "stress evaluation",
            "figure generation",
            "release validation",
        ],
        "protocol_seeds": {
            "development": config["allocation"]["tune_protocol_seeds"],
            "evaluation": config["allocation"]["evaluation_protocol_seeds"],
        },
        "input_artifacts": file_manifest(inputs, recursive=True),
        "verification_checks_passed": verification["checks_passed"],
        "artifacts": file_manifest(output, recursive=True),
    }
    write_json(output / "reproduction_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sentrysem",
        description="Reproduce and validate the SentrySem semantic cache-reuse experiments.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser("verify", help="validate committed or regenerated results")
    verify.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    verify.add_argument("--results", type=Path)
    verify.add_argument("--inputs", type=Path)
    verify.set_defaults(handler=command_verify)

    reproduce = subparsers.add_parser("reproduce", help="rerun every paper experiment")
    reproduce.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    reproduce.add_argument("--inputs", type=Path)
    reproduce.add_argument("--output", type=Path)
    reproduce.add_argument("--overwrite", action="store_true")
    reproduce.set_defaults(handler=command_reproduce)

    figures = subparsers.add_parser("figures", help="regenerate all seven paper figures")
    figures.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    figures.add_argument("--results", type=Path)
    figures.add_argument("--inputs", type=Path)
    figures.add_argument("--output", type=Path)
    figures.set_defaults(handler=command_figures)

    extract = subparsers.add_parser(
        "extract-embeddings",
        help="rebuild frozen semantic embeddings from official CIFAR-10 images",
    )
    extract.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    extract.add_argument("--output", type=Path, default=Path("artifacts/generated_inputs"))
    extract.add_argument("--data-root", type=Path, default=Path("data/cache"))
    extract.add_argument("--no-download", action="store_true")
    extract.set_defaults(handler=command_extract)

    smoke = subparsers.add_parser("smoke", help="run a fast installation and protocol check")
    smoke.add_argument("--config", type=Path, default=Path("configs/smoke.json"))
    smoke.set_defaults(handler=command_smoke)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        return int(arguments.handler(arguments))
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
