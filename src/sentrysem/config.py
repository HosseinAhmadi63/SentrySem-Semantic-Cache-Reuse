"""Validated configuration objects for the published SentrySem experiments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from .statistics import binomial_boundary, observed_mismatch_probability
from .traffic import REFRESH_BITS, fixed_audit_traffic


@dataclass(frozen=True, slots=True)
class ViewConfig:
    """Deterministic repeated-view generator parameters in uint8 pixel units."""

    reflection_pad: int = 3
    horizontal_flip_probability: float = 0.5
    brightness_range: tuple[float, float] = (0.90, 1.10)
    contrast_range: tuple[float, float] = (0.90, 1.10)
    sensor_noise_sigma: float = 2.0
    strong_reflection_pad: int = 5
    strong_rotation_degrees: tuple[float, float] = (-12.0, 12.0)
    strong_brightness_range: tuple[float, float] = (0.75, 1.25)
    strong_contrast_range: tuple[float, float] = (0.75, 1.25)
    strong_sensor_noise_sigma: float = 6.0

    def validate(self) -> None:
        if self.reflection_pad < 0 or self.strong_reflection_pad < 0:
            raise ValueError("view padding must be non-negative")
        if not 0.0 <= self.horizontal_flip_probability <= 1.0:
            raise ValueError("horizontal_flip_probability must lie in [0, 1]")
        for name in (
            "brightness_range",
            "contrast_range",
            "strong_brightness_range",
            "strong_contrast_range",
        ):
            low, high = getattr(self, name)
            if not 0.0 < low <= high:
                raise ValueError(f"{name} must be positive and ordered")
        low_rotation, high_rotation = self.strong_rotation_degrees
        if low_rotation > high_rotation:
            raise ValueError("strong_rotation_degrees must be ordered")
        if self.sensor_noise_sigma < 0.0 or self.strong_sensor_noise_sigma < 0.0:
            raise ValueError("noise standard deviations must be non-negative")


@dataclass(frozen=True, slots=True)
class DataConfig:
    """CIFAR-10 identity counts and deterministic split offsets."""

    development_cache_per_class: int = 100
    calibration_tune_per_class: int = 50
    calibration_validate_per_class: int = 50
    final_holdout_skip_per_class: int = 200
    evaluation_cache_per_class: int = 100
    repeated_queries_per_class: int = 50
    novel_queries_per_class: int = 20
    test_identity_offset: int = 1_000_000
    tune_view_seed_offset: int = 10_000_000
    validate_view_seed_offset: int = 20_000_000
    repeated_view_seed_offset: int = 30_000_000
    strong_view_seed_offset: int = 40_000_000
    novel_view_seed_offset: int = 50_000_000
    views: ViewConfig = ViewConfig()

    def validate(self) -> None:
        counts = (
            self.development_cache_per_class,
            self.calibration_tune_per_class,
            self.calibration_validate_per_class,
            self.evaluation_cache_per_class,
            self.repeated_queries_per_class,
            self.novel_queries_per_class,
        )
        if any(value <= 0 for value in counts):
            raise ValueError("all per-class sample counts must be positive")
        if (
            self.calibration_tune_per_class + self.calibration_validate_per_class
            > self.development_cache_per_class
        ):
            raise ValueError("calibration subsets exceed the development cache")
        if self.repeated_queries_per_class > self.evaluation_cache_per_class:
            raise ValueError("repeated queries exceed the evaluation cache")
        if (
            self.final_holdout_skip_per_class
            + self.evaluation_cache_per_class
            + self.novel_queries_per_class
            > 1000
        ):
            raise ValueError("requested official-test samples exceed CIFAR-10 class size")
        self.views.validate()


@dataclass(frozen=True, slots=True)
class EncoderConfig:
    """Frozen semantic encoder and preprocessing specification."""

    architecture: str = "resnet18"
    weights: str = "IMAGENET1K_V1"
    embedding_dimension: int = 512
    input_pixels: int = 128
    interpolation: str = "bilinear"
    antialias: bool = True
    normalization_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalization_std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    def validate(self) -> None:
        if self.architecture != "resnet18" or self.weights != "IMAGENET1K_V1":
            raise ValueError("the published encoder is ResNet-18 with ImageNet-1K V1 weights")
        if self.embedding_dimension != 512 or self.input_pixels <= 0:
            raise ValueError("the published encoder emits 512-dimensional features")
        if self.interpolation != "bilinear" or not self.antialias:
            raise ValueError("the published preprocessing uses antialiased bilinear resizing")
        if len(self.normalization_mean) != 3 or len(self.normalization_std) != 3:
            raise ValueError("normalization must contain three RGB channels")
        if any(value <= 0.0 for value in self.normalization_std):
            raise ValueError("normalization standard deviations must be positive")


@dataclass(frozen=True, slots=True)
class ProtocolConfig:
    """Fixed SentrySem protocol and sequential diagnostic parameters."""

    proposal_bits: int = 256
    audit_bits: int = 768
    total_source_signs: int = 1024
    alpha: float = 0.01
    bsc_probabilities: tuple[float, float] = (0.0, 0.05)
    shortlist: int = 32
    sequential_proposal_bits: int = 64
    audit_batch_bits: int = 128
    sequential_audit_max_bits: int = 768
    sequential_max_attempts: int = 3
    mixture_points: int = 8

    def validate(self) -> None:
        if self.proposal_bits <= 0 or self.audit_bits <= 0:
            raise ValueError("proposal_bits and audit_bits must be positive")
        if self.proposal_bits + self.audit_bits != self.total_source_signs:
            raise ValueError("proposal and audit allocation must equal total_source_signs")
        if self.total_source_signs != 1024:
            raise ValueError("the published allocation uses 1024 source signs")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        if any(not 0.0 <= probability <= 0.5 for probability in self.bsc_probabilities):
            raise ValueError("BSC probabilities must lie in [0, 0.5]")
        if self.shortlist <= 0 or self.sequential_proposal_bits <= 0:
            raise ValueError("shortlist and sequential proposal size must be positive")
        if self.audit_batch_bits <= 0 or self.sequential_audit_max_bits % self.audit_batch_bits:
            raise ValueError("sequential audit maximum must be a multiple of its batch size")
        if not 1 <= self.sequential_max_attempts <= 3 or self.mixture_points <= 0:
            raise ValueError(
                "sequential attempts and mixture points are outside their valid ranges"
            )


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Replication counts used for selection, evaluation, and uncertainty."""

    protocol_seeds: int = 5
    cluster_bootstrap_trials: int = 2000
    boundary_trials: int = 10_000
    sequential_null_trials: int = 50_000
    calibration_min_pairs: int = 200
    calibration_tune_precision: float = 0.95
    calibration_validate_wilson_lower: float = 0.90
    ratio_search_bits: tuple[int, ...] = (128, 256, 384, 512, 640, 768, 896)

    def validate(self) -> None:
        if self.protocol_seeds < 2:
            raise ValueError("at least two protocol seeds are required")
        if (
            min(self.cluster_bootstrap_trials, self.boundary_trials, self.sequential_null_trials)
            <= 0
        ):
            raise ValueError("all replication counts must be positive")
        if self.calibration_min_pairs <= 0:
            raise ValueError("calibration_min_pairs must be positive")
        if not 0.0 < self.calibration_tune_precision <= 1.0:
            raise ValueError("calibration_tune_precision must lie in (0, 1]")
        if not 0.0 < self.calibration_validate_wilson_lower <= 1.0:
            raise ValueError("calibration_validate_wilson_lower must lie in (0, 1]")
        if not self.ratio_search_bits or any(
            value <= 0 or value >= 1024 for value in self.ratio_search_bits
        ):
            raise ValueError("ratio candidates must be strictly between zero and 1024")


@dataclass(frozen=True, slots=True)
class SentrySemConfig:
    """Complete configuration for the paper reproduction workflow."""

    seed: int = 20260907
    data: DataConfig = DataConfig()
    encoder: EncoderConfig = EncoderConfig()
    protocol: ProtocolConfig = ProtocolConfig()
    evaluation: EvaluationConfig = EvaluationConfig()

    def validate(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        self.data.validate()
        self.encoder.validate()
        self.protocol.validate()
        self.evaluation.validate()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable nested dictionary."""

        return asdict(self)

    def spawn_protocol_seeds(self, groups: int = 1) -> tuple[tuple[int, ...], ...]:
        """Create deterministic, disjoint uint64 seed groups."""

        if groups <= 0:
            raise ValueError("groups must be positive")
        count = self.evaluation.protocol_seeds
        children = np.random.SeedSequence(self.seed).spawn(groups * count)
        values = [int(child.generate_state(1, dtype=np.uint64)[0]) for child in children]
        result = tuple(
            tuple(values[start : start + count]) for start in range(0, len(values), count)
        )
        if len({value for group in result for value in group}) != groups * count:
            raise AssertionError("protocol seed groups are not disjoint")
        return result


ConfigType = TypeVar("ConfigType")


def _construct_dataclass(cls: type[ConfigType], values: dict[str, Any]) -> ConfigType:
    allowed = {field.name for field in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown {cls.__name__} fields: {', '.join(sorted(unknown))}")
    return cls(**values)


def config_from_dict(values: dict[str, Any]) -> SentrySemConfig:
    """Construct and validate a configuration from decoded JSON data."""

    if not isinstance(values, dict):
        raise TypeError("configuration root must be a JSON object")
    root = dict(values)
    data_values = dict(root.pop("data", {}))
    view_values = dict(data_values.pop("views", {}))
    if view_values:
        for name in (
            "brightness_range",
            "contrast_range",
            "strong_rotation_degrees",
            "strong_brightness_range",
            "strong_contrast_range",
        ):
            if name in view_values:
                view_values[name] = tuple(view_values[name])
        data_values["views"] = _construct_dataclass(ViewConfig, view_values)
    encoder_values = dict(root.pop("encoder", {}))
    for name in ("normalization_mean", "normalization_std"):
        if name in encoder_values:
            encoder_values[name] = tuple(encoder_values[name])
    protocol_values = dict(root.pop("protocol", {}))
    if "bsc_probabilities" in protocol_values:
        protocol_values["bsc_probabilities"] = tuple(protocol_values["bsc_probabilities"])
    evaluation_values = dict(root.pop("evaluation", {}))
    if "ratio_search_bits" in evaluation_values:
        evaluation_values["ratio_search_bits"] = tuple(evaluation_values["ratio_search_bits"])
    allowed_root = {"seed"}
    unknown_root = set(root) - allowed_root
    if unknown_root:
        raise ValueError(f"unknown SentrySemConfig fields: {', '.join(sorted(unknown_root))}")
    config = SentrySemConfig(
        seed=int(root.get("seed", 20260907)),
        data=_construct_dataclass(DataConfig, data_values),
        encoder=_construct_dataclass(EncoderConfig, encoder_values),
        protocol=_construct_dataclass(ProtocolConfig, protocol_values),
        evaluation=_construct_dataclass(EvaluationConfig, evaluation_values),
    )
    config.validate()
    return config


def load_config(path: str | Path) -> SentrySemConfig:
    """Load the typed scientific parameters from either supported JSON schema."""

    source = Path(path)
    if source.suffix.lower() != ".json":
        raise ValueError("SentrySem configuration files use the .json format")
    values = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(values, dict) and "schema_version" in values:
        validate_experiment_config(values)
        return scientific_config_from_experiment(values)
    return config_from_dict(values)


def save_config(config: SentrySemConfig, path: str | Path) -> None:
    """Write a validated configuration as deterministic, readable JSON."""

    config.validate()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8")


def publication_config() -> SentrySemConfig:
    """Return the frozen configuration used for the manuscript results."""

    config = SentrySemConfig()
    config.validate()
    return config


def _relative_repository_path(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must remain inside the repository")
    return value


def scientific_config_from_experiment(values: dict[str, Any]) -> SentrySemConfig:
    """Project the repository's complete experiment JSON onto typed parameters."""

    data_source = values.get("data", {})
    encoder_source = values.get("encoder", {})
    protocol_source = values["protocol"]
    standard_view = values.get("views", {}).get("standard", {})
    strong_view = values.get("views", {}).get("strong", {})
    calibration = values.get("calibration", {})
    uncertainty = values.get("uncertainty", {})
    allocation = values.get("allocation", {})
    sequential = protocol_source.get("sequential", {})
    views = ViewConfig(
        reflection_pad=int(standard_view.get("reflection_pad", 3)),
        horizontal_flip_probability=float(standard_view.get("horizontal_flip_probability", 0.5)),
        brightness_range=tuple(standard_view.get("brightness_range", (0.90, 1.10))),
        contrast_range=tuple(standard_view.get("contrast_range", (0.90, 1.10))),
        sensor_noise_sigma=float(standard_view.get("noise_standard_deviation_pixels", 2.0)),
        strong_reflection_pad=int(strong_view.get("reflection_pad", 5)),
        strong_rotation_degrees=tuple(strong_view.get("rotation_degrees", (-12.0, 12.0))),
        strong_brightness_range=tuple(strong_view.get("brightness_range", (0.75, 1.25))),
        strong_contrast_range=tuple(strong_view.get("contrast_range", (0.75, 1.25))),
        strong_sensor_noise_sigma=float(strong_view.get("noise_standard_deviation_pixels", 6.0)),
    )
    data = DataConfig(
        development_cache_per_class=int(data_source.get("development_cache_per_class", 100)),
        calibration_tune_per_class=int(data_source.get("calibration_tune_per_class", 50)),
        calibration_validate_per_class=int(data_source.get("calibration_validate_per_class", 50)),
        final_holdout_skip_per_class=int(data_source.get("evaluation_holdout_skip_per_class", 200)),
        evaluation_cache_per_class=int(data_source.get("evaluation_cache_per_class", 100)),
        repeated_queries_per_class=int(data_source.get("evaluation_query_per_class", 50)),
        novel_queries_per_class=int(data_source.get("novel_query_per_class", 20)),
        test_identity_offset=int(data_source.get("wire_id_offset", 1_000_000)),
        views=views,
    )
    encoder = EncoderConfig(
        architecture=str(encoder_source.get("architecture", "resnet18")),
        weights=str(encoder_source.get("weights", "IMAGENET1K_V1")),
        embedding_dimension=int(encoder_source.get("embedding_dimension", 512)),
        input_pixels=int(encoder_source.get("input_size", 128)),
        interpolation=str(encoder_source.get("resize_interpolation", "bilinear")),
        antialias=bool(encoder_source.get("resize_antialias", True)),
        normalization_mean=tuple(encoder_source.get("normalization_mean", (0.485, 0.456, 0.406))),
        normalization_std=tuple(encoder_source.get("normalization_std", (0.229, 0.224, 0.225))),
    )
    proposal_bits = int(protocol_source["proposal_signs"])
    audit_bits = int(protocol_source["audit_signs"])
    protocol = ProtocolConfig(
        proposal_bits=proposal_bits,
        audit_bits=audit_bits,
        total_source_signs=proposal_bits + audit_bits,
        alpha=float(protocol_source["alpha"]),
        bsc_probabilities=tuple(protocol_source["bsc_probabilities"]),
        sequential_proposal_bits=int(
            sequential.get("proposal_signs", calibration.get("proposal_signs", 64))
        ),
        audit_batch_bits=int(sequential.get("audit_batch_signs", 128)),
        sequential_audit_max_bits=int(sequential.get("audit_max_signs_per_candidate", 768)),
        sequential_max_attempts=int(sequential.get("maximum_candidates", 3)),
        mixture_points=int(sequential.get("alternative_grid_points", 8)),
    )
    candidate_splits = allocation.get(
        "candidate_splits",
        [[value, 1024 - value] for value in range(128, 897, 128)],
    )
    evaluation = EvaluationConfig(
        protocol_seeds=len(allocation.get("evaluation_protocol_seeds", range(5))),
        cluster_bootstrap_trials=int(uncertainty.get("cluster_bootstrap_trials", 2000)),
        calibration_min_pairs=int(calibration.get("minimum_selected_pairs", 200)),
        calibration_tune_precision=float(calibration.get("tune_precision_target", 0.95)),
        calibration_validate_wilson_lower=float(
            calibration.get("validation_wilson_lower_target", 0.90)
        ),
        ratio_search_bits=tuple(int(pair[0]) for pair in candidate_splits),
    )
    result = SentrySemConfig(
        seed=int(values["base_seed"]),
        data=data,
        encoder=encoder,
        protocol=protocol,
        evaluation=evaluation,
    )
    result.validate()
    return result


def validate_experiment_config(values: dict[str, Any]) -> None:
    """Validate the complete paper or installation-check JSON schema."""

    required = {
        "schema_version",
        "experiment_name",
        "base_seed",
        "execution_device",
        "paths",
        "protocol",
        "traffic",
        "expected",
    }
    missing = required - set(values)
    if missing:
        raise ValueError(f"experiment configuration is missing: {', '.join(sorted(missing))}")
    if values["schema_version"] != 1:
        raise ValueError("unsupported experiment configuration schema")
    if not isinstance(values["experiment_name"], str) or not values["experiment_name"].strip():
        raise ValueError("experiment_name must be a non-empty string")
    if values["execution_device"] not in ("cpu", "cuda"):
        raise ValueError("execution_device must be 'cpu' or 'cuda'")
    for name, value in values["paths"].items():
        _relative_repository_path(value, f"paths.{name}")
    protocol = values["protocol"]
    if protocol.get("proposal_tie_break") != "lowest-cache-index":
        raise ValueError("proposal ties must resolve to the lowest cache index")
    if protocol.get("strict_reuse_threshold") is not True:
        raise ValueError("verifier membership must use a strict similarity threshold")
    traffic = values["traffic"]
    ledger = fixed_audit_traffic(int(protocol["proposal_signs"]), int(protocol["audit_signs"]))
    traffic_checks = {
        "frame_overhead_bytes": 18,
        "candidate_lock_payload_bytes": 16,
        "audit_seed_payload_bytes": 8,
        "feedback_payload_bytes": 1,
        "refresh_feature_values": 512,
        "refresh_value_bits": 8,
        "refresh_scale_bits": 32,
        "expected_prefallback_bits": ledger.protocol_bits,
        "expected_refresh_bits": REFRESH_BITS,
    }
    for name, expected in traffic_checks.items():
        if int(traffic.get(name, -1)) != expected:
            raise ValueError(f"traffic.{name} must equal {expected}")
    threshold = float(
        values.get("calibration", {}).get(
            "threshold", values["expected"].get("threshold", 0.7717004776000975)
        )
    )
    for probability in protocol["bsc_probabilities"]:
        key = str(float(probability))
        expected_boundaries = values["expected"].get("audit_boundaries", {})
        if key in expected_boundaries:
            q0 = float(observed_mismatch_probability(threshold, float(probability)))
            boundary = binomial_boundary(int(protocol["audit_signs"]), q0, float(protocol["alpha"]))
            if int(expected_boundaries[key]) != boundary:
                raise ValueError(f"expected audit boundary is inconsistent at BSC={key}")
    if "smoke" in values:
        smoke = values["smoke"]
        if int(smoke.get("queries", 0)) <= 0 or int(smoke.get("cache_entries", 0)) <= 0:
            raise ValueError("smoke query and cache sizes must be positive")
        if not smoke.get("protocol_seeds"):
            raise ValueError("smoke configuration requires at least one protocol seed")
        return
    full_sections = {
        "encoder",
        "data",
        "views",
        "calibration",
        "allocation",
        "comparators",
        "uncertainty",
        "stress",
    }
    missing_sections = full_sections - set(values)
    if missing_sections:
        raise ValueError(f"paper configuration is missing: {', '.join(sorted(missing_sections))}")
    scientific = scientific_config_from_experiment(values)
    encoder = values["encoder"]
    if encoder.get("l2_normalize") is not True:
        raise ValueError("encoder output must be L2 normalized")
    if int(encoder.get("expected_parameter_count", -1)) != 11_176_512:
        raise ValueError("unexpected frozen encoder parameter count")
    expected_state = str(encoder.get("expected_state_sha256", ""))
    if expected_state != "d7ecbcb063a266bf46ab86ce4e0d141732ba0d286c3fb25a8d55783bda06e41e":
        raise ValueError("unexpected frozen encoder digest")
    if values["data"].get("dataset") != "CIFAR10":
        raise ValueError("the publication data source must be CIFAR-10")
    tune_seeds = tuple(map(int, values["allocation"]["tune_protocol_seeds"]))
    evaluation_seeds = tuple(map(int, values["allocation"]["evaluation_protocol_seeds"]))
    if len(tune_seeds) != len(set(tune_seeds)) or len(evaluation_seeds) != len(
        set(evaluation_seeds)
    ):
        raise ValueError("protocol seed lists contain duplicates")
    if set(tune_seeds) & set(evaluation_seeds):
        raise ValueError("tuning and evaluation protocol seeds must be disjoint")
    generated_tune, generated_evaluation = scientific.spawn_protocol_seeds(2)
    if tune_seeds != generated_tune or evaluation_seeds != generated_evaluation:
        raise ValueError("protocol seeds do not match the declared base seed")


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a complete repository experiment configuration."""

    source = Path(path)
    if source.suffix.lower() != ".json":
        raise ValueError("experiment configuration files use the .json format")
    values = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise TypeError("configuration root must be a JSON object")
    validate_experiment_config(values)
    return values
