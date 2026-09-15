"""Reproducible implementation of reliable semantic cache reuse."""

from .config import (
    DataConfig,
    EncoderConfig,
    EvaluationConfig,
    ProtocolConfig,
    SentrySemConfig,
    ViewConfig,
    config_from_dict,
    load_config,
    load_experiment_config,
    publication_config,
)
from .protocol import (
    SELECTED_AUDIT_BITS,
    SELECTED_PROPOSAL_BITS,
    TOTAL_ALPHA,
    FixedAuditResult,
    SequentialAuditResult,
    ThresholdCalibration,
    calibrate_threshold,
    run_fixed_audit,
    run_sequential_cache_protocol,
)
from .statistics import (
    binomial_boundary,
    binomial_cdf,
    observed_mismatch_probability,
    wilson_interval,
    wilson_lower,
)
from .traffic import REFRESH_BITS, fixed_audit_traffic

__version__ = "1.0.0"

__all__ = [
    "DataConfig",
    "EncoderConfig",
    "EvaluationConfig",
    "FixedAuditResult",
    "ProtocolConfig",
    "REFRESH_BITS",
    "SELECTED_AUDIT_BITS",
    "SELECTED_PROPOSAL_BITS",
    "SequentialAuditResult",
    "SentrySemConfig",
    "TOTAL_ALPHA",
    "ThresholdCalibration",
    "ViewConfig",
    "binomial_boundary",
    "binomial_cdf",
    "calibrate_threshold",
    "config_from_dict",
    "fixed_audit_traffic",
    "load_config",
    "load_experiment_config",
    "observed_mismatch_probability",
    "publication_config",
    "run_fixed_audit",
    "run_sequential_cache_protocol",
    "wilson_interval",
    "wilson_lower",
]
