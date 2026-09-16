"""Quality metrics and gates for the commercial fork (license-clean)."""

from .quality_gates import (
    QualityGateConfig,
    aggregate,
    color_fidelity,
    evaluate,
    passes_quality_gate,
    sharpness,
    wasserstein_distance,
    weighted_score,
)
from .regression import (
    HIGHER_IS_BETTER,
    LOWER_IS_BETTER,
    TRACKED_METRICS,
    RegressionReport,
    compare_metrics,
)

__all__ = [
    "HIGHER_IS_BETTER",
    "LOWER_IS_BETTER",
    "TRACKED_METRICS",
    "QualityGateConfig",
    "RegressionReport",
    "aggregate",
    "color_fidelity",
    "compare_metrics",
    "evaluate",
    "passes_quality_gate",
    "sharpness",
    "wasserstein_distance",
    "weighted_score",
]
