"""Metric registry helpers."""

from evaluation.metrics.calibration import metric_names as calibration_metric_names
from evaluation.metrics.fairness import metric_names as fairness_metric_names
from evaluation.metrics.fidelity import metric_names as fidelity_metric_names
from evaluation.metrics.privacy import metric_names as privacy_metric_names


def available_metric_groups() -> dict[str, tuple[str, ...]]:
    """Return the metric groups exposed by this package."""
    return {
        "fidelity": fidelity_metric_names(),
        "calibration": calibration_metric_names(),
        "fairness": fairness_metric_names(),
        "privacy": privacy_metric_names(),
    }
