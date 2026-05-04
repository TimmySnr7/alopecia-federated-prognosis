"""Calibration metric stubs."""


def metric_names() -> tuple[str, ...]:
    return ("ece", "brier_score", "interval_coverage")
