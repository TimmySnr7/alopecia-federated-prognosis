from evaluation.metrics import available_metric_groups


def test_metric_groups_are_registered() -> None:
    groups = available_metric_groups()
    assert "fidelity" in groups
    assert "calibration" in groups
    assert "fairness" in groups
    assert "privacy" in groups
    assert "lpips" in groups["fidelity"]
