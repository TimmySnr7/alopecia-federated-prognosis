"""Minimal severity grading baselines for Experiment 1."""

from dataclasses import dataclass

from torch import nn
from torchvision.models import resnet18

@dataclass
class NorwoodClassifierConfig:
    backbone: str = "resnet18"
    class_count: int = 7


def baseline_tag(config: NorwoodClassifierConfig) -> str:
    return f"{config.backbone}-norwood-{config.class_count}"


def build_baseline_classifier(config: NorwoodClassifierConfig) -> nn.Module:
    """Construct a lightweight image classifier for severity smoke tests."""
    if config.backbone != "resnet18":
        raise ValueError(f"Unsupported backbone: {config.backbone}")

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, config.class_count)
    return model
