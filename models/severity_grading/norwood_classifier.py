"""Norwood severity grading scaffold."""

from dataclasses import dataclass


@dataclass
class NorwoodClassifierConfig:
    backbone: str = "resnet18"
    class_count: int = 7


def baseline_tag(config: NorwoodClassifierConfig) -> str:
    return f"{config.backbone}-norwood-{config.class_count}"
