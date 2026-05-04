"""Likert and NASA-TLX walkthrough survey scaffold."""

from dataclasses import dataclass


@dataclass
class WalkthroughSurveyConfig:
    scale_points: int = 7
    include_nasa_tlx: bool = True
