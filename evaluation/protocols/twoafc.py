"""2AFC realism protocol scaffold."""

from dataclasses import dataclass


@dataclass
class TwoAFCProtocol:
    minimum_raters: int = 3
    pair_count: int = 20
