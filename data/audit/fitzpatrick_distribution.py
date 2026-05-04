"""Scaffold for Fitzpatrick skin-tone auditing."""

from dataclasses import dataclass


@dataclass
class FitzpatrickAuditSummary:
    dataset_name: str
    total_records: int
    labelled_records: int


def build_summary(dataset_name: str, total_records: int, labelled_records: int) -> FitzpatrickAuditSummary:
    """Return a minimal audit summary placeholder for later extension."""
    return FitzpatrickAuditSummary(
        dataset_name=dataset_name,
        total_records=total_records,
        labelled_records=labelled_records,
    )
