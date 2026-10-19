"""Frailty construct-overlap exclusion list.

Lets pipelines read the list without importing the A2 audit script.
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FRAILTY_OVERLAP_AUDIT_CSV = ROOT / "config" / "frailty_overlap_audit.csv"


def get_overlap_excluded_vars(codes: str = "ab") -> list[str]:
    """Return frailty predictors with the requested overlap audit codes."""
    with FRAILTY_OVERLAP_AUDIT_CSV.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [row["predictor"] for row in rows if row["overlap_code"] in codes]
