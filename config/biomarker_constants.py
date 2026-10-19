"""Canonical biomarker definitions shared by preprocessing and analyses."""

from __future__ import annotations

# The allostatic-load composite: 16 biomarkers in five systems. BMI and waist-hip
# ratio are Tier 2 predictors and stay out of the outcome.
# (standard_name, suffix after the wave prefix, system, direction)
# direction: "HIGH" = high-risk quartile is top 75th; "LOW" = bottom 25th.
BIOMARKERS = [
    ("SBP", "P1GS", "cardiovascular", "HIGH"),
    ("DBP", "P1GD", "cardiovascular", "HIGH"),
    ("pulse", "P1D", "cardiovascular", "HIGH"),
    ("cholHDL_ratio", "BRTHDL", "metabolic", "HIGH"),
    ("HDL", "BHDL", "metabolic", "LOW"),
    ("triglycerides", "BTRIGL", "metabolic", "HIGH"),
    ("HbA1c", "BHA1C", "metabolic", "HIGH"),
    ("glucose", "BGLUC", "metabolic", "HIGH"),
    ("insulin", "BINSLN", "metabolic", "HIGH"),
    ("IGF1", "BIGF1", "metabolic", "LOW"),
    ("DHEAS", "BDHEAS", "hpa_axis", "LOW"),
    ("CRP", "BCRP", "inflammatory", "HIGH"),
    ("IL6", "BIL6", "inflammatory", "HIGH"),
    ("fibrinogen", "BFGN", "inflammatory", "HIGH"),
    ("sIL6R", "BSIL6R", "inflammatory", "HIGH"),
    ("creatinine", "BSCREA", "renal", "HIGH"),
]

MED_FLAGS = {
    "antihypertensive": [
        "XTCS_42",
        "XTCS_47",
        "XTCS_49",
        "XTCS_55",
        "XPC_957",
        "XPC_958",
        "XPC_1156",
    ],
    "antilipidemic": [
        "XTCS_19",
        "XPC_1171",
    ],
    "antidiabetic": [
        "XTCS_99",
    ],
}

MED_ADJUSTMENTS = {
    "antihypertensive": {"SBP": 10, "DBP": 5},
    "antilipidemic": {"cholHDL_ratio": None, "HDL": -5},
    "antidiabetic": {"HbA1c": 1.0, "glucose": 20},
}

SYSTEMS = sorted(set(b[2] for b in BIOMARKERS))

# The column the allostatic-load stage writes, mapped to the wave-1 and wave-2
# outcome names the analysis layer uses.
AL_COMPOSITE_COLUMNS = {
    "ALcomp_zscore": ("ALcomp_W1", "ALcomp_W2"),
}
