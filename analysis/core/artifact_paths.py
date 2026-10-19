"""Step output directories and the allostatic-load outcome stem.

Kept free of matplotlib imports (unlike `pub_style`) so preprocess stages can
use it.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output"

# Step output directories
CS1_DIR = OUTPUT / "cs1_cross_sectional"
A3_DIR = OUTPUT / "a3_domain_divergence"
A4_DIR = OUTPUT / "a4_longitudinal"
A4B_DIR = OUTPUT / "a4b_longitudinal"
A6_DIR = OUTPUT / "a6_sibling"
A7_DIR = OUTPUT / "a7_mortality"
A9_DIR = OUTPUT / "a9_concurrent"

# Stem of the allostatic-load outcome columns (ALcomp_W1, ALcomp_W2).
AL_STEM = "ALcomp"

