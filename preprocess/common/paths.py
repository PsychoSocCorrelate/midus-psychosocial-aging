"""Repository path constants for preprocessing entrypoints."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
KEY_MAP_DIR = DATA_DIR / "key_map"
CONFIG_DIR = ROOT / "config"

RAW_BIOMARKER_DIR = RAW_DIR / "biomarker"
PROCESSED_BIOMARKER_DIR = PROCESSED_DIR / "biomarker"
