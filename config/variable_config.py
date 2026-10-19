"""Helpers for reading the project predictor tier configuration."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_YAML = ROOT / "config" / "variable_config.yaml"

# Variables with >50% missingness, excluded from primary models by default.
HIGH_MISSING = {"familyHeartAttack", "weightChangeExercise", "weightChangeHealthIssue"}


def load_variable_config(path: str | Path = CONFIG_YAML) -> dict:
    """Load the variable tier YAML configuration."""
    with open(path) as f:
        return yaml.safe_load(f)


def get_tier_variables(
    tiers: list[int],
    exclude_high_missing: bool = True,
    path: str | Path = CONFIG_YAML,
) -> list[str]:
    """Return ordered predictor names for the requested tiers."""
    cfg = load_variable_config(path)
    variables: list[str] = []
    for t in sorted(set(tiers)):
        key = f"tier{t}"
        if key not in cfg:
            raise ValueError(f"Tier {t} not found in variable_config.yaml")
        variables.extend(cfg[key])

    if exclude_high_missing:
        variables = [v for v in variables if v not in HIGH_MISSING]

    seen: set[str] = set()
    unique: list[str] = []
    for var in variables:
        if var not in seen:
            seen.add(var)
            unique.append(var)
    return unique
