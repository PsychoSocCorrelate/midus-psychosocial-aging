#!/usr/bin/env python3
"""
Derive the recurring-predictor profile from the outcome-by-layer top-20 lists.

Rule
----
A feature is in the recurring profile when it appears in at least four of the
six top-20 lists: one per primary outcome at each of the two layers, the
cross-sectional models and the baseline-conditioned longitudinal ones.
`phenoAge_res` is not counted.

A top-20 list is the top twenty features by mean |SHAP| over all 177 predictors,
as stored by a3 in `source_top20_features`. Tier 1 features can enter; `tier` is
reported per row.

Writes
------
    output/recurring_predictors/recurring_predictors.csv

Reads only a3's two cross-prediction tables, so it runs without the
participant-level master dataset.

Usage:
    uv run python analysis/pipelines/recurring_predictors.py
"""

from __future__ import annotations

import collections
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.core.pub_style import get_subdomain  # noqa: E402
from config.variable_config import get_tier_variables  # noqa: E402

A3_DIR = ROOT / "output/a3_domain_divergence"
OUT = ROOT / "output/recurring_predictors/recurring_predictors.csv"

# The six outcome-by-layer lists
LAYERS = {
    "cross_sectional": (A3_DIR / "a3_cross_prediction_cs1.csv",
                        ["FInoPsy_W1", "btactComp_W1", "ALcomp_W1"]),
    "longitudinal": (A3_DIR / "a3_cross_prediction.csv",
                     ["FInoPsy_W2", "btactComp_W2", "ALcomp_W2"]),
}
MIN_LISTS = 4
N_LISTS = 6


def top20_lists() -> dict[str, list[str]]:
    """One top-20 list per outcome per layer, keyed `<layer>:<outcome>`."""
    out: dict[str, list[str]] = {}
    for layer, (path, outcomes) in LAYERS.items():
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        for oc in outcomes:
            rows = df[df["source_outcome"] == oc]
            if rows.empty:
                raise ValueError(f"{path.name} has no source_outcome {oc!r}")
            feats = [f.strip() for f in rows.iloc[0]["source_top20_features"].split(",")]
            if len(feats) != 20:
                raise ValueError(f"{layer}:{oc} carries {len(feats)} features, expected 20")
            out[f"{layer}:{oc}"] = feats
    if len(out) != N_LISTS:
        raise ValueError(f"expected {N_LISTS} lists, built {len(out)}")
    return out


def build() -> pd.DataFrame:
    lists = top20_lists()
    counts = collections.Counter(f for feats in lists.values() for f in feats)
    t1 = set(get_tier_variables([1]))

    rows = []
    for feat, n in counts.items():
        rows.append({
            "feature": feat,
            "n_lists": n,
            "n_lists_total": N_LISTS,
            "in_profile": n >= MIN_LISTS,
            "tier": "T1" if feat in t1 else "T2",
            "subdomain": get_subdomain(feat),
            "layers": ";".join(sorted(k for k, v in lists.items() if feat in v)),
        })
    df = pd.DataFrame(rows).sort_values(
        ["n_lists", "feature"], ascending=[False, True]).reset_index(drop=True)
    return df


def profile(min_lists: int = MIN_LISTS) -> list[str]:
    """The recurring profile, read from the produced file.

    Reads the file main() writes, so every consumer uses the same set.
    """
    if not OUT.exists():
        raise FileNotFoundError(
            f"{OUT.relative_to(ROOT)} not found; run "
            "`uv run python analysis/pipelines/recurring_predictors.py` first")
    df = pd.read_csv(OUT)
    return df.loc[df["n_lists"] >= min_lists, "feature"].tolist()


def main() -> None:
    df = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    sel = df[df["in_profile"]]
    print(f"Recurring profile: {len(sel)} features in >= {MIN_LISTS} of {N_LISTS} top-20 lists")
    for _, r in sel.iterrows():
        print(f"  {r.feature:<22} {r.n_lists}/{N_LISTS}  {r.tier}  {r.subdomain}")
    t1 = sel[sel["tier"] == "T1"]["feature"].tolist()
    if t1:
        print(f"\n  Tier 1 members: {', '.join(t1)}")
    near = df[(~df["in_profile"]) & (df["n_lists"] == MIN_LISTS - 1)]["feature"].tolist()
    if near:
        print(f"\n  Just below threshold at {MIN_LISTS - 1}/{N_LISTS}: {', '.join(near)}")
    print(f"\n-> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
