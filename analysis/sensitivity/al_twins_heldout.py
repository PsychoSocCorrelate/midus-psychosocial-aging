#!/usr/bin/env python3
"""Allostatic-load estimates re-fit with every complete sibling/twin pair held out.

The AL pipeline keeps complete sibling/twin pairs (SIBLING_EXCLUSION_EXEMPT), so
AL is the only outcome whose twins can straddle CV folds. This re-estimates the
composite's Tier 2 increments and top-20 with those pairs dropped from
training (load_master(..., force_exclude_siblings=True)) and compares them with
the siblings-retained canonical values, read from cs1 and a4 rather than copied.

Outputs (output/al_twins_heldout/):
  al_twins_heldout_summary.csv  one row per layer: n, tier R2s, delta-R2, canonical
  al_twins_heldout_top20.csv    held-out vs canonical top-20 ranks
  ALcomp_W1_heldout_bootstrap_shap.csv

Run:  uv run python analysis/sensitivity/al_twins_heldout.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.core.data_loader import load_master  # noqa: E402
from analysis.core.xgboost_pipeline import XGBoostPipeline  # noqa: E402
from analysis.core import artifact_paths as ap  # noqa: E402

OUT = ROOT / "output" / "al_twins_heldout"
USE_GPU = False
N_FOLDS = 5
N_BOOT = 200  # matches cs1's default, so the top-20 comparison is like for like

CS_OUTCOME = "ALcomp_W1"
LONG_OUTCOME = "ALcomp_W2"


def canonical_r2(table: Path, outcome: str) -> dict[str, tuple[float, int]]:
    df = pd.read_csv(table)
    df = df[df["outcome"] == outcome]
    if "cohort" in df.columns:
        df = df[df["cohort"] == "pooled"]
    return {r.tier_config: (float(r.r2_mean), int(r.n)) for r in df.itertuples()}


def tiered_r2(outcome, tier_configs, *, longitudinal):
    rows = {}
    for label, tiers in tier_configs.items():
        X, y, features, age = load_master(
            outcome, tiers=tiers, longitudinal=longitudinal,
            age_residualize=False, force_exclude_siblings=True,
        )
        cv = XGBoostPipeline(n_folds=N_FOLDS, use_gpu=USE_GPU).run_cv(X, y, features, age=age)
        rows[label] = (float(cv["r2_mean"]), int(y.notna().sum()))
        print(f"  [held-out] {outcome} {label}: R2={rows[label][0]:.4f} n={rows[label][1]}")
    return rows


def layer_row(layer, outcome, held, canon, base, full):
    return {
        "layer": layer, "outcome": outcome,
        "n_heldout": held[full][1], "n_canonical": canon[full][1],
        "r2_base_heldout": held[base][0], "r2_full_heldout": held[full][0],
        "delta_r2_T2_heldout": held[full][0] - held[base][0],
        "r2_base_canonical": canon[base][0], "r2_full_canonical": canon[full][0],
        "delta_r2_T2_canonical": canon[full][0] - canon[base][0],
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []

    print("\n== Cross-sectional (siblings held out) ==")
    held = tiered_r2(CS_OUTCOME, {"T1": [1], "T1T2": [1, 2]}, longitudinal=False)
    canon = canonical_r2(ap.CS1_DIR / "cs1_r2_table.csv", CS_OUTCOME)
    rows.append(layer_row("cross_sectional", CS_OUTCOME, held, canon, "T1", "T1T2"))

    print("\n== Baseline-conditioned change (siblings held out) ==")
    held = tiered_r2(LONG_OUTCOME, {"Yw1": [], "Yw1T1": [1], "Yw1T1T2": [1, 2]},
                     longitudinal=True)
    canon = canonical_r2(ap.A4_DIR / "a4_r2_table.csv", LONG_OUTCOME)
    rows.append(layer_row("longitudinal", LONG_OUTCOME, held, canon, "Yw1T1", "Yw1T1T2"))

    print("\n== Top-20 (siblings held out, bootstrap SHAP) ==")
    X, y, features, _ = load_master(CS_OUTCOME, tiers=[1, 2], age_residualize=True,
                                    force_exclude_siblings=True)
    pipe = XGBoostPipeline(n_folds=N_FOLDS, n_bootstrap=N_BOOT, use_gpu=USE_GPU)
    shap_df = pipe.run_bootstrap_shap(X, y, features).sort_values("mean_abs_shap", ascending=False)
    shap_df.to_csv(OUT / f"{CS_OUTCOME}_heldout_bootstrap_shap.csv")

    canon_path = ap.CS1_DIR / "shap" / f"{CS_OUTCOME}_T1T2_bootstrap_shap.csv"
    canon_df = pd.read_csv(canon_path, index_col=0).sort_values("mean_abs_shap", ascending=False)
    held20, canon20 = list(shap_df.index[:20]), list(canon_df.index[:20])
    inter = set(held20) & set(canon20)
    jacc = len(inter) / len(set(held20) | set(canon20))

    feats = sorted(set(held20) | set(canon20),
                   key=lambda f: (held20.index(f) if f in held20 else 99,
                                  canon20.index(f) if f in canon20 else 99))
    pd.DataFrame({
        "feature": feats,
        "rank_heldout": [held20.index(f) + 1 if f in held20 else pd.NA for f in feats],
        "rank_canonical": [canon20.index(f) + 1 if f in canon20 else pd.NA for f in feats],
    }).to_csv(OUT / "al_twins_heldout_top20.csv", index=False)

    # The top-20 is cross-sectional only; the longitudinal row leaves these blank.
    rows[0].update({"top20_overlap": len(inter), "top20_jaccard": jacc,
                    "top_feature_heldout": held20[0], "top_feature_canonical": canon20[0]})
    summary = pd.DataFrame(rows)
    summary["top20_overlap"] = summary["top20_overlap"].astype("Int64")
    summary.to_csv(OUT / "al_twins_heldout_summary.csv", index=False)
    print(summary.T.to_string())


if __name__ == "__main__":
    main()
