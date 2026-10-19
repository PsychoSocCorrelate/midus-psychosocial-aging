#!/usr/bin/env python3
"""
A9: Concurrent Cross-Sectional Analysis

Cross-sectional model at W2 using concurrent predictor values:
  CS-1: Y_w1 ~ f(X_w1)       [W1 cross-sectional]
  A4:   Y_w2 ~ f(X_w1, Y_w1) [W1 predicting W2 change]
  A9:   Y_w2 ~ f(X_w2)       [W2 cross-sectional]

Comparison of CS-1 vs A4 vs A9 reveals which predictors are robust
across specification (cross-sectional, longitudinal, concurrent).

Usage:
    uv run python analysis/pipelines/a9_concurrent.py
    uv run python analysis/pipelines/a9_concurrent.py --outcomes FInoPsy_W2
    uv run python analysis/pipelines/a9_concurrent.py --no_bootstrap   # CV R² only
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.core.output_log import write_output_log
from analysis.core import age_method
from analysis.core import provenance
from analysis.core.data_loader import load_master, deduplicate_longitudinal_outcomes
from analysis.core.xgboost_pipeline import XGBoostPipeline
from analysis.core.shap_dependence import plot_dependence_panel
from analysis.core.pub_style import setup_rcparams
from analysis.core import artifact_paths as ap

OUTPUT_DIR = ROOT / "output/a9_concurrent"
CS1_DIR = ROOT / "output/cs1_cross_sectional"
A4_DIR = ROOT / "output/a4_longitudinal"

PRIMARY_LONG = ["FInoPsy_W2", "btactComp_W2", "ALcomp_W2"]


def parse_args():
    p = argparse.ArgumentParser(description="A9 Concurrent model analysis")
    p.add_argument("--outcomes", nargs="+", default=["all"])
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--n_bootstrap", type=int, default=250)
    p.add_argument("--no_bootstrap", action="store_true")
    p.add_argument("--no_gpu", action="store_true")
    return p.parse_args()


def resolve_outcomes(names: list[str]) -> list[str]:
    """Resolve outcome names to W2 form (A9 is concurrent: Y_w2 ~ X_w2 only)."""
    if "all" in names:
        return PRIMARY_LONG
    resolved = deduplicate_longitudinal_outcomes(names)  # maps W1 names to W2
    # Reject any outcome still named W1.
    for o in resolved:
        if o.endswith("_W1"):
            raise ValueError(
                f"A9 concurrent mode requires W2 outcomes, got '{o}'. "
                f"Pass the W2 name (e.g. FInoPsy_W2) instead."
            )
    return resolved


# Phase 1: Tiered ΔR² (concurrent)

def run_tiered_r2_concurrent(
    outcome: str, n_folds: int, use_gpu: bool,
) -> list[dict]:
    """Tiered ΔR² with concurrent (W2) predictors."""
    from analysis.core.data_loader import get_tier_variables

    TIER_CONFIGS = {"T1": [1], "T1T2": [1, 2]}
    rows = []
    for tier_label, tiers in TIER_CONFIGS.items():
        print(f"\n  [R²] {outcome} | {tier_label} | concurrent")
        X, y, features, age = load_master(
            outcome, tiers=tiers, concurrent=True, age_residualize=False,
        )
        pipe = XGBoostPipeline(n_folds=n_folds, use_gpu=use_gpu)
        cv = pipe.run_cv(X, y, features, age=age)
        rows.append({
            "outcome": outcome,
            "mode": "concurrent",
            "tier_config": tier_label,
            "r2_mean": cv["r2_mean"],
            "r2_std": cv["r2_std"],
            "n": int(y.notna().sum()),
            "n_features": len(features),
        })
    return rows


# Phase 2: Bootstrap SHAP (concurrent)

def run_bootstrap_shap_concurrent(
    outcome: str, n_bootstrap: int, use_gpu: bool,
    output_dir: Path = OUTPUT_DIR,
) -> pd.DataFrame:
    """Bootstrap SHAP with concurrent (W2) predictors.

    Saves outputs to output_dir/{shap,shap_raw,figures}/ matching CS-1 layout.
    """
    print(f"\nBOOTSTRAP SHAP (concurrent): {outcome} (T1T2, {n_bootstrap}×)")

    X, y, features, age = load_master(
        outcome, tiers=[1, 2], concurrent=True, age_residualize=True,
    )
    pipe = XGBoostPipeline(n_bootstrap=n_bootstrap, use_gpu=use_gpu)
    shap_df, shap_raw = pipe.run_bootstrap_shap_parallel(
        X, y, features, n_gpus=2, return_raw=True, save_individual=True,
    )

    safe = outcome.replace("/", "_")
    shap_dir = output_dir / "shap"
    raw_dir = output_dir / "shap_raw"
    fig_dir = output_dir / "figures"
    for d in (shap_dir, raw_dir, fig_dir):
        d.mkdir(parents=True, exist_ok=True)

    shap_df.to_csv(shap_dir / f"{safe}_concurrent_bootstrap_shap.csv")
    print(f"  Saved: shap/{safe}_concurrent_bootstrap_shap.csv")

    np.save(raw_dir / f"{safe}_concurrent_bootstrap_shap_raw.npy", shap_raw)
    pd.Series(features).to_csv(
        shap_dir / f"{safe}_concurrent_features.csv", index=False, header=False,
    )
    print(f"  Saved: shap_raw/{safe}_concurrent_bootstrap_shap_raw.npy ({shap_raw.shape})")

    robust_shap = pipe.robust_shap_
    pd.DataFrame(robust_shap, columns=features).to_csv(
        raw_dir / f"{safe}_concurrent_robust_shap.csv", index=False
    )
    np.save(raw_dir / f"{safe}_concurrent_oob_count.npy", pipe.oob_count_)
    print(f"  Saved: shap_raw/{safe}_concurrent_robust_shap.csv ({robust_shap.shape})")

    # Dependence panel (stable features only)
    setup_rcparams()
    import matplotlib.pyplot as plt
    fig = plot_dependence_panel(
        X, robust_shap, features, ncols=4,
        outcome_label=outcome,
        bootstrap_shap_df=shap_df,
    )
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"{safe}_concurrent_dependence.{ext}",
                    format=ext, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: figures/{safe}_concurrent_dependence.pdf/.png")

    return shap_df


# Phase 3: CS-1 vs A9 comparison (primary)

def _load_shap(path: Path) -> pd.DataFrame | None:
    """Load a bootstrap SHAP CSV, return None if missing."""
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0)


def _signal_features(df: pd.DataFrame, quantile: float = 0.75) -> set[str]:
    """Features with mean_abs_shap above the given quantile (default: top 25%).

    Used as an outcome-adaptive 'with signal' filter, avoiding arbitrary
    absolute SHAP thresholds that differ across outcomes.
    """
    threshold = df["mean_abs_shap"].quantile(quantile)
    return set(df[df["mean_abs_shap"] > threshold].index)


def _shap_rank_correlation(
    df_a: pd.DataFrame, df_b: pd.DataFrame,
    signal_features: set[str] | None = None,
) -> tuple[list[str], float, float, int]:
    """Spearman ρ on shared features between two SHAP DataFrames.

    Args:
        signal_features: If provided, restrict to this subset of features
            (e.g. top-quartile features from the reference model).
    """
    shared = sorted(set(df_a.index) & set(df_b.index))
    if signal_features is not None:
        shared = [f for f in shared if f in signal_features]
    if len(shared) < 5:
        return shared, np.nan, np.nan, 0
    vals_a = df_a.loc[shared, "mean_abs_shap"].values
    vals_b = df_b.loc[shared, "mean_abs_shap"].values
    rho, p = spearmanr(vals_a, vals_b)
    return shared, rho, p, len(shared)


def _top_k_overlap(df_a: pd.DataFrame, df_b: pd.DataFrame, k: int = 20) -> int:
    """Count overlap between top-k features of two SHAP DataFrames."""
    top_a = set(df_a.head(k).index)
    top_b = set(df_b.head(k).index)
    return len(top_a & top_b)


def _ci_overlap(
    df_a: pd.DataFrame, df_b: pd.DataFrame,
) -> pd.DataFrame:
    """Check bootstrap CI overlap per shared feature.

    Returns DataFrame with columns: variable, ci_overlap (bool),
    cs1_lower, cs1_upper, a9_lower, a9_upper.
    """
    shared = sorted(set(df_a.index) & set(df_b.index))
    rows = []
    for var in shared:
        a_lo, a_hi = df_a.loc[var, "ci_lower"], df_a.loc[var, "ci_upper"]
        b_lo, b_hi = df_b.loc[var, "ci_lower"], df_b.loc[var, "ci_upper"]
        overlap = a_lo <= b_hi and b_lo <= a_hi
        rows.append({
            "variable": var,
            "ci_overlap": overlap,
            "cs1_lower": a_lo, "cs1_upper": a_hi,
            "a9_lower": b_lo, "a9_upper": b_hi,
        })
    return pd.DataFrame(rows)


def _bidirectional_shift(
    df_cs1: pd.DataFrame, df_a9: pd.DataFrame, top_k: int = 15,
) -> list[dict]:
    """Bidirectional rank shift: CS-1 top-k + A9 risers not in CS-1 top-k."""
    cs1_ranked = df_cs1.reset_index()
    cs1_ranked.columns = ["variable"] + list(cs1_ranked.columns[1:])
    cs1_ranked["rank_cs1"] = range(1, len(cs1_ranked) + 1)

    a9_ranked = df_a9.reset_index()
    a9_ranked.columns = ["variable"] + list(a9_ranked.columns[1:])
    a9_ranked["rank_a9"] = range(1, len(a9_ranked) + 1)

    cs1_top = set(cs1_ranked.head(top_k)["variable"])
    rows = []

    # CS-1 top-k and their A9 ranks
    for _, row in cs1_ranked.head(top_k).iterrows():
        var = row["variable"]
        r_cs1 = row["rank_cs1"]
        shap_cs1 = row["mean_abs_shap"]
        a9_match = a9_ranked[a9_ranked["variable"] == var]
        if len(a9_match):
            r_a9 = int(a9_match.iloc[0]["rank_a9"])
            shap_a9 = a9_match.iloc[0]["mean_abs_shap"]
        else:
            r_a9 = np.nan
            shap_a9 = np.nan
        rows.append({
            "variable": var,
            "rank_cs1": r_cs1,
            "rank_a9": r_a9,
            "rank_shift": r_cs1 - r_a9 if not np.isnan(r_a9) else np.nan,
            "shap_cs1": shap_cs1,
            "shap_a9": shap_a9,
            "shap_ratio": shap_a9 / shap_cs1 if shap_cs1 > 0 and not np.isnan(shap_a9) else np.nan,
            "source": "cs1_top15",
        })

    # A9 top-k risers not in CS-1 top-k
    for _, row in a9_ranked.head(top_k).iterrows():
        var = row["variable"]
        if var in cs1_top:
            continue  # already included above
        r_a9 = row["rank_a9"]
        shap_a9 = row["mean_abs_shap"]
        cs1_match = cs1_ranked[cs1_ranked["variable"] == var]
        if len(cs1_match):
            r_cs1 = int(cs1_match.iloc[0]["rank_cs1"])
            shap_cs1 = cs1_match.iloc[0]["mean_abs_shap"]
        else:
            r_cs1 = np.nan
            shap_cs1 = np.nan
        rows.append({
            "variable": var,
            "rank_cs1": r_cs1,
            "rank_a9": int(r_a9),
            "rank_shift": r_cs1 - r_a9 if not np.isnan(r_cs1) else np.nan,
            "shap_cs1": shap_cs1,
            "shap_a9": shap_a9,
            "shap_ratio": shap_a9 / shap_cs1 if not np.isnan(shap_cs1) and shap_cs1 > 0 else np.nan,
            "source": "a9_riser",
        })

    return rows


def compare_cs1_vs_a9(outcomes: list[str]) -> dict:
    """Compare CS-1 (W1 cross-sectional) vs A9 (W2 concurrent) SHAP rankings.

    Returns dict with keys: r2_rows, rho_rows, shift_rows, ci_rows
    for downstream summary generation.
    """
    print("\nCS-1 vs A9 SHAP COMPARISON (primary)")

    r2_rows: list[dict] = []
    rho_rows: list[dict] = []
    shift_rows: list[dict] = []
    ci_summary_rows: list[dict] = []

    # Load R² tables
    cs1_r2_path = CS1_DIR / "cs1_r2_table.csv"
    a9_r2_path = OUTPUT_DIR / "a9_r2_table.csv"
    cs1_r2 = pd.read_csv(cs1_r2_path) if cs1_r2_path.exists() else None
    a9_r2 = pd.read_csv(a9_r2_path) if a9_r2_path.exists() else None

    for outcome in outcomes:
        safe = outcome.replace("/", "_")
        outcome_stem = outcome.replace("_W2", "").replace("_W1", "")

        # Load SHAP
        cs1_shap_path = ap.CS1_DIR / "shap" / f"{outcome_stem}_W1_T1T2_bootstrap_shap.csv"
        a9_shap_path = OUTPUT_DIR / "shap" / f"{safe}_concurrent_bootstrap_shap.csv"
        cs1_df = _load_shap(cs1_shap_path)
        a9_df = _load_shap(a9_shap_path)

        if cs1_df is None:
            print(f"  [SKIP] CS-1 SHAP not found: {cs1_shap_path.name}")
            continue
        if a9_df is None:
            print(f"  [SKIP] A9 SHAP not found: {a9_shap_path.name}")
            continue

        # ΔR²_T2 comparison
        if cs1_r2 is not None and a9_r2 is not None:
            for tier in ["T1", "T1T2"]:
                cs1_row = cs1_r2[(cs1_r2["outcome"] == f"{outcome_stem}_W1") &
                                 (cs1_r2["tier_config"] == tier)]
                a9_row = a9_r2[(a9_r2["outcome"] == outcome) &
                               (a9_r2["tier_config"] == tier)]
                if len(cs1_row) and len(a9_row):
                    cs1_val = cs1_row.iloc[0]["r2_mean"]
                    a9_val = a9_row.iloc[0]["r2_mean"]
                    r2_rows.append({
                        "outcome": outcome,
                        "tier": tier,
                        "CS1_R2": round(cs1_val, 4),
                        "A9_R2": round(a9_val, 4),
                        "delta": round(a9_val - cs1_val, 4),
                        "ratio": round(a9_val / cs1_val, 3) if cs1_val > 0 else np.nan,
                        "CS1_n": int(cs1_row.iloc[0]["n"]),
                        "A9_n": int(a9_row.iloc[0]["n"]),
                    })

        # SHAP rank correlation: full + signal
        sig_feats = _signal_features(cs1_df, quantile=0.75)   # top 25% by CS-1 importance
        shared_full, rho_full, p_full, n_full = _shap_rank_correlation(cs1_df, a9_df)
        _, rho_sig, p_sig, n_sig = _shap_rank_correlation(cs1_df, a9_df, signal_features=sig_feats)
        top20_overlap = _top_k_overlap(cs1_df, a9_df, k=20)

        rho_rows.append({
            "outcome": outcome,
            "n_full": n_full,
            "rho_full": round(rho_full, 3) if not np.isnan(rho_full) else np.nan,
            "p_full": p_full,
            "n_signal": n_sig,
            "rho_signal": round(rho_sig, 3) if not np.isnan(rho_sig) else np.nan,
            "p_signal": p_sig,
            "top20_overlap": f"{top20_overlap}/20",
        })

        # Bootstrap CI overlap
        ci_df = _ci_overlap(cs1_df, a9_df)
        n_overlap = ci_df["ci_overlap"].sum()
        n_disjoint = (~ci_df["ci_overlap"]).sum()
        ci_summary_rows.append({
            "outcome": outcome,
            "n_features": len(ci_df),
            "n_ci_overlap": int(n_overlap),
            "n_ci_disjoint": int(n_disjoint),
            "pct_overlap": round(100 * n_overlap / len(ci_df), 1) if len(ci_df) else 0,
        })
        # Save per-variable CI detail
        ci_df.insert(0, "outcome", outcome)
        ci_out = OUTPUT_DIR
        ci_out.mkdir(parents=True, exist_ok=True)
        ci_df.to_csv(ci_out / f"a9_cs1_ci_overlap_{safe}.csv", index=False)

        # Bidirectional rank shift
        shifts = _bidirectional_shift(cs1_df, a9_df, top_k=15)
        for s in shifts:
            s["outcome"] = outcome
        shift_rows.extend(shifts)

        print(f"  {outcome}: ρ_full={rho_full:.3f} ρ_sig={rho_sig:.3f}(n={n_sig}), "
              f"top-20 overlap={top20_overlap}/20, CI overlap={n_overlap}/{len(ci_df)}")

    # Save CSVs
    if r2_rows:
        pd.DataFrame(r2_rows).to_csv(OUTPUT_DIR / "a9_cs1_r2_comparison.csv", index=False)
        print(f"\n  Saved: a9_cs1_r2_comparison.csv")
    if rho_rows:
        pd.DataFrame(rho_rows).to_csv(OUTPUT_DIR / "a9_cs1_shap_rank_correlation.csv", index=False)
        print(f"  Saved: a9_cs1_shap_rank_correlation.csv")
    if shift_rows:
        pd.DataFrame(shift_rows).to_csv(OUTPUT_DIR / "a9_cs1_variable_shift.csv", index=False)
        print(f"  Saved: a9_cs1_variable_shift.csv")
    if ci_summary_rows:
        pd.DataFrame(ci_summary_rows).to_csv(OUTPUT_DIR / "a9_cs1_ci_summary.csv", index=False)
        print(f"  Saved: a9_cs1_ci_summary.csv")

    return {
        "r2_rows": r2_rows,
        "rho_rows": rho_rows,
        "shift_rows": shift_rows,
        "ci_summary_rows": ci_summary_rows,
    }


# Phase 4: Three-way ρ triangle

def compute_three_way_rho(outcomes: list[str]) -> list[dict]:
    """Compute ρ(CS-1,A9), ρ(CS-1,A4), ρ(A4,A9) for each outcome."""
    print("\nTHREE-WAY ρ TRIANGLE")

    rows = []
    for outcome in outcomes:
        safe = outcome.replace("/", "_")
        outcome_stem = outcome.replace("_W2", "").replace("_W1", "")

        # Load all three SHAP DataFrames
        cs1_df = _load_shap(ap.CS1_DIR / "shap" / f"{outcome_stem}_W1_T1T2_bootstrap_shap.csv")
        a9_df = _load_shap(OUTPUT_DIR / "shap" / f"{safe}_concurrent_bootstrap_shap.csv")

        # A4: try adjusted first, then standard
        a4_df = None
        for suffix in ["_long_adjusted_bootstrap_shap.csv", "_long_bootstrap_shap.csv"]:
            a4_path = ap.A4_DIR / "shap" / f"{safe}{suffix}"
            if a4_path.exists():
                a4_df = _load_shap(a4_path)
                break

        if any(df is None for df in [cs1_df, a9_df, a4_df]):
            print(f"  [SKIP] Missing SHAP for {outcome}")
            continue

        # Exclude autoregressive term from A4
        a4_clean = a4_df.drop(index=f"{outcome_stem}_W1", errors="ignore")

        # CS-1 top-quartile features as anchor for signal-ρ (same set for all three pairs)
        cs1_sig = _signal_features(cs1_df, quantile=0.75)
        n_sig = len([f for f in cs1_sig if f in a9_df.index and f in a4_clean.index])

        # Full ρ
        _, rho_cs1_a9, p_cs1_a9, _ = _shap_rank_correlation(cs1_df, a9_df)
        _, rho_cs1_a4, p_cs1_a4, _ = _shap_rank_correlation(cs1_df, a4_clean)
        _, rho_a4_a9, p_a4_a9, _ = _shap_rank_correlation(a4_clean, a9_df)

        # Signal ρ (CS-1 top-quartile anchor)
        _, rho_cs1_a9_s, _, _ = _shap_rank_correlation(cs1_df, a9_df, signal_features=cs1_sig)
        _, rho_cs1_a4_s, _, _ = _shap_rank_correlation(cs1_df, a4_clean, signal_features=cs1_sig)
        _, rho_a4_a9_s, _, _ = _shap_rank_correlation(a4_clean, a9_df, signal_features=cs1_sig)

        row = {
            "outcome": outcome,
            "n_signal": n_sig,
            # full ρ (all shared features)
            "rho_CS1_A9": round(rho_cs1_a9, 3),
            "rho_CS1_A4": round(rho_cs1_a4, 3),
            "rho_A4_A9": round(rho_a4_a9, 3),
            # signal ρ (CS-1 top-quartile anchor, outcome-adaptive)
            "rho_sig_CS1_A9": round(rho_cs1_a9_s, 3) if not np.isnan(rho_cs1_a9_s) else np.nan,
            "rho_sig_CS1_A4": round(rho_cs1_a4_s, 3) if not np.isnan(rho_cs1_a4_s) else np.nan,
            "rho_sig_A4_A9": round(rho_a4_a9_s, 3) if not np.isnan(rho_a4_a9_s) else np.nan,
            # p-values for full ρ
            "p_CS1_A9": p_cs1_a9,
            "p_CS1_A4": p_cs1_a4,
            "p_A4_A9": p_a4_a9,
        }
        rows.append(row)
        print(f"  {outcome} full:   ρ(CS1,A9)={rho_cs1_a9:.3f}  ρ(CS1,A4)={rho_cs1_a4:.3f}  ρ(A4,A9)={rho_a4_a9:.3f}")
        print(f"  {outcome} signal(n={n_sig}): ρ(CS1,A9)={rho_cs1_a9_s:.3f}  ρ(CS1,A4)={rho_cs1_a4_s:.3f}  ρ(A4,A9)={rho_a4_a9_s:.3f}")

    if rows:
        pd.DataFrame(rows).to_csv(OUTPUT_DIR / "a9_three_way_rho.csv", index=False)
        print(f"\n  Saved: a9_three_way_rho.csv")
    return rows


# Phase 5: A4 vs A9 comparison (secondary)

def compare_a4_vs_a9(outcomes: list[str]) -> dict:
    """Compare baseline (A4) vs concurrent (A9) SHAP rankings (secondary).

    Returns dict with keys: rho_rows, shift_rows for summary generation.
    """
    print("\nA4 vs A9 SHAP COMPARISON (secondary)")

    r2_rows = []
    rho_rows = []
    shift_rows = []

    for outcome in outcomes:
        safe = outcome.replace("/", "_")
        outcome_stem = outcome.replace("_W2", "").replace("_W1", "")

        # Load A4 SHAP
        for suffix in ["_long_adjusted_bootstrap_shap.csv", "_long_bootstrap_shap.csv"]:
            a4_path = ap.A4_DIR / "shap" / f"{safe}{suffix}"
            if a4_path.exists():
                break
        if not a4_path.exists():
            print(f"  [SKIP] A4 SHAP not found for {outcome}")
            continue
        a4_df = pd.read_csv(a4_path, index_col=0)

        # Load A9 SHAP
        a9_path = OUTPUT_DIR / "shap" / f"{safe}_concurrent_bootstrap_shap.csv"
        if not a9_path.exists():
            print(f"  [SKIP] A9 SHAP not found for {outcome}")
            continue
        a9_df = pd.read_csv(a9_path, index_col=0)

        # Load R²
        a4_r2_path = A4_DIR / "a4_r2_table.csv"
        a9_r2_path = OUTPUT_DIR / "a9_r2_table.csv"
        if a4_r2_path.exists() and a9_r2_path.exists():
            a4_r2 = pd.read_csv(a4_r2_path)
            a9_r2 = pd.read_csv(a9_r2_path)
            # A4 uses "Yw1T1T2" tier naming (includes autoregressive term);
            # A9 uses "T1T2" (no autoregressive term)
            a4_t1t2 = a4_r2[(a4_r2["outcome"] == outcome) &
                            (a4_r2["tier_config"] == "Yw1T1T2")]
            a9_t1t2 = a9_r2[(a9_r2["outcome"] == outcome) &
                            (a9_r2["tier_config"] == "T1T2")]
            if len(a4_t1t2) and len(a9_t1t2):
                a4_val = a4_t1t2.iloc[0]["r2_mean"]
                a9_val = a9_t1t2.iloc[0]["r2_mean"]
                # Also get A4 pooled from a4_r2 (cohort == "pooled")
                a4_pooled = a4_r2[(a4_r2["outcome"] == outcome) &
                                  (a4_r2["tier_config"] == "Yw1T1T2") &
                                  (a4_r2["cohort"] == "pooled")]
                a4_r2_val = a4_pooled.iloc[0]["r2_mean"] if len(a4_pooled) else a4_val
                r2_rows.append({
                    "outcome": outcome,
                    "A4_R2": round(a4_r2_val, 3),
                    "A9_R2": round(a9_val, 3),
                    "delta_R2": round(a9_val - a4_r2_val, 3),
                })

        # SHAP rank correlation
        # Exclude autoregressive term from A4 (A9 has none)
        a4_clean = a4_df.drop(index=f"{outcome_stem}_W1", errors="ignore")
        a9_clean = a9_df

        shared = sorted(set(a4_clean.index) & set(a9_clean.index))
        if len(shared) < 10:
            print(f"  [SKIP] Only {len(shared)} shared features for {outcome}")
            continue

        a4_vals = a4_clean.loc[shared, "mean_abs_shap"].values
        a9_vals = a9_clean.loc[shared, "mean_abs_shap"].values
        rho_all, p_all = spearmanr(a4_vals, a9_vals)

        # Top-20 overlap
        a4_top20 = set(a4_clean.head(20).index)
        a9_top20 = set(a9_clean.head(20).index)
        top20_overlap = len(a4_top20 & a9_top20)

        rho_rows.append({
            "outcome": outcome,
            "n_shared": len(shared),
            "spearman_rho": round(rho_all, 3),
            "p_value": p_all,
            "top20_overlap": f"{top20_overlap}/20",
        })

        # Per-variable SHAP shift (top-15 from A4)
        a4_ranked = a4_clean.reset_index()
        a4_ranked.columns = ["variable"] + list(a4_ranked.columns[1:])
        a4_ranked["rank_a4"] = range(1, len(a4_ranked) + 1)

        a9_ranked = a9_clean.reset_index()
        a9_ranked.columns = ["variable"] + list(a9_ranked.columns[1:])
        a9_ranked["rank_a9"] = range(1, len(a9_ranked) + 1)

        for i, row in a4_ranked.head(15).iterrows():
            var = row["variable"]
            rank_a4 = row["rank_a4"]
            shap_a4 = row["mean_abs_shap"]

            a9_match = a9_ranked[a9_ranked["variable"] == var]
            if len(a9_match):
                rank_a9 = a9_match.iloc[0]["rank_a9"]
                shap_a9 = a9_match.iloc[0]["mean_abs_shap"]
            else:
                rank_a9 = np.nan
                shap_a9 = np.nan

            shift_rows.append({
                "outcome": outcome,
                "variable": var,
                "rank_a4": rank_a4,
                "rank_a9": rank_a9,
                "rank_shift": rank_a4 - rank_a9 if not np.isnan(rank_a9) else np.nan,
                "shap_a4": shap_a4,
                "shap_a9": shap_a9,
                "shap_ratio": shap_a9 / shap_a4 if shap_a4 > 0 and not np.isnan(shap_a9) else np.nan,
            })

        print(f"  {outcome}: ρ={rho_all:.3f}, top-20 overlap={top20_overlap}/20")

    # Save comparison CSVs
    if r2_rows:
        pd.DataFrame(r2_rows).to_csv(OUTPUT_DIR / "a9_r2_comparison.csv", index=False)
        print(f"\n  Saved: a9_r2_comparison.csv")
    if rho_rows:
        pd.DataFrame(rho_rows).to_csv(OUTPUT_DIR / "a9_shap_rank_correlation.csv", index=False)
        print(f"  Saved: a9_shap_rank_correlation.csv")
    if shift_rows:
        pd.DataFrame(shift_rows).to_csv(OUTPUT_DIR / "a9_variable_shift.csv", index=False)
        print(f"  Saved: a9_variable_shift.csv")

    return {"rho_rows": rho_rows, "shift_rows": shift_rows}


# Output log generation

def write_pipeline_log(
    args: argparse.Namespace,
    outcomes: list[str],
    cs1_results: dict,
    three_way_rows: list[dict],
    a4_results: dict,
) -> None:
    outputs = [
        {"path": "a9_r2_table.csv", "kind": "CSV"},
        {"path": "a9_cs1_r2_comparison.csv", "kind": "CSV", "status": "written" if cs1_results.get("r2_rows") else "not written"},
        {"path": "a9_cs1_shap_rank_correlation.csv", "kind": "CSV", "status": "written" if cs1_results.get("rho_rows") else "not written"},
        {"path": "a9_cs1_variable_shift.csv", "kind": "CSV", "status": "written" if cs1_results.get("shift_rows") else "not written"},
        {"path": "a9_cs1_ci_summary.csv", "kind": "CSV", "status": "written" if cs1_results.get("ci_summary_rows") else "not written"},
        {"path": "a9_three_way_rho.csv", "kind": "CSV", "status": "written" if three_way_rows else "not written"},
        {"path": "a9_r2_comparison.csv", "kind": "CSV", "status": "written" if a4_results.get("r2_rows") else "not written"},
        {"path": "a9_shap_rank_correlation.csv", "kind": "CSV", "status": "written" if a4_results.get("rho_rows") else "not written"},
        {"path": "a9_variable_shift.csv", "kind": "CSV", "status": "written" if a4_results.get("shift_rows") else "not written"},
    ]
    for outcome in outcomes:
        safe = outcome.replace("/", "_")
        shap_status = "written" if not args.no_bootstrap else "skipped"
        outputs.extend([
            {"path": OUTPUT_DIR / "shap" / f"{safe}_concurrent_bootstrap_shap.csv", "kind": "CSV", "status": shap_status},
            {"path": OUTPUT_DIR / "shap" / f"{safe}_concurrent_features.csv", "kind": "CSV", "status": shap_status},
            {"path": OUTPUT_DIR / "shap_raw" / f"{safe}_concurrent_bootstrap_shap_raw.npy", "kind": "NPY", "status": shap_status},
            {"path": OUTPUT_DIR / "shap_raw" / f"{safe}_concurrent_robust_shap.csv", "kind": "CSV", "status": shap_status},
            {"path": OUTPUT_DIR / "shap_raw" / f"{safe}_concurrent_oob_count.npy", "kind": "NPY", "status": shap_status},
            {"path": OUTPUT_DIR / "figures" / f"{safe}_concurrent_dependence.pdf", "kind": "PDF", "status": shap_status},
            {"path": OUTPUT_DIR / "figures" / f"{safe}_concurrent_dependence.png", "kind": "PNG", "status": shap_status},
            {"path": OUTPUT_DIR / f"a9_cs1_ci_overlap_{safe}.csv", "kind": "CSV"},
        ])

    run_args = vars(args).copy()
    run_args["resolved_outcomes"] = outcomes
    write_output_log(OUTPUT_DIR, "a9_output_log.md", Path(__file__).name, outputs, run_args)


# Main

def main():
    # Record input provenance and the age-adjustment method beside the outputs.
    provenance.stamp_at_exit(OUTPUT_DIR, "a9_concurrent", ["data/processed/master_analysis.csv", "data/processed/master_analysis_concurrent.csv", "output/a4_longitudinal", "output/cs1_cross_sectional"])
    age_method.emit_at_exit(OUTPUT_DIR, "a9_concurrent")
    args = parse_args()
    outcomes = resolve_outcomes(args.outcomes)
    use_gpu = not args.no_gpu

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"A9 CONCURRENT MODEL ANALYSIS: {datetime.now().isoformat()}")
    print(f"Outcomes: {outcomes}")

    # Phase 1: Tiered ΔR²
    r2_rows = []
    for outcome in outcomes:
        r2_rows.extend(run_tiered_r2_concurrent(outcome, args.n_folds, use_gpu))
    r2_df = pd.DataFrame(r2_rows)
    r2_df.to_csv(OUTPUT_DIR / "a9_r2_table.csv", index=False)
    print(f"\n  Saved: a9_r2_table.csv ({len(r2_df)} rows)")

    print("\n  Concurrent Tiered ΔR²:")
    for outcome in outcomes:
        sub = r2_df[r2_df["outcome"] == outcome]
        for _, row in sub.iterrows():
            print(f"    {outcome} | {row['tier_config']:8s} | R²={row['r2_mean']:.3f} (±{row['r2_std']:.3f})")

    # Phase 2: Bootstrap SHAP
    if not args.no_bootstrap:
        for outcome in outcomes:
            run_bootstrap_shap_concurrent(outcome, args.n_bootstrap, use_gpu)

    # Phase 3: CS-1 vs A9 comparison (primary)
    cs1_results = compare_cs1_vs_a9(outcomes)

    # Phase 4: Three-way ρ triangle
    three_way_rows = compute_three_way_rho(outcomes)

    # Phase 5: A4 vs A9 comparison (secondary)
    a4_results = compare_a4_vs_a9(outcomes)

    # Phase 6: Write output log
    write_pipeline_log(args, outcomes, cs1_results, three_way_rows, a4_results)

    print(f"\nDone. Results in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
