#!/usr/bin/env python3
"""
A4b: Longitudinal Prediction with Concurrent (W2) Predictors

Formula:   y_w2 ~ y_w1 + X_w2
  - Target: Y_w2, age-residualized with age_W2
  - Autoregressive feature: Y_w1, age-residualized with age_W1
  - Predictors X: T1/T2 values measured at Wave 2 (from master_analysis_concurrent.csv)

Contrast with sibling analyses:
  - A4  (a4_longitudinal.py):     y_w2 ~ y_w1 + X_w1: how does W1 state predict W2 change?
  - A4b (this file):              y_w2 ~ y_w1 + X_w2: does concurrent W2 state predict W2 change above autoregression?
  - A9  (a9_concurrent.py):       y_w2 ~ X_w2: pure W2 cross-sectional, no autoregression

Design inherited from A4:
  - Kept:   age-residualization (Y_w1 and Y_w2), tiered ΔR² pooled + MIDUS2/MIDUSR cohort split,
            16-marker allostatic-load composite (ALcomp), 5-fold CV, 200× bootstrap SHAP, FInoPsy overlap-adjusted model.
  - Skipped: SHAP dependence plots, sibling holdout validation.
  - Rank comparison: against A4 (X_w1 vs X_w2), not CS-1.

Bootstrap SHAP (200×, 2-GPU parallel).

Outputs:
  1. Tiered ΔR² (Yw1, Yw1+T1, Yw1+T1T2), pooled and per cohort
  2. Bootstrap SHAP rankings (mean |SHAP|, 95% CI, top-20 stability)
  3. A4 (X_w1) vs A4b (X_w2) SHAP ρ
  4. FInoPsy overlap-adjusted bootstrap SHAP

Usage:
    uv run python analysis/pipelines/a4b_longitudinal.py
    uv run python analysis/pipelines/a4b_longitudinal.py --outcomes FInoPsy_W2
    uv run python analysis/pipelines/a4b_longitudinal.py --no_bootstrap
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.core.output_log import write_output_log
from analysis.core import age_method
from analysis.core import provenance
from analysis.core.data_loader import AL_OUTCOME_COLS, load_master, get_tier_variables, deduplicate_longitudinal_outcomes
from analysis.core.overlap_sources import get_overlap_excluded_vars
from analysis.core.xgboost_pipeline import XGBoostPipeline

OUTPUT_DIR = ROOT / "output/a4b_longitudinal"
A4_DIR = ROOT / "output/a4_longitudinal"

PRIMARY_LONG = ["FInoPsy_W2", "btactComp_W2", "ALcomp_W2"]

TIER_CONFIGS = {
    "Yw1": [],
    "Yw1T1": [1],
    "Yw1T1T2": [1, 2],
}

# Allostatic load has no Refresher wave-2 source, so cohort splits are skipped
# for every AL construction.
COHORT_SPLIT_SKIP = set(AL_OUTCOME_COLS)

# Outcomes that get the frailty content-overlap sensitivity (R2 and SHAP).
OVERLAP_ADJUSTED_OUTCOMES = ["FInoPsy_W2"]

# Arms of the overlap-adjusted tiered R2 table: (label, tiers, drop overlap set).
OVERLAP_R2_ARMS = (
    ("Yw1T1", [1], False),
    ("Yw1T1T2", [1, 2], False),
    ("Yw1T1T2_excl_ab", [1, 2], True),
)


def parse_args():
    p = argparse.ArgumentParser(description="A4b Longitudinal prediction (W2 predictors)")
    p.add_argument("--outcomes", nargs="+", default=["all"],
                    help="Outcome names or 'all'")
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--n_bootstrap", type=int, default=200)
    p.add_argument("--no_bootstrap", action="store_true")
    p.add_argument("--no_gpu", action="store_true")
    p.add_argument("--overlap_r2_only", action="store_true",
                   help="Run only the overlap-adjusted tiered R2 phase and exit. "
                        "Writes a4b_overlap_adjusted_r2.csv and touches nothing else.")
    return p.parse_args()


def resolve_outcomes(names: list[str]) -> list[str]:
    if "all" in names:
        return PRIMARY_LONG
    return deduplicate_longitudinal_outcomes(names)


# Phase 1: Tiered ΔR²

# The concurrent CSV uses W2-era cohort labels (MIDUS3 / MR2) for the same
# people a4's master_analysis.csv labels as MIDUS2 / MIDUSR. Translate so the
# output table keeps a4-matching cohort names while filtering on the right tag.
COHORT_W1_TO_W2 = {"MIDUS2": "MIDUS3", "MIDUSR": "MR2"}


def run_tiered_r2_a4b(
    outcome: str, n_folds: int, use_gpu: bool,
    cohort: str | None = None,
) -> list[dict]:
    """Tiered ΔR² for one outcome in a4b mode (Y_w1 autoregressive + X_w2)."""
    rows = []
    cohort_label = cohort or "pooled"
    cohort_filter = COHORT_W1_TO_W2.get(cohort, cohort)

    for tier_label, tiers in TIER_CONFIGS.items():
        print(f"\n  [R²] {outcome} | {tier_label} | {cohort_label}")
        X, y, features, age = load_master(
            outcome, tiers=tiers, longitudinal=True, concurrent=True,
            age_residualize=False, cohort_filter=cohort_filter,
        )
        pipe = XGBoostPipeline(n_folds=n_folds, use_gpu=use_gpu)
        cv = pipe.run_cv(X, y, features, age=age)
        rows.append({
            "outcome": outcome,
            "cohort": cohort_label,
            "tier_config": tier_label,
            "r2_mean": cv["r2_mean"],
            "r2_std": cv["r2_std"],
            "n": int(y.notna().sum()),
            "n_features": len(features),
        })

    return rows


# Phase 1c: Overlap-adjusted tiered ΔR²

def run_overlap_adjusted_r2_a4b(
    outcome: str, n_folds: int, use_gpu: bool,
) -> list[dict]:
    """Tiered ΔR² with and without the frailty content-overlap predictors.

    The a4b counterpart of the a4 phase of the same name, and the arm that
    bears on common-occasion reporting variance: here the Tier 2 predictors and
    the outcome are both taken at Wave 2, so removing the 29 direct and
    proximal overlap items bounds how much of the concurrent increment those
    items carry. Both increments run on identical folds and an identical
    sample, and the excluded set is entirely Tier 2, so one ``Yw1T1`` baseline
    serves both.

    The adjusted feature set is the one the adjusted bootstrap SHAP model fits,
    so the two artifacts describe one model.
    """
    overlap_vars = get_overlap_excluded_vars("ab")
    rows = []

    for label, tiers, drop_overlap in OVERLAP_R2_ARMS:
        print(f"\n  [R² overlap-adjusted, X_w2] {outcome} | {label}")
        X, y, features, age = load_master(
            outcome, tiers=tiers, longitudinal=True, concurrent=True,
            age_residualize=False,
        )
        n_dropped = 0
        if drop_overlap:
            kept = [f for f in features if f not in overlap_vars]
            n_dropped = len(features) - len(kept)
            features, X = kept, X[kept]
        pipe = XGBoostPipeline(n_folds=n_folds, use_gpu=use_gpu)
        cv = pipe.run_cv(X, y, features, age=age)
        rows.append({
            "outcome": outcome,
            "cohort": "pooled",
            "tier_config": label,
            "r2_mean": cv["r2_mean"],
            "r2_std": cv["r2_std"],
            "n": int(y.notna().sum()),
            "n_features": len(features),
            "n_overlap_excluded": n_dropped,
        })

    return _add_overlap_r2_deltas(rows)


def _add_overlap_r2_deltas(rows: list[dict]) -> list[dict]:
    """Attach the Tier 2 increment over the shared Yw1T1 baseline."""
    baseline = {
        r["outcome"]: r["r2_mean"] for r in rows if r["tier_config"] == "Yw1T1"
    }
    for r in rows:
        base = baseline.get(r["outcome"])
        r["delta_r2_T2_vs_Yw1T1"] = (
            np.nan if base is None or r["tier_config"] == "Yw1T1"
            else r["r2_mean"] - base
        )
    return rows


def write_overlap_adjusted_r2(rows: list[dict], output_dir: Path) -> None:
    if not rows:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "a4b_overlap_adjusted_r2.csv", index=False)
    print(f"\n  Saved: a4b_overlap_adjusted_r2.csv ({len(rows)} rows)")


# Phase 2: Bootstrap SHAP

def run_bootstrap_shap_a4b(
    outcome: str, n_bootstrap: int, use_gpu: bool,
    output_dir: Path, run_adjusted: bool = False,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Bootstrap SHAP on T1T2(@W2)+Y_w1 for one a4b outcome."""
    print(f"\nBOOTSTRAP SHAP (a4b / X_w2): {outcome} (T1T2@W2+Yw1, {n_bootstrap}×)")

    X, y, features, age = load_master(
        outcome, tiers=[1, 2], longitudinal=True, concurrent=True,
        age_residualize=True,
    )
    pipe = XGBoostPipeline(n_bootstrap=n_bootstrap, use_gpu=use_gpu)
    shap_df, shap_raw = pipe.run_bootstrap_shap_parallel(
        X, y, features, n_gpus=2, return_raw=True, save_individual=True,
    )

    safe = outcome.replace("/", "_")
    shap_dir = output_dir / "shap"
    raw_dir = output_dir / "shap_raw"
    for d in (shap_dir, raw_dir):
        d.mkdir(parents=True, exist_ok=True)

    shap_df.to_csv(shap_dir / f"{safe}_a4b_bootstrap_shap.csv")
    np.save(raw_dir / f"{safe}_a4b_bootstrap_shap_raw.npy", shap_raw)
    pd.Series(features).to_csv(
        shap_dir / f"{safe}_a4b_features.csv", index=False, header=False,
    )
    print(f"  Saved: shap/{safe}_a4b_bootstrap_shap.csv ({shap_raw.shape})")

    robust_shap = pipe.robust_shap_
    pd.DataFrame(robust_shap, columns=features).to_csv(
        raw_dir / f"{safe}_a4b_robust_shap.csv", index=False
    )
    np.save(raw_dir / f"{safe}_a4b_oob_count.npy", pipe.oob_count_)
    print(f"  Saved: shap_raw/{safe}_a4b_robust_shap.csv ({robust_shap.shape})")

    # FInoPsy overlap-adjusted
    if run_adjusted:
        overlap_vars = get_overlap_excluded_vars("ab")
        adj_features = [f for f in features if f not in overlap_vars]
        X_adj = X[adj_features]

        print(f"\n  ADJUSTED MODEL: {len(adj_features)} features "
              f"({len(features) - len(adj_features)} excluded)")

        pipe_adj = XGBoostPipeline(n_bootstrap=n_bootstrap, use_gpu=use_gpu)
        shap_adj_df, shap_adj_raw = pipe_adj.run_bootstrap_shap_parallel(
            X_adj, y, adj_features, n_gpus=2, return_raw=True,
            save_individual=True,
        )
        shap_adj_df.to_csv(shap_dir / f"{safe}_a4b_adjusted_bootstrap_shap.csv")
        np.save(raw_dir / f"{safe}_a4b_adjusted_bootstrap_shap_raw.npy", shap_adj_raw)
        pd.Series(adj_features).to_csv(
            shap_dir / f"{safe}_a4b_adjusted_features.csv", index=False, header=False,
        )
        pd.DataFrame(pipe_adj.robust_shap_, columns=adj_features).to_csv(
            raw_dir / f"{safe}_a4b_adjusted_robust_shap.csv", index=False
        )
        print(f"  Saved: shap/{safe}_a4b_adjusted_bootstrap_shap.csv")

    return shap_df, shap_raw


# Phase 3: A4 (X_w1) vs A4b (X_w2) SHAP ρ

def compute_a4_vs_a4b_rho(outcomes: list[str], output_dir: Path) -> pd.DataFrame:
    """Spearman ρ between A4 and A4b bootstrap SHAP rankings.

    A4 uses X_w1, A4b uses X_w2; both include Y_w1 autoregressively. The ρ
    tests whether moving predictors from W1 to W2 reorders variable importance.
    """
    rows = []
    for outcome in outcomes:
        safe = outcome.replace("/", "_")

        a4_path = A4_DIR / "shap" / f"{safe}_long_bootstrap_shap.csv"
        if not a4_path.exists():
            print(f"  [SKIP] A4 SHAP not found for {outcome}: {a4_path.name}")
            continue
        a4_df = pd.read_csv(a4_path, index_col=0)

        a4b_path = output_dir / "shap" / f"{safe}_a4b_bootstrap_shap.csv"
        if not a4b_path.exists():
            print(f"  [SKIP] A4b SHAP not found for {outcome}")
            continue
        a4b_df = pd.read_csv(a4b_path, index_col=0)

        # Align on shared features (Y_w1 appears in both by design)
        shared = sorted(set(a4_df.index) & set(a4b_df.index))
        if len(shared) < 10:
            print(f"  [SKIP] Only {len(shared)} shared features for {outcome}")
            continue

        a4_vals = a4_df.loc[shared, "mean_abs_shap"].values
        a4b_vals = a4b_df.loc[shared, "mean_abs_shap"].values

        rho, pval = spearmanr(a4_vals, a4b_vals)

        a4_top20 = set(a4_df.head(20).index)
        a4b_top20 = set(a4b_df.head(20).index)
        overlap = len(a4_top20 & a4b_top20)

        rows.append({
            "outcome": outcome,
            "n_shared_features": len(shared),
            "spearman_rho": rho,
            "p_value": pval,
            "top20_overlap": f"{overlap}/{min(20, len(a4b_top20))}",
        })
        print(f"  {outcome}: ρ={rho:.3f}, p={pval:.2e}, top-20 overlap={overlap}")

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_dir / "a4b_vs_a4_rho.csv", index=False)
        return df
    return pd.DataFrame()


# Summary

def write_pipeline_outputs_and_log(
    args: argparse.Namespace,
    outcomes: list[str],
    r2_rows: list[dict],
    bootstrap_outcomes: list[str],
    rho_df: pd.DataFrame,
    output_dir: Path,
    adj_r2_rows: list[dict] | None = None,
) -> None:
    if r2_rows:
        pd.DataFrame(r2_rows).to_csv(output_dir / "a4b_r2_table.csv", index=False)

    outputs = [
        {"path": "a4b_r2_table.csv", "kind": "CSV", "status": "written" if r2_rows else "not written"},
        {"path": "a4b_overlap_adjusted_r2.csv", "kind": "CSV",
         "status": "written" if adj_r2_rows else "skipped"},
        {"path": "a4b_vs_a4_rho.csv", "kind": "CSV", "status": "written" if rho_df is not None and len(rho_df) else "skipped"},
    ]
    for outcome in outcomes:
        safe = outcome.replace("/", "_")
        shap_status = "written" if not args.no_bootstrap else (
            "reused" if outcome in bootstrap_outcomes else "skipped"
        )
        outputs.extend([
            {"path": output_dir / "shap" / f"{safe}_a4b_bootstrap_shap.csv", "kind": "CSV", "status": shap_status},
            {"path": output_dir / "shap" / f"{safe}_a4b_features.csv", "kind": "CSV", "status": shap_status},
            {"path": output_dir / "shap_raw" / f"{safe}_a4b_bootstrap_shap_raw.npy", "kind": "NPY", "status": shap_status},
            {"path": output_dir / "shap_raw" / f"{safe}_a4b_robust_shap.csv", "kind": "CSV", "status": shap_status},
            {"path": output_dir / "shap_raw" / f"{safe}_a4b_oob_count.npy", "kind": "NPY", "status": shap_status},
        ])
        if outcome in OVERLAP_ADJUSTED_OUTCOMES:
            outputs.extend([
                {"path": output_dir / "shap" / f"{safe}_a4b_adjusted_bootstrap_shap.csv", "kind": "CSV", "status": shap_status},
                {"path": output_dir / "shap" / f"{safe}_a4b_adjusted_features.csv", "kind": "CSV", "status": shap_status},
                {"path": output_dir / "shap_raw" / f"{safe}_a4b_adjusted_bootstrap_shap_raw.npy", "kind": "NPY", "status": shap_status},
                {"path": output_dir / "shap_raw" / f"{safe}_a4b_adjusted_robust_shap.csv", "kind": "CSV", "status": shap_status},
            ])

    run_args = vars(args).copy()
    run_args["resolved_outcomes"] = outcomes
    write_output_log(output_dir, "a4b_output_log.md", Path(__file__).name, outputs, run_args)


# Main

def main():
    # Record input provenance and the age-adjustment method beside the outputs.
    provenance.stamp_at_exit(OUTPUT_DIR, "a4b_concurrent_sensitivity", ["data/processed/master_analysis.csv", "data/processed/master_analysis_concurrent.csv", "output/a4_longitudinal"])
    age_method.emit_at_exit(OUTPUT_DIR, "a4b_longitudinal")
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    use_gpu = not args.no_gpu
    outcomes = resolve_outcomes(args.outcomes)

    print("A4B: LONGITUDINAL PREDICTION (W2 PREDICTORS)")
    print(f"Outcomes: {outcomes}")
    print(f"Bootstrap: {'disabled' if args.no_bootstrap else f'{args.n_bootstrap}×'}")
    print()

    adj_outcomes = [o for o in outcomes if o in OVERLAP_ADJUSTED_OUTCOMES]

    # Phase 1c standalone: overlap-adjusted ΔR² only
    # Writes its own table and leaves a4b_r2_table.csv untouched, so the
    # sensitivity can be produced without rerunning the primary arms.
    if args.overlap_r2_only:
        adj_rows = []
        for outcome in adj_outcomes:
            adj_rows.extend(run_overlap_adjusted_r2_a4b(outcome, args.n_folds, use_gpu))
        write_overlap_adjusted_r2(adj_rows, OUTPUT_DIR)
        print(f"\nDone. Results in: {OUTPUT_DIR}")
        return

    # Phase 1: Tiered ΔR² (pooled + per-cohort)
    all_r2_rows = []
    for outcome in outcomes:
        # Pooled
        rows = run_tiered_r2_a4b(outcome, args.n_folds, use_gpu)
        all_r2_rows.extend(rows)

        # Cohort split (not for AL: MIDUS 2 only)
        if outcome not in COHORT_SPLIT_SKIP:
            for cohort in ["MIDUS2", "MIDUSR"]:
                rows = run_tiered_r2_a4b(
                    outcome, args.n_folds, use_gpu, cohort=cohort,
                )
                all_r2_rows.extend(rows)

    # Phase 1c: Overlap-adjusted ΔR² (FInoPsy only)
    all_adj_r2_rows = []
    for outcome in adj_outcomes:
        all_adj_r2_rows.extend(
            run_overlap_adjusted_r2_a4b(outcome, args.n_folds, use_gpu)
        )
    write_overlap_adjusted_r2(all_adj_r2_rows, OUTPUT_DIR)

    # Phase 2: Bootstrap SHAP
    bootstrap_done = []
    if not args.no_bootstrap:
        for outcome in outcomes:
            run_adjusted = outcome in OVERLAP_ADJUSTED_OUTCOMES
            run_bootstrap_shap_a4b(
                outcome, args.n_bootstrap, use_gpu, OUTPUT_DIR,
                run_adjusted=run_adjusted,
            )
            bootstrap_done.append(outcome)

    # Detect existing SHAP files if bootstrap was skipped
    if args.no_bootstrap:
        for outcome in outcomes:
            safe = outcome.replace("/", "_")
            csv_path = OUTPUT_DIR / "shap" / f"{safe}_a4b_bootstrap_shap.csv"
            if csv_path.exists():
                bootstrap_done.append(outcome)
                print(f"  [reuse] {csv_path.name}")

    # Phase 3: A4 vs A4b SHAP ρ
    rho_df = pd.DataFrame()
    if bootstrap_done:
        rho_df = compute_a4_vs_a4b_rho(bootstrap_done, OUTPUT_DIR)

    write_pipeline_outputs_and_log(
        args, outcomes, all_r2_rows, bootstrap_done, rho_df, OUTPUT_DIR,
        adj_r2_rows=all_adj_r2_rows,
    )

    print(f"\nDone. Results in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
