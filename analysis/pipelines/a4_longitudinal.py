#!/usr/bin/env python3
"""
A4: Longitudinal Prediction (Primary Analysis)

Target = Y_w2 (age-residualized within-fold).
Bootstrap SHAP (200×, 2-GPU parallel).

Outputs:
  1. Tiered ΔR² (Yw1, Yw1+T1, Yw1+T1T2), pooled and per cohort
  2. Bootstrap SHAP rankings (mean |SHAP|, 95% CI, top-20 stability)
  3. Cross-sectional vs longitudinal SHAP ρ
  4. FInoPsy overlap-adjusted bootstrap SHAP

Usage:
    uv run python analysis/pipelines/a4_longitudinal.py
    uv run python analysis/pipelines/a4_longitudinal.py --outcomes FInoPsy_W2
    uv run python analysis/pipelines/a4_longitudinal.py --no_bootstrap
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
from analysis.core.data_loader import AL_OUTCOME_COLS, load_master, load_sibling_holdout, get_tier_variables, deduplicate_longitudinal_outcomes
from analysis.core.overlap_sources import get_overlap_excluded_vars
from analysis.core.xgboost_pipeline import XGBoostPipeline
from analysis.core.shap_dependence import plot_dependence_panel
from analysis.core.pub_style import setup_rcparams

OUTPUT_DIR = ROOT / "output/a4_longitudinal"
CS1_DIR = ROOT / "output/cs1_cross_sectional"

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
    p = argparse.ArgumentParser(description="A4 Longitudinal prediction")
    p.add_argument("--outcomes", nargs="+", default=["all"],
                    help="Outcome names or 'all'")
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--n_bootstrap", type=int, default=200)
    p.add_argument("--no_bootstrap", action="store_true")
    p.add_argument("--no_gpu", action="store_true")
    p.add_argument("--overlap_r2_only", action="store_true",
                   help="Run only the overlap-adjusted tiered R2 phase and exit. "
                        "Writes a4_overlap_adjusted_r2.csv and touches nothing else.")
    return p.parse_args()


def resolve_outcomes(names: list[str]) -> list[str]:
    if "all" in names:
        return PRIMARY_LONG
    return deduplicate_longitudinal_outcomes(names)


# Phase 1: Tiered ΔR²

def run_tiered_r2_longitudinal(
    outcome: str, n_folds: int, use_gpu: bool,
    cohort: str | None = None,
) -> list[dict]:
    """Tiered ΔR² for one outcome in longitudinal mode."""
    rows = []
    cohort_label = cohort or "pooled"

    for tier_label, tiers in TIER_CONFIGS.items():
        print(f"\n  [R²] {outcome} | {tier_label} | {cohort_label}")
        X, y, features, age = load_master(
            outcome, tiers=tiers, longitudinal=True,
            age_residualize=False, cohort_filter=cohort,
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

def run_overlap_adjusted_r2(
    outcome: str, n_folds: int, use_gpu: bool,
) -> list[dict]:
    """Tiered ΔR² with and without the frailty content-overlap predictors.

    The Tier 2 increment is reported twice on identical folds and an identical
    sample: once over all 161 Tier 2 predictors, once with the 29 direct and
    proximal overlap items removed. This is the longitudinal counterpart of
    ``analysis/sensitivity/a2_frailty_overlap.py`` at the cross-sectional layer
    and it uses the same audit list, so the adjusted increment here is
    comparable to the adjusted increment there.

    The excluded set is entirely Tier 2, so the ``Yw1T1`` baseline is shared by
    both increments and one baseline arm suffices. The adjusted feature set is
    the same one the adjusted bootstrap SHAP model fits, so the two artifacts
    describe one model.
    """
    overlap_vars = get_overlap_excluded_vars("ab")
    rows = []

    for label, tiers, drop_overlap in OVERLAP_R2_ARMS:
        print(f"\n  [R² overlap-adjusted] {outcome} | {label}")
        X, y, features, age = load_master(
            outcome, tiers=tiers, longitudinal=True,
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
    pd.DataFrame(rows).to_csv(output_dir / "a4_overlap_adjusted_r2.csv", index=False)
    print(f"\n  Saved: a4_overlap_adjusted_r2.csv ({len(rows)} rows)")


# Phase 2: Bootstrap SHAP

def run_bootstrap_shap_longitudinal(
    outcome: str, n_bootstrap: int, use_gpu: bool,
    output_dir: Path, run_adjusted: bool = False,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Bootstrap SHAP on T1T2+Y_w1 for one longitudinal outcome."""
    print(f"\nBOOTSTRAP SHAP (longitudinal): {outcome} (T1T2+Yw1, {n_bootstrap}×)")

    X, y, features, age = load_master(
        outcome, tiers=[1, 2], longitudinal=True, age_residualize=True,
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

    shap_df.to_csv(shap_dir / f"{safe}_long_bootstrap_shap.csv")
    np.save(raw_dir / f"{safe}_long_bootstrap_shap_raw.npy", shap_raw)
    pd.Series(features).to_csv(
        shap_dir / f"{safe}_long_features.csv", index=False, header=False,
    )
    print(f"  Saved: shap/{safe}_long_bootstrap_shap.csv ({shap_raw.shape})")

    robust_shap = pipe.robust_shap_
    pd.DataFrame(robust_shap, columns=features).to_csv(
        raw_dir / f"{safe}_long_robust_shap.csv", index=False
    )
    np.save(raw_dir / f"{safe}_long_oob_count.npy", pipe.oob_count_)
    print(f"  Saved: shap_raw/{safe}_long_robust_shap.csv ({robust_shap.shape})")

    # Dependence panel (stable features only)
    setup_rcparams()
    import matplotlib.pyplot as plt
    fig = plot_dependence_panel(
        X, robust_shap, features, ncols=4,
        outcome_label=f"{outcome} (long)",
        bootstrap_shap_df=shap_df,
        top_n=20,
    )
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"{safe}_long_dependence.{ext}",
                    format=ext, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: figures/{safe}_long_dependence.pdf/.png")

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
        shap_adj_df.to_csv(shap_dir / f"{safe}_long_adjusted_bootstrap_shap.csv")
        np.save(raw_dir / f"{safe}_long_adjusted_bootstrap_shap_raw.npy", shap_adj_raw)
        pd.Series(adj_features).to_csv(
            shap_dir / f"{safe}_long_adjusted_features.csv", index=False, header=False,
        )
        pd.DataFrame(pipe_adj.robust_shap_, columns=adj_features).to_csv(
            raw_dir / f"{safe}_long_adjusted_robust_shap.csv", index=False
        )
        print(f"  Saved: shap/{safe}_long_adjusted_bootstrap_shap.csv")

    return shap_df, shap_raw


# Phase 3: Cross-sectional vs longitudinal SHAP ρ

def compute_cs_vs_long_rho(outcomes: list[str], output_dir: Path) -> pd.DataFrame:
    """Spearman ρ between CS-1 and A4 bootstrap SHAP rankings."""
    rows = []
    for outcome in outcomes:
        safe = outcome.replace("/", "_")

        w1_name = outcome.replace("_W2", "_W1")
        w1_safe = w1_name.replace("/", "_")

        cs1_path = CS1_DIR / "shap" / f"{w1_safe}_T1T2_bootstrap_shap.csv"
        if not cs1_path.exists():
            print(f"  [SKIP] CS-1 SHAP not found for {w1_name}")
            continue
        cs1_df = pd.read_csv(cs1_path, index_col=0)

        a4_path = output_dir / "shap" / f"{safe}_long_bootstrap_shap.csv"
        if not a4_path.exists():
            print(f"  [SKIP] A4 SHAP not found for {outcome}")
            continue
        a4_df = pd.read_csv(a4_path, index_col=0)

        # Align on shared features (exclude Y_w1 from longitudinal)
        shared = sorted(set(cs1_df.index) & set(a4_df.index))
        if len(shared) < 10:
            print(f"  [SKIP] Only {len(shared)} shared features for {outcome}")
            continue

        cs1_vals = cs1_df.loc[shared, "mean_abs_shap"].values
        a4_vals = a4_df.loc[shared, "mean_abs_shap"].values

        rho, pval = spearmanr(cs1_vals, a4_vals)

        cs1_top20 = set(cs1_df.head(20).index)
        a4_top20 = set(a4_df.head(20).index) - {outcome.replace("_W2", "_W1")}
        overlap = len(cs1_top20 & a4_top20)

        rows.append({
            "outcome": outcome,
            "n_shared_features": len(shared),
            "spearman_rho": rho,
            "p_value": pval,
            "top20_overlap": f"{overlap}/{min(20, len(a4_top20))}",
        })
        print(f"  {outcome}: ρ={rho:.3f}, p={pval:.2e}, top-20 overlap={overlap}")

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_dir / "a4_cs_vs_long_rho.csv", index=False)
        return df
    return pd.DataFrame()


# Summary

def write_pipeline_outputs_and_log(
    args: argparse.Namespace,
    outcomes: list[str],
    r2_rows: list[dict],
    holdout_results: list[dict],
    bootstrap_outcomes: list[str],
    rho_df: pd.DataFrame,
    output_dir: Path,
    adj_r2_rows: list[dict] | None = None,
) -> None:
    if r2_rows:
        r2_df = pd.DataFrame(r2_rows)
        if holdout_results:
            holdout_rows = []
            for h in holdout_results:
                holdout_rows.append({
                    "outcome": h["outcome"],
                    "cohort": "sibling_holdout",
                    "tier_config": h["tier"],
                    "r2_mean": h["holdout_r2"],
                    "r2_std": np.nan,
                    "n": h["holdout_n"],
                    "n_features": np.nan,
                })
            r2_df = pd.concat([r2_df, pd.DataFrame(holdout_rows)], ignore_index=True)
        r2_df.to_csv(output_dir / "a4_r2_table.csv", index=False)

    outputs = [
        {"path": "a4_r2_table.csv", "kind": "CSV", "status": "written" if r2_rows else "not written"},
        {"path": "a4_overlap_adjusted_r2.csv", "kind": "CSV",
         "status": "written" if adj_r2_rows else "skipped"},
        {"path": "a4_holdout_results.json", "kind": "JSON", "status": "written" if holdout_results else "skipped"},
        {"path": "a4_cs_vs_long_rho.csv", "kind": "CSV", "status": "written" if rho_df is not None and len(rho_df) else "skipped"},
    ]
    for outcome in outcomes:
        safe = outcome.replace("/", "_")
        shap_status = "written" if not args.no_bootstrap else (
            "reused" if outcome in bootstrap_outcomes else "skipped"
        )
        outputs.extend([
            {"path": output_dir / "shap" / f"{safe}_long_bootstrap_shap.csv", "kind": "CSV", "status": shap_status},
            {"path": output_dir / "shap" / f"{safe}_long_features.csv", "kind": "CSV", "status": shap_status},
            {"path": output_dir / "shap_raw" / f"{safe}_long_bootstrap_shap_raw.npy", "kind": "NPY", "status": shap_status},
            {"path": output_dir / "shap_raw" / f"{safe}_long_robust_shap.csv", "kind": "CSV", "status": shap_status},
            {"path": output_dir / "shap_raw" / f"{safe}_long_oob_count.npy", "kind": "NPY", "status": shap_status},
            {"path": output_dir / "figures" / f"{safe}_long_dependence.pdf", "kind": "PDF", "status": shap_status},
            {"path": output_dir / "figures" / f"{safe}_long_dependence.png", "kind": "PNG", "status": shap_status},
        ])
        if outcome in OVERLAP_ADJUSTED_OUTCOMES:
            outputs.extend([
                {"path": output_dir / "shap" / f"{safe}_long_adjusted_bootstrap_shap.csv", "kind": "CSV", "status": shap_status},
                {"path": output_dir / "shap" / f"{safe}_long_adjusted_features.csv", "kind": "CSV", "status": shap_status},
                {"path": output_dir / "shap_raw" / f"{safe}_long_adjusted_bootstrap_shap_raw.npy", "kind": "NPY", "status": shap_status},
                {"path": output_dir / "shap_raw" / f"{safe}_long_adjusted_robust_shap.csv", "kind": "CSV", "status": shap_status},
            ])

    run_args = vars(args).copy()
    run_args["resolved_outcomes"] = outcomes
    write_output_log(output_dir, "a4_output_log.md", Path(__file__).name, outputs, run_args)


# Main

def main():
    # Record input provenance and the age-adjustment method beside the outputs.
    provenance.stamp_at_exit(OUTPUT_DIR, "a4_longitudinal", ["data/processed/master_analysis.csv", "data/processed/master_analysis_concurrent.csv", "output/cs1_cross_sectional"])
    age_method.emit_at_exit(OUTPUT_DIR, "a4_longitudinal")
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    use_gpu = not args.no_gpu
    outcomes = resolve_outcomes(args.outcomes)

    print("A4: LONGITUDINAL PREDICTION")
    print(f"Outcomes: {outcomes}")
    print(f"Bootstrap: {'disabled' if args.no_bootstrap else f'{args.n_bootstrap}×'}")
    print()

    adj_outcomes = [o for o in outcomes if o in OVERLAP_ADJUSTED_OUTCOMES]

    # Phase 1c standalone: overlap-adjusted ΔR² only
    # Writes its own table and leaves a4_r2_table.csv untouched, so the
    # sensitivity can be produced without rerunning the primary arms.
    if args.overlap_r2_only:
        adj_rows = []
        for outcome in adj_outcomes:
            adj_rows.extend(run_overlap_adjusted_r2(outcome, args.n_folds, use_gpu))
        write_overlap_adjusted_r2(adj_rows, OUTPUT_DIR)
        print(f"\nDone. Results in: {OUTPUT_DIR}")
        return

    # Phase 1: Tiered ΔR² (pooled + per-cohort)
    all_r2_rows = []
    for outcome in outcomes:
        # Pooled
        rows = run_tiered_r2_longitudinal(outcome, args.n_folds, use_gpu)
        all_r2_rows.extend(rows)

        # Cohort split (not for AL: MIDUS 2 only)
        if outcome not in COHORT_SPLIT_SKIP:
            for cohort in ["MIDUS2", "MIDUSR"]:
                rows = run_tiered_r2_longitudinal(
                    outcome, args.n_folds, use_gpu, cohort=cohort,
                )
                all_r2_rows.extend(rows)

    # Phase 1c: Overlap-adjusted ΔR² (FInoPsy only)
    all_adj_r2_rows = []
    for outcome in adj_outcomes:
        all_adj_r2_rows.extend(
            run_overlap_adjusted_r2(outcome, args.n_folds, use_gpu)
        )
    write_overlap_adjusted_r2(all_adj_r2_rows, OUTPUT_DIR)

    # Phase 1b: Sibling holdout validation (CV ensemble)
    # FInoPsy & btactComp only; validate both T1 and T1T2 tiers; AL skipped
    HOLDOUT_OUTCOMES = ["FInoPsy_W2", "btactComp_W2"]
    HOLDOUT_TIERS = {"Yw1": [], "Yw1T1": [1], "Yw1T1T2": [1, 2]}
    holdout_results = []
    for outcome in [o for o in outcomes if o in HOLDOUT_OUTCOMES]:
        for tier_label, tiers in HOLDOUT_TIERS.items():
            print(f"\nSIBLING HOLDOUT (CV ensemble): {outcome} | {tier_label}")
            X, y, features, age = load_master(
                outcome, tiers=tiers, longitudinal=True, age_residualize=False,
            )
            pipe = XGBoostPipeline(n_folds=5, tune=True, use_gpu=use_gpu)
            cv = pipe.run_cv(X, y, features, age=age)
            X_hold, y_hold, _, age_hold = load_sibling_holdout(
                outcome, tiers=tiers, longitudinal=True, age_residualize=False,
            )
            holdout = pipe.predict_holdout_ensemble(
                cv["fold_models"], X_hold, y_hold, age_holdout=age_hold,
            )
            holdout_results.append({
                "outcome": outcome, "tier": tier_label, "source": "A4",
                "cv_r2": cv["r2_mean"], "cv_r2_std": cv["r2_std"],
                "holdout_r2": holdout["r2"], "holdout_n": holdout["n"],
                "gap": cv["r2_mean"] - holdout["r2"],
            })
            print(f"  CV R²: {cv['r2_mean']:.4f} ± {cv['r2_std']:.4f}")
            print(f"  Holdout R²: {holdout['r2']:.4f} (gap: {cv['r2_mean'] - holdout['r2']:+.4f})")

    if holdout_results:
        import json
        holdout_path = OUTPUT_DIR / "a4_holdout_results.json"
        with open(holdout_path, "w") as f:
            json.dump(holdout_results, f, indent=2)
        print(f"\nSaved: {holdout_path}")

    # Phase 2: Bootstrap SHAP
    bootstrap_done = []
    if not args.no_bootstrap:
        for outcome in outcomes:
            run_adjusted = outcome in OVERLAP_ADJUSTED_OUTCOMES
            run_bootstrap_shap_longitudinal(
                outcome, args.n_bootstrap, use_gpu, OUTPUT_DIR,
                run_adjusted=run_adjusted,
            )
            bootstrap_done.append(outcome)

    # Detect existing SHAP files if bootstrap was skipped
    if args.no_bootstrap:
        for outcome in outcomes:
            safe = outcome.replace("/", "_")
            csv_path = OUTPUT_DIR / "shap" / f"{safe}_long_bootstrap_shap.csv"
            if csv_path.exists():
                bootstrap_done.append(outcome)
                print(f"  [reuse] {csv_path.name}")

    # Phase 3: CS vs Long SHAP ρ
    rho_df = pd.DataFrame()
    if bootstrap_done:
        rho_df = compute_cs_vs_long_rho(bootstrap_done, OUTPUT_DIR)

    write_pipeline_outputs_and_log(
        args, outcomes, all_r2_rows, holdout_results, bootstrap_done, rho_df,
        OUTPUT_DIR, adj_r2_rows=all_adj_r2_rows,
    )

    print(f"\nDone. Results in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
