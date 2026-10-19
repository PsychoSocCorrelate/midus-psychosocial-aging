#!/usr/bin/env python3
"""
CS-1: Cross-Sectional Tiered ΔR² with Bootstrap SHAP

Bootstrap SHAP (200× by default) for all primary and supplemental outcomes.
For FInoPsy, also runs the A2 overlap-adjusted model.

Saves raw bootstrap SHAP matrices (.npy) for A3 domain divergence test.

Usage:
    uv run python analysis/pipelines/cs1_cross_sectional.py
    uv run python analysis/pipelines/cs1_cross_sectional.py --outcomes FInoPsy_W1 btactComp_W1
    uv run python analysis/pipelines/cs1_cross_sectional.py --outcomes all --no_tabpfn
    uv run python analysis/pipelines/cs1_cross_sectional.py --no_bootstrap  # R² only

Output: output/cs1_cross_sectional/
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
from analysis.core.data_loader import load_master, load_sibling_holdout, get_tier_variables
from analysis.core.overlap_sources import get_overlap_excluded_vars
from analysis.core.xgboost_pipeline import XGBoostPipeline
from analysis.core.shap_dependence import plot_dependence_panel
from analysis.core.pub_style import setup_rcparams
from analysis.core import artifact_paths as ap

OUTPUT_DIR = ROOT / "output/cs1_cross_sectional"

PRIMARY = ["FInoPsy_W1", "btactComp_W1", "ALcomp_W1", "phenoAge_res"]
SUPPLEMENTAL = ["grimAgeN_res", "dunedinPace_res", "episoMem_W1", "efFactor_W1"]

TIER_CONFIGS = {
    "T1": [1],
    "T1T2": [1, 2],
}


def parse_args():
    p = argparse.ArgumentParser(description="CS-1 Cross-sectional bootstrap SHAP")
    p.add_argument(
        "--outcomes", nargs="+", default=["all"],
        help="Outcome names or 'all' / 'primary' / 'supplemental'",
    )
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--n_bootstrap", type=int, default=200)
    p.add_argument("--no_tabpfn", action="store_true")
    p.add_argument("--no_bootstrap", action="store_true",
                    help="Skip bootstrap SHAP (R² and single-run SHAP only)")
    p.add_argument("--no_gpu", action="store_true")
    p.add_argument("--replot_dependence", action="store_true",
                   help="Only redraw the dependence panels from persisted SHAP; fit nothing")
    return p.parse_args()


def resolve_outcomes(names: list[str]) -> list[str]:
    from analysis.core.data_loader import W2_TO_W1
    if "all" in names:
        return PRIMARY + SUPPLEMENTAL
    if "primary" in names:
        return PRIMARY
    if "supplemental" in names:
        return SUPPLEMENTAL
    # Normalize W2 names to W1 (CS-1 is cross-sectional) and deduplicate
    seen: set[str] = set()
    result: list[str] = []
    for name in names:
        canonical = W2_TO_W1.get(name, name)
        if canonical not in seen:
            seen.add(canonical)
            result.append(canonical)
    return result


def run_tiered_r2(
    outcome: str, n_folds: int, use_gpu: bool
) -> list[dict]:
    """Run tiered ΔR² for one outcome (T1, T1T2)."""
    rows = []
    r2_cache = {}

    for tier_label, tiers in TIER_CONFIGS.items():
        print(f"\n  [R²] {outcome} | {tier_label}")
        X, y, features, age = load_master(outcome, tiers=tiers, age_residualize=False)
        pipe = XGBoostPipeline(n_folds=n_folds, use_gpu=use_gpu)
        cv = pipe.run_cv(X, y, features, age=age)
        r2_cache[tier_label] = cv["r2_mean"]
        rows.append({
            "outcome": outcome,
            "tier_config": tier_label,
            "r2_mean": cv["r2_mean"],
            "r2_std": cv["r2_std"],
            "n": int(y.notna().sum()),
            "n_features": len(features),
        })

    return rows, r2_cache


def run_bootstrap_shap_for_outcome(
    outcome: str,
    n_bootstrap: int,
    use_gpu: bool,
    output_dir: Path,
    run_adjusted: bool = False,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Run bootstrap SHAP on T1T2 for one outcome. Returns (summary_df, raw_array)."""
    print(f"\nBOOTSTRAP SHAP: {outcome} (T1T2, {n_bootstrap} iterations)")

    X, y, features, age = load_master(outcome, tiers=[1, 2], age_residualize=True)
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

    shap_df.to_csv(shap_dir / f"{safe}_T1T2_bootstrap_shap.csv")
    print(f"  Saved: shap/{safe}_T1T2_bootstrap_shap.csv")

    # Raw array and feature names, for A3
    np.save(raw_dir / f"{safe}_T1T2_bootstrap_shap_raw.npy", shap_raw)
    pd.Series(features).to_csv(
        shap_dir / f"{safe}_T1T2_features.csv", index=False, header=False
    )
    print(f"  Saved: shap_raw/{safe}_T1T2_bootstrap_shap_raw.npy ({shap_raw.shape})")

    robust_shap = pipe.robust_shap_
    pd.DataFrame(robust_shap, columns=features).to_csv(
        raw_dir / f"{safe}_T1T2_robust_shap.csv", index=False
    )
    np.save(raw_dir / f"{safe}_T1T2_oob_count.npy", pipe.oob_count_)
    print(f"  Saved: shap_raw/{safe}_T1T2_robust_shap.csv ({robust_shap.shape})")

    render_dependence(outcome, X, robust_shap, features, shap_df, fig_dir)

    # FInoPsy adjusted model
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

        shap_adj_df.to_csv(shap_dir / f"{safe}_T1T2_adjusted_bootstrap_shap.csv")
        np.save(raw_dir / f"{safe}_T1T2_adjusted_bootstrap_shap_raw.npy", shap_adj_raw)
        pd.Series(adj_features).to_csv(
            shap_dir / f"{safe}_T1T2_adjusted_features.csv", index=False, header=False
        )
        pd.DataFrame(pipe_adj.robust_shap_, columns=adj_features).to_csv(
            raw_dir / f"{safe}_T1T2_adjusted_robust_shap.csv", index=False
        )
        print(f"  Saved: shap/{safe}_T1T2_adjusted_bootstrap_shap.csv")

    return shap_df, shap_raw


# Figure titles for the outcomes the SI shows as dependence figures (S2-S5).
DEPENDENCE_TITLE = {
    "FInoPsy_W1": "Frailty",
    "btactComp_W1": "Cognition",
    f"{ap.AL_STEM}_W1": "Allostatic load",
    "phenoAge_res": "PhenoAge",
}


def render_dependence(outcome, X, robust_shap, features, shap_df, fig_dir) -> None:
    """Dependence panel of the stable features, titled with the outcome's display name."""
    setup_rcparams()
    import matplotlib.pyplot as plt
    fig = plot_dependence_panel(
        X, robust_shap, features, ncols=4,
        outcome_label=DEPENDENCE_TITLE.get(outcome, outcome),
        bootstrap_shap_df=shap_df,
        top_n=20,
    )
    safe = outcome.replace("/", "_")
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"{safe}_T1T2_dependence.{ext}",
                    format=ext, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: figures/{safe}_T1T2_dependence.pdf/.png")


def replot_dependence(outcome: str) -> None:
    """Redraw a dependence panel from the persisted SHAP, without refitting."""
    safe = outcome.replace("/", "_")
    X, _, features, _ = load_master(outcome, tiers=[1, 2], age_residualize=True)
    robust = pd.read_csv(OUTPUT_DIR / "shap_raw" / f"{safe}_T1T2_robust_shap.csv")
    if list(robust.columns) != list(features) or len(robust) != len(X):
        raise ValueError(f"{outcome}: persisted robust SHAP does not match the loaded design")
    shap_df = pd.read_csv(OUTPUT_DIR / "shap" / f"{safe}_T1T2_bootstrap_shap.csv", index_col=0)
    fig_dir = OUTPUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    render_dependence(outcome, X, robust.to_numpy(), features, shap_df, fig_dir)


def run_tabpfn_r2(
    outcome: str, n_folds: int, output_dir: Path
) -> dict | None:
    """Run TabPFN 5-fold CV for R² benchmarking only (no SHAP).

    Returns an r2_info dict, or None if TabPFN is unavailable.
    """
    try:
        from tabpfn import TabPFNRegressor
    except ImportError as e:
        print(f"  [TabPFN] Import failed: {e}")
        return None

    from sklearn.linear_model import LinearRegression as _LR
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score

    print(f"\n  [TabPFN R²] {outcome} | T1T2")
    X_df, y_raw, feat_names, age = load_master(
        outcome, tiers=[1, 2], age_residualize=False,
    )

    # TabPFN expects numeric input: encode categoricals as ordinal codes.
    # (XGBoost uses native pd.Categorical; TabPFN needs float32.)
    X_enc = X_df.copy()
    for col in X_enc.columns:
        if X_enc[col].dtype.name == 'category' or X_enc[col].dtype == object:
            X_enc[col] = X_enc[col].cat.codes.replace(-1, np.nan).astype('float32')
    X_np = X_enc.to_numpy(dtype=np.float32, na_value=np.nan)
    y_np = y_raw.to_numpy(dtype=np.float64)
    valid_mask = ~np.isnan(y_np)
    X_np, y_np = X_np[valid_mask], y_np[valid_mask]
    age_np = age[valid_mask] if age is not None else None
    n = len(y_np)

    MAX_TRAIN = 3000  # TabPFN in-context learning cap
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    r2_folds = []

    for fold_i, (train_idx, test_idx) in enumerate(kf.split(X_np)):
        X_tr, X_te = X_np[train_idx], X_np[test_idx]
        y_tr, y_te = y_np[train_idx], y_np[test_idx]

        # Within-fold age residualization (same standard as XGBoost)
        if age_np is not None:
            age_tr = age_np[train_idx].reshape(-1, 1)
            age_te = age_np[test_idx].reshape(-1, 1)
            ok_tr = ~np.isnan(age_tr.ravel())
            if ok_tr.sum() > 5:
                ar = _LR().fit(age_tr[ok_tr], y_tr[ok_tr])
                y_tr = y_tr - ar.predict(age_tr).ravel()
                y_te = y_te - ar.predict(age_te).ravel()

        # Subsample to TabPFN cap
        rng = np.random.default_rng(42 + fold_i)
        if len(train_idx) > MAX_TRAIN:
            sub = rng.choice(len(X_tr), MAX_TRAIN, replace=False)
            X_tr_fit, y_tr_fit = X_tr[sub], y_tr[sub]
        else:
            X_tr_fit, y_tr_fit = X_tr, y_tr

        model = TabPFNRegressor(device="cuda", n_estimators=8)
        try:
            model.fit(X_tr_fit, y_tr_fit)
        except Exception as e:
            print(f"    Fold {fold_i+1} TabPFN fit failed: {e}")
            r2_folds.append(np.nan)
            continue

        y_hat = model.predict(X_te)
        r2 = r2_score(y_te, y_hat)
        r2_folds.append(r2)
        print(f"    Fold {fold_i+1}/{n_folds}: R²={r2:.4f}")

    r2_arr = np.array([r for r in r2_folds if not np.isnan(r)])
    info = {
        "outcome": outcome, "model": "TabPFN", "tier_config": "T1T2",
        "r2_mean": float(r2_arr.mean()) if len(r2_arr) > 0 else np.nan,
        "r2_std": float(r2_arr.std()) if len(r2_arr) > 0 else np.nan,
        "n": n, "n_features": len(feat_names),
    }
    print(f"  [TabPFN] {outcome} T1T2: R²={info['r2_mean']:.4f} ± {info['r2_std']:.4f}")
    return info


def write_pipeline_log(
    args: argparse.Namespace,
    outcomes: list[str],
    bootstrap_outcomes: list[str],
    tabpfn_rows: list[dict],
) -> None:
    outputs = [
        {"path": "cs1_r2_table.csv", "kind": "CSV"},
        {"path": "cs1_holdout_results.json", "kind": "JSON"},
        {
            "path": "cs1_tabpfn_r2.csv",
            "kind": "CSV",
            "status": "skipped" if args.no_tabpfn else ("written" if tabpfn_rows else "not written"),
        },
    ]
    for outcome in outcomes:
        safe = outcome.replace("/", "_")
        shap_status = "written" if not args.no_bootstrap else (
            "reused" if outcome in bootstrap_outcomes else "skipped"
        )
        outputs.extend([
            {"path": OUTPUT_DIR / "shap" / f"{safe}_T1T2_bootstrap_shap.csv", "kind": "CSV", "status": shap_status},
            {"path": OUTPUT_DIR / "shap" / f"{safe}_T1T2_features.csv", "kind": "CSV", "status": shap_status},
            {"path": OUTPUT_DIR / "shap_raw" / f"{safe}_T1T2_bootstrap_shap_raw.npy", "kind": "NPY", "status": shap_status},
            {"path": OUTPUT_DIR / "shap_raw" / f"{safe}_T1T2_robust_shap.csv", "kind": "CSV", "status": shap_status},
            {"path": OUTPUT_DIR / "shap_raw" / f"{safe}_T1T2_oob_count.npy", "kind": "NPY", "status": shap_status},
            {"path": OUTPUT_DIR / "figures" / f"{safe}_T1T2_dependence.pdf", "kind": "PDF", "status": shap_status},
            {"path": OUTPUT_DIR / "figures" / f"{safe}_T1T2_dependence.png", "kind": "PNG", "status": shap_status},
        ])
        if outcome == "FInoPsy_W1":
            outputs.extend([
                {"path": OUTPUT_DIR / "shap" / f"{safe}_T1T2_adjusted_bootstrap_shap.csv", "kind": "CSV", "status": shap_status},
                {"path": OUTPUT_DIR / "shap" / f"{safe}_T1T2_adjusted_features.csv", "kind": "CSV", "status": shap_status},
                {"path": OUTPUT_DIR / "shap_raw" / f"{safe}_T1T2_adjusted_bootstrap_shap_raw.npy", "kind": "NPY", "status": shap_status},
                {"path": OUTPUT_DIR / "shap_raw" / f"{safe}_T1T2_adjusted_robust_shap.csv", "kind": "CSV", "status": shap_status},
            ])

    run_args = vars(args).copy()
    run_args["resolved_outcomes"] = outcomes
    write_output_log(OUTPUT_DIR, "cs1_output_log.md", Path(__file__).name, outputs, run_args)


def merge_by_outcome(new: pd.DataFrame, path: Path) -> pd.DataFrame:
    """This run's rows plus the file's rows for outcomes this run did not touch.

    A run restricted by --outcomes therefore replaces only its own outcomes and
    cannot drop anyone else's. Sorted so row order does not depend on the run.
    """
    if path.exists():
        prior = pd.read_csv(path)
        kept = prior[~prior["outcome"].isin(set(new["outcome"]))]
        if len(kept):
            print(f"  keeping {len(kept)} row(s) of {path.name} for outcomes this run did not "
                  f"touch: {sorted(kept['outcome'].unique())}")
        new = pd.concat([kept, new], ignore_index=True)
    return new.sort_values(["outcome", "tier_config"], kind="mergesort", ignore_index=True)


def main():
    args = parse_args()
    if args.replot_dependence:
        for outcome in resolve_outcomes(args.outcomes):
            replot_dependence(outcome)
        return
    # Record input provenance and the age-adjustment method beside the outputs.
    provenance.stamp_at_exit(OUTPUT_DIR, "cs1_cross_sectional", ["data/processed/master_analysis.csv", "data/processed/master_analysis_concurrent.csv"])
    age_method.emit_at_exit(OUTPUT_DIR, "cs1_cross_sectional")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    use_gpu = not args.no_gpu
    outcomes = resolve_outcomes(args.outcomes)

    print("CS-1: CROSS-SECTIONAL TIERED ΔR² WITH BOOTSTRAP SHAP")
    print(f"Outcomes: {outcomes}")
    print(f"Bootstrap: {'disabled' if args.no_bootstrap else f'{args.n_bootstrap} iterations'}")
    print(f"TabPFN: {'disabled' if args.no_tabpfn else 'enabled'}")
    print()

    # Phase 1: Tiered R² for primary outcomes
    all_r2_rows = []
    for outcome in outcomes:
        if outcome in PRIMARY:
            rows, _ = run_tiered_r2(outcome, args.n_folds, use_gpu)
            all_r2_rows.extend(rows)

    # Merge with existing rows so a run restricted by --outcomes keeps the other outcomes.
    if all_r2_rows:
        r2_path = OUTPUT_DIR / "cs1_r2_table.csv"
        r2_df = merge_by_outcome(pd.DataFrame(all_r2_rows), r2_path)
        r2_df.to_csv(r2_path, index=False)
        print(f"\nSaved: cs1_r2_table.csv ({len(r2_df)} rows)")

    # Phase 1b: Sibling holdout validation (CV ensemble)
    # FInoPsy & btactComp only; validate both T1 and T1T2 tiers; AL skipped
    HOLDOUT_OUTCOMES = ["FInoPsy_W1", "btactComp_W1"]
    HOLDOUT_TIERS = {"T1": [1], "T1T2": [1, 2]}
    holdout_results = []
    for outcome in [o for o in outcomes if o in HOLDOUT_OUTCOMES]:
        for tier_label, tiers in HOLDOUT_TIERS.items():
            print(f"\nSIBLING HOLDOUT (CV ensemble): {outcome} | {tier_label}")
            X, y, features, age = load_master(
                outcome, tiers=tiers, age_residualize=False,
            )
            pipe = XGBoostPipeline(n_folds=5, tune=True, use_gpu=use_gpu)
            cv = pipe.run_cv(X, y, features, age=age)
            X_hold, y_hold, _, age_hold = load_sibling_holdout(
                outcome, tiers=tiers, longitudinal=False, age_residualize=False,
            )
            holdout = pipe.predict_holdout_ensemble(
                cv["fold_models"], X_hold, y_hold, age_holdout=age_hold,
            )
            holdout_results.append({
                "outcome": outcome, "tier": tier_label, "source": "CS1",
                "cv_r2": cv["r2_mean"], "cv_r2_std": cv["r2_std"],
                "holdout_r2": holdout["r2"], "holdout_n": holdout["n"],
                "gap": cv["r2_mean"] - holdout["r2"],
            })
            print(f"  CV R²: {cv['r2_mean']:.4f} ± {cv['r2_std']:.4f}")
            print(f"  Holdout R²: {holdout['r2']:.4f} (gap: {cv['r2_mean'] - holdout['r2']:+.4f})")

    if holdout_results:
        import json
        holdout_path = OUTPUT_DIR / "cs1_holdout_results.json"
        with open(holdout_path, "w") as f:
            json.dump(holdout_results, f, indent=2)
        print(f"\nSaved: {holdout_path}")

    # Phase 2: Bootstrap SHAP for all outcomes
    bootstrap_done = []
    if not args.no_bootstrap:
        for outcome in outcomes:
            run_adjusted = (outcome == "FInoPsy_W1")
            run_bootstrap_shap_for_outcome(
                outcome, args.n_bootstrap, use_gpu, OUTPUT_DIR,
                run_adjusted=run_adjusted,
            )
            bootstrap_done.append(outcome)

    # Detect existing SHAP files if bootstrap was skipped
    if args.no_bootstrap:
        for outcome in outcomes:
            safe = outcome.replace("/", "_")
            csv_path = OUTPUT_DIR / "shap" / f"{safe}_T1T2_bootstrap_shap.csv"
            if csv_path.exists():
                bootstrap_done.append(outcome)
                print(f"  [reuse] {csv_path.name}")

    # Phase 3: TabPFN R² benchmark (primary outcomes, no SHAP).
    tabpfn_rows = []
    if not args.no_tabpfn:
        for outcome in [o for o in outcomes if o in PRIMARY]:
            info = run_tabpfn_r2(outcome, args.n_folds, OUTPUT_DIR)
            if info is not None:
                tabpfn_rows.append(info)
        if tabpfn_rows:
            tabpfn_df = merge_by_outcome(pd.DataFrame(tabpfn_rows),
                                         OUTPUT_DIR / "cs1_tabpfn_r2.csv")
            tabpfn_df.to_csv(OUTPUT_DIR / "cs1_tabpfn_r2.csv", index=False)
            print(f"\nSaved: cs1_tabpfn_r2.csv ({len(tabpfn_df)} rows)")
            # Print XGB vs TabPFN comparison
            if all_r2_rows:
                r2_df = pd.DataFrame(all_r2_rows)
                for _, row in tabpfn_df.iterrows():
                    xgb_match = r2_df[(r2_df["outcome"] == row["outcome"])
                                      & (r2_df["tier_config"] == "T1T2")]
                    if not xgb_match.empty:
                        delta = row["r2_mean"] - xgb_match.iloc[0]["r2_mean"]
                        print(f"  {row['outcome']}: XGB={xgb_match.iloc[0]['r2_mean']:.4f} "
                              f"TabPFN={row['r2_mean']:.4f} (Δ={delta:+.4f})")

    write_pipeline_log(args, outcomes, bootstrap_done, tabpfn_rows)

    print(f"\nDone. Results in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
