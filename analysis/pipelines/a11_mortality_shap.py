#!/usr/bin/env python3
"""
A11: Mortality SHAP + Lethality-vs-Aging Scatter

Train XGBoost (binary:logistic) on the baseline analytic sample predicting
all-cause mortality through the 2025 MIDUS mortality release, extract
bootstrap SHAP values, then produce a dual
scatter plot comparing mortality SHAP vs CS-1 cross-sectional SHAP:
  Panel A: mortality importance × frailty importance (FInoPsy_W1)
  Panel B: mortality importance × cognition importance (btactComp_W1)

Each point = one T1/T2 variable, colored by subdomain.
Uses CS-1 SHAP (not A4 longitudinal) for the aging-domain axis.

Usage:
    uv run python analysis/pipelines/a11_mortality_shap.py
    uv run python analysis/pipelines/a11_mortality_shap.py --endpoint fixed_9y --n_bootstrap 50 --no_plots
    uv run python analysis/pipelines/a11_mortality_shap.py --no_bootstrap
    uv run python analysis/pipelines/a11_mortality_shap.py --no_cv --no_bootstrap --plot_only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.core import provenance

from analysis.core.output_log import write_output_log
from analysis.core.data_loader import (
    get_tier_variables, CATEGORICALS,
)
from analysis.core.mortality_data import (
    ENDPOINTS,
    load_survival_data,
    output_dir_for_endpoint,
    summarize_survival_frame,
)
from analysis.core.xgboost_pipeline import XGBoostPipeline
from analysis.core.pub_style import (
    setup_rcparams, display_name, get_subdomain, get_tier,
    SUBDOMAIN_COLORS, PNAS_FULL, FONT_TITLE, FONT_LABEL, FONT_ANNOT,
    FONT_TICK,
)
from analysis.core import artifact_paths as ap

OUTPUT_DIR = ROOT / "output/a11_mortality_shap"

# Excluded from the scatter (CS-1 has no autoregressive term, so only age).
SCATTER_EXCLUDE = {"age_W1"}


def parse_args():
    p = argparse.ArgumentParser(description="A11 Mortality SHAP")
    p.add_argument("--endpoint", choices=sorted(ENDPOINTS), default="through_2025")
    p.add_argument("--n_bootstrap", type=int, default=200)
    p.add_argument("--n_gpus", type=int, default=2)
    p.add_argument("--no_bootstrap", action="store_true")
    p.add_argument("--no_cv", action="store_true")
    p.add_argument("--no_gpu", action="store_true")
    p.add_argument("--no_plots", action="store_true")
    p.add_argument("--plot_only", action="store_true",
                    help="Skip model training, only generate scatter plots")
    return p.parse_args()


# Data Loading

def load_mortality_features(
    endpoint: str = "through_2025",
    use_gpu: bool = True,
    exclude_siblings: bool = True,
) -> tuple[pd.DataFrame, pd.Series, list[str], dict]:
    """Load baseline features + mortality outcome for XGBoost classification.

    Returns (X, y_binary, feature_names, endpoint_summary) where y_binary is
    the endpoint-specific deceased indicator.
    """
    df = load_survival_data(endpoint=endpoint)
    endpoint_summary = summarize_survival_frame(df)

    # Sibling exclusion (same logic as data_loader.py)
    if exclude_siblings and "pair_id" in df.columns:
        n_before = len(df)
        if "is_full_pair" in df.columns:
            df = df[~(df["is_full_pair"] == True)].reset_index(drop=True)
        else:
            df = df[df["pair_id"].isna()].reset_index(drop=True)
        n_removed = n_before - len(df)
        print(f"  [sibling exclusion] removed {n_removed} complete-pair rows; "
              f"training N={len(df):,}")
        endpoint_summary["training_n_after_sibling_exclusion"] = int(len(df))
        endpoint_summary["training_events_after_sibling_exclusion"] = int(df["deceased"].sum())
        endpoint_summary["sibling_excluded_rows"] = int(n_removed)

    # Feature set: T1 + T2 + age_W1
    predictor_vars = get_tier_variables([1, 2])
    feature_cols = list(predictor_vars)
    if "age_W1" not in feature_cols and "age_W1" in df.columns:
        feature_cols.append("age_W1")

    available = set(df.columns)
    missing = [v for v in feature_cols if v not in available]
    if missing:
        print(f"  [WARN] {len(missing)} vars not in CSV: {missing[:5]}...")
    feature_cols = [v for v in feature_cols if v in available]

    X = df[feature_cols].copy()

    # Encode categoricals as pd.Categorical (same as data_loader.py)
    for col in X.columns:
        if X[col].dtype == object or col in CATEGORICALS:
            X[col] = X[col].astype(str).replace("nan", pd.NA).replace("<NA>", pd.NA)
            X[col] = pd.Categorical(X[col]).remove_unused_categories()

    # Drop all-NaN categorical columns
    empty_cat_cols = [c for c in X.columns
                      if hasattr(X[c], 'cat') and len(X[c].cat.categories) == 0]
    if empty_cat_cols:
        print(f"  [data] dropping {len(empty_cat_cols)} all-NaN categorical(s): {empty_cat_cols}")
        X = X.drop(columns=empty_cat_cols)
        feature_cols = [c for c in feature_cols if c not in empty_cat_cols]

    y = df["deceased"].astype(float)

    n_events = int(y.sum())
    print(f"  load_mortality_features: N={len(df):,}, events={n_events} "
          f"({n_events/len(df)*100:.1f}%), features={len(feature_cols)}")

    return X, y, feature_cols, endpoint_summary


# CV Evaluation

def run_mortality_cv(
    X: pd.DataFrame, y: pd.Series, features: list[str],
    use_gpu: bool = True,
) -> dict:
    """5-fold CV with AUC for mortality classification."""
    print("\nMORTALITY CV (5-fold, AUC)")

    pipe = XGBoostPipeline(
        n_folds=5, tune=True, use_gpu=use_gpu, task="classification",
    )
    cv = pipe.run_cv(X, y, features)

    print(f"\n  CV AUC: {cv['r2_mean']:.4f} ± {cv['r2_std']:.4f}")
    return cv


# Bootstrap SHAP

def available_gpu_count() -> int:
    """Return visible NVIDIA GPU count, or 0 on CPU-only hosts."""
    import subprocess

    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            text=True,
            capture_output=True,
            timeout=3,
            check=False,
        )
    except Exception:
        return 0
    if result.returncode != 0:
        return 0
    return sum(1 for line in result.stdout.splitlines() if line.strip().startswith("GPU "))


def run_mortality_shap(
    X: pd.DataFrame, y: pd.Series, features: list[str],
    output_dir: Path,
    n_bootstrap: int = 200,
    use_gpu: bool = True,
    n_gpus: int = 2,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Bootstrap SHAP for mortality model."""
    print(f"\nBOOTSTRAP SHAP (mortality, binary:logistic, {n_bootstrap}×)")

    # Fixed subdirectories: these artifacts belong to the mortality model, not to an outcome.
    shap_dir = output_dir / "shap"
    raw_dir = output_dir / "shap_raw"
    for d in (shap_dir, raw_dir):
        d.mkdir(parents=True, exist_ok=True)

    pipe = XGBoostPipeline(
        n_bootstrap=n_bootstrap, use_gpu=use_gpu, task="classification",
    )
    gpu_count = available_gpu_count() if use_gpu else 0
    if gpu_count:
        shap_df, shap_raw = pipe.run_bootstrap_shap_parallel(
            X,
            y,
            features,
            n_gpus=max(1, min(n_gpus, gpu_count)),
            return_raw=True,
            save_individual=True,
        )
    else:
        shap_df, shap_raw = pipe.run_bootstrap_shap(
            X,
            y,
            features,
            n_bootstrap=n_bootstrap,
            return_raw=True,
            save_individual=True,
        )

    shap_df.to_csv(shap_dir / "mortality_bootstrap_shap.csv")
    np.save(raw_dir / "mortality_bootstrap_shap_raw.npy", shap_raw)
    pd.Series(features).to_csv(
        shap_dir / "mortality_features.csv", index=False, header=False,
    )
    pd.DataFrame(pipe.robust_shap_, columns=features).to_csv(
        raw_dir / "mortality_robust_shap.csv", index=False,
    )
    np.save(raw_dir / "mortality_oob_count.npy", pipe.oob_count_)

    print(f"  Saved: shap/mortality_bootstrap_shap.csv ({shap_raw.shape})")
    print(f"  Saved: shap_raw/mortality_robust_shap.csv ({pipe.robust_shap_.shape})")

    return shap_df, shap_raw


# Scatter Plot

def _load_and_align_shap(
    mort_shap_path: Path,
    outcome_shap_path: Path,
) -> pd.DataFrame:
    """Load two SHAP CSVs, intersect variables, return merged DataFrame."""
    mort = pd.read_csv(mort_shap_path, index_col=0)
    outcome = pd.read_csv(outcome_shap_path, index_col=0)

    # Intersect variables, excluding age
    shared = sorted(
        (set(mort.index) & set(outcome.index)) - SCATTER_EXCLUDE
    )
    print(f"  Shared variables (excluding {SCATTER_EXCLUDE}): {len(shared)}")

    df = pd.DataFrame({
        "mort_shap": mort.loc[shared, "mean_abs_shap"],
        "outcome_shap": outcome.loc[shared, "mean_abs_shap"],
        "mort_stable": mort.loc[shared, "stable"],
        "outcome_stable": outcome.loc[shared, "stable"],
    }, index=shared)

    # Percentile rank normalization (0 = least important, 1 = most important)
    n = len(df)
    df["mort_rank"] = stats.rankdata(df["mort_shap"]) / n
    df["outcome_rank"] = stats.rankdata(df["outcome_shap"]) / n

    # Subdomain lookup
    df["subdomain"] = [get_subdomain(v) or "demographics" for v in df.index]
    df["tier"] = [get_tier(v) for v in df.index]
    df["display"] = [display_name(v) for v in df.index]

    return df


def plot_single_scatter(
    df: pd.DataFrame,
    outcome_label: str,
    ax=None,
    n_annotate: int = 5,
):
    """Draw one scatter panel on the given axes."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()

    # Group by subdomain for legend
    subdomains = sorted(df["subdomain"].unique())

    # Stable in at least one model: larger, opaque
    any_stable = df["mort_stable"] | df["outcome_stable"]

    # Draw all points colored by subdomain; stable ones larger
    for sub in subdomains:
        mask = df["subdomain"] == sub
        if not mask.any():
            continue
        color = SUBDOMAIN_COLORS.get(sub, "#888888")
        sub_df = df[mask]

        # Unstable: small, semi-transparent
        unstable_mask = ~any_stable[mask]
        if unstable_mask.any():
            ax.scatter(
                sub_df.loc[unstable_mask, "mort_rank"],
                sub_df.loc[unstable_mask, "outcome_rank"],
                c=color, s=12, alpha=0.30, edgecolors="none", zorder=1,
            )

        # Stable in at least one model: larger, opaque
        stable_mask = any_stable[mask]
        if stable_mask.any():
            ax.scatter(
                sub_df.loc[stable_mask, "mort_rank"],
                sub_df.loc[stable_mask, "outcome_rank"],
                c=color, s=28, alpha=0.85, edgecolors="white",
                linewidths=0.3, zorder=2,
                label=sub.replace("_", " ").title(),
            )

    # Annotate top-N per axis, deduplicated
    top_mort = df.nlargest(n_annotate, "mort_rank")
    top_outcome = df.nlargest(n_annotate, "outcome_rank")
    to_annotate = pd.concat([top_mort, top_outcome]).drop_duplicates()

    from adjustText import adjust_text
    texts = []
    xs, ys = [], []
    for var, row in to_annotate.iterrows():
        xs.append(row["mort_rank"])
        ys.append(row["outcome_rank"])
        texts.append(ax.text(
            row["mort_rank"], row["outcome_rank"],
            row["display"],
            fontsize=8, color="#333333",
        ))
    adjust_text(
        texts, x=xs, y=ys, ax=ax,
        force_text=(1.2, 1.2),
        force_points=(1.5, 1.5),
        expand=(1.3, 1.5),
        arrowprops=dict(arrowstyle="-", color="#999999", lw=0.4),
        ensure_inside_axes=True,
    )

    # Diagonal reference
    ax.plot([0, 1], [0, 1], ls="--", c="#BBBBBB", lw=0.5, zorder=0)

    # Median crosshairs
    ax.axhline(0.5, ls=":", c="#DDDDDD", lw=0.4, zorder=0)
    ax.axvline(0.5, ls=":", c="#DDDDDD", lw=0.4, zorder=0)

    # Spearman rho
    rho, pval = stats.spearmanr(df["mort_shap"], df["outcome_shap"])
    ax.text(
        0.03, 0.60,
        f"$\\rho_s$ = {rho:.2f}\np = {pval:.1e}",
        transform=ax.transAxes, fontsize=FONT_ANNOT * 1.25,
        va="top", ha="left", color="#555555",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1),
    )

    ax.set_xlabel("Mortality importance (percentile rank)", fontsize=FONT_LABEL * 1.25)
    ax.set_ylabel(f"{outcome_label}\nimportance (percentile rank)",
                   fontsize=FONT_LABEL * 1.25)
    ax.set_xlim(-0.03, 1.06)
    ax.set_ylim(-0.03, 1.06)
    ax.tick_params(labelsize=FONT_TICK * 1.25)

    return ax


def plot_dual_scatter(
    mort_shap_path: Path,
    fi_shap_path: Path,
    cog_shap_path: Path,
    output_dir: Path,
):
    """Side-by-side dual scatter: mortality × frailty and mortality × cognition SHAP."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    setup_rcparams()

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df_fi = _load_and_align_shap(mort_shap_path, fi_shap_path)
    df_cog = _load_and_align_shap(mort_shap_path, cog_shap_path)

    _w = PNAS_FULL * 1.25 * 1.25
    _h = (PNAS_FULL * 0.48 + 0.6) * 1.25
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(_w, _h))

    plot_single_scatter(
        df_fi, "Frailty importance (FInoPsy)", ax=ax1, n_annotate=5,
    )
    ax1.set_title("A", fontsize=FONT_TITLE * 1.25, fontweight="bold", loc="left",
                   pad=4)

    plot_single_scatter(
        df_cog, "Cognition importance (BTACT)", ax=ax2, n_annotate=5,
    )
    ax2.set_title("B", fontsize=FONT_TITLE * 1.25, fontweight="bold", loc="left",
                   pad=4)

    # Shared legend below, deduplicated across panels
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    seen = set()
    handles, labels = [], []
    for h, l in list(zip(h1, l1)) + list(zip(h2, l2)):
        if l not in seen:
            seen.add(l)
            handles.append(h)
            labels.append(l)
    if handles:
        fig.legend(
            handles, labels, loc="lower center",
            ncol=min(5, len(handles)), fontsize=(FONT_ANNOT - 0.5) * 1.25,
            frameon=False, bbox_to_anchor=(0.5, -0.03),
            handletextpad=0.3, columnspacing=1.0,
        )

    fig.subplots_adjust(
        left=0.10, right=0.97, bottom=0.20, top=0.95, wspace=0.40,
    )

    for ext in ("pdf", "png"):
        fig.savefig(
            fig_dir / f"mortality_dual_scatter.{ext}",
            format=ext, dpi=300, bbox_inches="tight",
        )
    plt.close(fig)
    print(f"  Saved: figures/mortality_dual_scatter.pdf/.png")

    # Also save individual panels (larger, more room for labels)
    for df_plot, label, fname in [
        (df_fi, "Frailty importance (FInoPsy)", "mortality_vs_frailty_scatter"),
        (df_cog, "Cognition importance (BTACT)", "mortality_vs_cognition_scatter"),
    ]:
        w = PNAS_FULL * 0.58 * 1.25 * 1.25
        fig_s, ax_s = plt.subplots(figsize=(w, w + 0.55 * 1.25))
        plot_single_scatter(df_plot, label, ax=ax_s, n_annotate=7)
        handles, labels = ax_s.get_legend_handles_labels()
        if handles:
            ax_s.legend(
                handles, labels, loc="lower left",
                fontsize=(FONT_ANNOT - 1) * 1.25, frameon=True, ncol=2,
                handletextpad=0.3, columnspacing=0.6,
                fancybox=False, edgecolor="#DDDDDD",
                facecolor="white", framealpha=0.9,
            )
        fig_s.subplots_adjust(left=0.17, bottom=0.12, right=0.97, top=0.97)
        for ext in ("pdf", "png"):
            fig_s.savefig(
                fig_dir / f"{fname}.{ext}",
                format=ext, dpi=300, bbox_inches="tight",
            )
        plt.close(fig_s)
        print(f"  Saved: figures/{fname}.pdf/.png")

    df_fi.to_csv(output_dir / "mortality_vs_frailty_data.csv")
    df_cog.to_csv(output_dir / "mortality_vs_cognition_data.csv")


# Structured outputs and output log

def write_model_outputs(
    cv_result: dict | None,
    output_dir: Path,
    endpoint: str,
    endpoint_summary: dict | None,
) -> None:
    if endpoint_summary:
        pd.DataFrame([endpoint_summary]).to_csv(
            output_dir / "mortality_endpoint_summary.csv", index=False,
        )

    if cv_result:
        pd.DataFrame(
            {
                "endpoint": [endpoint],
                "auc_mean": [cv_result["r2_mean"]],
                "auc_std": [cv_result["r2_std"]],
                "fold_auc": [";".join(f"{v:.6f}" for v in cv_result["r2_folds"])],
            }
        ).to_csv(output_dir / "mortality_cv_metrics.csv", index=False)


def write_pipeline_log(
    args: argparse.Namespace,
    output_dir: Path,
    cv_result: dict | None,
    endpoint_summary: dict | None,
) -> None:
    shap_status = "skipped" if args.no_bootstrap or args.plot_only else "written"
    cv_status = "skipped" if args.no_cv or args.plot_only else ("written" if cv_result else "not written")
    endpoint_status = "skipped" if args.plot_only else ("written" if endpoint_summary else "not written")
    plot_status = "skipped" if args.no_plots else "written if source files available"
    outputs = [
        {"path": "mortality_endpoint_summary.csv", "kind": "CSV", "status": endpoint_status},
        {"path": "mortality_cv_metrics.csv", "kind": "CSV", "status": cv_status},
        {"path": output_dir / "shap" / "mortality_bootstrap_shap.csv", "kind": "CSV", "status": shap_status},
        {"path": output_dir / "shap" / "mortality_features.csv", "kind": "CSV", "status": shap_status},
        {"path": output_dir / "shap_raw" / "mortality_bootstrap_shap_raw.npy", "kind": "NPY", "status": shap_status},
        {"path": output_dir / "shap_raw" / "mortality_robust_shap.csv", "kind": "CSV", "status": shap_status},
        {"path": output_dir / "shap_raw" / "mortality_oob_count.npy", "kind": "NPY", "status": shap_status},
        {"path": output_dir / "figures" / "mortality_dual_scatter.pdf", "kind": "PDF", "status": plot_status},
        {"path": output_dir / "figures" / "mortality_dual_scatter.png", "kind": "PNG", "status": plot_status},
        {"path": output_dir / "figures" / "mortality_vs_frailty_scatter.pdf", "kind": "PDF", "status": plot_status},
        {"path": output_dir / "figures" / "mortality_vs_frailty_scatter.png", "kind": "PNG", "status": plot_status},
        {"path": output_dir / "figures" / "mortality_vs_cognition_scatter.pdf", "kind": "PDF", "status": plot_status},
        {"path": output_dir / "figures" / "mortality_vs_cognition_scatter.png", "kind": "PNG", "status": plot_status},
        {"path": "mortality_vs_frailty_data.csv", "kind": "CSV", "status": plot_status},
        {"path": "mortality_vs_cognition_data.csv", "kind": "CSV", "status": plot_status},
    ]
    run_args = vars(args).copy()
    write_output_log(output_dir, "a11_output_log.md", Path(__file__).name, outputs, run_args)


# Main

def main():
    # Record the inputs this output was built from.
    provenance.stamp_at_exit(OUTPUT_DIR, "a11_mortality_shap", ["data/processed/master_analysis.csv", "data/processed/master_analysis_concurrent.csv", "data/raw/mortality/MIDUS_Core_MortalityCauseData_N2822_20260511.sav", "data/raw/mortality/MIDUS_Refresher_MortalityCauseData_N454_20260511.sav", "data/raw/wave1/M2_P1_SURVEY_N4963_20200720.sav", "data/raw/wave1/MR1_P1_SURVEY_N3577_20250407.sav", "output/cs1_cross_sectional"])
    args = parse_args()
    output_dir = output_dir_for_endpoint(OUTPUT_DIR, args.endpoint)
    output_dir.mkdir(parents=True, exist_ok=True)
    use_gpu = not args.no_gpu

    print("A11: MORTALITY SHAP + LETHALITY-vs-AGING SCATTER")
    print(f"Endpoint: {args.endpoint}")

    cv_result = None
    shap_df = None
    endpoint_summary = None

    if not args.plot_only:
        X, y, features, endpoint_summary = load_mortality_features(
            endpoint=args.endpoint,
            use_gpu=use_gpu,
        )

        if not args.no_cv:
            cv_result = run_mortality_cv(X, y, features, use_gpu=use_gpu)

        if not args.no_bootstrap:
            shap_df, _ = run_mortality_shap(
                X, y, features,
                output_dir=output_dir,
                n_bootstrap=args.n_bootstrap,
                use_gpu=use_gpu,
                n_gpus=args.n_gpus,
            )

        write_model_outputs(cv_result, output_dir, args.endpoint, endpoint_summary)

    # Scatter plots (use saved SHAP if available)
    mort_shap_path = output_dir / "shap" / "mortality_bootstrap_shap.csv"
    fi_shap_path = ap.CS1_DIR / "shap" / "FInoPsy_W1_T1T2_bootstrap_shap.csv"
    cog_shap_path = ap.CS1_DIR / "shap" / "btactComp_W1_T1T2_bootstrap_shap.csv"

    if args.no_plots:
        print("\n  [SKIP] Scatter plots (--no_plots)")
    elif mort_shap_path.exists() and fi_shap_path.exists() and cog_shap_path.exists():
        print("\nSCATTER PLOTS")
        plot_dual_scatter(mort_shap_path, fi_shap_path, cog_shap_path, output_dir)
    else:
        missing = [p for p in [mort_shap_path, fi_shap_path, cog_shap_path]
                    if not p.exists()]
        print(f"\n  [SKIP] Scatter plots: missing SHAP files:")
        for p in missing:
            print(f"    {p}")

    write_pipeline_log(args, output_dir, cv_result, endpoint_summary)
    print(f"\nDone. Results in: {output_dir}")


if __name__ == "__main__":
    main()
