#!/usr/bin/env python3
"""Frailty index without the loneliness item: score and model sensitivity.

The canonical MIDUS frailty key keeps `lonely` as a non-psychiatric item. This
step recomputes the index without it and refits the Tier 1 and Tier 1+2
cross-sectional models, reporting how much the score and Delta R2 change.

Run from project root:
    uv run python analysis/sensitivity/frailty_lonely_sensitivity.py
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.core.data_loader import load_master  # noqa: E402
from analysis.core.xgboost_pipeline import XGBoostPipeline  # noqa: E402

OUT = ROOT / "output/frailty_lonely_sensitivity"
SUPP = ROOT / "manuscript/supplement"


def _fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NA"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.{digits}f}"


def compute_no_lonely_scores() -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    wave_files = {
        "W1": ROOT / "data/processed/wave1/pooled_frailty_M2R.csv",
        "W2": ROOT / "data/processed/wave2/pooled_frailty_M3MR2.csv",
    }
    score_rows: list[dict[str, object]] = []
    score_series: dict[str, pd.Series] = {}

    for wave, path in wave_files.items():
        df = pd.read_csv(path)
        # Items are every column but the id and the FInoPsy* bookkeeping columns. The
        # primary score is the row mean over the items a cohort collected, as in
        # master_analysis.csv; the pooled file's own FInoPsy column is the
        # 38-shared-item variant, so it is not the comparator at wave 2.
        item_cols = [c for c in df.columns if c != "id" and not c.startswith("FInoPsy")]
        no_lonely_cols = [c for c in item_cols if c != "lonely"]
        no_lonely = df[no_lonely_cols].mean(axis=1, skipna=True)
        primary = df[item_cols].mean(axis=1, skipna=True)
        lonely = df["lonely"].astype(float)
        valid = primary.notna() & no_lonely.notna()
        diff = primary[valid] - no_lonely[valid]
        score_rows.append(
            {
                "wave": wave,
                "source_file": str(path.relative_to(ROOT)),
                "n_primary_nonmissing": int(primary.notna().sum()),
                "n_no_lonely_nonmissing": int(no_lonely.notna().sum()),
                "n_overlap": int(valid.sum()),
                "primary_mean": float(primary[valid].mean()),
                "no_lonely_mean": float(no_lonely[valid].mean()),
                "mean_difference_primary_minus_no_lonely": float(diff.mean()),
                "median_abs_difference": float(diff.abs().median()),
                "max_abs_difference": float(diff.abs().max()),
                "pearson_r": float(primary[valid].corr(no_lonely[valid], method="pearson")),
                "spearman_r": float(primary[valid].corr(no_lonely[valid], method="spearman")),
                "lonely_mean": float(lonely[valid].mean()),
                "item_count_primary": int(df[item_cols].notna().sum(axis=1).max()),
                "item_count_primary_min": int(df[item_cols].notna().sum(axis=1).min()),
                "item_count_no_lonely": len(no_lonely_cols),
            }
        )
        score_series[wave] = pd.Series(no_lonely.to_numpy(), index=df["id"], name=f"FInoPsyNoLonely_{wave}")

    return pd.DataFrame(score_rows), score_series


def run_cross_sectional_model(no_lonely_w1: pd.Series, n_folds: int, tuned_model: bool) -> pd.DataFrame:
    master = pd.read_csv(ROOT / "data/processed/master_analysis.csv", low_memory=False)
    master["FInoPsyNoLonely_W1"] = master["M2MRID"].map(no_lonely_w1.to_dict())

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        master.to_csv(tmp_path, index=False)
        rows = []
        for outcome, outcome_label in [
            ("FInoPsy_W1", "canonical"),
            ("FInoPsyNoLonely_W1", "excluding_lonely"),
        ]:
            for tier_label, tiers in [("T1_only", [1]), ("T1T2", [1, 2])]:
                X, y, features, age = load_master(
                    outcome,
                    tiers=tiers,
                    age_residualize=False,
                    data_path=tmp_path,
                )
                pipe = XGBoostPipeline(n_folds=n_folds, tune=tuned_model)
                cv = pipe.run_cv(X, y, features, age=age)
                rows.append(
                    {
                        "model": f"{outcome_label}_{tier_label}",
                        "outcome": outcome,
                        "outcome_label": outcome_label,
                        "tier_config": tier_label,
                        "estimator": "tuned_xgboost" if tuned_model else "fixed_xgboost",
                        "n": int(y.notna().sum()),
                        "n_features": len(features),
                        "r2_mean": cv["r2_mean"],
                        "r2_std": cv["r2_std"],
                    }
                )
    finally:
        os.unlink(tmp_path)

    model_df = pd.DataFrame(rows)
    model_df["delta_r2"] = np.nan
    for outcome in model_df["outcome"].unique():
        baseline = float(
            model_df[
                model_df["outcome"].eq(outcome)
                & model_df["tier_config"].eq("T1_only")
            ]["r2_mean"].iloc[0]
        )
        model_df.loc[model_df["outcome"].eq(outcome), "delta_r2"] = model_df.loc[
            model_df["outcome"].eq(outcome), "r2_mean"
        ] - baseline

    primary = pd.read_csv(ROOT / "output/cs1_cross_sectional/cs1_r2_table.csv")
    primary_frail = primary[primary["outcome"].eq("FInoPsy_W1")].copy()
    primary_frail["model"] = primary_frail["tier_config"].map({"T1": "T1_only_primary", "T1T2": "T1T2_primary"})
    primary_frail["outcome_label"] = "canonical"
    primary_frail = primary_frail.rename(columns={"tier_config": "source_tier_config"})
    primary_frail["delta_r2"] = primary_frail["r2_mean"] - float(
        primary_frail.loc[primary_frail["source_tier_config"].eq("T1"), "r2_mean"].iloc[0]
    )
    primary_frail["outcome"] = "FInoPsy_W1"
    primary_frail["estimator"] = "canonical_cs1_tuned_xgboost"
    primary_frail["tier_config"] = primary_frail["source_tier_config"].replace({"T1": "T1_only"})
    primary_frail = primary_frail[["model", "outcome", "outcome_label", "tier_config", "estimator", "n", "n_features", "r2_mean", "r2_std", "delta_r2"]]

    return pd.concat([primary_frail, model_df], ignore_index=True)


def write_summary(score_df: pd.DataFrame, model_df: pd.DataFrame, skip_model: bool, tuned_model: bool) -> None:
    w1 = score_df[score_df["wave"].eq("W1")].iloc[0]
    w2 = score_df[score_df["wave"].eq("W2")].iloc[0]
    lines = [
        "# FInoPsy Excluding-Lonely Sensitivity",
        "",
        "`lonely` is retained in the canonical MIDUS frailty key, but it is",
        "psychosocially adjacent. This sensitivity recomputes FInoPsy after",
        "dropping `lonely` from the denominator and reports score-level and",
        "primary cross-sectional predictive checks.",
        "",
        "## Score-Level Sensitivity",
        "",
        "| Wave | N overlap | Primary mean | No-lonely mean | Mean difference | Pearson r | Spearman r |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| W1 | {int(w1['n_overlap'])} | {_fmt(w1['primary_mean'])} | {_fmt(w1['no_lonely_mean'])} | {_fmt(w1['mean_difference_primary_minus_no_lonely'])} | {_fmt(w1['pearson_r'])} | {_fmt(w1['spearman_r'])} |",
        f"| W2 | {int(w2['n_overlap'])} | {_fmt(w2['primary_mean'])} | {_fmt(w2['no_lonely_mean'])} | {_fmt(w2['mean_difference_primary_minus_no_lonely'])} | {_fmt(w2['pearson_r'])} | {_fmt(w2['spearman_r'])} |",
        "",
    ]
    if skip_model:
        lines.extend(
            [
                "## Predictive Sensitivity",
                "",
                "Predictive sensitivity was skipped for this run.",
            ]
        )
    else:
        canonical_ref = model_df[model_df["estimator"].eq("canonical_cs1_tuned_xgboost")]
        fast = model_df[model_df["estimator"].eq("tuned_xgboost" if tuned_model else "fixed_xgboost")]
        primary_t1 = canonical_ref[canonical_ref["tier_config"].eq("T1_only")].iloc[0]
        primary_t1t2 = canonical_ref[canonical_ref["tier_config"].eq("T1T2")].iloc[0]
        fixed_can_t1 = fast[fast["model"].eq("canonical_T1_only")].iloc[0]
        fixed_can_t1t2 = fast[fast["model"].eq("canonical_T1T2")].iloc[0]
        no_t1 = fast[fast["model"].eq("excluding_lonely_T1_only")].iloc[0]
        no_t1t2 = fast[fast["model"].eq("excluding_lonely_T1T2")].iloc[0]
        r2_shift = float(no_t1t2["r2_mean"] - fixed_can_t1t2["r2_mean"])
        delta_shift = float(no_t1t2["delta_r2"] - fixed_can_t1t2["delta_r2"])
        lines.extend(
            [
                "## Primary Cross-Sectional Predictive Sensitivity",
                "",
                "The no-lonely contamination check uses "
                + ("the same tuned XGBoost grid as the canonical model." if tuned_model else "fixed-parameter XGBoost for a fast QA sensitivity; use `--tuned-model` for a heavier rerun."),
                "",
                "| Estimator | Outcome | Model | N | Features | R2 | Delta R2 |",
                "|---|---|---|---:|---:|---:|---:|",
                f"| canonical tuned reference | Canonical FInoPsy | T1 only | {int(primary_t1['n'])} | {int(primary_t1['n_features'])} | {_fmt(primary_t1['r2_mean'])} | {_fmt(primary_t1['delta_r2'])} |",
                f"| canonical tuned reference | Canonical FInoPsy | T1+T2 | {int(primary_t1t2['n'])} | {int(primary_t1t2['n_features'])} | {_fmt(primary_t1t2['r2_mean'])} | {_fmt(primary_t1t2['delta_r2'])} |",
                f"| {'tuned' if tuned_model else 'fixed'} QA check | Canonical FInoPsy | T1 only | {int(fixed_can_t1['n'])} | {int(fixed_can_t1['n_features'])} | {_fmt(fixed_can_t1['r2_mean'])} | {_fmt(fixed_can_t1['delta_r2'])} |",
                f"| {'tuned' if tuned_model else 'fixed'} QA check | Canonical FInoPsy | T1+T2 | {int(fixed_can_t1t2['n'])} | {int(fixed_can_t1t2['n_features'])} | {_fmt(fixed_can_t1t2['r2_mean'])} | {_fmt(fixed_can_t1t2['delta_r2'])} |",
                f"| {'tuned' if tuned_model else 'fixed'} QA check | Excluding lonely | T1 only | {int(no_t1['n'])} | {int(no_t1['n_features'])} | {_fmt(no_t1['r2_mean'])} | {_fmt(no_t1['delta_r2'])} |",
                f"| {'tuned' if tuned_model else 'fixed'} QA check | Excluding lonely | T1+T2 | {int(no_t1t2['n'])} | {int(no_t1t2['n_features'])} | {_fmt(no_t1t2['r2_mean'])} | {_fmt(no_t1t2['delta_r2'])} |",
                "",
                f"Within the same {'tuned' if tuned_model else 'fixed'} QA estimator, dropping `lonely` shifts T1+T2 R2 by {_fmt(r2_shift)} and Delta R2 by {_fmt(delta_shift)}. The fixed-estimator Delta R2 is a sensitivity check; the reported estimate is the tuned cs1_cross_sectional value.",
            ]
        )
    lines.extend(
        [
            "",
            "Source files:",
            "",
            "- `data/processed/wave1/pooled_frailty_M2R.csv`",
            "- `data/processed/wave2/pooled_frailty_M3MR2.csv`",
            "- `data/processed/master_analysis.csv`",
            "",
            "Regeneration command: `uv run python analysis/sensitivity/frailty_lonely_sensitivity.py`.",
        ]
    )
    (SUPP / f"finopsy_excluding_lonely_sensitivity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument(
        "--tuned-model",
        action="store_true",
        help="Use the same tuned XGBoost grid as canonical CS-1; slower, intended as a heavy rerun.",
    )
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    SUPP.mkdir(parents=True, exist_ok=True)

    score_df, score_series = compute_no_lonely_scores()
    score_out = OUT / f"finopsy_excluding_lonely_score_sensitivity.csv"
    score_supp = SUPP / f"finopsy_excluding_lonely_score_sensitivity.csv"
    score_df.to_csv(score_out, index=False)
    score_df.to_csv(score_supp, index=False)
    print(f"Wrote {score_out}")
    print(f"Wrote {score_supp}")

    if args.skip_model:
        model_df = pd.DataFrame()
    else:
        model_df = run_cross_sectional_model(
            score_series["W1"],
            n_folds=args.n_folds,
            tuned_model=args.tuned_model,
        )
        model_out = OUT / f"finopsy_excluding_lonely_model_sensitivity.csv"
        model_supp = SUPP / f"finopsy_excluding_lonely_model_sensitivity.csv"
        model_df.to_csv(model_out, index=False)
        model_df.to_csv(model_supp, index=False)
        print(f"Wrote {model_out}")
        print(f"Wrote {model_supp}")

    write_summary(score_df, model_df, skip_model=args.skip_model, tuned_model=args.tuned_model)
    print(f"Wrote {SUPP / f'finopsy_excluding_lonely_sensitivity.md'}")


if __name__ == "__main__":
    main()
