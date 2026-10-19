#!/usr/bin/env python3
"""
A7 - Mortality validation.

Cox PH with tiered delta-C statistics. Mortality data and endpoint semantics
are defined in analysis.core.mortality_data.

Usage:
    uv run python analysis/pipelines/a7_mortality.py
    uv run python analysis/pipelines/a7_mortality.py --endpoint fixed_9y
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.core import provenance

from analysis.core.output_log import write_output_log
from analysis.core.data_loader import get_tier_variables
from analysis.core.mortality_data import (
    ENDPOINTS,
    MORTALITY_SOURCE_LABEL,
    endpoint_config,
    load_survival_data,
    output_dir_for_endpoint,
    summarize_survival_frame,
)
from analysis.core import artifact_paths as ap


A4_DIR = ROOT / "output/a4_longitudinal"
OUTPUT_DIR = ROOT / "output/a7_mortality"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A7 mortality Cox validation")
    p.add_argument(
        "--endpoint",
        choices=sorted(ENDPOINTS),
        default="through_2025",
        help="Mortality endpoint to analyze; primary output uses through_2025.",
    )
    return p.parse_args()


def load_a4_focal_vars(top_k: int = 10) -> list[str]:
    """Load top variables from A4 bootstrap SHAP (union of FInoPsy + BTACT)."""
    focal: list[str] = []
    for outcome in ["FInoPsy_W2", "btactComp_W2"]:
        safe = outcome.replace("/", "_")
        path = ap.A4_DIR / "shap" / f"{safe}_long_bootstrap_shap.csv"
        if path.exists():
            shap_df = pd.read_csv(path, index_col=0)
            w1_col = outcome.replace("_W2", "_W1")
            # Rank explicitly rather than relying on the row order a4 writes.
            ranked = shap_df.sort_values("mean_abs_shap", ascending=False)
            candidates = [v for v in ranked.index if v != w1_col]
            focal.extend(candidates[:top_k])
    return list(dict.fromkeys(focal))


def encode_for_cox(df: pd.DataFrame) -> pd.DataFrame:
    """Factorize object predictors so lifelines receives numeric columns."""
    out = df.copy()
    skip = {
        "UNDERCOD_ICD10",
        "person_id",
        "pair_id",
        "cohort",
        "wave_baseline",
        "sib_type",
        "mortality_endpoint",
        "mortality_endpoint_label",
        "mortality_source_file",
    }
    for col in out.columns:
        if col in skip:
            continue
        if out[col].dtype == object:
            out[col] = pd.factorize(out[col])[0].astype(float)
            out[col] = out[col].replace(-1, np.nan)
    return out


def fit_cox(
    df: pd.DataFrame,
    feature_cols: list[str],
    label: str,
    penalizer: float = 0.01,
) -> dict | None:
    """Fit a Cox PH model and return a compact result dictionary."""
    from lifelines import CoxPHFitter

    available = list(dict.fromkeys(c for c in feature_cols if c in df.columns))
    if not available:
        return None

    surv_cols = available + ["follow_up_years", "deceased"]
    sub = df[surv_cols].copy().dropna(subset=["follow_up_years", "deceased"])

    for c in available:
        if sub[c].isna().any():
            sub[c] = sub[c].fillna(sub[c].median())

    if sub["deceased"].sum() < 10:
        return None

    keep = []
    for c in available:
        sd = float(sub[c].std()) if sub[c].notna().any() else 0.0
        if np.isfinite(sd) and sd > 1e-10:
            keep.append(c)
    if not keep:
        return None

    cph = CoxPHFitter(penalizer=penalizer)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cph.fit(
                sub[keep + ["follow_up_years", "deceased"]],
                duration_col="follow_up_years",
                event_col="deceased",
            )
    except Exception as exc:
        print(f"    [WARN] Cox PH failed for {label}: {exc}")
        return None

    return {
        "label": label,
        "n": len(sub),
        "n_events": int(sub["deceased"].sum()),
        "n_features": len(keep),
        "c_index": cph.concordance_index_,
        "summary": cph.summary,
        "model": cph,
    }


def run_validation(endpoint: str) -> tuple[pd.DataFrame, list[dict]]:
    """Load data and run the tiered Cox models."""
    df = load_survival_data(endpoint=endpoint)
    df = encode_for_cox(df)

    t1_vars = get_tier_variables([1])
    t2_vars = get_tier_variables([2])
    t1t2_vars = get_tier_variables([1, 2])
    a4_vars = load_a4_focal_vars(top_k=10)

    print(f"\n  T1: {len(t1_vars)} vars, T2: {len(t2_vars)} vars")
    print(f"  A4 focal: {len(a4_vars)} vars: {a4_vars[:5]}...")

    models = [
        ("Age only", ["age_W1"]),
        ("T1", ["age_W1"] + t1_vars),
        ("T1+T2", ["age_W1"] + t1t2_vars),
        ("T1+A4top", ["age_W1"] + t1_vars + a4_vars),
    ]

    results: list[dict] = []
    for label, cols in models:
        result = fit_cox(df, cols, label)
        if result:
            results.append(result)
            print(
                f"  {label}: C={result['c_index']:.3f} "
                f"(n={result['n']:,}, events={result['n_events']:,})"
            )

    return df, results


def write_outputs(
    df: pd.DataFrame,
    results: list[dict],
    endpoint: str,
    output_dir: Path,
) -> None:
    """Write A7 CSV outputs and the output log for one endpoint."""
    cfg = endpoint_config(endpoint)
    output_dir.mkdir(parents=True, exist_ok=True)
    r_dict = {r["label"]: r for r in results}
    surv_summary = summarize_survival_frame(df)

    c_table = pd.DataFrame(
        [
            {
                "endpoint": endpoint,
                "model": r["label"],
                "n": r["n"],
                "n_events": r["n_events"],
                "n_features": r["n_features"],
                "c_index": r["c_index"],
                "mortality_file": MORTALITY_SOURCE_LABEL,
                "censor_year": cfg.censor_year,
                "horizon_years": cfg.horizon_years,
            }
            for r in results
        ]
    )
    c_table.to_csv(output_dir / "a7_c_index_table.csv", index=False)
    pd.DataFrame([surv_summary]).to_csv(output_dir / "a7_endpoint_summary.csv", index=False)

    has_t1t2 = "T1+T2" in r_dict
    if has_t1t2:
        cox = r_dict["T1+T2"]["summary"]
        cox.to_csv(output_dir / "a7_cox_t1t2_summary.csv", index_label="covariate")

        hr = cox[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].copy()
        hr = hr.sort_values("p")
        hr.to_csv(output_dir / "a7_hazard_ratios.csv", index_label="covariate")

    outputs = [
        {"path": "a7_c_index_table.csv", "kind": "CSV"},
        {"path": "a7_endpoint_summary.csv", "kind": "CSV"},
        {"path": "a7_cox_t1t2_summary.csv", "kind": "CSV", "status": "written" if has_t1t2 else "not written"},
        {"path": "a7_hazard_ratios.csv", "kind": "CSV", "status": "written" if has_t1t2 else "not written"},
    ]
    write_output_log(output_dir, "a7_output_log.md", Path(__file__).name, outputs, {"endpoint": endpoint})


def main() -> None:
    # Record the inputs this output was built from.
    provenance.stamp_at_exit(OUTPUT_DIR, "a7_mortality_primary", ["data/processed/master_analysis.csv", "data/processed/master_analysis_concurrent.csv", "data/raw/mortality/MIDUS_Core_MortalityCauseData_N2822_20260511.sav", "data/raw/mortality/MIDUS_Refresher_MortalityCauseData_N454_20260511.sav", "data/raw/wave1/M2_P1_SURVEY_N4963_20200720.sav", "data/raw/wave1/MR1_P1_SURVEY_N3577_20250407.sav", "output/a4_longitudinal"])
    args = parse_args()
    output_dir = output_dir_for_endpoint(OUTPUT_DIR, args.endpoint)

    print("A7 - MORTALITY VALIDATION")
    print(f"Endpoint: {args.endpoint}")

    df, results = run_validation(args.endpoint)
    write_outputs(df, results, args.endpoint, output_dir)
    print(f"Done. Results in: {output_dir}")


if __name__ == "__main__":
    main()
