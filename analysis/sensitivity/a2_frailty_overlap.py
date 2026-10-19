#!/usr/bin/env python3
"""
A2: frailty construct-overlap audit

Determines how much FInoPsy Tier 2 ΔR² is inflated by content overlap
between Tier 2 predictors and FInoPsy items.

Reads the manual overlap audit CSV, excludes overlapping variables, and
re-runs FInoPsy XGBoost to compute adjusted ΔR².

Usage:
    uv run python analysis/sensitivity/a2_frailty_overlap.py

Output: output/a2_frailty_overlap/
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.core.data_loader import load_master, get_tier_variables
from analysis.core import age_method
from analysis.core import provenance
from analysis.core.overlap_sources import FRAILTY_OVERLAP_AUDIT_CSV, get_overlap_excluded_vars
from analysis.core.xgboost_pipeline import XGBoostPipeline
from analysis.core.output_log import write_output_log

AUDIT_CSV = FRAILTY_OVERLAP_AUDIT_CSV
OUTPUT_DIR = ROOT / "output/a2_frailty_overlap"


def run_model(
    X: pd.DataFrame,
    y: pd.Series,
    features: list[str],
    age: np.ndarray,
    label: str,
    n_folds: int = 5,
) -> dict:
    """Run XGBoost CV and return R² summary."""
    print(f"\n{'-'*50}")
    print(f"Model: {label} ({len(features)} features)")
    pipe = XGBoostPipeline(n_folds=n_folds)
    cv = pipe.run_cv(X[features], y, features, age=age)
    return {
        "model": label,
        "n_features": len(features),
        "r2_mean": cv["r2_mean"],
        "r2_std": cv["r2_std"],
    }


def main():
    # Write provenance.json and age_adjustment_method.csv on exit.
    provenance.stamp_at_exit(OUTPUT_DIR, "a2_frailty_overlap", ["data/processed/master_analysis.csv", "data/processed/master_analysis_concurrent.csv"])
    age_method.emit_at_exit(OUTPUT_DIR, "a2_frailty_overlap")
    parser = argparse.ArgumentParser(description="A2 Frailty Overlap Audit")
    parser.add_argument("--n_folds", type=int, default=5)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X_full, y, features_full, age = load_master(
        "FInoPsy_W1", tiers=[1, 2], age_residualize=False
    )

    t1_vars = get_tier_variables([1])
    t1_features = [f for f in t1_vars if f in X_full.columns]

    overlap_ab = get_overlap_excluded_vars("ab")
    overlap_a_only = get_overlap_excluded_vars("a")

    adjusted_ab_features = [f for f in features_full if f not in overlap_ab]
    adjusted_a_features = [f for f in features_full if f not in overlap_a_only]

    print("A2 frailty construct-overlap audit")
    print(f"Total T2 variables: {len(features_full) - len(t1_features)}")
    print(f"Code (a) direct overlap: {len(overlap_a_only)} vars excluded")
    print(f"Code (a)+(b) overlap: {len(overlap_ab)} vars excluded")
    print(f"Adjusted T1+T2 features (a-only): {len(adjusted_a_features)}")
    print(f"Adjusted T1+T2 features (a+b): {len(adjusted_ab_features)}")

    # Models: T1, T1+T2, minus (a), minus (a)+(b)
    results = []
    results.append(run_model(X_full, y, t1_features, age, "T1_only", args.n_folds))

    results.append(
        run_model(X_full, y, features_full, age, "T1T2_full", args.n_folds)
    )

    results.append(
        run_model(
            X_full, y, adjusted_a_features, age,
            "T1T2_excl_a", args.n_folds,
        )
    )

    results.append(
        run_model(
            X_full, y, adjusted_ab_features, age,
            "T1T2_excl_ab", args.n_folds,
        )
    )

    r2_t1 = results[0]["r2_mean"]
    r2_full = results[1]["r2_mean"]
    r2_excl_a = results[2]["r2_mean"]
    r2_excl_ab = results[3]["r2_mean"]

    delta_full = r2_full - r2_t1
    delta_excl_a = r2_excl_a - r2_t1
    delta_excl_ab = r2_excl_ab - r2_t1

    for r in results:
        r["delta_r2"] = r["r2_mean"] - r2_t1

    df = pd.DataFrame(results)
    csv_path = OUTPUT_DIR / "a2_adjusted_r2.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    atten_a = (delta_full - delta_excl_a) / delta_full * 100 if delta_full > 0 else 0
    atten_ab = (delta_full - delta_excl_ab) / delta_full * 100 if delta_full > 0 else 0

    lines = [
        "# A2 — Frailty Construct Overlap Audit Results",
        f"*Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        "",
        "## Overlap Coding Summary",
        "",
        f"- Total Tier 2 variables: {len(features_full) - len(t1_features)}",
        f"- Code (a) direct overlap: {len(overlap_a_only)} vars",
        f"  - {', '.join(overlap_a_only)}",
        f"- Code (b) definitionally proximal: {len(overlap_ab) - len(overlap_a_only)} vars",
        f"  - {', '.join(v for v in overlap_ab if v not in overlap_a_only)}",
        f"- Code (c) distal: {len(features_full) - len(t1_features) - len(overlap_ab)} vars",
        "",
        "## Tiered ΔR² Results",
        "",
        "| Model | Features | R² (mean±SD) | ΔR²_T2 |",
        "|-------|----------|-------------|--------|",
        f"| T1 only | {results[0]['n_features']} | {results[0]['r2_mean']:.4f} ± {results[0]['r2_std']:.4f} | — |",
        f"| T1+T2 (unadjusted) | {results[1]['n_features']} | {results[1]['r2_mean']:.4f} ± {results[1]['r2_std']:.4f} | **{delta_full:+.4f}** |",
        f"| T1+T2 excl (a) | {results[2]['n_features']} | {results[2]['r2_mean']:.4f} ± {results[2]['r2_std']:.4f} | {delta_excl_a:+.4f} |",
        f"| T1+T2 excl (a)+(b) | {results[3]['n_features']} | {results[3]['r2_mean']:.4f} ± {results[3]['r2_std']:.4f} | {delta_excl_ab:+.4f} |",
        "",
        "## Attenuation",
        "",
        f"- Excluding code (a) only: ΔR² attenuated by **{atten_a:.1f}%** ({delta_full:.4f} → {delta_excl_a:.4f})",
        f"- Excluding code (a)+(b): ΔR² attenuated by **{atten_ab:.1f}%** ({delta_full:.4f} → {delta_excl_ab:.4f})",
        "",
        "## Interpretation",
        "",
        "Attenuation quantifies how much of the original ΔR² was due to",
        "content overlap between Tier 2 predictors and FInoPsy items.",
        "The adjusted ΔR² represents the information"
        "carried by psychosocial variables beyond item-level overlap.",
        "",
        f"Source: `config/frailty_overlap_audit.csv`",
    ]

    # Summary goes into the output log as notes.
    write_output_log(
        OUTPUT_DIR, "a2_output_log.md", Path(__file__).name,
        [
            {"path": "a2_adjusted_r2.csv", "kind": "CSV"},
            {"path": "age_adjustment_method.csv", "kind": "CSV"},
        ],
        notes=lines,
    )

    print("\nResults")
    print(f"  Unadjusted ΔR²_T2:       {delta_full:+.4f}")
    print(f"  Adjusted ΔR²_T2 (a):     {delta_excl_a:+.4f}  (attenuation: {atten_a:.1f}%)")
    print(f"  Adjusted ΔR²_T2 (a+b):   {delta_excl_ab:+.4f}  (attenuation: {atten_ab:.1f}%)")


if __name__ == "__main__":
    main()
