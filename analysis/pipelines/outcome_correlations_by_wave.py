#!/usr/bin/env python3
"""Table S15: correlations among the age-residualized aging outcomes at
baseline (Wave 1) and 9-year follow-up (Wave 2), with 95% CIs.

Convention matches d1_descriptive.py: each outcome is age-residualized (OLS on
the participant's age at that wave -- W1 on age_W1, W2 on age_W2), then Pearson
correlations are computed on pairwise-complete rows. 95% CIs use the Fisher
z-transform (se = 1/sqrt(n-3)). W1 values reproduce output/d1_descriptive/
outcome_correlations.csv as a consistency check.

Run:  uv run python analysis/pipelines/outcome_correlations_by_wave.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.core.artifact_paths import AL_STEM  # noqa: E402
MASTER = ROOT / "data" / "processed" / "master_analysis.csv"
OUT_D1 = ROOT / "output" / "d1_descriptive"
OUT = ROOT / "output" / "outcome_correlations_by_wave"

OUTCOMES = [("FInoPsy", "Frailty Index"),
            ("btactComp", "Cognitive Composite"),
            (AL_STEM, "Allostatic Load")]
WAVES = [("W1", "Baseline (Wave 1)", "age_W1"),
         ("W2", "9-year follow-up (Wave 2)", "age_W2")]


def age_residualize(values: np.ndarray, age: np.ndarray) -> np.ndarray:
    """OLS-residualize values on age; NaN preserved (matches d1_descriptive)."""
    values = values.astype(float)
    valid = ~np.isnan(values) & ~np.isnan(age)
    if valid.sum() < 10:
        return values
    reg = LinearRegression().fit(age[valid].reshape(-1, 1), values[valid])
    out = np.full_like(values, np.nan)
    out[valid] = values[valid] - reg.predict(age[valid].reshape(-1, 1))
    return out


def fisher_ci(r: float, n: int, alpha: float = 0.05):
    if n < 4 or not np.isfinite(r) or abs(r) >= 1:
        return (np.nan, np.nan)
    z, se = np.arctanh(r), 1.0 / np.sqrt(n - 3)
    zc = stats.norm.ppf(1 - alpha / 2)
    return float(np.tanh(z - zc * se)), float(np.tanh(z + zc * se))


def main():
    df = pd.read_csv(MASTER, low_memory=False)
    rows = []
    for wcode, wlabel, agecol in WAVES:
        age = df[agecol].values.astype(float)
        res = {}
        for stem, label in OUTCOMES:
            col = f"{stem}_{wcode}"
            if col not in df.columns:
                print(f"  [WARN] missing {col}")
                continue
            res[label] = age_residualize(df[col].values, age)
        res = pd.DataFrame(res)
        labels = list(res.columns)
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a, b = labels[i], labels[j]
                mask = res[a].notna() & res[b].notna()
                n = int(mask.sum())
                if n > 5:
                    r, p = stats.pearsonr(res.loc[mask, a], res.loc[mask, b])
                    lo, hi = fisher_ci(r, n)
                else:
                    r = p = lo = hi = np.nan
                rows.append({"wave": wcode, "wave_label": wlabel,
                             "outcome_1": a, "outcome_2": b,
                             "r": round(r, 3), "ci_lo": round(lo, 3),
                             "ci_hi": round(hi, 3), "n": n, "p": p})

    tidy = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    tidy_path = OUT / "outcome_correlations_by_wave.csv"
    tidy.to_csv(tidy_path, index=False)

    # Publication-formatted "r [95% CI] (n)" long table
    def cell(row):
        if not np.isfinite(row["r"]):
            return "-"
        return f"{row['r']:+.2f} [{row['ci_lo']:+.2f}, {row['ci_hi']:+.2f}] (n={row['n']})"
    tidy["formatted"] = tidy.apply(cell, axis=1)
    fmt = tidy.pivot_table(index=["outcome_1", "outcome_2"], columns="wave_label",
                           values="formatted", aggfunc="first")
    fmt_path = OUT / "outcome_correlations_by_wave_formatted.csv"
    fmt.to_csv(fmt_path)

    print("\nTable S15: age-residualized outcome correlations by wave\n")
    print(tidy[["wave", "outcome_1", "outcome_2", "r", "ci_lo", "ci_hi", "n"]].to_string(index=False))
    print("\n--- formatted r [95% CI] (n) ---\n")
    print(fmt.to_string())

    # Consistency check against d1_descriptive's outcome correlations (W1)
    s4 = OUT_D1 / "outcome_correlations.csv"
    if s4.exists():
        m = pd.read_csv(s4, index_col=0)

        def _ref(a: str, b: str) -> float:
            return m.loc[a, b]

        print("\nW1 consistency check vs d1_descriptive")
        for _, rr in tidy[tidy.wave == "W1"].iterrows():
            ref = _ref(rr["outcome_1"], rr["outcome_2"])
            ok = abs(ref - rr["r"]) < 0.005
            print(f"  {rr['outcome_1']} x {rr['outcome_2']}: S15={rr['r']:+.3f} "
                  f"d1={ref:+.3f} {'OK' if ok else 'MISMATCH'}")

    print(f"\nSaved: {tidy_path}\nSaved: {fmt_path}")


if __name__ == "__main__":
    main()
