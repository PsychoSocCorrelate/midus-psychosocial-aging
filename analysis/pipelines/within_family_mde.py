"""Within-family minimum detectable effect and power.

Reads the within-family sibling tables on both layers:
  - change,     all pairs : a6_All_paired_{outcome}.csv
  - change,     MZ subset : a6_MZ_sensitivity_{outcome}.csv
  - concurrent, all pairs : a6_cs_All_paired_{outcome}.csv
  - concurrent, MZ subset : a6_cs_MZ_sensitivity_{outcome}.csv

For each predictor (within-family / discordant-pair slope) compute the
standardized minimum detectable effect at 80% power, two-sided alpha .05:

    MDE_std = 2.802 * SE_std
    SE_std  = beta_W_se * (beta_W_std / beta_W)        [undefined only at beta_W = 0]

2.802 = z_{0.975} + z_{0.80} = 1.95996 + 0.84162.

Observed |z| for the within-family slope is beta_W / beta_W_se (invariant to
standardization). Observed power at the observed effect:
    power = Phi(|z| - 1.95996) + Phi(-|z| - 1.95996)
Power >= .80 requires |z| >= 2.802.

Pairs needed for a target MDE_std of 0.10 (SE scales as 1/sqrt(n_pairs)):
    n_needed = n_pairs * (MDE_std_observed / 0.10)**2

Writes one pair of files per layer: within_family_mde.csv and
within_family_mde_summary.csv for change, within_family_mde_concurrent.csv and
within_family_mde_concurrent_summary.csv for concurrent. Prints per-outcome
medians (All vs MZ) for each.

Also writes within_family_mde_reference.csv, the reference the change-layer MDE
is read against: per sample and outcome, the median change MDE beside the median
|beta_W_std| of the concurrent within-family associations surviving FDR
(beta_W_q <= .05), and their ratio. Concurrent survivors are selected on
significance, so their median overstates the typical concurrent effect and the
ratio understates how far the change MDE sits from it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.core.artifact_paths import A6_DIR, AL_STEM, OUTPUT  # noqa: E402

OUT_DIR = OUTPUT / "within_family_mde"
OUTCOMES = [AL_STEM, "btactComp", "FInoPsy"]
OUTCOME_LABEL = {AL_STEM: "Allostatic Load",
                 "btactComp": "Cognition",
                 "FInoPsy": "Frailty"}

Z_ALPHA = norm.ppf(0.975)          # 1.95996
Z_POWER = norm.ppf(0.80)           # 0.84162
K_MDE = Z_ALPHA + Z_POWER          # 2.80158
TARGET_MDE = 0.10
FDR_Q = 0.05                       # Benjamini-Hochberg threshold, as in a6


def load_layer(prefix: str, sample: str) -> pd.DataFrame:
    frames = []
    for outcome in OUTCOMES:
        df = pd.read_csv(A6_DIR / f"{prefix}_{outcome}.csv")
        df = df[["focal_var", "n_pairs", "beta_W", "beta_W_se", "beta_W_p",
                 "beta_W_std", "sd_x_within"]].copy()
        df["outcome_code"] = outcome
        df["outcome"] = OUTCOME_LABEL[outcome]
        df["sample"] = sample
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Add the MDE, power and pairs-needed columns to a loaded layer."""
    beta = df["beta_W"].values.astype(float)
    se = df["beta_W_se"].values.astype(float)
    bstd = df["beta_W_std"].values.astype(float)

    # Standardization factor beta_W_std / beta_W, which is sd_x_within / sd_y_within
    # whatever the size of beta_W, so it is undefined only at beta_W = 0. An absolute
    # floor on |beta_W| would wrongly drop predictors on large scales such as household income.
    with np.errstate(divide="ignore", invalid="ignore"):
        sf = np.where(beta != 0, bstd / beta, np.nan)
    se_std = se * np.abs(sf)
    mde_std = K_MDE * se_std

    obs_z = np.where(se > 0, np.abs(beta) / se, np.nan)
    obs_power = norm.cdf(obs_z - Z_ALPHA) + norm.cdf(-obs_z - Z_ALPHA)

    n_pairs = df["n_pairs"].values.astype(float)

    df = df.copy()
    df["SE_std"] = se_std
    df["MDE_std"] = mde_std
    df["obs_z"] = obs_z
    df["obs_power"] = obs_power
    df["power_ge_80"] = obs_power >= 0.80
    df["n_needed_mde010"] = n_pairs * (mde_std / TARGET_MDE) ** 2
    df["beta_guarded"] = beta == 0
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(["sample", "outcome_code"])
              .apply(lambda g: pd.Series({
                  "n_predictors": len(g),
                  "n_pairs_min": int(g["n_pairs"].min()),
                  "n_pairs_max": int(g["n_pairs"].max()),
                  "median_MDE_std": np.nanmedian(g["MDE_std"]),
                  "n_power_ge_80": int(g["power_ge_80"].sum()),
                  "median_n_needed_mde010": np.nanmedian(g["n_needed_mde010"]),
              }), include_groups=False)
              .reset_index())


def report(summ: pd.DataFrame) -> None:
    # Sort outcomes in OUTCOMES order
    order = {o: i for i, o in enumerate(OUTCOMES)}
    summ = summ.sort_values(["sample", "outcome_code"],
                            key=lambda s: s.map(order).fillna(s))
    for _, r in summ.iterrows():
        print(f"  {r['sample']:>3} {r['outcome_code']:>10}: "
              f"median MDE_std={r['median_MDE_std']:.3f} | "
              f"n_pairs {int(r['n_pairs_min'])}-{int(r['n_pairs_max'])} | "
              f"power>=.80: {int(r['n_power_ge_80'])}/{int(r['n_predictors'])} | "
              f"median pairs needed for MDE_std=0.10: {r['median_n_needed_mde010']:.0f}")


def reference(change_summ: pd.DataFrame) -> pd.DataFrame:
    """Change-layer median MDE against the concurrent FDR-surviving effects."""
    rows = []
    for sample, prefix in (("All", "a6_cs_All_paired"), ("MZ", "a6_cs_MZ_sensitivity")):
        for outcome in OUTCOMES:
            cs = pd.read_csv(A6_DIR / f"{prefix}_{outcome}.csv")
            surv = cs.loc[cs["beta_W_q"] <= FDR_Q, "beta_W_std"].abs()
            mde = change_summ.loc[(change_summ["sample"] == sample)
                                  & (change_summ["outcome_code"] == outcome),
                                  "median_MDE_std"].iloc[0]
            cs_med = surv.median() if len(surv) else np.nan
            rows.append({"sample": sample, "outcome_code": outcome,
                         "outcome": OUTCOME_LABEL[outcome],
                         "change_median_MDE_std": mde,
                         "cs_n_fdr_surviving": len(surv),
                         "cs_median_abs_beta_W_std_surviving": cs_med,
                         "change_mde_over_cs_surviving": mde / cs_med})
    return pd.DataFrame(rows)


def run_layer(all_prefix: str, mz_prefix: str, stem: str, label: str) -> pd.DataFrame:
    df = compute(pd.concat(
        [load_layer(all_prefix, "All"), load_layer(mz_prefix, "MZ")],
        ignore_index=True,
    ))

    out_cols = ["outcome", "outcome_code", "sample", "focal_var", "n_pairs",
                "beta_W", "beta_W_se", "beta_W_std", "sd_x_within",
                "SE_std", "MDE_std", "obs_z", "obs_power", "power_ge_80",
                "n_needed_mde010", "beta_W_p", "beta_guarded"]
    out = (df[out_cols].sort_values(["sample", "outcome", "MDE_std"])
                       .reset_index(drop=True))
    out_path = OUT_DIR / f"{stem}.csv"
    out.to_csv(out_path, index=False)

    summ = summarize(df)
    summ_path = OUT_DIR / f"{stem}_summary.csv"
    summ.to_csv(summ_path, index=False)

    print(f"Wrote {out_path} ({len(out)} predictor rows)")
    print(f"Wrote {summ_path}")
    print(f"--- {label}: median standardized MDE and power ---")
    report(summ)
    return summ


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"K_MDE (z.975 + z.80) = {K_MDE:.5f}")
    change = run_layer("a6_All_paired", "a6_MZ_sensitivity",
                       "within_family_mde", "change")
    run_layer("a6_cs_All_paired", "a6_cs_MZ_sensitivity",
              "within_family_mde_concurrent", "concurrent")

    ref = reference(change)
    ref_path = OUT_DIR / "within_family_mde_reference.csv"
    ref.to_csv(ref_path, index=False)
    print(f"Wrote {ref_path}")
    print("--- change MDE against concurrent FDR-surviving |beta_W_std| ---")
    for _, r in ref.iterrows():
        print(f"  {r['sample']:>3} {r['outcome_code']:>11}: MDE {r['change_median_MDE_std']:.3f}"
              f" | concurrent survivors {r['cs_n_fdr_surviving']},"
              f" median {r['cs_median_abs_beta_W_std_surviving']:.3f}"
              f" | ratio {r['change_mde_over_cs_surviving']:.2f}")


if __name__ == "__main__":
    main()
