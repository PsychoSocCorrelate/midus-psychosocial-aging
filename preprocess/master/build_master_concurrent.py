#!/usr/bin/env python3
"""
Concurrent-predictor master dataset for the a9 analyses.

Same structure as master_analysis.csv, but T1/T2 predictor values come from
W2 (MIDUS3/MR2) instead of W1 (MIDUS2/MIDUSR).

Base: data/processed/wave2/all_M3MR2_analysis.csv  (W2 predictor values, post-composite)
Outcomes: same W1/W2 outcomes as master_analysis.csv
Family structure: same IGEMS twin-pair merge

Composites (psychWellbeing, religiosity, etc.) are pre-joined by build_analysis_ready.py.
Some composites may have fewer non-NA values at W2 due to structural missingness.

Usage:
    uv run python -m preprocess.master.build_master_concurrent
"""

import pandas as pd
import numpy as np
import pyreadstat
import warnings
from config.variable_config import get_tier_variables
from preprocess.common.paths import ROOT
from preprocess.common.guards import concat_cohorts, merge_no_fanout
from config.biomarker_constants import AL_COMPOSITE_COLUMNS

# Silence only FutureWarning and DeprecationWarning; merge and dtype warnings stay.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

print("Building master_analysis_concurrent.csv (W2 predictors)...")

# Load W2 predictors as base
w2_pred = pd.read_csv(ROOT / "data/processed/wave2/all_M3MR2_analysis.csv", low_memory=False)
w2_pred = w2_pred.rename(columns={"age": "age_W2"})
print(f"  W2 predictor base: {len(w2_pred):,} rows x {len(w2_pred.columns)} cols")

# age_W1 for OLS residualization
w1 = pd.read_csv(ROOT / "data/processed/wave1/all_M2MR_py.csv", low_memory=False,
                  usecols=["M2MRID", "age", "cohort"])
w1 = w1.rename(columns={"age": "age_W1"})

master = w2_pred.merge(w1[["M2MRID", "age_W1"]], on="M2MRID", how="left")
print(f"  After age_W1 merge: {len(master):,}")

# W1 outcomes (needed as autoregressive features)
# FInoPsy_W1
fi_w1 = pd.read_csv(ROOT / "data/processed/wave1/pooled_frailty_M2R.csv")
fi_w1 = fi_w1[["id", "FInoPsy"]].rename(columns={"id": "M2MRID", "FInoPsy": "FInoPsy_W1"})
master = master.merge(fi_w1, on="M2MRID", how="left")
print(f"  FInoPsy_W1: {master['FInoPsy_W1'].notna().sum():,}")

# btactComp_W1
cog_M2 = pd.read_csv(ROOT / "data/processed/cognition/cog_M2.csv")
cog_MR1 = pd.read_csv(ROOT / "data/processed/cognition/cog_MR1.csv")
for cog_df, idcol in [(cog_M2, "M2ID"), (cog_MR1, "MRID")]:
    cog_df.rename(columns={idcol: "M2MRID"}, inplace=True)
    if "flag_any_problematic" in cog_df.columns:
        for col in ["btactComp", "episoMem", "efFactor"]:
            if col in cog_df.columns:
                cog_df.loc[cog_df["flag_any_problematic"] == 1, col] = np.nan
    cog_df.rename(columns={
        "btactComp": "btactComp_W1", "episoMem": "episoMem_W1", "efFactor": "efFactor_W1"
    }, inplace=True)
cog_W1 = pd.concat([
    cog_M2[["M2MRID", "btactComp_W1", "episoMem_W1", "efFactor_W1"]],
    cog_MR1[["M2MRID", "btactComp_W1", "episoMem_W1", "efFactor_W1"]],
], ignore_index=True)
master = master.merge(cog_W1, on="M2MRID", how="left")
print(f"  btactComp_W1: {master['btactComp_W1'].notna().sum():,}")

# W1 allostatic load
al_M2 = pd.read_csv(ROOT / "data/processed/biomarker/allostatic_load_M2.csv")
al_MR1 = pd.read_csv(ROOT / "data/processed/biomarker/allostatic_load_MR1.csv")
_al_w1_present_m2 = [c for c in AL_COMPOSITE_COLUMNS if c in al_M2.columns]
_al_w1_present_mr1 = [c for c in AL_COMPOSITE_COLUMNS if c in al_MR1.columns]
_al_w1_cols_m2 = ["M2ID"] + _al_w1_present_m2
_al_w1_cols_mr1 = ["MRID"] + _al_w1_present_mr1
if "wave" in al_M2.columns:
    al_M2 = al_M2[al_M2["wave"] == "M2"][_al_w1_cols_m2].copy()
if "wave" in al_MR1.columns:
    al_MR1 = al_MR1[al_MR1["wave"] == "MR1"][_al_w1_cols_mr1].copy()
al_M2 = al_M2.rename(columns={
    "M2ID": "M2MRID", **{c: AL_COMPOSITE_COLUMNS[c][0] for c in _al_w1_present_m2}})
al_MR1 = al_MR1.rename(columns={
    "MRID": "M2MRID", **{c: AL_COMPOSITE_COLUMNS[c][0] for c in _al_w1_present_mr1}})
# concat_cohorts checks the M and R ids do not overlap.
master = master.merge(
    concat_cohorts([al_M2, al_MR1], "M2MRID", "W1 allostatic load M2+MR1"),
    on="M2MRID", how="left")
print(f"  ALcomp_W1: {master['ALcomp_W1'].notna().sum():,}")

# W2 outcomes
# FInoPsy_W2
_fi_m3 = pd.read_csv(ROOT / "data/processed/wave2/midus3_frailty.csv")
_fi_m3 = _fi_m3[["M2ID", "FInoPsy"]].rename(columns={"M2ID": "M2MRID"})
_fi_mr2 = pd.read_csv(ROOT / "data/processed/wave2/mr2_frailty.csv")
_fi_mr2["M2MRID"] = pd.to_numeric(_fi_mr2["MRID"], errors="coerce")
_fi_mr2 = _fi_mr2[["M2MRID", "FInoPsy"]]
w2_fi = concat_cohorts([_fi_m3, _fi_mr2], "M2MRID", "W2 frailty M3+MR2"
                       ).rename(columns={"FInoPsy": "FInoPsy_W2"})
master = master.merge(w2_fi, on="M2MRID", how="left")
print(f"  FInoPsy_W2: {master['FInoPsy_W2'].notna().sum():,}")

# btactComp_W2
_cog_m3 = pd.read_csv(ROOT / "data/processed/cognition/cog_M3.csv")
_cog_m3 = _cog_m3.rename(columns={"M2ID": "M2MRID"})
_cog_mr2 = pd.read_csv(ROOT / "data/processed/cognition/cog_MR2.csv")
_cog_mr2["M2MRID"] = pd.to_numeric(_cog_mr2["MRID"], errors="coerce")
for _cdf in [_cog_m3, _cog_mr2]:
    for col in ["btactComp", "episoMem", "efFactor"]:
        if "flag_any_problematic" in _cdf.columns and col in _cdf.columns:
            _cdf.loc[_cdf["flag_any_problematic"] == 1, col] = np.nan
w2_cog = pd.concat([
    _cog_m3[["M2MRID", "btactComp", "episoMem", "efFactor"]],
    _cog_mr2[["M2MRID", "btactComp", "episoMem", "efFactor"]],
], ignore_index=True).rename(columns={c: f"{c}_W2" for c in ["btactComp", "episoMem", "efFactor"]})
master = master.merge(w2_cog, on="M2MRID", how="left")
print(f"  btactComp_W2: {master['btactComp_W2'].notna().sum():,}")

# W2 allostatic load
w2_al = pd.read_csv(ROOT / "data/processed/biomarker/allostatic_load_M3.csv")
if "wave" in w2_al.columns:
    w2_al = w2_al[w2_al["wave"] == "M3"]
_al_w2_present = [c for c in AL_COMPOSITE_COLUMNS if c in w2_al.columns]
w2_al = w2_al[["M2ID"] + _al_w2_present].rename(columns={
    "M2ID": "M2MRID", **{c: AL_COMPOSITE_COLUMNS[c][1] for c in _al_w2_present},
})
master = master.merge(w2_al, on="M2MRID", how="left")
print(f"  ALcomp_W2: {master['ALcomp_W2'].notna().sum():,}")

# Family structure (same as build_master_dataset.py)
print("Family structure...")
try:
    admin_path = ROOT / "data/raw/admin_midus_20nov2024.sav"
    # The missing-reason mask for this read is written by build_master_dataset.py.
    admin, _ = pyreadstat.read_sav(str(admin_path))
    pair_df = admin[["M2ID", "zygos", "pairid"]].dropna(subset=["pairid"]).copy()
    pair_df = pair_df.rename(columns={"M2ID": "M2MRID"})
    pair_df["M2MRID"] = pd.to_numeric(pair_df["M2MRID"], errors="coerce")
    pair_df["pair_id"] = pd.to_numeric(pair_df["pairid"], errors="coerce")
    sib_map = {1.0: "MZ", 2.0: "DZ_SS", 3.0: "DZ_OS"}
    pair_df["sib_type"] = pair_df["zygos"].map(sib_map)
    pair_df["family_id"] = pair_df["pair_id"].astype("Int64").astype(str)
    # The twin-pair file has 61 rows with a null key; pandas would match them to any
    # null master id, so merge_no_fanout drops them and checks the row count.
    master = merge_no_fanout(
        master, pair_df[["M2MRID", "family_id", "sib_type", "pair_id"]],
        on="M2MRID", where="IGEMS twin-pair merge",
    )
    master["is_full_pair"] = master["pair_id"].notna()
    print(f"  Pairs: {pair_df['pair_id'].nunique():,}")
except FileNotFoundError as e:
    # Only a missing twin-pair file is tolerated; family columns are then null.
    print(f"  [WARN] IGEMS twin-pair file not found ({e}); family structure will be null, "
          f"so sibling exclusion downstream will not work")
    master["family_id"] = np.nan
    master["sib_type"] = np.nan
    master["pair_id"] = np.nan
    master["is_full_pair"] = False

# Availability flags
master["has_biomarker_W1"] = master["ALcomp_W1"].notna()
master["has_biomarker_W2"] = master["ALcomp_W2"].notna()
master["has_frailty_W1"] = master["FInoPsy_W1"].notna()
master["has_frailty_W2"] = master["FInoPsy_W2"].notna()
master["has_cog_W1"] = master["btactComp_W1"].notna()
master["has_cog_W2"] = master["btactComp_W2"].notna()
master["person_id"] = master["M2MRID"].astype("Int64").astype(str)

_BASE_WAVE = 2

# Write the ids present in the other wave but not in this base. The two masters
# use different base waves, so each excludes a different part of the attrition.
_w1_ids = set(pd.read_csv(ROOT / "data/processed/wave1/all_M2MR_analysis.csv",
                          usecols=["M2MRID"], low_memory=False)["M2MRID"])
_w2_ids = set(pd.read_csv(ROOT / "data/processed/wave2/all_M3MR2_analysis.csv",
                          usecols=["M2MRID"], low_memory=False)["M2MRID"])
_base_ids = set(master["M2MRID"])
_excluded = (_w2_ids - _w1_ids) if _BASE_WAVE == 1 else (_w1_ids - _w2_ids)
print("\nBase exclusions")
print(f"  base: wave {_BASE_WAVE}, {len(_base_ids):,} participants")
print(f"  present in the other wave but absent from this base: {len(_excluded):,}")
_excl_path = ROOT / f"data/processed/excluded_by_base_wave{_BASE_WAVE}.csv"
pd.DataFrame({"M2MRID": sorted(_excluded)}).to_csv(_excl_path, index=False)
print(f"  written: {_excl_path.name}")

# Write
out = ROOT / "data/processed/master_analysis_concurrent.csv"
master.to_csv(out, index=False)
print(f"\nWrote {out}")
print(f"   {len(master):,} rows x {len(master.columns):,} cols")

# Validation
t1t2 = get_tier_variables([1, 2], exclude_high_missing=False)
in_csv = [v for v in t1t2 if v in master.columns]
missing = [v for v in t1t2 if v not in master.columns]
print("\nPredictor coverage")
print(f"  T1+T2 in concurrent CSV: {len(in_csv)}/179")
print(f"  Missing (W1-only): {missing}")

_al_cols = [n for pair in AL_COMPOSITE_COLUMNS.values() for n in pair]
for col in (["FInoPsy_W1", "FInoPsy_W2", "btactComp_W1", "btactComp_W2"]
            + _al_cols + ["age_W1", "age_W2"]):
    if col in master.columns:
        n = master[col].notna().sum()
        print(f"  {col}: {n:,}/{len(master):,}")

# Verify predictors differ from W1
w1_full = pd.read_csv(ROOT / "data/processed/wave1/all_M2MR_analysis.csv", low_memory=False)
shared_ids = sorted(set(master["M2MRID"]) & set(w1_full["M2MRID"]))
test_vars = ["negAffect", "controlOverHealth", "bmi", "education", "workStatus"]
test_vars = [v for v in test_vars if v in master.columns and v in w1_full.columns]
print(f"\nW1-W2 predictor divergence (n={len(shared_ids):,} shared IDs)")
for v in test_vars:
    m_conc = master.set_index("M2MRID").loc[shared_ids, v]
    m_base = w1_full.set_index("M2MRID").loc[shared_ids, v]
    try:
        corr = m_conc.astype(float).corr(m_base.astype(float))
    except (ValueError, TypeError):
        corr = float("nan")
    same_frac = (m_conc.astype(str) == m_base.astype(str)).mean()
    print(f"  {v}: W1-W2 corr={corr:.3f}, exact_match={same_frac:.1%}")

print(f"\nDone {pd.Timestamp.now()}")
