#!/usr/bin/env python3
"""
Longitudinal master dataset, one row per participant.
W1 base (M1 + MR cohort), W2 outcomes merged onto the same row by M2MRID.
Usage: run from repo root with `uv run python -m preprocess.master.build_master_dataset`
"""

import pandas as pd
import numpy as np
import pyreadstat

from preprocess.survey.missing_reasons import write_reason_artifacts
from pathlib import Path
import warnings
from preprocess.common.paths import PROCESSED_DIR, RAW_DIR, ROOT
from preprocess.common.guards import concat_cohorts, merge_no_fanout
from config.biomarker_constants import AL_COMPOSITE_COLUMNS

# Silence only FutureWarning and DeprecationWarning; merge and dtype warnings stay.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

RAW_M3 = RAW_DIR / "wave2/M3_P1_SURVEY_N3294_20251029.sav"
RAW_MR2 = RAW_DIR / "wave2/MR2_P1_SURVEY_N2154_20251003.sav"
DNAM_AGE_PATH = PROCESSED_DIR / "biomarker/dnam_age_residualized.csv"


def _clean_negatives(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Replace SPSS sentinel negatives with NaN."""
    for col in cols:
        df.loc[df[col] < 0, col] = np.nan
    return df


def _read_w2_age_from_raw(
    path: Path,
    id_col: str,
    age_col: str,
    birth_year_col: str,
    interview_year_col: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Required raw W2 SPSS file missing for age_W2 recovery: {path}"
        )

    df, _ = pyreadstat.read_sav(
        str(path),
        usecols=[id_col, age_col, birth_year_col, interview_year_col],
    )
    write_reason_artifacts(
        sav_path=path,
        columns=[age_col, birth_year_col, interview_year_col],
        id_col=id_col,
        out_dir=ROOT / "data/processed/missing_reasons",
        stem="age_w2_recovery",
        rename={id_col: "M2MRID"},
        id_name="M2MRID",
    )
    df = df.rename(columns={id_col: "M2MRID"})
    df = _clean_negatives(df, [age_col, birth_year_col, interview_year_col])
    df["calc_age"] = df[interview_year_col] - df[birth_year_col]
    df["age_W2_recovered"] = df[age_col].fillna(df["calc_age"])
    df["age_source"] = np.where(
        df[age_col].notna(),
        age_col,
        np.where(
            df["calc_age"].notna(),
            f"calc_{interview_year_col}-{birth_year_col}",
            "missing",
        ),
    )
    return df[["M2MRID", "age_W2_recovered", "age_source"]].dropna(
        subset=["age_W2_recovered"]
    )


def _build_age_w2_raw_lookup() -> pd.DataFrame:
    """Recover W2 age for outcome cases excluded from the W2 QC base."""
    m3 = _read_w2_age_from_raw(RAW_M3, "M2ID", "C1PRAGE", "C1PBYEAR", "C1PIDATE_YR")
    mr2 = _read_w2_age_from_raw(RAW_MR2, "MRID", "RB1PRAGE", "RB1PBYEAR", "RB1PIDATE_YR")
    combined = pd.concat([m3, mr2], ignore_index=True)
    return combined.drop_duplicates(subset=["M2MRID"], keep="first")


def _read_dnam_age_residuals(path: Path = DNAM_AGE_PATH) -> pd.DataFrame:
    """Load residualized DNAm clocks produced by preprocess/biomarkers/dnam_age.py."""
    if not path.exists():
        raise FileNotFoundError(
            "Required residualized DNAm age file missing: "
            f"{path}. Run `uv run python -m preprocess.biomarkers.dnam_age` "
            "before building master_analysis.csv."
        )

    required = ["M2MRID", "phenoAge_res", "dunedinPace_res", "grimAgeN_res"]
    dnam = pd.read_csv(path)
    missing = [col for col in required if col not in dnam.columns]
    if missing:
        raise ValueError(
            f"Residualized DNAm age file is missing required columns: {missing}"
        )
    return dnam[required].drop_duplicates(subset=["M2MRID"], keep="first")

# Load
print("Loading base files...")
w1 = pd.read_csv(ROOT / "data/processed/wave1/all_M2MR_analysis.csv", low_memory=False)
w2 = pd.read_csv(ROOT / "data/processed/wave2/all_M3MR2_analysis.csv", low_memory=False)
print(f"  W1: {len(w1):,}, W2: {len(w2):,}")

w1 = w1.rename(columns={"age": "age_W1"})
w2 = w2.rename(columns={"age": "age_W2"})

# Build W2 outcomes dataframe
# W2 outcomes are empty in the W2 base; they come from the stage outputs.
# W2 cognitive: M3 (M2ID) + MR2 (MRID used as M2MRID, no prefix)
_cog_m3 = pd.read_csv(ROOT / "data/processed/cognition/cog_M3.csv")
_cog_m3 = _cog_m3.rename(columns={"M2ID": "M2MRID"})
_cog_mr2 = pd.read_csv(ROOT / "data/processed/cognition/cog_MR2.csv")
_cog_mr2["M2MRID"] = pd.to_numeric(_cog_mr2["MRID"], errors="coerce")  # MIDUSR uses raw MRID as M2MRID in qc_pipeline
for _cdf in [_cog_m3, _cog_mr2]:
    for col in ["btactComp", "episoMem", "efFactor"]:
        if "flag_any_problematic" in _cdf.columns and col in _cdf.columns:
            _cdf.loc[_cdf["flag_any_problematic"] == 1, col] = np.nan
w2_cog = pd.concat([
    _cog_m3[["M2MRID", "btactComp", "episoMem", "efFactor"]],
    _cog_mr2[["M2MRID", "btactComp", "episoMem", "efFactor"]],
], ignore_index=True).rename(columns={c: f"{c}_W2" for c in ["btactComp", "episoMem", "efFactor"]})

w2_al = pd.read_csv(ROOT / "data/processed/biomarker/allostatic_load_M3.csv")
if "wave" in w2_al.columns:
    w2_al = w2_al[w2_al["wave"] == "M3"]
_al_w2_present = [c for c in AL_COMPOSITE_COLUMNS if c in w2_al.columns]
w2_al = w2_al[["M2ID"] + _al_w2_present].rename(columns={
    "M2ID": "M2MRID", **{c: AL_COMPOSITE_COLUMNS[c][1] for c in _al_w2_present},
})

# W2 frailty: M3 + MR2
_fi_m3 = pd.read_csv(ROOT / "data/processed/wave2/midus3_frailty.csv")
_fi_m3 = _fi_m3[["M2ID", "FInoPsy"]].rename(columns={"M2ID": "M2MRID"})
_fi_mr2 = pd.read_csv(ROOT / "data/processed/wave2/mr2_frailty.csv")
_fi_mr2["M2MRID"] = pd.to_numeric(_fi_mr2["MRID"], errors="coerce")
_fi_mr2 = _fi_mr2[["M2MRID", "FInoPsy"]]
# M and R ids share M2MRID without a prefix; concat_cohorts checks they do not overlap.
w2_fi = concat_cohorts([_fi_m3, _fi_mr2], "M2MRID", "W2 frailty M3+MR2"
                       ).rename(columns={"FInoPsy": "FInoPsy_W2"})

# Merge W2 outcomes onto w1 (one row per M2MRID)
master = w1.merge(w2[["M2MRID", "age_W2"]], on="M2MRID", how="left")
master = master.merge(w2_cog, on="M2MRID", how="left")
master = master.merge(w2_al, on="M2MRID", how="left")
master = master.merge(w2_fi, on="M2MRID", how="left")
print(f"  After W2 merges: {len(master):,} rows")

# Recover missing age_W2 from raw SPSS (birth year + interview year)
# Some W2 participants (STATUS != 4) have cognitive/outcome data from phone
# assessments but were excluded from the main QC pipeline, losing their age.
_age_rec = _build_age_w2_raw_lookup()
_before = master["age_W2"].isna().sum()
_fill_mask = master["age_W2"].isna() & master["M2MRID"].isin(_age_rec["M2MRID"])
_age_lookup = _age_rec.set_index("M2MRID")["age_W2_recovered"]
_source_lookup = _age_rec.set_index("M2MRID")["age_source"]
_used_sources = (
    master.loc[_fill_mask, "M2MRID"].map(_source_lookup).value_counts().to_dict()
)
master.loc[_fill_mask, "age_W2"] = master.loc[_fill_mask, "M2MRID"].map(_age_lookup)
_after = master["age_W2"].isna().sum()
print(f"  age_W2 recovered: {_before - _after} (from {_before} to {_after} missing)")
print(f"  age_W2 recovery sources: {_used_sources}")

# Drop empty outcome columns from the W1 base; values come from the merges below.
for _stale in ["FInoPsy", "FInoPsy_W1", "btactComp", "btactComp_W1", "episoMem", "episoMem_W1",
               "efFactor", "efFactor_W1", "wordList"]:
    if _stale in master.columns:
        master.drop(columns=[_stale], inplace=True)

# W1 outcomes
fi_w1 = pd.read_csv(ROOT / "data/processed/wave1/pooled_frailty_M2R.csv")
fi_w1 = fi_w1[["id", "FInoPsy"]].rename(columns={"id": "M2MRID", "FInoPsy": "FInoPsy_W1"})
if "FInoPsy_W1" in master.columns:
    master = master.drop(columns=["FInoPsy_W1"])
master = master.merge(fi_w1, on="M2MRID", how="left")
print(f"  FInoPsy_W1: {master['FInoPsy_W1'].notna().sum():,}")

# W1 cognitive (M2 + MR1 combined, non-overlapping IDs)
cog_M2 = pd.read_csv(ROOT / "data/processed/cognition/cog_M2.csv")
cog_MR1 = pd.read_csv(ROOT / "data/processed/cognition/cog_MR1.csv")
for cog_df, idcol in [(cog_M2, "M2ID"), (cog_MR1, "MRID")]:
    cog_df.rename(columns={idcol: "M2MRID"}, inplace=True)
    if "flag_any_problematic" in cog_df.columns:
        mask = cog_df["flag_any_problematic"] == 1
        for col in ["btactComp", "episoMem", "efFactor"]:
            if col in cog_df.columns:
                cog_df.loc[mask, col] = np.nan
    cog_df.rename(columns={"btactComp": "btactComp_W1", "episoMem": "episoMem_W1", "efFactor": "efFactor_W1"}, inplace=True)
# M2 and MR1 ids do not overlap, so concat then merge once
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
master = master.merge(
    concat_cohorts([al_M2, al_MR1], "M2MRID", "W1 allostatic load M2+MR1"),
    on="M2MRID", how="left")
print(f"  ALcomp_W1: {master['ALcomp_W1'].notna().sum():,}")

# Epigenetic clocks
dnam = _read_dnam_age_residuals()
master = master.merge(dnam, on="M2MRID", how="left")
print(f"  phenoAge_res: {master['phenoAge_res'].notna().sum():,}")

# Family structure
print("Family structure...")
try:
    admin_path = ROOT / "data/raw/admin_midus_20nov2024.sav"
    admin, _ = pyreadstat.read_sav(str(admin_path))
    # The IGEMS twin-pair file declares no user-missing ranges on any of its 64 variables,
    # so the mask is empty; the summary records that sentinel codes reach the
    # merge unfiltered.
    write_reason_artifacts(
        sav_path=admin_path,
        columns=["zygos", "pairid"],
        id_col="M2ID",
        out_dir=ROOT / "data/processed/missing_reasons",
        stem="admin_family",
    )
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
master["has_dna_methylation"] = master["phenoAge_res"].notna()
master["has_biomarker_W1"] = master["ALcomp_W1"].notna()
master["has_biomarker_W2"] = master["ALcomp_W2"].notna()
master["has_frailty_W1"] = master["FInoPsy_W1"].notna()
master["has_frailty_W2"] = master["FInoPsy_W2"].notna()
master["has_cog_W1"] = master["btactComp_W1"].notna()
master["has_cog_W2"] = master["btactComp_W2"].notna()
master["person_id"] = master["M2MRID"].astype("Int64").astype(str)

# Apply key exclusions
key_path = ROOT / "data/key_map/midus2R_key_extended.xlsx"
if key_path.exists():
    key = pd.read_excel(key_path, sheet_name=0)
    name_col = "variable" if "variable" in key.columns else "value"
    if name_col in key.columns and "remove" in key.columns:
        remove_vars = key[key["remove"] == 1][name_col].dropna().tolist()
        existing = [v for v in remove_vars if v in master.columns]
        master.drop(columns=existing, inplace=True)
        print(f"  Dropped {len(existing)} remove=1 vars from key map")

_BASE_WAVE = 1

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
print("Writing master_analysis.csv...")
out = ROOT / "data/processed/master_analysis.csv"
master.to_csv(out, index=False)
print(f"  Wrote {out} ({len(master):,} rows x {len(master.columns):,} cols)")

# Validation
long = (master["FInoPsy_W1"].notna() & master["FInoPsy_W2"].notna()).sum()
long_age = master.loc[master["FInoPsy_W1"].notna() & master["FInoPsy_W2"].notna(), "age_W1"].isna().sum()
print("\nValidation")
for col in ["FInoPsy_W1", "FInoPsy_W2", "btactComp_W1", "btactComp_W2",
            "ALcomp_W1", "ALcomp_W2",
            "phenoAge_res", "pair_id"]:
    if col in master.columns:
        n = master[col].notna().sum()
        print(f"  {col}: {n:,}/{len(master):,} ({n/len(master)*100:.1f}%)")
print(f"\n  Longitudinal (both waves): {long:,}")
print(f"  Longitudinal with age_W1 NaN: {long_age}")
print(f"\nDone {pd.Timestamp.now()}")
