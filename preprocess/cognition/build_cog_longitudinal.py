"""
Cognition preprocessing for MIDUS BTACT data.

Extracts cognitive composites (btactComp, episoMem, efFactor, wordList) from
MIDUS cognitive SAV files with national-sample z-scores, subtest-level
problematic-response flags, and raw subtest totals.

Output mirrors data/processed/biomarker/allostatic_load_*.csv structure:
  - Per-cohort files: cog_M2.csv, cog_M3.csv, cog_MR1.csv, cog_MR2.csv
  - Longitudinal wide frame: built for the summary, not written

Run from project root: uv run python -m preprocess.cognition.build_cog_longitudinal
"""
import warnings

import numpy as np
import pandas as pd
import pyreadstat

from preprocess.survey.missing_reasons import write_reason_artifacts
from scipy import stats

from preprocess.common.paths import PROCESSED_DIR, RAW_DIR, ROOT

warnings.filterwarnings("ignore", category=UserWarning)

# Configuration

OUT_DIR = PROCESSED_DIR / "cognition"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Composite z-score columns per wave
# M1 cohort: Z1 = MIDUS national sample (N=4206)
# MR cohort: Z1 = Refresher national sample (N=2550); MR2 _Long = standardised to MR1 M/SD
COMPOSITE_MAP = {
    "btactComp": {"M2": "B3TCOMPZ1", "M3": "C3TCOMP", "MR1": "RA3TCOMPZ1", "MR2": "RB3TCOMPZ_Long"},
    "episoMem":  {"M2": "B3TEMZ1",   "M3": "C3TEM",   "MR1": "RA3TEMZ1",   "MR2": "RB3TEMZ_Long"},
    "efFactor":  {"M2": "B3TEFZ1",   "M3": "C3TEF",   "MR1": "RA3TEFZ1",   "MR2": "RB3TEFZ_Long"},
    "wordList":  {"M2": "B3TWLF",    "M3": "C3TWLF",  "MR1": "RA3TWLF",    "MR2": "RB3TWLF"},
}

# Flag columns (1=problematic, 2=valid), 6 subtests per wave
FLAG_SUFFIXES = {
    "flag_wli": "TWLIFP",   # Word List Immediate
    "flag_db":  "TDBFP",    # Digits Backward
    "flag_ctf": "TCTFFP",   # Category Fluency
    "flag_ns":  "TNSFP",    # Number Series
    "flag_bk":  "TBKFP",    # Backward Counting
    "flag_wld": "TWLDFP",   # Word List Delayed
}

# Raw subtest totals
RAW_SUFFIXES = {
    "raw_wli_total": "TWLITU",  # Word List Immediate total
    "raw_wld_total": "TWLDTU",  # Word List Delayed total
    "raw_dbs":       "TDBS",    # Digits Backward span
    "raw_ctf":       "TCTFLU",  # Category Fluency count
    "raw_ns_total":  "TNSTOT",  # Number Series total
    "raw_bk_total":  "TBKTOT",  # Backward Counting total
}

# Wave-level configuration
WAVES = {
    "M2": {
        "sav_path": RAW_DIR / "wave1" / "M2_P3_BTACT_N4512_20211123.sav",
        "id_col": "M2ID",
        "prefix": "B3T",
        "age_col": "B1PAGE_M2",
        "sex_col": "B1PRSEX",
    },
    "M3": {
        "sav_path": RAW_DIR / "wave2" / "M3_P3_BTACT_N3291_20210922.sav",
        "id_col": "M2ID",
        "prefix": "C3T",
        "age_col": "C1PRAGE",
        "sex_col": "C1PRSEX",
    },
    "MR1": {
        "sav_path": RAW_DIR / "wave1" / "MR1_P3_BTACT_N2763_20251119.SAV",
        "id_col": "MRID",
        "prefix": "RA3T",
        "age_col": "RA1PRAGE",
        "sex_col": "RA1PRSEX",
    },
    "MR2": {
        "sav_path": RAW_DIR / "wave2" / "MR2_P3_BTACT+MOCA_N1934_20260224.sav",
        "id_col": "MRID",
        "prefix": "RB3T",
        "age_col": "RB1PRAGE",
        "sex_col": "RB1PRSEX",
    },
}

COMPOSITES = ["btactComp", "episoMem", "efFactor", "wordList"]
FLAG_NAMES = list(FLAG_SUFFIXES.keys())
RAW_NAMES = list(RAW_SUFFIXES.keys())


# Load a single wave

def load_wave(wave: str) -> pd.DataFrame:
    """Read cognitive SAV, extract composites/flags/raw subtests, standardise names."""
    cfg = WAVES[wave]
    prefix = cfg["prefix"]

    # Build column selection map: standard_name -> SAV_column
    rename_map = {}
    rename_map[cfg["id_col"]] = "id"
    rename_map[cfg["age_col"]] = "age"
    rename_map[cfg["sex_col"]] = "sex"

    for std_name, wave_map in COMPOSITE_MAP.items():
        rename_map[wave_map[wave]] = std_name

    for std_name, suffix in FLAG_SUFFIXES.items():
        rename_map[prefix + suffix.removeprefix("T")] = std_name  # e.g. B3TWLIFP

    for std_name, suffix in RAW_SUFFIXES.items():
        rename_map[prefix + suffix.removeprefix("T")] = std_name  # e.g. B3TWLITU

    # Read SAV (only needed columns)
    needed_cols = list(rename_map.keys())
    df, _ = pyreadstat.read_sav(str(cfg["sav_path"]), usecols=needed_cols)

    # Missing-reason mask. The BTACT files use their own vocabulary: REFUSED OR
    # MISSING and INVALID OR MISSING already merge two reasons in the source,
    # INCORRECT and INVALID mark an attempted item that cannot be scored, and
    # PROBLEM HEARING/UNDERSTANDING/RECORDING marks an administration failure.
    write_reason_artifacts(
        sav_path=cfg["sav_path"],
        columns=[c for c in needed_cols if c != cfg["id_col"]],
        id_col=cfg["id_col"],
        out_dir=PROCESSED_DIR / "missing_reasons",
        stem=f"cognition_{wave}",
        rename=rename_map,
        id_name="id",
    )

    df = df.rename(columns=rename_map)

    # Zap SPSS labels to plain numeric
    for col in df.columns:
        if hasattr(df[col], "cat"):
            df[col] = df[col].astype(float)

    # Recode flags: 1=problematic stays 1, 2=valid becomes 0
    for flag in FLAG_NAMES:
        if flag in df.columns:
            df[flag] = df[flag].map({1.0: 1, 2.0: 0})

    # Composite flag: any subtest flagged problematic
    flag_cols = [c for c in FLAG_NAMES if c in df.columns]
    df["flag_any_problematic"] = df[flag_cols].max(axis=1)

    # Blank btactComp, episoMem and efFactor for flagged sessions, as the master
    # builds do, so the per-cohort files agree with them. wordList is not blanked.
    _flagged = df["flag_any_problematic"] == 1
    for _c in ["btactComp", "episoMem", "efFactor"]:
        if _c in df.columns:
            df.loc[_flagged, _c] = np.nan
    if int(_flagged.sum()):
        print(f"    blanked {int(_flagged.sum())} flagged sessions")

    df["wave"] = wave
    df["id"] = df["id"].astype(int)

    col_order = ["id", "age", "sex"] + COMPOSITES + FLAG_NAMES + ["flag_any_problematic"] + RAW_NAMES + ["wave"]
    col_order = [c for c in col_order if c in df.columns]
    return df[col_order]


# Process all waves, save per-cohort CSVs

def process_all_waves() -> dict[str, pd.DataFrame]:
    """Load all 4 waves and save per-cohort CSVs."""
    results = {}
    for wave in WAVES:
        df = load_wave(wave)
        # Use original ID column name in output
        id_name = WAVES[wave]["id_col"]
        out = df.rename(columns={"id": id_name})
        out_path = OUT_DIR / f"cog_{wave}.csv"
        out.to_csv(out_path, index=False)
        print(f"  {wave}: n={len(df):,} -> {out_path.relative_to(ROOT)}")
        results[wave] = df
    return results


# Build longitudinal wide file

def build_longitudinal(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge W1 and W2 within each cohort into wide format with delta columns."""
    long_vars = ["age"] + COMPOSITES + ["flag_any_problematic"]
    parts = []

    for w1, w2, cohort_name, id_transform in [
        ("M2", "M3", "M1", lambda x: x),                              # M2ID stays as-is
        ("MR1", "MR2", "MR", lambda x: x),                              # raw MRID (no collision with M2ID)
    ]:
        df1 = results[w1][["id", "sex"] + long_vars].copy()
        df2 = results[w2][["id"] + long_vars].copy()

        merged = df1.merge(df2, on="id", suffixes=("_W1", "_W2"))

        # Delta columns for composites
        for comp in COMPOSITES:
            c1, c2 = f"{comp}_W1", f"{comp}_W2"
            if c1 in merged.columns and c2 in merged.columns:
                merged[f"{comp}_delta"] = merged[c2] - merged[c1]

        merged["M2MRID"] = merged["id"].apply(id_transform)
        merged["cohort"] = cohort_name

        keep = (
            ["M2MRID", "age_W1", "age_W2", "sex"]
            + [f"{c}_{s}" for c in COMPOSITES for s in ("W1", "W2")]
            + [f"{c}_delta" for c in COMPOSITES]
            + ["flag_any_W1", "flag_any_W2", "cohort"]
        )
        merged = merged.rename(columns={
            "flag_any_problematic_W1": "flag_any_W1",
            "flag_any_problematic_W2": "flag_any_W2",
        })
        keep = [c for c in keep if c in merged.columns]
        parts.append(merged[keep])

    longitudinal = pd.concat(parts, ignore_index=True)
    # Not written to disk; returned for the summary.
    print(f"  Longitudinal: n={len(longitudinal):,} matched pairs (not written)")
    return longitudinal


# Summary diagnostics

def print_summary(results: dict[str, pd.DataFrame], longitudinal: pd.DataFrame):
    """Print per-wave and longitudinal summary statistics."""
    print("\nPer-wave summary")
    for wave, df in results.items():
        n = len(df)
        print(f"\n  {wave} (n={n:,}):")
        for comp in COMPOSITES:
            s = df[comp].dropna()
            miss = (n - len(s)) / n * 100
            print(f"    {comp:12s}: mean={s.mean():+.3f}  sd={s.std():.3f}  miss={miss:.1f}%")
        flagged = df["flag_any_problematic"].sum()
        print(f"    flag_any_problematic: {int(flagged)} ({flagged/n*100:.1f}%)")

    print("\nLongitudinal summary")
    for cohort in ["M1", "MR"]:
        sub = longitudinal[longitudinal["cohort"] == cohort]
        print(f"\n  {cohort} cohort: n={len(sub):,} matched pairs")
        age_gap = sub["age_W2"] - sub["age_W1"]
        print(f"    Age gap: mean={age_gap.mean():.1f} yrs (sd={age_gap.std():.1f})")
        for comp in COMPOSITES:
            c1, c2 = f"{comp}_W1", f"{comp}_W2"
            valid = sub[[c1, c2]].dropna()
            if len(valid) > 2:
                r, p = stats.pearsonr(valid[c1], valid[c2])
                delta = (valid[c2] - valid[c1]).mean()
                print(f"    {comp:12s}: r={r:.3f} (p={p:.1e})  delta={delta:+.3f}  n={len(valid):,}")


# Main

def main():
    print("Cognition longitudinal pipeline (MIDUS)")

    print("\nSaving per-cohort files:")
    results = process_all_waves()

    print("\nBuilding longitudinal file:")
    longitudinal = build_longitudinal(results)

    print_summary(results, longitudinal)

    print("\nDone.")


if __name__ == "__main__":
    main()
