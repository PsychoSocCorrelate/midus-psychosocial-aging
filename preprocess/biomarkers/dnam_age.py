#!/usr/bin/env python3
"""
Build residualized DNAm aging measures for the MIDUS 2 / Refresher 1 sample.

Reads the raw DNAmAge SPSS export, keeps samples with QC correlation >= 0.95 and
a matching sex, residualizes each clock on plate (BRA6DMPLATE) as a factor, and
writes the file used by the master dataset build.

Run from project root:
    uv run python -m preprocess.biomarkers.dnam_age
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

from preprocess.survey.missing_reasons import write_reason_artifacts

from preprocess.common.paths import PROCESSED_BIOMARKER_DIR, PROCESSED_DIR, RAW_BIOMARKER_DIR

RAW_PATH = RAW_BIOMARKER_DIR / "M2MR1_GEN_DNAmAge_N2118_20230822.sav"
ANALYSIS_READY_PATH = PROCESSED_DIR / "wave1/all_M2MR_analysis.csv"
OUT_DIR = PROCESSED_BIOMARKER_DIR
OUT_PATH = OUT_DIR / "dnam_age_residualized.csv"

MIN_QC_CORR = 0.95
ID_COL = "M2MRID"
PLATE_COL = "BRA6DMPLATE"
RAW_COLUMNS = [
    ID_COL,
    "SAMPLMAJ",
    "BRA6DMAGEPHENOAGE",
    "BRA6DMAGEDUNEDINPACE",
    "BRA6DMAGEGRIMAGE2",
    "BRA6DMQCCORR",
    "BRA6DMQCFEMALE",
    PLATE_COL,
]
CLOCK_RENAMES = {
    "BRA6DMAGEPHENOAGE": "phenoAge",
    "BRA6DMAGEDUNEDINPACE": "dunedinPace",
    "BRA6DMAGEGRIMAGE2": "grimAgeN",
}
CLOCKS = ["phenoAge", "dunedinPace", "grimAgeN"]
OUTPUT_COLUMNS = [
    ID_COL,
    "SAMPLMAJ",
    "dnam_sample_group",
    "phenoAge",
    "dunedinPace",
    "grimAgeN",
    PLATE_COL,
    "BRA6DMQCCORR",
    "BRA6DMQCFEMALE",
    "female",
    "dnam_qc_corr_pass",
    "dnam_sex_match",
    "phenoAge_res",
    "dunedinPace_res",
    "grimAgeN_res",
]


def _require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def _sample_group(sample_code: pd.Series) -> pd.Series:
    core_codes = {1.0, 2.0, 3.0, 4.0, 13.0}
    refresher_codes = {20.0, 21.0}
    return np.select(
        [sample_code.isin(core_codes), sample_code.isin(refresher_codes)],
        ["core", "refresher"],
        default="other",
    )


def load_raw_dnam_age(path: Path = RAW_PATH) -> pd.DataFrame:
    """Read the raw DNAmAge SPSS file and normalize clock names."""
    _require_file(path, "raw DNAmAge SPSS file")
    raw, _ = pyreadstat.read_sav(str(path), usecols=RAW_COLUMNS)
    missing = sorted(set(RAW_COLUMNS) - set(raw.columns))
    if missing:
        raise ValueError(f"Raw DNAmAge file is missing required columns: {missing}")

    # Missing-reason mask. The clock file labels its missing codes INAPP,
    # INAPP-M3NEWTWIN and MISSING; the default read collapses all three to NaN.
    write_reason_artifacts(
        sav_path=path,
        columns=[c for c in RAW_COLUMNS if c != ID_COL],
        id_col=ID_COL,
        out_dir=PROCESSED_DIR / "missing_reasons",
        stem="dnam_age",
        rename=CLOCK_RENAMES,
    )

    raw = raw.rename(columns=CLOCK_RENAMES)
    raw[ID_COL] = pd.to_numeric(raw[ID_COL], errors="coerce")
    raw["dnam_sample_group"] = _sample_group(raw["SAMPLMAJ"])
    return raw


def load_analysis_ready_sex(path: Path = ANALYSIS_READY_PATH) -> pd.DataFrame:
    """Load sex from the analysis-ready wave-1 file for the sex-match check."""
    _require_file(path, "analysis-ready Wave-1 file for DNAm sex QC")
    sex = pd.read_csv(path, usecols=[ID_COL, "female"])
    sex[ID_COL] = pd.to_numeric(sex[ID_COL], errors="coerce")
    sex = sex.dropna(subset=[ID_COL]).drop_duplicates(subset=[ID_COL], keep="first")
    return sex


def apply_qc(raw: pd.DataFrame, sex: pd.DataFrame) -> pd.DataFrame:
    """Keep rows passing the correlation and sex-match checks."""
    merged = raw.merge(sex, on=ID_COL, how="left", validate="many_to_one")
    merged["dnam_qc_corr_pass"] = merged["BRA6DMQCCORR"] >= MIN_QC_CORR
    merged["dnam_sex_match"] = (
        merged["female"].notna()
        & merged["BRA6DMQCFEMALE"].notna()
        & (merged["female"] == merged["BRA6DMQCFEMALE"])
    )

    qc = merged.loc[merged["dnam_qc_corr_pass"] & merged["dnam_sex_match"]].copy()
    return qc


def residualize_on_plate_factor(df: pd.DataFrame) -> pd.DataFrame:
    """Residualize each clock on BRA6DMPLATE as a categorical factor.

    Plate is a batch identifier, so it is treated as a factor rather than an
    ordered covariate. A saturated factor model fits each plate's mean, so the
    residual is the within-plate deviation, computed here directly.

    Single-observation plates residualize to 0; their count is printed.
    """
    df = df.copy()
    for clock in CLOCKS:
        res_col = f"{clock}_res"
        valid = df[clock].notna() & df[PLATE_COL].notna()
        if valid.sum() < 2:
            raise ValueError(f"Not enough complete rows to residualize {clock}")

        sub = df.loc[valid, [clock, PLATE_COL]].copy()
        grp = sub.groupby(PLATE_COL)[clock]
        resid = sub[clock].astype(float) - grp.transform("mean")

        df[res_col] = np.nan
        df.loc[valid, res_col] = resid.to_numpy()

        sizes = grp.size()
        singletons = int((sizes == 1).sum())
        print(
            f"  {res_col}: N={int(valid.sum()):,}, plates={len(sizes)}, "
            f"singleton plates={singletons}, "
            f"between-plate SD={grp.mean().std():.6g}, residual SD={resid.std():.6g}"
        )
    return df


def print_diagnostics(raw: pd.DataFrame, qc: pd.DataFrame) -> None:
    """Print QC counts."""
    corr_pass = int((raw["BRA6DMQCCORR"] >= MIN_QC_CORR).sum())
    in_analysis_ready = int(
        raw.loc[raw["BRA6DMQCCORR"] >= MIN_QC_CORR, ID_COL]
        .isin(qc[ID_COL])
        .sum()
    )
    print(f"Raw DNAmAge rows: {len(raw):,}")
    print(f"  QC correlation >= {MIN_QC_CORR}: {corr_pass:,}")
    print(f"  QC correlation + sex match: {len(qc):,}")
    print(f"  Master-eligible QC rows: {in_analysis_ready:,}")
    print("  Sample groups after QC:")
    for group, n in qc["dnam_sample_group"].value_counts(dropna=False).sort_index().items():
        print(f"    {group}: {n:,}")


def main() -> None:
    print("DNAm age preprocessing")
    print(f"Raw SPSS: {RAW_PATH}")
    raw = load_raw_dnam_age()
    sex = load_analysis_ready_sex()
    qc = apply_qc(raw, sex)
    print_diagnostics(raw, qc)

    processed = residualize_on_plate_factor(qc)
    processed = processed[OUTPUT_COLUMNS].sort_values(ID_COL).reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    processed.to_csv(OUT_PATH, index=False)
    print(f"Saved: {OUT_PATH} ({len(processed):,} rows x {processed.shape[1]:,} cols)")


if __name__ == "__main__":
    main()
