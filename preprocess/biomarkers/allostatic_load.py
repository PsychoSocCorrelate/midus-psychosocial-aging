"""
Allostatic load (AL) preprocessing for MIDUS biomarker data.

Scores the 16-biomarker composite (config/biomarker_constants.BIOMARKERS) for
MIDUS 2, MIDUS 3 and Refresher 1, after standard medication adjustments.

Run from project root: uv run python -m preprocess.biomarkers.allostatic_load
"""

import warnings

import numpy as np
import pandas as pd
import pyreadstat
from scipy import stats

from config.biomarker_constants import BIOMARKERS, MED_FLAGS, SYSTEMS
from preprocess.common.paths import (
    PROCESSED_BIOMARKER_DIR,
    PROCESSED_DIR,
    RAW_BIOMARKER_DIR,
)
from preprocess.survey.missing_reasons import write_reason_artifacts

warnings.filterwarnings("ignore", category=UserWarning)

# Configuration

RAW_BIO = RAW_BIOMARKER_DIR
OUT_DIR = PROCESSED_BIOMARKER_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Wave-specific file paths and variable prefixes
WAVES = {
    "M2": {
        "bio_file": RAW_BIO / "M2_BIO_AGGREGATE_N1255_20240405.sav",
        "med_file": RAW_BIO / "M2_BIO_MEDICATION_STACKED_N7174_20240404.SAV",
        "prefix": "B4",
        "id_col": "M2ID",
        "age_col": "B4ZAGE",
    },
    "M3": {
        "bio_file": RAW_BIO / "M3_BIO_AGGREGATE_N747_20231016.sav",
        "med_file": RAW_BIO / "M3_BIO_MEDICATION_STACKED_N5751_20231016.SAV",
        "prefix": "C4",
        "id_col": "M2ID",
        "age_col": "C4ZAGE",
    },
    "MR1": {
        "bio_file": RAW_BIO / "MR1_BIO_AGGREGATE_N863_20240405.sav",
        "med_file": RAW_BIO / "MR1_BIO_MEDICATION_STACKED_N5118_20240404.sav",
        "prefix": "RA4",
        "id_col": "MRID",
        "age_col": "RA4ZAGE",
    },
}

# Data Loading

def load_wave(wave_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load biomarker aggregate and medication files for one wave."""
    cfg = WAVES[wave_name]
    prefix = cfg["prefix"]
    id_col = cfg["id_col"]

    bio_df, _ = pyreadstat.read_sav(str(cfg["bio_file"]))
    print(f"  {wave_name} biomarker: N={len(bio_df)}, cols={bio_df.shape[1]}")

    # Load medication data (stacked: one row per drug per person)
    med_df, _ = pyreadstat.read_sav(str(cfg["med_file"]))
    print(f"  {wave_name} medication: N_rows={len(med_df)}, "
          f"N_persons={med_df[id_col].nunique()}")

    _emit_biomarker_missing_reasons(wave_name, cfg, prefix, id_col)

    return bio_df, med_df


def _emit_biomarker_missing_reasons(wave_name, cfg, prefix: str, id_col: str) -> None:
    """Write missing-reason masks for the biomarker and medication files.

    Besides INAPP, MISSING and DON'T KNOW, the aggregate files distinguish NOT
    CALCULATED due to missing data from NOT CALCULATED due to INAPP data (a biomarker
    the lab could not derive vs one the participant was not eligible for). A default
    read makes all of them NaN. The medication INAPP codes separate an unmatched drug
    from a participant with no medication record.
    """
    out_dir = PROCESSED_DIR / "missing_reasons"
    bio_rename = {f"{prefix}{suffix}": std for std, suffix, _, _ in BIOMARKERS}
    write_reason_artifacts(
        sav_path=cfg["bio_file"],
        columns=sorted(bio_rename),
        id_col=id_col,
        out_dir=out_dir,
        stem=f"al_{wave_name}_biomarkers",
        rename=bio_rename,
    )
    med_cols = sorted({f"{prefix}{s}" for suffixes in MED_FLAGS.values() for s in suffixes})
    write_reason_artifacts(
        sav_path=cfg["med_file"],
        columns=med_cols,
        id_col=id_col,
        out_dir=out_dir,
        stem=f"al_{wave_name}_medication",
    )


# Medication Adjustment

def get_med_flags(med_df: pd.DataFrame, prefix: str, id_col: str) -> pd.DataFrame:
    """Collapse stacked medication file to per-person binary flags."""
    # Recode flag columns: 1=YES -> 1, 2=NO -> 0 (MIDUS SPSS coding)
    med_df = med_df.copy()
    all_flag_cols = []
    for suffixes in MED_FLAGS.values():
        for s in suffixes:
            col = f"{prefix}{s}"
            if col in med_df.columns:
                med_df[col] = (med_df[col] == 1).astype(int)
                all_flag_cols.append(col)

    flags = {}
    for med_class, suffixes in MED_FLAGS.items():
        cols = [f"{prefix}{s}" for s in suffixes]
        available = [c for c in cols if c in med_df.columns]
        if not available:
            print(f"    WARNING: No {med_class} flags found for prefix {prefix}")
            flags[med_class] = pd.Series(0, index=med_df[id_col].unique())
            continue
        # Any drug row with a flag == 1 puts the person in that med class
        person_flag = (
            med_df.groupby(id_col)[available]
            .max()
            .max(axis=1)
            .fillna(0)
            .astype(int)
        )
        flags[med_class] = person_flag

    return pd.DataFrame(flags)


def apply_medication_adjustments(
    df: pd.DataFrame, med_flags: pd.DataFrame, id_col: str
) -> pd.DataFrame:
    """Apply additive medication corrections to biomarker values."""
    df = df.copy()

    if id_col in med_flags.index.names or med_flags.index.name == id_col:
        med_flags = med_flags.reset_index()
    df = df.merge(med_flags, on=id_col, how="left")
    for mc in MED_FLAGS:
        df[mc] = df[mc].fillna(0).astype(int)

    n_adj = {}

    # Antihypertensive: SBP += 10, DBP += 5
    mask_bp = df["antihypertensive"] == 1
    n_adj["antihypertensive"] = int(mask_bp.sum())
    df.loc[mask_bp, "SBP"] = df.loc[mask_bp, "SBP"] + 10
    df.loc[mask_bp, "DBP"] = df.loc[mask_bp, "DBP"] + 5

    # Antilipidemic: Total chol += 20 (need raw total chol to recalculate ratio)
    # HDL adjustment: subtract 5 (medication artificially raises HDL)
    mask_lip = df["antilipidemic"] == 1
    n_adj["antilipidemic"] = int(mask_lip.sum())
    # Statins raise observed HDL, so subtract
    df.loc[mask_lip, "HDL"] = df.loc[mask_lip, "HDL"] - 5
    # Rebuild the ratio: adjusted = (orig_total + 20) / (orig_HDL - 5), with
    # orig_total = ratio * orig_HDL.
    if "cholHDL_ratio" in df.columns and "HDL" in df.columns:
        orig_hdl = df.loc[mask_lip, "HDL"] + 5  # HDL before the adjustment above
        orig_total = df.loc[mask_lip, "cholHDL_ratio"] * orig_hdl
        adj_total = orig_total + 20
        adj_hdl = df.loc[mask_lip, "HDL"]  # already adjusted
        df.loc[mask_lip, "cholHDL_ratio"] = adj_total / adj_hdl

    # Antidiabetic: HbA1c += 1%, glucose += 20
    mask_dm = df["antidiabetic"] == 1
    n_adj["antidiabetic"] = int(mask_dm.sum())
    df.loc[mask_dm, "HbA1c"] = df.loc[mask_dm, "HbA1c"] + 1.0
    df.loc[mask_dm, "glucose"] = df.loc[mask_dm, "glucose"] + 20

    print(f"    Medication adjustments applied: "
          f"antihypertensive={n_adj['antihypertensive']}, "
          f"antilipidemic={n_adj['antilipidemic']}, "
          f"antidiabetic={n_adj['antidiabetic']}")

    # Keep the class flags, class count and a record-present indicator, so outputs
    # show who was adjusted and who counts as unmedicated only because they have no
    # medication record.
    df["med_n_classes"] = df[list(MED_FLAGS.keys())].sum(axis=1).astype(int)
    df["med_record_present"] = df[id_col].isin(med_flags[id_col]).astype(int) \
        if id_col in med_flags.columns else 1
    df = df.rename(columns={k: f"med_{k}" for k in MED_FLAGS})
    return df


# Extract & Standardize Biomarker Names

def extract_biomarkers(bio_df: pd.DataFrame, wave_name: str,
                       biomarkers: list[tuple] = BIOMARKERS) -> pd.DataFrame:
    """Extract `biomarkers` (BIOMARKERS by default) from wave data under standard names."""
    cfg = WAVES[wave_name]
    prefix = cfg["prefix"]
    id_col = cfg["id_col"]
    age_col = cfg["age_col"]

    rename_map = {}
    for std_name, suffix, _, _ in biomarkers:
        src_col = f"{prefix}{suffix}"
        if src_col in bio_df.columns:
            rename_map[src_col] = std_name
        else:
            print(f"    WARNING: {src_col} ({std_name}) not found in {wave_name}")

    # Select ID, age, and biomarker columns
    keep_cols = [id_col]
    if age_col in bio_df.columns:
        keep_cols.append(age_col)
    keep_cols.extend(rename_map.keys())

    df = bio_df[keep_cols].copy()
    df = df.rename(columns=rename_map)
    if age_col in df.columns:
        df = df.rename(columns={age_col: "age"})

    print(f"    Extracted {len(rename_map)} biomarkers for {wave_name}")
    return df


# Compute AL Scores

# AL is standardised against a wave-1 reference per cohort: M2 and M3 against
# M2, MR1 against MR1. Centring each wave on its own sample would force every
# wave's mean composite to about zero and hide change between waves. The reference
# is saved so it can be audited and reused for later waves.
REFERENCE_PATH = PROCESSED_BIOMARKER_DIR / "al_reference_wave1.csv"
WAVE_TO_ANCHOR = {"M2": "M2", "M3": "M2", "MR1": "MR1"}   # MR2 has no biomarker data


def build_wave1_reference(frames: dict[str, pd.DataFrame], biomarkers: list[str]) -> pd.DataFrame:
    """Mean, SD and quartile cut-offs from each wave-1 cohort, saved as a stage output."""
    rows = []
    for anchor in ("M2", "MR1"):
        d = frames[anchor]
        for bm in biomarkers:
            if bm not in d.columns:
                continue
            v = d[bm].dropna()
            rows.append({"anchor": anchor, "biomarker": bm, "n": len(v),
                         "mean": v.mean(), "sd": v.std(),
                         "q25": v.quantile(0.25), "q75": v.quantile(0.75)})
    ref = pd.DataFrame(rows)
    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    ref.to_csv(REFERENCE_PATH, index=False)
    print(f"  Wave-1 reference written: {REFERENCE_PATH} ({len(ref)} rows)")
    return ref


def compute_al_scores(df: pd.DataFrame, id_col: str,
                      reference: pd.DataFrame | None = None,
                      anchor: str | None = None) -> pd.DataFrame:
    """Score the composite and the per-system high-risk flags.

    ALcomp_zscore is the mean of the risk-oriented z-scores, computed when at
    least 80% of the biomarkers (13 of 16) are observed. AL_sys_<system> is 1 when
    any biomarker in the system falls in its high-risk quartile.
    """
    names = [b[0] for b in BIOMARKERS if b[0] in df.columns]
    directions = {b[0]: b[3] for b in BIOMARKERS}
    systems = {b[0]: b[2] for b in BIOMARKERS}

    result = df[[id_col]].copy()
    if "age" in df.columns:
        result["age"] = df["age"]

    # Carry the medication columns into the new frame.
    for _c in [c for c in df.columns if c.startswith("med_")]:
        result[_c] = df[_c]

    # Quartile flags and z-scores
    high_risk = pd.DataFrame(index=df.index)
    z_scores = pd.DataFrame(index=df.index)

    for bm in names:
        vals = df[bm]
        direction = directions[bm]

        # Quartile-based binary (high-risk = 1). Cut-offs come from the wave-1
        # reference when one is supplied, so they do not move with the wave scored.
        ref_row = None
        if reference is not None and anchor is not None:
            _m = reference[(reference["anchor"] == anchor) & (reference["biomarker"] == bm)]
            ref_row = _m.iloc[0] if len(_m) else None
        if direction == "HIGH":
            cutoff = ref_row["q75"] if ref_row is not None else vals.quantile(0.75)
            high_risk[bm] = (vals >= cutoff).astype(float)
        else:  # LOW
            cutoff = ref_row["q25"] if ref_row is not None else vals.quantile(0.25)
            high_risk[bm] = (vals <= cutoff).astype(float)

        high_risk.loc[vals.isna(), bm] = np.nan

        # Z-score (sign-flip for LOW direction)
        mean_val = ref_row["mean"] if ref_row is not None else vals.mean()
        std_val = ref_row["sd"] if ref_row is not None else vals.std()
        if std_val > 0:
            z = (vals - mean_val) / std_val
            if direction == "LOW":
                z = -z  # flip so higher z = worse
            z_scores[bm] = z
        else:
            z_scores[bm] = np.nan

    min_required = int(np.ceil(len(names) * 0.80))
    n_available = z_scores[names].notna().sum(axis=1)
    result["ALcomp_zscore"] = z_scores[names].mean(axis=1)
    result.loc[n_available < min_required, "ALcomp_zscore"] = np.nan
    result["ALcomp_n_biomarkers"] = n_available

    for sys_name in SYSTEMS:
        sys_bms = [bm for bm in names if systems[bm] == sys_name]
        if sys_bms:
            result[f"AL_sys_{sys_name}"] = high_risk[sys_bms].max(axis=1)

    return result


# Process All Waves

def prepare_wave(wave_name: str, biomarkers: list[tuple] = BIOMARKERS) -> pd.DataFrame:
    """Load, extract and medication-adjust one wave (no scoring).

    Scoring is separate because the wave-1 reference is built from the adjusted
    frames of every wave first.
    """
    print(f"\nPreparing {wave_name}")
    cfg = WAVES[wave_name]
    bio_df, med_df = load_wave(wave_name)
    df = extract_biomarkers(bio_df, wave_name, biomarkers)
    med_flags = get_med_flags(med_df, cfg["prefix"], cfg["id_col"])
    df = apply_medication_adjustments(df, med_flags, cfg["id_col"])
    return df


# Diagnostics

def print_diagnostics(results: dict[str, pd.DataFrame]) -> None:
    """Print the composite's distribution, availability and M2-M3 stability."""
    print("\nValidation diagnostics")
    for wave_name, df in results.items():
        vals = df["ALcomp_zscore"].dropna()
        n_bm = df["ALcomp_n_biomarkers"]
        print(f"\n--- {wave_name} (N={len(df)}) ---")
        print(f"  ALcomp_zscore: mean={vals.mean():.2f}, sd={vals.std():.2f}, "
              f"range=[{vals.min():.2f}, {vals.max():.2f}], N_valid={len(vals)}")
        print(f"  Biomarkers observed per person: min={n_bm.min():.0f}, max={n_bm.max():.0f} "
              f"of {len(BIOMARKERS)}")

    m2 = results["M2"][["M2ID", "ALcomp_zscore"]]
    m3 = results["M3"][["M2ID", "ALcomp_zscore"]]
    pair = m2.merge(m3, on="M2ID", suffixes=("_W1", "_W2")).dropna()
    if len(pair) > 10:
        r, p = stats.pearsonr(pair["ALcomp_zscore_W1"], pair["ALcomp_zscore_W2"])
        print(f"\n  M2-M3 ALcomp_zscore: r={r:.3f}, p={p:.4f}, N={len(pair)}")


# Main

def main():
    print("Allostatic load pipeline: MIDUS biomarker data")
    print(f"Output directory: {OUT_DIR}")

    # Prepare all waves, build the wave-1 reference from the adjusted wave-1 frames,
    # then score each wave against its cohort anchor.
    prepared = {w: prepare_wave(w) for w in ["M2", "M3", "MR1"]}
    all_bm = [b[0] for b in BIOMARKERS]
    reference = build_wave1_reference(prepared, all_bm)

    results = {}
    for wave_name in ["M2", "M3", "MR1"]:
        anchor = WAVE_TO_ANCHOR[wave_name]
        print(f"\n  Scoring {wave_name} against the {anchor} wave-1 reference")
        df = compute_al_scores(prepared[wave_name], WAVES[wave_name]["id_col"],
                               reference=reference, anchor=anchor)
        df["wave"] = wave_name
        df["al_reference_anchor"] = anchor
        out_path = OUT_DIR / f"allostatic_load_{wave_name}.csv"
        df.to_csv(out_path, index=False)
        print(f"  Saved: {out_path} ({len(df)} rows, {df.shape[1]} cols)")
        results[wave_name] = df

    print_diagnostics(results)

    print("\nDone.")


if __name__ == "__main__":
    main()
