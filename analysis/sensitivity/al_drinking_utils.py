"""Shared helpers for AL drinking-status sensitivity analyses."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
MASTER_CSV = ROOT / "data" / "processed" / "master_analysis.csv"
CONCURRENT_CSV = ROOT / "data" / "processed" / "master_analysis_concurrent.csv"
BIO_DIR = ROOT / "data" / "processed" / "biomarker"

RAW_WAVES = {
    "MIDUS2": {
        "path": ROOT / "data" / "raw" / "wave1" / "M2_P1_SURVEY_N4963_20200720.sav",
        "prefix": "B1",
        "id_col": "M2ID",
    },
    "MIDUSR": {
        "path": ROOT / "data" / "raw" / "wave1" / "MR1_P1_SURVEY_N3577_20250407.sav",
        "prefix": "RA1",
        "id_col": "MRID",
    },
}

CURRENT_CATS = [
    "Past-month abstainer",
    "Infrequent-non-binge",
    "Frequent-non-binge",
    "Binge",
]
CURRENT_DRINKER_CATS = ["Infrequent-non-binge", "Frequent-non-binge", "Binge"]
REFINED_CATS = [
    "abstainer_never_or_former_light",
    "former_moderate",
    "former_heavy",
    "Infrequent-non-binge",
    "Frequent-non-binge",
    "Binge",
]
HISTORY6_CATS = [
    "lifelong_or_former_light",
    "former_moderate",
    "former_heavy",
    "current_light",
    "current_moderate",
    "current_heavy",
]
ABSTAINER_REFINED = [
    "abstainer_never_or_former_light",
    "former_moderate",
    "former_heavy",
]

def normalize_drink_labels(s: pd.Series) -> pd.Series:
    """drinkStatus as an object series, so missing values stay None-like for grouping."""
    return s.astype("object")


def numeric(s: pd.Series) -> pd.Series:
    """Coerce a series to numeric values."""
    return pd.to_numeric(s, errors="coerce")


def alcohol_frequency_days_per_week(
    freq: pd.Series,
    monthly_freq: pd.Series | None = None,
) -> pd.Series:
    """Convert MIDUS alcohol-frequency codes to approximate days/week."""
    f = numeric(freq)
    m = numeric(monthly_freq) if monthly_freq is not None else pd.Series(np.nan, index=f.index)
    out = pd.Series(np.nan, index=f.index, dtype=float)

    out.loc[f == 1] = 7.0
    out.loc[f == 2] = 5.5
    out.loc[f == 3] = 3.5
    out.loc[f == 4] = 1.5

    monthly_days = pd.Series(np.nan, index=f.index, dtype=float)
    monthly_days.loc[m == 1] = 3.5
    monthly_days.loc[m == 2] = 1.5
    monthly_days.loc[m == 3] = 0.5
    monthly_days.loc[m == 4] = 0.0
    out.loc[f == 5] = monthly_days.loc[f == 5] / 4.345

    out.loc[f == 6] = 0.0
    return out


def clean_quantity(s: pd.Series) -> pd.Series:
    """Clean drink-count fields and common missing/refused codes."""
    x = numeric(s)
    x = x.mask(x.isin([97, 98, 99]))
    x = x.mask(x < 0)
    return x


def weekly_volume(
    freq: pd.Series,
    monthly_freq: pd.Series | None,
    drinks_per_day: pd.Series,
) -> pd.Series:
    """Approximate drinks/week from frequency and usual drinks per drinking day."""
    days = alcohol_frequency_days_per_week(freq, monthly_freq)
    drinks = clean_quantity(drinks_per_day)
    vol = days * drinks
    vol.loc[days == 0] = 0.0
    return vol


def volume_class(volume: pd.Series, female: pd.Series) -> pd.Series:
    """Classify weekly volume as light/moderate/heavy using sex-specific cutoffs."""
    vol = numeric(volume)
    fem = numeric(female)
    threshold = pd.Series(np.nan, index=vol.index, dtype=float)
    threshold.loc[fem == 1] = 8.0
    threshold.loc[fem == 0] = 15.0

    out = pd.Series(pd.NA, index=vol.index, dtype="object")
    out.loc[(vol >= 0) & (vol < 1)] = "light"
    out.loc[(vol >= 1) & (vol < threshold)] = "moderate"
    out.loc[(vol >= threshold) & threshold.notna()] = "heavy"
    return out


def raw_current_drink_status(df: pd.DataFrame, prefix: str) -> pd.Series:
    """Recreate the current 4-level drinkStatus from raw PA50/PA51/PA53."""
    pa50 = numeric(df[f"{prefix}PA50"])
    pa51 = numeric(df[f"{prefix}PA51"])
    pa51 = pa51.mask(pa51 == 6)
    pa53 = clean_quantity(df[f"{prefix}PA53"])
    med51 = pa51.median(skipna=True)

    out = pd.Series(pd.NA, index=df.index, dtype="object")
    out.loc[pa50 == 2] = "Past-month abstainer"
    out.loc[(pa50 == 1) & (pa53 > 1)] = "Binge"
    out.loc[(pa50 == 1) & (pa53 <= 1) & (pa51 < med51)] = "Frequent-non-binge"
    out.loc[(pa50 == 1) & (pa53 <= 1) & (pa51 >= med51)] = "Infrequent-non-binge"
    return out


def derive_raw_alcohol_history_for_wave(cohort: str, cfg: dict) -> pd.DataFrame:
    """Read raw PA50-PA57 alcohol fields for one wave and derive volumes."""
    prefix = cfg["prefix"]
    id_col = cfg["id_col"]
    cols = [
        id_col,
        f"{prefix}PA50",
        f"{prefix}PA51",
        f"{prefix}PA51A",
        f"{prefix}PA52",
        f"{prefix}PA53",
        f"{prefix}PA54",
        f"{prefix}PA54A",
        f"{prefix}PA55",
        f"{prefix}PA56",
        f"{prefix}PA57",
    ]
    df, _ = pyreadstat.read_sav(str(cfg["path"]), usecols=cols)

    return pd.DataFrame(
        {
            "M2MRID": df[id_col],
            "cohort_raw_alcohol": cohort,
            "raw_current_status": raw_current_drink_status(df, prefix),
            "current_weekly_drinks": weekly_volume(
                df[f"{prefix}PA51"],
                df[f"{prefix}PA51A"],
                df[f"{prefix}PA52"],
            ),
            "max_weekly_drinks": weekly_volume(
                df[f"{prefix}PA54"],
                df[f"{prefix}PA54A"],
                df[f"{prefix}PA55"],
            ),
            "age_started_max_drinking": clean_quantity(df[f"{prefix}PA56"]),
            "years_at_max_drinking": clean_quantity(df[f"{prefix}PA57"]),
        }
    )


def derive_raw_alcohol_history(raw_waves: dict | None = None) -> pd.DataFrame:
    """Read both baseline cohorts and return raw alcohol-history derivatives."""
    waves = raw_waves or RAW_WAVES
    pieces = [derive_raw_alcohol_history_for_wave(cohort, cfg) for cohort, cfg in waves.items()]
    raw = pd.concat(pieces, ignore_index=True)
    raw["M2MRID"] = numeric(raw["M2MRID"])
    return raw.drop_duplicates("M2MRID", keep="first")


def add_refined_drinking_variables(master: pd.DataFrame, raw: pd.DataFrame | None) -> pd.DataFrame:
    """Merge raw alcohol history and derive refined drinking-history categories."""
    df = master.copy()
    if raw is not None and "M2MRID" in raw.columns:
        raw_add = raw[[c for c in raw.columns if c == "M2MRID" or c not in df.columns]].copy()
        if len(raw_add.columns) > 1:
            raw_add["M2MRID"] = numeric(raw_add["M2MRID"])
            df["M2MRID"] = numeric(df["M2MRID"])
            df = df.merge(raw_add, on="M2MRID", how="left")

    if "drinkStatus" not in df.columns:
        df["drinkStatus"] = pd.NA
    df["drinkStatus_original"] = normalize_drink_labels(df["drinkStatus"])

    for col in ("current_weekly_drinks", "max_weekly_drinks"):
        if col not in df.columns:
            df[col] = np.nan

    current_class = volume_class(df["current_weekly_drinks"], df["female"])
    max_class = volume_class(df["max_weekly_drinks"], df["female"])
    df["current_volume_class"] = current_class
    df["max_volume_class"] = max_class

    refined = pd.Series(pd.NA, index=df.index, dtype="object")
    history6 = pd.Series(pd.NA, index=df.index, dtype="object")

    is_abstainer = df["drinkStatus_original"] == "Past-month abstainer"
    is_current = df["drinkStatus_original"].isin(CURRENT_DRINKER_CATS)

    refined.loc[is_abstainer & (max_class == "light")] = "abstainer_never_or_former_light"
    refined.loc[is_abstainer & (max_class == "moderate")] = "former_moderate"
    refined.loc[is_abstainer & (max_class == "heavy")] = "former_heavy"
    refined.loc[is_current] = df.loc[is_current, "drinkStatus_original"]

    history6.loc[is_abstainer & (max_class == "light")] = "lifelong_or_former_light"
    history6.loc[is_abstainer & (max_class == "moderate")] = "former_moderate"
    history6.loc[is_abstainer & (max_class == "heavy")] = "former_heavy"
    history6.loc[is_current & (current_class == "light")] = "current_light"
    history6.loc[is_current & (current_class == "moderate")] = "current_moderate"
    history6.loc[is_current & (current_class == "heavy")] = "current_heavy"

    df["drinkStatus_refined"] = pd.Categorical(refined, categories=REFINED_CATS)
    df["drinkHistory6"] = pd.Categorical(history6, categories=HISTORY6_CATS)
    return df


def load_refined_master(master_csv: str | Path = MASTER_CSV) -> pd.DataFrame:
    """Load the W1 master file and add raw-derived refined drinking variables."""
    master = pd.read_csv(master_csv, low_memory=False)
    raw = derive_raw_alcohol_history()
    return add_refined_drinking_variables(master, raw)


def load_baseline_biomarkers() -> pd.DataFrame:
    """Load pooled W1 allostatic-load biomarker rows."""
    bio_m2 = pd.read_csv(BIO_DIR / "allostatic_load_M2.csv")
    bio_m2 = bio_m2.rename(columns={"M2ID": "M2MRID"})
    bio_m2["cohort_biomarker"] = "MIDUS2"

    bio_mr1 = pd.read_csv(BIO_DIR / "allostatic_load_MR1.csv")
    bio_mr1 = bio_mr1.rename(columns={"MRID": "M2MRID"})
    bio_mr1["cohort_biomarker"] = "MIDUSR"

    bio = pd.concat([bio_m2, bio_mr1], ignore_index=True)
    bio["M2MRID"] = numeric(bio["M2MRID"])
    return bio.drop_duplicates("M2MRID", keep="first")


def biomarker_analysis_table(df: pd.DataFrame) -> pd.DataFrame:
    """Merge refined drinking fields onto W1 AL rows."""
    keep_cols = [
        "M2MRID",
        "female",
        "age_W1",
        "drinkStatus_original",
        "drinkStatus_refined",
        "drinkHistory6",
        "current_weekly_drinks",
        "max_weekly_drinks",
        "current_volume_class",
        "max_volume_class",
    ]
    base = df[[c for c in keep_cols if c in df.columns]].copy()
    bio = load_baseline_biomarkers()
    merged = bio.merge(base, on="M2MRID", how="inner")
    return merged[merged["drinkStatus_refined"].notna()].copy()


def summarize_scores(df: pd.DataFrame, group_col: str, categories: list[str]) -> pd.DataFrame:
    """Summarize AL score columns by drinking group."""
    score_cols = ["ALcomp_zscore"]
    rows = []
    for score in score_cols:
        if score not in df.columns:
            continue
        for cat in categories:
            vals = numeric(df.loc[df[group_col].astype("object") == cat, score]).dropna()
            rows.append(
                {
                    "group_variable": group_col,
                    "group": cat,
                    "score": score,
                    "n": int(len(vals)),
                    "mean": float(vals.mean()) if len(vals) else np.nan,
                    "sd": float(vals.std()) if len(vals) > 1 else np.nan,
                    "median": float(vals.median()) if len(vals) else np.nan,
                    "min": float(vals.min()) if len(vals) else np.nan,
                    "max": float(vals.max()) if len(vals) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def subsystem_summary(
    df: pd.DataFrame,
    group_col: str,
    categories: list[str],
    systems: list[str],
) -> pd.DataFrame:
    """Summarize system-level AL dysregulation flags by drinking group."""
    rows = []
    for sys_name in systems:
        col = f"AL_sys_{sys_name}"
        if col not in df.columns:
            continue
        valid = df[col].notna() & df[group_col].notna()
        ct = pd.crosstab(df.loc[valid, group_col].astype("object"), df.loc[valid, col])
        chi2 = p = np.nan
        if ct.shape[0] >= 2 and ct.shape[1] >= 2:
            chi2, p, _, _ = stats.chi2_contingency(ct)
        for cat in categories:
            vals = numeric(df.loc[df[group_col].astype("object") == cat, col]).dropna()
            rows.append(
                {
                    "group_variable": group_col,
                    "group": cat,
                    "subsystem": sys_name,
                    "n": int(len(vals)),
                    "prop_dysregulated": float(vals.mean()) if len(vals) else np.nan,
                    "chi2": float(chi2) if not np.isnan(chi2) else np.nan,
                    "chi2_p": float(p) if not np.isnan(p) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_drinking_transition_table(w1: pd.DataFrame, w2: pd.DataFrame) -> pd.DataFrame:
    """Join W1 and W2 drinking status and classify broad drinking transitions."""
    base_cols = [
        "M2MRID",
        "drinkStatus",
        "drinkStatus_refined",
        "drinkHistory6",
        "age_W1",
        "female",
        "education",
        "FInoPsy_W1",
        "seeMedDoc",
        "mentalHealth",
        "controlOverHealth",
    ]
    left = w1[[c for c in base_cols if c in w1.columns]].copy()
    left = left.rename(columns={"drinkStatus": "drinkStatus_W1"})
    right = w2[[c for c in ["M2MRID", "drinkStatus"] if c in w2.columns]].copy()
    right = right.rename(columns={"drinkStatus": "drinkStatus_W2"})

    out = left.merge(right, on="M2MRID", how="left")
    out["drinkStatus_W1"] = normalize_drink_labels(out["drinkStatus_W1"])
    out["drinkStatus_W2"] = normalize_drink_labels(out["drinkStatus_W2"])

    w1_abst = out["drinkStatus_W1"] == "Past-month abstainer"
    w1_cur = out["drinkStatus_W1"].isin(CURRENT_DRINKER_CATS)
    w2_abst = out["drinkStatus_W2"] == "Past-month abstainer"
    w2_cur = out["drinkStatus_W2"].isin(CURRENT_DRINKER_CATS)
    w2_obs = out["drinkStatus_W2"].notna()

    transition = pd.Series(pd.NA, index=out.index, dtype="object")
    transition.loc[w1_abst & w2_abst] = "remained_abstainer"
    transition.loc[w1_abst & w2_cur] = "initiated_current_drinking"
    transition.loc[w1_cur & w2_abst] = "became_past_month_abstainer"
    transition.loc[w1_cur & w2_cur] = "remained_current_drinker"
    out["drink_transition_W1_W2"] = transition
    out["became_abstainer_W2"] = np.where(
        w1_cur & w2_obs,
        np.where(w2_abst, 1.0, 0.0),
        np.nan,
    )
    return out
