"""
Derived demographic, work, and relationship variables.

Each function accepts a DataFrame with raw SPSS column names (passed as
parameters, not hard-coded) so it works across all four cohorts (MIDUS2,
MIDUSR, MIDUS3, MR2). Return value is a pandas Series (single derived
column) or a DataFrame (when noted).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

__all__ = [
    # helpers
    "create_datetime",
    "year_gap",
    "get_latest_date",
    # demographics
    "derive_female",
    "derive_hispanic",
    "derive_race",
    "derive_sexual_orientation",
    # work
    "derive_work_for_pay",
    "derive_work_status",
    "derive_manage_role",
    "derive_occ_group",
    "derive_work_industry",
    "derive_occ_prestige",
    # relationships
    "derive_relat_status",
    "derive_marr_duration",
    "derive_times_married",
    "derive_relat_cohab_length",
    "derive_partner_age_gap",
    "derive_partner_edu_gap",
    "derive_sex_partner",
    # caregiving
    "derive_caregiv_status",
    "derive_caregiv_intensity",
    # smoking
    "derive_smoke_status",
    "derive_smoke_duration",
    "derive_smoke_amount",
    "derive_smoke_exposure",
    "derive_smoke_exposure_3item",
    # drinking
    "derive_drink_status",
    "derive_drink_exposure",
    "derive_drink_exposure_pa59",
    # family history
    "derive_family_heart_attack",
    "derive_family_cancer",
    # psychological
    "derive_sub_achievement",
    # ACE composites
    "derive_ace_composite",
    "derive_age_split_event",
    # physical activity
    "derive_physical_activity",
    # health outcomes
    "derive_health_affect_work",
    "derive_start_middle_age",
    "derive_end_middle_age",
]


# Helpers

def _safe_col(df: pd.DataFrame, col: str) -> pd.Series:
    """Return column if it exists, else a Series of NaN."""
    if col in df.columns:
        return df[col].copy()
    return pd.Series(np.nan, index=df.index, name=col)


def create_datetime(month: pd.Series, year: pd.Series) -> pd.Series:
    """
    Build a datetime64 from month (1-12) and year columns.

    Returns NaT for any row where month is not in [1, 12] or year is
    non-positive / NaN.
    """
    m = pd.to_numeric(month, errors="coerce")
    y = pd.to_numeric(year, errors="coerce")
    valid = m.between(1, 12) & y.notna() & (y > 0)
    out = pd.Series(pd.NaT, index=month.index, dtype="datetime64[ns]")
    if valid.any():
        # Build "YYYY-MM-01" strings for valid rows
        date_str = (
            y[valid].astype(int).astype(str)
            + "-"
            + m[valid].astype(int).astype(str).str.zfill(2)
            + "-01"
        )
        out[valid] = pd.to_datetime(date_str, format="%Y-%m-%d", errors="coerce")
    return out


def year_gap(start_date: pd.Series, end_date: pd.Series) -> pd.Series:
    """
    Difference in years (float) between two datetime Series.

    Uses 365.25 days/year.
    """
    delta = end_date - start_date
    return delta.dt.total_seconds() / (365.25 * 24 * 3600)


def get_latest_date(*date_cols: pd.Series) -> pd.Series:
    """Element-wise max across date columns, ignoring NaT."""
    combined = pd.concat(list(date_cols), axis=1)
    return combined.max(axis=1)


# Demographics

def derive_female(df: pd.DataFrame, prsex_col: str) -> pd.Series:
    """PRSEX == 2 -> 1 (female), else 0."""
    sex = _safe_col(df, prsex_col)
    return np.where(sex == 2, 1, np.where(sex.isna(), np.nan, 0)).astype(float)


def derive_hispanic(df: pd.DataFrame, pf1_col: str) -> pd.Series:
    """
    PF1 != 1 -> 1 (hispanic), else 0. Uses PF1, not PF2.
    """
    val = _safe_col(df, pf1_col)
    return np.where(val.isna(), np.nan, np.where(val != 1, 1, 0)).astype(float)


def derive_race(
    df: pd.DataFrame,
    race_col: str,
) -> pd.Series:
    """
    Race from a single cohort-specific PF7A column, with recode 4/5->3, 6->4.

    Columns per cohort:
        midus2: B1PF7A, midusr: RA1PF7A, midus3: C1PF7A, mr2: RB1PF7A
    """
    val = _safe_col(df, race_col)
    val = val.replace({4: 3, 5: 3, 6: 4})
    return val


def derive_sexual_orientation(df: pd.DataFrame, col: str) -> pd.Series:
    """Passthrough -- return the column as-is."""
    return _safe_col(df, col)


# Work

def derive_work_for_pay(
    df: pd.DataFrame,
    pb4n_col: str,
    pb6_col: str,
    pbwork_col: str,
) -> pd.Series:
    """
    Works for pay: 1 if PB4N is missing, PB6 == 1, or PBWORK in {1, 2};
    otherwise 0 if PB6 == 2 or PBWORK in {4, 5, 6, 10}; else NA.
    """
    pb4n = _safe_col(df, pb4n_col)
    pb6 = _safe_col(df, pb6_col)
    pbwork = _safe_col(df, pbwork_col)

    out = pd.Series(np.nan, index=df.index, dtype=float)
    cond1 = pb4n.isna() | (pb6 == 1) | pbwork.isin([1, 2])
    cond0 = (pb6 == 2) | pbwork.isin([4, 5, 6, 10])
    # Assign lowest priority first so higher priority overwrites.
    out = out.where(~cond0, 0)
    out = out.where(~cond1, 1)
    return out


def derive_work_status(df: pd.DataFrame, pbwork_col: str) -> pd.Series:
    """
    8-level categorical recode from PBWORK.
    Returns string labels (object dtype), NaN where PBWORK is unmatched.
    """
    pbwork = _safe_col(df, pbwork_col)

    conditions = [
        pbwork.isin([1, 9]),
        pbwork == 2,
        pbwork.isin([3, 4]),
        pbwork == 5,
        pbwork == 6,
        pbwork.isin([7, 8]),
        pbwork == 10,
        pbwork == 11,
    ]
    choices = [
        "Working now",
        "Self-employed",
        "Laid-off/Seeking-work",
        "Retired",
        "Homemaker",
        "Student",
        "Disabled",
        "Other work",
    ]
    out = np.select(conditions, choices, default=None)
    # np.select fills '0' for no match; set to NaN.
    result = pd.Series(out, index=df.index, dtype=object)
    result[result == "0"] = np.nan
    return result


def derive_manage_role(
    df: pd.DataFrame,
    pb7_col: str,
) -> pd.Series:
    """
    Management role: PB7 as-is.

    Not restricted to current workers, for consistency
    across waves.
    """
    return _safe_col(df, pb7_col)


def derive_occ_group(
    df: pd.DataFrame,
    pocmaj_col: str,
    pocpmj_col: str | None = None,
    cohort: str = "midus2",
    psoc_col: str | None = None,
) -> pd.Series:
    """
    Occupation group with fallback + cohort-specific recoding.

    midus2: POCMAJ/POCPMJ fallback, then B1PSOC starting with '55-' -> 10.
    midusr/midus3/mr2: POCMAJ/POCPMJ fallback, then remap category codes.
    """
    primary = _safe_col(df, pocmaj_col)
    if pocpmj_col is not None and pocpmj_col in df.columns:
        fallback = df[pocpmj_col]
        result = primary.where(primary.notna(), fallback)
    else:
        result = primary

    if cohort == "midus2" and psoc_col is not None and psoc_col in df.columns:
        mask = df[psoc_col].astype(str).str.startswith("55-")
        result = result.where(~mask, 10)
    elif cohort in ("midusr", "midus3", "mr2"):
        remap = {1: 1, 2: 2, 3: 6, 4: 4, 5: 5, 6: 7, 7: 7, 8: 3, 9: 8, 10: 9, 11: 10}
        result = result.map(remap)

    return result


def derive_work_industry(
    df: pd.DataFrame,
    pindmj_col: str,
    ppinmj_col: str | None = None,
    cohort: str = "midus2",
    pind_raw_col: str | None = None,
    pindp_raw_col: str | None = None,
    industry_map: dict | None = None,
) -> pd.Series:
    """
    Work industry with fallback.

    midus2: map raw census codes (B1PIND/B1PINDP) -> industry_code via lookup.
    midusr/midus3/mr2: PINDMJ/PPINMJ fallback (already industry codes).
    """
    if cohort == "midus2" and pind_raw_col is not None and industry_map is not None:
        raw = _safe_col(df, pind_raw_col)
        if pindp_raw_col is not None and pindp_raw_col in df.columns:
            raw_fallback = df[pindp_raw_col]
            raw = raw.where(raw.notna(), raw_fallback)
        return raw.map(industry_map)

    primary = _safe_col(df, pindmj_col)
    if ppinmj_col is not None and ppinmj_col in df.columns:
        fallback = df[ppinmj_col]
        return primary.where(primary.notna(), fallback)
    return primary


def derive_occ_prestige(df: pd.DataFrame, ppinmj_col: str) -> pd.Series:
    """Passthrough -- occupational prestige score."""
    return _safe_col(df, ppinmj_col)


# Relationships

def derive_relat_status(
    df: pd.DataFrame,
    pb19_col: str,
    pb30_col: str,
) -> pd.Series:
    """
    6-level relationship status. Rules, first match wins:
        PB19 == 1              -> Married
        PB30 == 1              -> Cohabiting(not married)
        PB30 == 0 & PB19 == 2  -> Separated(not cohabiting)
        PB19 == 3 / 4 / 5      -> Divorced / Widowed / Never married (not cohabiting)
        otherwise NA
    """
    pb19 = _safe_col(df, pb19_col)
    pb30 = _safe_col(df, pb30_col)

    out = pd.Series(np.nan, index=df.index, dtype=object)
    # Assign lowest priority first so higher priority overwrites.
    out = out.where(~(pb19 == 5), "Never married(not cohabiting)")
    out = out.where(~(pb19 == 4), "Widowed(not cohabiting)")
    out = out.where(~(pb19 == 3), "Divorced(not cohabiting)")
    out = out.where(~((pb30 == 0) & (pb19 == 2)), "Separated(not cohabiting)")
    out = out.where(~(pb30 == 1), "Cohabiting(not married)")
    out = out.where(~(pb19 == 1), "Married")
    return out


def derive_marr_duration(
    df: pd.DataFrame,
    *,
    interview_mo_col: str,
    interview_yr_col: str,
    first_marr_mo_col: str,
    first_marr_yr_col: str,
    first_died_mo_col: str,
    first_died_yr_col: str,
    first_sep_mo_col: str,
    first_sep_yr_col: str,
    first_div_mo_col: str,
    first_div_yr_col: str,
    recent_marr_mo_col: str,
    recent_marr_yr_col: str,
    recent_sep_mo_col: str,
    recent_sep_yr_col: str,
    recent_div_mo_col: str,
    recent_div_yr_col: str,
    recent_died_mo_col: str,
    recent_died_yr_col: str,
    age_col: str,
    relat_status: pd.Series,
) -> pd.Series:
    """
    Marriage duration in years.

    Steps:
      1. Build 8 date columns (first/recent marriage/death/separation/divorce).
      2. Current marriage start: latest of firstMarr, recentMarr.
      3. End date: interview date if the marriage is ongoing, else the latest end date.
      4. year_gap(start, end).
      5. Never-married -> 0; clamp to [0, age-14]; negative -> NA.
    """
    interview = create_datetime(_safe_col(df, interview_mo_col),
                                _safe_col(df, interview_yr_col))
    firstMarr = create_datetime(_safe_col(df, first_marr_mo_col),
                                _safe_col(df, first_marr_yr_col))
    firstDied = create_datetime(_safe_col(df, first_died_mo_col),
                                _safe_col(df, first_died_yr_col))
    firstSep = create_datetime(_safe_col(df, first_sep_mo_col),
                               _safe_col(df, first_sep_yr_col))
    firstDiv = create_datetime(_safe_col(df, first_div_mo_col),
                               _safe_col(df, first_div_yr_col))
    recentMarr = create_datetime(_safe_col(df, recent_marr_mo_col),
                                 _safe_col(df, recent_marr_yr_col))
    recentSep = create_datetime(_safe_col(df, recent_sep_mo_col),
                                _safe_col(df, recent_sep_yr_col))
    recentDiv = create_datetime(_safe_col(df, recent_div_mo_col),
                                _safe_col(df, recent_div_yr_col))
    recentDied = create_datetime(_safe_col(df, recent_died_mo_col),
                                 _safe_col(df, recent_died_yr_col))

    lastMarrEnd = get_latest_date(firstDied, firstSep, firstDiv)
    recentMarrEnd = get_latest_date(recentSep, recentDied, recentDiv)
    currMarr = get_latest_date(firstMarr, recentMarr)

    # endMarr logic:
    #   If (recentMarr known AND recentMarrEnd unknown) OR
    #      (currMarr known AND lastMarrEnd unknown AND recentMarrEnd unknown)
    #   => marriage is ongoing, use interview date.
    #   Otherwise: latest of lastMarrEnd, recentMarrEnd.
    ongoing = (
        (recentMarr.notna() & recentMarrEnd.isna())
        | (currMarr.notna() & lastMarrEnd.isna() & recentMarrEnd.isna())
    )
    endMarr = get_latest_date(lastMarrEnd, recentMarrEnd)
    endMarr = endMarr.where(~ongoing, interview)

    duration = year_gap(currMarr, endMarr)

    age = _safe_col(df, age_col)
    never = relat_status == "Never married(not cohabiting)"
    duration = duration.where(~never, 0)
    invalid = (duration > (age - 14)) | (duration < 0)
    duration = duration.where(~(invalid & ~never), np.nan)

    return duration


def derive_times_married(
    df: pd.DataFrame,
    pb20_col: str,
    pb19_col: str,
) -> pd.Series:
    """Times married: PB20 if present, 0 if never married (PB19 == 5), else NA."""
    pb20 = _safe_col(df, pb20_col)
    pb19 = _safe_col(df, pb19_col)
    out = pd.Series(np.nan, index=df.index, dtype=float)
    out = out.where(~(pb19 == 5), 0)
    out = out.where(~pb20.notna(), pb20)
    return out


def derive_relat_cohab_length(
    df: pd.DataFrame,
    pb31m_col: str,
    pb31n_col: str,
    marr_duration: pd.Series,
) -> pd.Series:
    """
    Cohabitation length in years.

    PB31N converted by its unit PB31M (2 = weeks, 3 = months, 4 = years);
    falls back to marriage duration where that is missing.
    """
    pb31m = _safe_col(df, pb31m_col)
    pb31n = _safe_col(df, pb31n_col)

    cohab = pd.Series(np.nan, index=df.index, dtype=float)
    cohab = cohab.where(~(pb31m == 4), pb31n)
    cohab = cohab.where(~(pb31m == 3), pb31n / 12)
    cohab = cohab.where(~(pb31m == 2), pb31n / 52)

    return cohab.where(cohab.notna(), marr_duration)


def derive_partner_age_gap(
    df: pd.DataFrame,
    age_col: str,
    partner_age_col: str,
) -> pd.Series:
    """Respondent age minus partner age."""
    return _safe_col(df, age_col) - _safe_col(df, partner_age_col)


def derive_partner_edu_gap(
    df: pd.DataFrame,
    edu_col: str,
    partner_edu_col: str,
) -> pd.Series:
    """Respondent education minus partner education."""
    return _safe_col(df, edu_col) - _safe_col(df, partner_edu_col)


def derive_sex_partner(
    df: pd.DataFrame,
    gate_col: str | None,
    count_col: str,
) -> pd.Series:
    """
    Sex partner count.

    midus2: single count column (B1SM7, range 0-6).
    midusr/midus3/mr2: gate question (1=had partner, 2=no) + count (1-6).
        If gate == 2 -> 0, else use count value.
    """
    if gate_col is None:
        return _safe_col(df, count_col)

    gate = _safe_col(df, gate_col)
    count = _safe_col(df, count_col)
    result = count.copy()
    result[gate == 2] = 0
    return result


# Caregiving

def derive_caregiv_status(
    df: pd.DataFrame,
    pd1_col: str,
    pd7_col: str,
    pd18_col: str,
) -> pd.Series:
    """
    3-level caregiving status (Never / Former / Current). Rules, first match wins:
        Never:   PD18 and PD7 missing, or PD1 == 0 and PD18 missing
        Former:  PD1 == 1 and PD7 == 0, or PD1 == 0 and PD18 present
        Current: PD1 == 1 and PD7 == 1
        otherwise NA
    """
    d1 = _safe_col(df, pd1_col)
    d7 = _safe_col(df, pd7_col)
    d18 = _safe_col(df, pd18_col)

    out = pd.Series(np.nan, index=df.index, dtype=object)

    # Current caregiver (lowest priority, assigned first)
    cond_curr = (d1 == 1) & (d7 == 1)
    out = out.where(~cond_curr, "Current caregiver")

    # Former caregiver
    cond_former = ((d1 == 1) & (d7 == 0)) | ((d1 == 0) & d18.notna())
    out = out.where(~cond_former, "Former caregiver")

    # Never caregiver (highest priority)
    cond_never = (d18.isna() & d7.isna()) | ((d1 == 0) & d18.isna())
    out = out.where(~cond_never, "Never caregiver")

    return out


def derive_caregiv_intensity(
    df: pd.DataFrame,
    pd14_col: str,
    pd15_col: str,
    caregiv_status: pd.Series,
) -> pd.Series:
    """
    Hours of caregiving per day: PD14 * PD15 / 24, set to 0 for never
    caregivers and for former caregivers with PD14 missing.
    """
    d14 = _safe_col(df, pd14_col)
    d15 = _safe_col(df, pd15_col)

    raw = d14 * d15 / 24.0
    out = raw.copy()
    out = out.where(
        ~((caregiv_status == "Former caregiver") & d14.isna()), 0
    )
    out = out.where(~(caregiv_status == "Never caregiver"), 0)
    return out


# Smoking

def derive_smoke_status(
    df: pd.DataFrame,
    pa38a_col: str,
    pa39_col: str,
    pa40_col: str,
    pa43_col: str,
    pa44_col: str,
) -> pd.Series:
    """
    5-level smoking classification. Rules, first match wins:
        PA38A == 0                               -> Never smoked regular
        PA39 == 0 & (PA43 < median or missing)   -> Former light smoker
        PA39 == 0 & PA43 >= median               -> Former heavy smoker
        PA39 == 1 & (PA40 < median or missing)   -> Current light smoker
        PA39 == 1 & PA40 >= median               -> Current heavy smoker
        PA38A missing & PA44 != 1                -> Never smoked regular
        otherwise NA

    PA43 is cigarettes/day when quit, PA40 current cigarettes/day. Light/heavy
    splits use the sample median, so thresholds are data-dependent.
    """
    a38a = _safe_col(df, pa38a_col)
    a39 = _safe_col(df, pa39_col)
    a40 = _safe_col(df, pa40_col)
    a43 = _safe_col(df, pa43_col)
    a44 = _safe_col(df, pa44_col)

    med43 = a43.median()
    med40 = a40.median()

    out = pd.Series(np.nan, index=df.index, dtype=object)

    # Assign bottom-up (highest priority overwrites)
    # 6. PA38A missing & PA44 != 1 -> Never smoked
    c6 = a38a.isna() & (a44 != 1)
    out = out.where(~c6, "Never smoked regular")

    # 5. Current heavy
    c5 = (a39 == 1) & (a40 >= med40)
    out = out.where(~c5, "Current heavy smoker")

    # 4. Current light
    c4 = (a39 == 1) & ((a40 < med40) | a40.isna())
    out = out.where(~c4, "Current light smoker")

    # 3. Former heavy
    c3 = (a39 == 0) & (a43 >= med43)
    out = out.where(~c3, "Former heavy smoker")

    # 2. Former light
    c2 = (a39 == 0) & ((a43 < med43) | a43.isna())
    out = out.where(~c2, "Former light smoker")

    # 1. Never smoked (highest priority)
    c1 = a38a == 0
    out = out.where(~c1, "Never smoked regular")

    return out


def derive_smoke_duration(
    df: pd.DataFrame,
    pa38a_col: str,
    pa38b_col: str,
    pa39_col: str,
    pa42_col: str,
    pa44_col: str,
    age_col: str,
) -> pd.Series:
    """
    Years of smoking.

    Never smokers (PA38A == 0, or PA38A missing and PA44 != 1) get 0; former
    smokers (PA39 == 0) PA42 - PA38B; current smokers (PA39 == 1) age - PA38B.
    Negative durations -> NA.
    """
    a38a = _safe_col(df, pa38a_col)
    a38b = _safe_col(df, pa38b_col)
    a39 = _safe_col(df, pa39_col)
    a42 = _safe_col(df, pa42_col)
    a44 = _safe_col(df, pa44_col)
    age = _safe_col(df, age_col)

    out = pd.Series(np.nan, index=df.index, dtype=float)

    # Current smoker
    out = out.where(~(a39 == 1), age - a38b)
    # Former smoker
    out = out.where(~(a39 == 0), a42 - a38b)
    # Never smoked
    never = (a38a == 0) | (a38a.isna() & (a44 != 1))
    out = out.where(~never, 0)

    # Negative -> NA
    out = out.where(~(out < 0), np.nan)
    return out


def derive_smoke_amount(
    df: pd.DataFrame,
    pa38a_col: str,
    pa40_col: str,
    pa43_col: str,
    pa44_col: str,
) -> pd.Series:
    """
    Cigarettes per day (current or former peak). Rules, first match wins:
        PA38A missing & PA44 != 1  -> 0
        PA40 present               -> PA40
        PA43 present               -> PA43
        PA38A == 2                 -> 1
        otherwise NA
    """
    a38a = _safe_col(df, pa38a_col)
    a40 = _safe_col(df, pa40_col)
    a43 = _safe_col(df, pa43_col)
    a44 = _safe_col(df, pa44_col)

    out = pd.Series(np.nan, index=df.index, dtype=float)

    # lowest priority first
    out = out.where(~(a38a == 2), 1)
    out = out.where(~a43.notna(), a43)
    out = out.where(~a40.notna(), a40)
    out = out.where(~(a38a.isna() & (a44 != 1)), 0)

    return out


def derive_smoke_exposure(
    df: pd.DataFrame,
    item_cols: Sequence[str],
    min_nonmissing: int = 4,
) -> pd.Series:
    """
    Mean of smoke-exposure items when at least `min_nonmissing` are present.

    W1: 6 items (PA45A/B/C, PA46, PA47, PA48), >= 4 present.
    W2: 3 items (PA46, PA47, PA48), >= 2 present.
    """
    cols_present = [c for c in item_cols if c in df.columns]
    if not cols_present:
        return pd.Series(np.nan, index=df.index, dtype=float)
    sub = df[cols_present].astype(float)
    n_valid = sub.notna().sum(axis=1)
    means = sub.mean(axis=1)
    return means.where(n_valid >= min_nonmissing, np.nan)


def derive_smoke_exposure_3item(
    df: pd.DataFrame,
    pa46_col: str,
    pa47_col: str,
    pa48_col: str,
) -> pd.Series:
    """
    3-item adult-exposure-only smoke exposure (W2-comparable).

    Mean of PA46, PA47, PA48 when >= 2 are present.
    """
    return derive_smoke_exposure(df, [pa46_col, pa47_col, pa48_col], min_nonmissing=2)


# Drinking

def derive_drink_status(
    df: pd.DataFrame,
    pa50_col: str,
    pa51_col: str,
    pa53_col: str,
) -> pd.Series:
    """
    4-level drinking classification (2-axis: frequency x binge).

    PA50: "Had at least one drink past month" (0/1 after binary recode).
    PA51: "How often drink past month"; ordinal, reverse-coded:
          1=every day, 2=5-6 days/wk, 3=3-4 days/wk,
          4=1-2 days/wk, 5=<1 day/wk, 6=never (volunteered).
          Lower value = more frequent. PA51=6 set to NaN (contradicts PA50=1).
    PA53: "Times had 5+ drinks same occasion past month" (0-31).

    Categories (independent axes, no priority cascade):
      - 'Past-month abstainer'  : PA50 == 0
      - 'Infrequent-non-binge'  : PA51 >= median (<=2 days/wk) & PA53 <= 1
      - 'Frequent-non-binge'    : PA51 <  median (>=3 days/wk) & PA53 <= 1
      - 'Binge'                 : PA53 > 1  (any frequency)
    """
    a50 = _safe_col(df, pa50_col)
    a51 = _safe_col(df, pa51_col)
    a53 = _safe_col(df, pa53_col)

    # PA51=6 ("never drink" volunteered) contradicts PA50=1; treat as missing
    a51 = a51.where(a51 != 6, np.nan)

    med51 = a51.median()

    out = pd.Series(np.nan, index=df.index, dtype=object)

    # Binge: PA53 > 1 (any drinking frequency)
    out[a53 > 1] = "Binge"

    # Frequent-non-binge: PA51 < median (>=3 days/wk) & PA53 <= 1
    out[(a51 < med51) & (a53 <= 1)] = "Frequent-non-binge"

    # Infrequent-non-binge: PA51 >= median (<=2 days/wk) & PA53 <= 1
    out[(a51 >= med51) & (a53 <= 1)] = "Infrequent-non-binge"

    # Past-month abstainer: PA50 == 0 (overrides all; skip pattern)
    out[a50 == 0] = "Past-month abstainer"

    return out


def derive_drink_exposure(
    df: pd.DataFrame,
    pa58_col: str,
    pa59_col: str,
) -> pd.Series:
    """Sum of PA58 and PA59 when at least one is present."""
    cols_present = [c for c in [pa58_col, pa59_col] if c in df.columns]
    if not cols_present:
        return pd.Series(np.nan, index=df.index, dtype=float)
    sub = df[cols_present].astype(float)
    n_valid = sub.notna().sum(axis=1)
    sums = sub.sum(axis=1, min_count=1)
    return sums.where(n_valid >= 1, np.nan)


def derive_drink_exposure_pa59(df: pd.DataFrame, pa59_col: str) -> pd.Series:
    """
    Single-item binary: PA59 only (for W1 <-> MR2 comparison; MR2 lacks PA58).
    """
    return _safe_col(df, pa59_col).astype(float)


# Family history

def derive_family_heart_attack(
    df: pd.DataFrame,
    pa10a_col: str,
    pa10b_col: str,
    pa10c_col: str,
    pa10d_col: str,
    pa10e_col: str,
    pa10f_col: str,
    pa10g_col: str,
    pa10h_col: str,
) -> pd.Series:
    """
    Family heart-attack history: No family history / Parent / Sibling / Children.

    Highest priority first: PA10A (none), PA10B/C (parent), PA10D-G (sibling),
    PA10H (children); NA if no item is endorsed.
    """
    a = _safe_col(df, pa10a_col)
    b = _safe_col(df, pa10b_col)
    c = _safe_col(df, pa10c_col)
    d = _safe_col(df, pa10d_col)
    e = _safe_col(df, pa10e_col)
    f = _safe_col(df, pa10f_col)
    g = _safe_col(df, pa10g_col)
    h = _safe_col(df, pa10h_col)

    out = pd.Series(np.nan, index=df.index, dtype=object)
    out = out.where(~(h == 1), "Children")
    out = out.where(~((d == 1) | (e == 1) | (f == 1) | (g == 1)), "Sibling")
    out = out.where(~((b == 1) | (c == 1)), "Parent")
    # Not "None": read_csv would parse it as NaN.
    out = out.where(~(a == 1), "No family history")
    return out


def derive_family_cancer(
    df: pd.DataFrame,
    pa30a_col: str,
    pa30b_col: str,
    pa30c_col: str,
    pa30d_col: str,
    pa30e_col: str,
    pa30f_col: str,
) -> pd.Series:
    """
    Family cancer history: No family history / Parent / Sibling / Children.

    Same priority as derive_family_heart_attack: PA30A, PA30B/C, PA30D/E, PA30F.
    """
    a = _safe_col(df, pa30a_col)
    b = _safe_col(df, pa30b_col)
    c = _safe_col(df, pa30c_col)
    d = _safe_col(df, pa30d_col)
    e = _safe_col(df, pa30e_col)
    f = _safe_col(df, pa30f_col)

    out = pd.Series(np.nan, index=df.index, dtype=object)
    out = out.where(~(f == 1), "Children")
    out = out.where(~((d == 1) | (e == 1)), "Sibling")
    out = out.where(~((b == 1) | (c == 1)), "Parent")
    out = out.where(~(a == 1), "No family history")
    return out


# Psychological

def derive_sub_achievement(
    df: pd.DataFrame,
    pg5_col: str,
    pg5a_col: str,
) -> pd.Series:
    """
    6-level subjective achievement.

    PG5 == 1 with PG5A 1/2/3 -> 6/5/4; PG5 == 2 with PG5A 3/2/1 -> 3/2/1;
    otherwise NA.
    """
    g5 = _safe_col(df, pg5_col)
    g5a = _safe_col(df, pg5a_col)

    out = pd.Series(np.nan, index=df.index, dtype=float)
    out = out.where(~((g5 == 2) & (g5a == 1)), 1)
    out = out.where(~((g5 == 2) & (g5a == 2)), 2)
    out = out.where(~((g5 == 2) & (g5a == 3)), 3)
    out = out.where(~((g5 == 1) & (g5a == 3)), 4)
    out = out.where(~((g5 == 1) & (g5a == 2)), 5)
    out = out.where(~((g5 == 1) & (g5a == 1)), 6)
    return out


# ACE Composites

def derive_ace_composite(
    df: pd.DataFrame,
    item_cols: Sequence[str],
    min_frac: float = 0.5,
) -> pd.Series:
    """
    Sum of binary ACE items when at least ``min_frac`` fraction are non-NA.

    Parameters
    ----------
    item_cols : list of column names
    min_frac  : fraction of items that must be non-NA (default 0.5)
    """
    cols_present = [c for c in item_cols if c in df.columns]
    if not cols_present:
        return pd.Series(np.nan, index=df.index, dtype=float)
    sub = df[cols_present].astype(float)
    n_valid = sub.notna().sum(axis=1)
    threshold = len(item_cols) * min_frac
    sums = sub.sum(axis=1, min_count=1)
    return sums.where(n_valid >= threshold, np.nan)


def derive_age_split_event(
    df: pd.DataFrame,
    event_col: str,
    age_at_event_col: str,
    age_threshold: int = 18,
) -> tuple[pd.Series, pd.Series]:
    """
    Split a life-event binary into "young" and "old" versions based on the
    age-at-event column.

    Where age at event is known, for_young is 0 if it exceeds the threshold
    and for_old is 0 if it does not (e.g. SE11R with age SE11R1).

    Returns (for_young, for_old) pair.
    """
    event = _safe_col(df, event_col)
    age_at = _safe_col(df, age_at_event_col)

    happened_old = age_at.notna() & (age_at > age_threshold)
    happened_young = age_at.notna() & (age_at <= age_threshold)

    for_young = event.where(~happened_old, 0)
    for_old = event.where(~happened_young, 0)

    return for_young, for_old


# Physical Activity

def _pair_mean(df: pd.DataFrame, col_a: str, col_b: str) -> pd.Series:
    """Mean of two columns where at least one is non-NA."""
    a = _safe_col(df, col_a)
    b = _safe_col(df, col_b)
    stacked = pd.concat([a, b], axis=1)
    n_valid = stacked.notna().sum(axis=1)
    means = stacked.mean(axis=1)
    return means.where(n_valid >= 1, np.nan)


def derive_physical_activity(
    df: pd.DataFrame,
    *,
    vig_cols: Sequence[str],
    mod_cols: Sequence[str],
    light_cols: Sequence[str],
) -> pd.DataFrame:
    """
    Derive all physical activity composites.

    Parameters
    ----------
    vig_cols : 6 columns [job_A, job_B, chores_C, chores_D, leisure_E, leisure_F]
        Vigorous activity items (SA30 for MIDUS2, SA26 for MIDUSR/MIDUS3/MR2).
    mod_cols : 6 columns [job_A, job_B, chores_C, chores_D, leisure_E, leisure_F]
        Moderate activity items (SA31 for MIDUS2, SA27 for MIDUSR etc).
    light_cols : 6 columns [job_A, job_B, chores_C, chores_D, leisure_E, leisure_F]
        Light activity items (SA32 for MIDUS2, SA28 for MIDUSR etc).

    Returns
    -------
    DataFrame with columns:
        vmJobAct, vmChoresAct, vmLeisureAct,
        vmJobAct_vig, vmChoresAct_vig, vmLeisureAct_vig,
        lightAct
    """
    # Vigorous
    vjob = _pair_mean(df, vig_cols[0], vig_cols[1])
    vchores = _pair_mean(df, vig_cols[2], vig_cols[3])
    vleisure = _pair_mean(df, vig_cols[4], vig_cols[5])

    # Moderate
    mjob = _pair_mean(df, mod_cols[0], mod_cols[1])
    mchores = _pair_mean(df, mod_cols[2], mod_cols[3])
    mleisure = _pair_mean(df, mod_cols[4], mod_cols[5])

    # Light
    ljob = _pair_mean(df, light_cols[0], light_cols[1])
    lchores = _pair_mean(df, light_cols[2], light_cols[3])
    lleisure = _pair_mean(df, light_cols[4], light_cols[5])

    # Combined scores
    vmJobAct = vjob * 2 + mjob * 1
    vmChoresAct = vchores * 2 + mchores * 1
    vmLeisureAct = vleisure * 2 + mleisure * 1

    # Vigorous-only (MIDUS3-compatible)
    vmJobAct_vig = vjob * 2
    vmChoresAct_vig = vchores * 2
    vmLeisureAct_vig = vleisure * 2

    # Light activity (mean of 3 domains, >=1 non-NA)
    light_stack = pd.concat([ljob, lchores, lleisure], axis=1)
    n_valid = light_stack.notna().sum(axis=1)
    lightAct = light_stack.mean(axis=1).where(n_valid >= 1, np.nan)

    return pd.DataFrame(
        {
            "vmJobAct": vmJobAct,
            "vmChoresAct": vmChoresAct,
            "vmLeisureAct": vmLeisureAct,
            "vmJobAct_vig": vmJobAct_vig,
            "vmChoresAct_vig": vmChoresAct_vig,
            "vmLeisureAct_vig": vmLeisureAct_vig,
            "lightAct": lightAct,
        },
        index=df.index,
    )


# Health Outcomes

def derive_health_affect_work(
    df: pd.DataFrame,
    pa4_col: str,
    pa5_col: str,
) -> pd.Series:
    """Sum of PA4 and PA5, treating missing as 0."""
    a4 = _safe_col(df, pa4_col).astype(float)
    a5 = _safe_col(df, pa5_col).astype(float)
    return a4.fillna(0) + a5.fillna(0)


def derive_start_middle_age(
    df: pd.DataFrame,
    prsex_col: str,
    male_col: str,
    female_col: str,
) -> pd.Series:
    """
    Sex-dependent middle-age start: male_col if PRSEX == 1, female_col if
    PRSEX == 2, else NA (SE16 / SE18 in MIDUS2).

    Columns already filtered to [10, 100] before this step.
    """
    sex = _safe_col(df, prsex_col)
    m = _safe_col(df, male_col)
    f = _safe_col(df, female_col)
    out = pd.Series(np.nan, index=df.index, dtype=float)
    out = out.where(~(sex == 2), f)
    out = out.where(~(sex == 1), m)
    return out


def derive_end_middle_age(
    df: pd.DataFrame,
    prsex_col: str,
    male_col: str,
    female_col: str,
) -> pd.Series:
    """
    Sex-dependent middle-age end, as derive_start_middle_age (SE17 / SE19
    in MIDUS2).
    """
    sex = _safe_col(df, prsex_col)
    m = _safe_col(df, male_col)
    f = _safe_col(df, female_col)
    out = pd.Series(np.nan, index=df.index, dtype=float)
    out = out.where(~(sex == 2), f)
    out = out.where(~(sex == 1), m)
    return out


# Validation

if __name__ == "__main__":
    np.random.seed(42)
    N = 200

    print("Helper validation")
    months = pd.Series([1, 6, 13, np.nan, 12])
    years = pd.Series([2000, 2010, 2020, 2015, 0])
    dt = create_datetime(months, years)
    assert dt.iloc[0] == pd.Timestamp("2000-01-01"), "create_datetime: basic"
    assert dt.iloc[1] == pd.Timestamp("2010-06-01"), "create_datetime: mid-year"
    assert pd.isna(dt.iloc[2]), "create_datetime: month>12 -> NaT"
    assert pd.isna(dt.iloc[3]), "create_datetime: NaN month -> NaT"
    assert pd.isna(dt.iloc[4]), "create_datetime: year=0 -> NaT"

    s1 = pd.Series(pd.to_datetime(["2000-01-01", "2005-06-15"]))
    s2 = pd.Series(pd.to_datetime(["2010-01-01", "2010-06-15"]))
    yg = year_gap(s1, s2)
    assert abs(yg.iloc[0] - 10.0) < 0.02, "year_gap: ~10 years"
    assert abs(yg.iloc[1] - 5.0) < 0.02, "year_gap: ~5 years"

    d1 = pd.Series(pd.to_datetime(["2000-01-01", pd.NaT, "2020-01-01"]))
    d2 = pd.Series(pd.to_datetime([pd.NaT, "2010-01-01", "2015-01-01"]))
    latest = get_latest_date(d1, d2)
    assert latest.iloc[0] == pd.Timestamp("2000-01-01"), "get_latest_date: one NaT"
    assert latest.iloc[1] == pd.Timestamp("2010-01-01"), "get_latest_date: one NaT(2)"
    assert latest.iloc[2] == pd.Timestamp("2020-01-01"), "get_latest_date: max"
    print("  Helpers: PASS")

    print("\nDemographics validation")
    df_demo = pd.DataFrame({
        "B1PRSEX": [1, 2, 2, 1, np.nan],
        "B1PF1": [1, 2, 3, 1, np.nan],
        "B1PF7A": [1, 2, 3, 6, np.nan],
        "B1PF8A": [np.nan, 1, np.nan, 2, np.nan],
        "B1SXORNT": [1, 2, 3, np.nan, 1],
    })
    female = derive_female(df_demo, "B1PRSEX")
    assert list(female[:4]) == [0, 1, 1, 0], "derive_female: basic"
    assert np.isnan(female[4]), "derive_female: NaN passthrough"

    hisp = derive_hispanic(df_demo, "B1PF1")
    assert list(hisp[:4]) == [0, 1, 1, 0], "derive_hispanic: basic"

    df_demo["B1PF7A"] = [1, 2, 4, 5, 6]
    race = derive_race(df_demo, "B1PF7A")
    assert race.iloc[0] == 1, "derive_race: passthrough 1"
    assert race.iloc[1] == 2, "derive_race: passthrough 2"
    assert race.iloc[2] == 3, "derive_race: recode 4->3"
    assert race.iloc[3] == 3, "derive_race: recode 5->3"
    assert race.iloc[4] == 4, "derive_race: recode 6->4"

    sxo = derive_sexual_orientation(df_demo, "B1SXORNT")
    assert sxo.iloc[0] == 1, "derive_sexual_orientation: passthrough"
    print("  Demographics: PASS")

    print("\nWork validation")
    df_work = pd.DataFrame({
        "B1PB4N": [np.nan, 10, 10, np.nan, 10],
        "B1PB6": [0, 1, 2, 0, 0],
        "B1PBWORK": [5, 1, 4, 2, 10],
        "B1PB7": [np.nan, 1, 0, np.nan, 1],
    })
    wfp = derive_work_for_pay(df_work, "B1PB4N", "B1PB6", "B1PBWORK")
    # row0: PB4N=NaN -> 1
    # row1: PB6==1 -> 1
    # row2: PB6==2 -> 0 (also PBWORK=4 -> 0, but cond1 not triggered since PB4N=10, PB6=2)
    # row3: PB4N=NaN -> 1
    # row4: PBWORK=10 -> 0
    assert wfp.iloc[0] == 1, "workForPay: PB4N=NaN -> 1"
    assert wfp.iloc[1] == 1, "workForPay: PB6==1 -> 1"
    assert wfp.iloc[2] == 0, "workForPay: PB6==2 -> 0"
    assert wfp.iloc[3] == 1, "workForPay: PB4N=NaN -> 1"
    assert wfp.iloc[4] == 0, "workForPay: PBWORK=10 -> 0"

    ws = derive_work_status(df_work, "B1PBWORK")
    assert ws.iloc[0] == "Retired", "workStatus: 5->Retired"
    assert ws.iloc[1] == "Working now", "workStatus: 1->Working now"
    assert ws.iloc[2] == "Laid-off/Seeking-work", "workStatus: 4->Laid-off"
    assert ws.iloc[3] == "Self-employed", "workStatus: 2->Self-employed"
    assert ws.iloc[4] == "Disabled", "workStatus: 10->Disabled"

    mr = derive_manage_role(df_work, "B1PB7")
    assert np.isnan(mr.iloc[0]), "manageRole: NaN -> NaN"
    assert mr.iloc[1] == 1, "manageRole: passthrough"
    print("  Work: PASS")

    print("\nRelationship validation")
    df_rel = pd.DataFrame({
        "B1PB19": [1, 2, 3, 4, 5, np.nan],
        "B1PB30": [np.nan, 0, np.nan, np.nan, np.nan, 1],
    })
    rs = derive_relat_status(df_rel, "B1PB19", "B1PB30")
    assert rs.iloc[0] == "Married", "relatStatus: 1->Married"
    assert rs.iloc[1] == "Separated(not cohabiting)", "relatStatus: PB19=2,PB30=0"
    assert rs.iloc[2] == "Divorced(not cohabiting)", "relatStatus: 3->Divorced"
    assert rs.iloc[3] == "Widowed(not cohabiting)", "relatStatus: 4->Widowed"
    assert rs.iloc[4] == "Never married(not cohabiting)", "relatStatus: 5->Never"
    assert rs.iloc[5] == "Cohabiting(not married)", "relatStatus: PB30=1->Cohab"
    print("  Relationships: PASS")

    print("\nCaregiving validation")
    df_cg = pd.DataFrame({
        "B1PD1": [0, 1, 1, 0, np.nan],
        "B1PD7": [np.nan, 0, 1, np.nan, np.nan],
        "B1PD18": [np.nan, np.nan, np.nan, 5.0, np.nan],
        "B1PD14": [np.nan, np.nan, 4, np.nan, np.nan],
        "B1PD15": [np.nan, np.nan, 6, np.nan, np.nan],
    })
    cgs = derive_caregiv_status(df_cg, "B1PD1", "B1PD7", "B1PD18")
    assert cgs.iloc[0] == "Never caregiver", "careGivStatus: D1=0, D18=NA"
    assert cgs.iloc[1] == "Former caregiver", "careGivStatus: D1=1, D7=0"
    assert cgs.iloc[2] == "Current caregiver", "careGivStatus: D1=1, D7=1"
    assert cgs.iloc[3] == "Former caregiver", "careGivStatus: D1=0, D18=5"

    cgi = derive_caregiv_intensity(df_cg, "B1PD14", "B1PD15", cgs)
    assert cgi.iloc[0] == 0, "careGivIntensity: Never -> 0"
    assert cgi.iloc[1] == 0, "careGivIntensity: Former, PD14=NaN -> 0"
    assert abs(cgi.iloc[2] - 4 * 6 / 24) < 0.001, "careGivIntensity: Current -> 1.0"
    print("  Caregiving: PASS")

    print("\nSmoking validation")
    df_smoke = pd.DataFrame({
        #                never  never  fmr_lt fmr_hv cur_lt cur_hv never   NA
        "B1PA38A": [       0,     0,     1,     1,     1,     1, np.nan, np.nan],
        "B1PA39":  [  np.nan,np.nan,     0,     0,     1,     1, np.nan, np.nan],
        "B1PA40":  [  np.nan,np.nan,np.nan,np.nan,     2,    30, np.nan, np.nan],
        "B1PA43":  [  np.nan,np.nan,     5,    30,np.nan,np.nan, np.nan, np.nan],
        "B1PA44":  [  np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,     2,      1],
    })
    # median(PA43) = median(5,30) = 17.5;  median(PA40) = median(2,30) = 16
    ss = derive_smoke_status(df_smoke, "B1PA38A", "B1PA39", "B1PA40", "B1PA43", "B1PA44")
    assert ss.iloc[0] == "Never smoked regular", "smokeStatus: PA38A=0"
    assert ss.iloc[1] == "Never smoked regular", "smokeStatus: PA38A=0 (dup)"
    assert ss.iloc[2] == "Former light smoker", "smokeStatus: quit, PA43=5 < 17.5"
    assert ss.iloc[3] == "Former heavy smoker", "smokeStatus: quit, PA43=30 >= 17.5"
    assert ss.iloc[4] == "Current light smoker", "smokeStatus: curr, PA40=2 < 16"
    assert ss.iloc[5] == "Current heavy smoker", "smokeStatus: curr, PA40=30 >= 16"
    assert ss.iloc[6] == "Never smoked regular", "smokeStatus: PA38A=NaN, PA44!=1"
    assert pd.isna(ss.iloc[7]), "smokeStatus: PA38A=NaN, PA44=1 -> NA"
    print("  Smoking: PASS")

    print("\nACE composite validation")
    df_ace = pd.DataFrame({
        "A": [1, 0, np.nan, 1],
        "B": [1, 1, np.nan, np.nan],
        "C": [0, 0, np.nan, 1],
        "D": [np.nan, 1, np.nan, np.nan],
    })
    ace = derive_ace_composite(df_ace, ["A", "B", "C", "D"], min_frac=0.5)
    assert ace.iloc[0] == 2, "ace: 1+1+0+NA = 2 (3/4 >= 2)"
    assert ace.iloc[1] == 2, "ace: 0+1+0+1 = 2"
    assert np.isnan(ace.iloc[2]), "ace: all NA -> NA"
    assert ace.iloc[3] == 2, "ace: 1+NA+1+NA = 2 (2/4 >= 2)"
    print("  ACE composites: PASS")

    print("\nAge-split validation")
    df_split = pd.DataFrame({
        "event": [1, 1, 0, 1, np.nan],
        "age_at": [10, 25, np.nan, 18, 5],
    })
    young, old = derive_age_split_event(df_split, "event", "age_at", age_threshold=18)
    assert young.iloc[0] == 1, "age_split: age=10 <= 18 -> young=event"
    assert young.iloc[1] == 0, "age_split: age=25 > 18 -> young=0"
    assert old.iloc[0] == 0, "age_split: age=10 <= 18 -> old=0"
    assert old.iloc[1] == 1, "age_split: age=25 > 18 -> old=event"
    assert young.iloc[3] == 1, "age_split: age=18 <= 18 -> young=event"
    assert old.iloc[3] == 0, "age_split: age=18 <= 18 -> old=0"
    print("  Age-split: PASS")

    print("\nPhysical activity validation")
    df_pa = pd.DataFrame({
        "VA": [3, np.nan], "VB": [5, np.nan],
        "VC": [2, 4], "VD": [4, np.nan],
        "VE": [1, 3], "VF": [np.nan, 3],
        "MA": [2, np.nan], "MB": [2, np.nan],
        "MC": [1, 1], "MD": [3, np.nan],
        "ME": [2, 2], "MF": [np.nan, 2],
        "LA": [1, np.nan], "LB": [1, np.nan],
        "LC": [2, 2], "LD": [2, np.nan],
        "LE": [3, 1], "LF": [np.nan, 1],
    })
    pa = derive_physical_activity(
        df_pa,
        vig_cols=["VA", "VB", "VC", "VD", "VE", "VF"],
        mod_cols=["MA", "MB", "MC", "MD", "ME", "MF"],
        light_cols=["LA", "LB", "LC", "LD", "LE", "LF"],
    )
    # row0: vjob=mean(3,5)=4.0, mjob=mean(2,2)=2.0 => vmJobAct=4*2+2=10
    assert abs(pa["vmJobAct"].iloc[0] - 10.0) < 0.001, "physAct: vmJobAct"
    # row0: vjob_vig = 4*2 = 8
    assert abs(pa["vmJobAct_vig"].iloc[0] - 8.0) < 0.001, "physAct: vmJobAct_vig"
    # row1: VA=NaN, VB=NaN => vjob=NaN => vmJobAct=NaN
    assert np.isnan(pa["vmJobAct"].iloc[1]), "physAct: all NaN vig -> NaN"
    print("  Physical activity: PASS")

    print("\nDrink status validation")
    # PA51: 1=every day .. 5=<1 day/wk (lower = more frequent). median of [2,3,5] = 3
    df_drink = pd.DataFrame({
        "B1PA50": [0, 1, 1, 1, 1, 1],
        "B1PA51": [np.nan, 5, 2, 5, 2, 6],
        "B1PA53": [np.nan, 0, 0, 5, 3, 0],
    })
    ds = derive_drink_status(df_drink, "B1PA50", "B1PA51", "B1PA53")
    assert ds.iloc[0] == "Past-month abstainer", "drinkStatus: PA50=0"
    assert ds.iloc[1] == "Infrequent-non-binge", "drinkStatus: PA51>=med, PA53=0"
    assert ds.iloc[2] == "Frequent-non-binge", "drinkStatus: PA51<med, PA53=0"
    assert ds.iloc[3] == "Binge", "drinkStatus: PA53=5>1, infrequent"
    assert ds.iloc[4] == "Binge", "drinkStatus: PA53=3>1, frequent"
    assert pd.isna(ds.iloc[5]), "drinkStatus: PA51=6 -> NaN (contradicts PA50=1)"
    print("  Drink status: PASS")

    print("\nFamily history validation")
    df_fam = pd.DataFrame({
        "PA10A": [1, 0, 0, 0],
        "PA10B": [0, 1, 0, 0],
        "PA10C": [0, 0, 0, 0],
        "PA10D": [0, 0, 1, 0],
        "PA10E": [0, 0, 0, 0],
        "PA10F": [0, 0, 0, 0],
        "PA10G": [0, 0, 0, 0],
        "PA10H": [0, 0, 0, 1],
    })
    fha = derive_family_heart_attack(
        df_fam, "PA10A", "PA10B", "PA10C", "PA10D", "PA10E", "PA10F", "PA10G", "PA10H"
    )
    assert fha.iloc[0] == "No family history", "familyHeartAttack: No family history"
    assert fha.iloc[1] == "Parent", "familyHeartAttack: Parent"
    assert fha.iloc[2] == "Sibling", "familyHeartAttack: Sibling"
    assert fha.iloc[3] == "Children", "familyHeartAttack: Children"
    print("  Family history: PASS")

    print("\nSub achievement validation")
    df_sub = pd.DataFrame({
        "PG5": [1, 1, 1, 2, 2, 2, np.nan],
        "PG5A": [1, 2, 3, 3, 2, 1, 1],
    })
    sa = derive_sub_achievement(df_sub, "PG5", "PG5A")
    assert list(sa[:6]) == [6, 5, 4, 3, 2, 1], "subAchievement: 6-level"
    assert np.isnan(sa.iloc[6]), "subAchievement: NaN"
    print("  Sub achievement: PASS")

    print("\nHealth outcomes validation")
    df_health = pd.DataFrame({
        "B1PA4": [1, 2, np.nan],
        "B1PA5": [3, np.nan, np.nan],
        "B1PRSEX": [1, 2, 1],
        "B1SE16": [35, np.nan, 40],
        "B1SE17": [55, np.nan, 60],
        "B1SE18": [np.nan, 33, np.nan],
        "B1SE19": [np.nan, 53, np.nan],
    })
    haw = derive_health_affect_work(df_health, "B1PA4", "B1PA5")
    assert haw.iloc[0] == 4, "healthAffectWork: 1+3=4"
    assert haw.iloc[1] == 2, "healthAffectWork: 2+0=2"

    sma = derive_start_middle_age(df_health, "B1PRSEX", "B1SE16", "B1SE18")
    assert sma.iloc[0] == 35, "startMiddleAge: male->SE16"
    assert sma.iloc[1] == 33, "startMiddleAge: female->SE18"

    ema = derive_end_middle_age(df_health, "B1PRSEX", "B1SE17", "B1SE19")
    assert ema.iloc[0] == 55, "endMiddleAge: male->SE17"
    assert ema.iloc[1] == 53, "endMiddleAge: female->SE19"
    print("  Health outcomes: PASS")

    print("\nAll validation tests passed")
