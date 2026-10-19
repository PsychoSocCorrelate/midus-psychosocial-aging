"""
Derived health behavior, ACE, activity, and CIRS variables.

All functions accept column name(s) as parameters so they work across cohorts
(MIDUS2 B1* / MIDUSR RA1* / MIDUS3 C1* / MR2 RB1*).

PANGIN 0-3 are angina classes (present); 6 is absent (per codebook).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    # Smoking
    "derive_smoke_status",
    "derive_smoke_duration",
    "derive_smoke_amount",
    "derive_smoke_exposure",
    "derive_smoke_exposure_3item",
    # Drinking
    "derive_drink_status",
    "derive_drink_exposure",
    "derive_drink_exposure_pa59",
    # ACE
    "derive_ace_composite",
    # Physical activity
    "derive_activity_composites",
    # CIRS
    "derive_cirs",
    # Health outcomes
    "derive_health_affect_work",
    "derive_middle_age",
]


# Helpers

def _safe_col(df: pd.DataFrame, col: str) -> pd.Series:
    """Return column as float64 Series, or all-NaN if column is absent."""
    if col in df.columns:
        return df[col].astype("float64")
    return pd.Series(np.nan, index=df.index, dtype="float64")


def _row_count_nonna(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Count non-NA values across *cols* (absent cols treated as all-NA)."""
    present = [c for c in cols if c in df.columns]
    if not present:
        return pd.Series(0, index=df.index, dtype="int64")
    return df[present].notna().sum(axis=1)


def _row_sum(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Row-wise sum across *cols*, treating NaN as 0.
    Absent cols are ignored."""
    present = [c for c in cols if c in df.columns]
    if not present:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return df[present].astype("float64").sum(axis=1, min_count=1)


def _row_mean(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Row-wise mean across *cols*, ignoring NaN. Absent cols are ignored."""
    present = [c for c in cols if c in df.columns]
    if not present:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return df[present].astype("float64").mean(axis=1, skipna=True)


def _row_mean_if(
    df: pd.DataFrame,
    cols: list[str],
    min_nonna: int = 1,
) -> pd.Series:
    """Row-wise mean if at least *min_nonna* items are non-NA, else NaN."""
    cnt = _row_count_nonna(df, cols)
    mean = _row_mean(df, cols)
    return mean.where(cnt >= min_nonna, other=np.nan)


# Smoking (5 output variables)

def derive_smoke_status(
    df: pd.DataFrame,
    pa38a: str,
    pa39: str,
    pa40: str,
    pa43: str,
    pa44: str,
) -> pd.Series:
    """5-level smoking status classification.

    Categories:
        - 'Never smoked regular'
        - 'Former light smoker'   (quit, amount < median)
        - 'Former heavy smoker'   (quit, amount >= median)
        - 'Current light smoker'  (current, amount < median)
        - 'Current heavy smoker'  (current, amount >= median)

    The median thresholds are computed within the Series, ignoring NaN.

    Parameters
    ----------
    df : pd.DataFrame
    pa38a : str  Column for "ever smoked regularly" (binary after recode: 0/1).
    pa39 : str   Column for "currently smoking" (binary: 0=no, 1=yes).
    pa40 : str   Column for "cigarettes/day (current)".
    pa43 : str   Column for "cigarettes/day (when quit)" or similar amount.
    pa44 : str   Column for "smoking status check" (pipe/cigar flag).

    Returns
    -------
    pd.Series
        Categorical string Series.
    """
    ever   = _safe_col(df, pa38a)
    curr   = _safe_col(df, pa39)
    cpd    = _safe_col(df, pa40)  # current cigs/day
    qamt   = _safe_col(df, pa43)  # amount when quit (or peak)
    flag44 = _safe_col(df, pa44)  # pipe/cigar screening

    med_quit = qamt.median()
    med_cpd  = cpd.median()

    result = pd.Series(np.nan, index=df.index, dtype="object")

    # Never smoked: PA38A == 0  OR  (PA38A is NA & PA44 != 1)
    never_mask = (ever == 0) | (ever.isna() & (flag44 != 1))
    result = result.where(~never_mask, "Never smoked regular")

    # Former light smoker: PA39==0 & (PA43 < median | PA43 is NA)
    former_light = (~never_mask) & (curr == 0) & ((qamt < med_quit) | qamt.isna())
    result = result.where(~former_light, "Former light smoker")

    # Former heavy smoker: PA39==0 & PA43 >= median
    former_heavy = (~never_mask) & (curr == 0) & (qamt >= med_quit) & qamt.notna()
    result = result.where(~former_heavy, "Former heavy smoker")

    # Current light smoker: PA39==1 & (PA40 < median | PA40 is NA)
    current_light = (~never_mask) & (curr == 1) & ((cpd < med_cpd) | cpd.isna())
    result = result.where(~current_light, "Current light smoker")

    # Current heavy smoker: PA39==1 & PA40 >= median
    current_heavy = (~never_mask) & (curr == 1) & (cpd >= med_cpd) & cpd.notna()
    result = result.where(~current_heavy, "Current heavy smoker")

    return result


def derive_smoke_duration(
    df: pd.DataFrame,
    pa38a: str,
    pa38b: str,
    pa39: str,
    pa42: str,
    age_col: str,
    pa44: str,
) -> pd.Series:
    """Compute smoking duration in years.

    Logic:
        - Never smoked (PA38A==0 or PA44!=1 when PA38A is NA) -> 0
        - Former smoker (PA39==0) -> PA42 - PA38B  (age quit - age started)
        - Current smoker (PA39==1) -> age - PA38B
        - Negative durations -> NaN

    Parameters
    ----------
    df : pd.DataFrame
    pa38a : str   "Ever smoked regularly" (binary 0/1).
    pa38b : str   "Age first smoked regularly".
    pa39 : str    "Currently smoking" (0/1).
    pa42 : str    "Age when quit".
    age_col : str Current age column.
    pa44 : str    Pipe/cigar screening flag.

    Returns
    -------
    pd.Series
        Duration in years (float64).
    """
    ever   = _safe_col(df, pa38a)
    start  = _safe_col(df, pa38b)
    curr   = _safe_col(df, pa39)
    quit_age = _safe_col(df, pa42)
    age    = _safe_col(df, age_col)
    flag44 = _safe_col(df, pa44)

    never = (ever == 0) | (ever.isna() & (flag44 != 1))

    duration = pd.Series(np.nan, index=df.index, dtype="float64")
    duration = duration.where(~never, 0.0)

    former = (~never) & (curr == 0)
    duration = np.where(former, quit_age - start, duration)

    current = (~never) & (curr == 1)
    duration = np.where(current, age - start, duration)

    duration = pd.Series(duration, index=df.index, dtype="float64")
    duration = duration.where(duration >= 0, other=np.nan)
    return duration


def derive_smoke_amount(
    df: pd.DataFrame,
    pa38a: str,
    pa40: str,
    pa43: str,
    pa44: str,
) -> pd.Series:
    """Derive cigarettes-per-day amount.

    Rules in priority order:
        1. PA38A is NA & PA44 != 1 -> 0  (never smoker, pipe/cigar check)
        2. PA40 not NA -> PA40  (current cigs/day)
        3. PA43 not NA -> PA43  (peak / quit cigs/day)
        4. PA38A == 2  -> 1     (smoked but cigar/pipe only; code as 1)
        5. else -> NaN

    Parameters
    ----------
    df : pd.DataFrame
    pa38a, pa40, pa43, pa44 : str  Column names.

    Returns
    -------
    pd.Series (float64)
    """
    ever   = _safe_col(df, pa38a)
    cpd    = _safe_col(df, pa40)
    qamt   = _safe_col(df, pa43)
    flag44 = _safe_col(df, pa44)

    result = pd.Series(np.nan, index=df.index, dtype="float64")

    # Apply rules lowest priority first so higher ones overwrite.
    # 5. else -> NaN  (already the default)
    # 4. PA38A == 2 -> 1
    mask4 = ever == 2
    result = result.where(~mask4, 1.0)
    # 3. PA43 not NA -> PA43
    mask3 = qamt.notna()
    result = np.where(mask3, qamt, result)
    result = pd.Series(result, index=df.index, dtype="float64")
    # 2. PA40 not NA -> PA40
    mask2 = cpd.notna()
    result = np.where(mask2, cpd, result)
    result = pd.Series(result, index=df.index, dtype="float64")
    # 1. PA38A is NA & PA44 != 1 -> 0
    mask1 = ever.isna() & (flag44 != 1)
    result = np.where(mask1, 0.0, result)
    result = pd.Series(result, index=df.index, dtype="float64")

    return result


def derive_smoke_exposure(
    df: pd.DataFrame,
    item_cols: list[str],
    threshold: int,
) -> pd.Series:
    """Row-wise mean of smoke exposure items if >= *threshold* are non-NA.

    W1 (MIDUS2/MIDUSR): 6 items (PA45A/B/C + PA46/47/48), threshold=4.
    W2 (MIDUS3/MR2):    3 items (PA46/47/48),              threshold=2.

    Parameters
    ----------
    df : pd.DataFrame
    item_cols : list[str]  Column names for exposure items.
    threshold : int        Minimum non-NA count to compute mean.

    Returns
    -------
    pd.Series (float64)
    """
    return _row_mean_if(df, item_cols, min_nonna=threshold)


def derive_smoke_exposure_3item(
    df: pd.DataFrame,
    pa46: str,
    pa47: str,
    pa48: str,
    threshold: int = 2,
) -> pd.Series:
    """3-item smoke exposure (adult-only, cross-wave comparable).

    Uses PA46 (home), PA47 (job ever/current), PA48 (job current/ever).
    Threshold = 2.

    Parameters
    ----------
    df : pd.DataFrame
    pa46, pa47, pa48 : str  Column names.
    threshold : int         Minimum non-NA items (default 2).

    Returns
    -------
    pd.Series (float64)
    """
    return _row_mean_if(df, [pa46, pa47, pa48], min_nonna=threshold)


# Drinking (3 output variables)

def derive_drink_status(
    df: pd.DataFrame,
    pa50: str,
    pa51: str,
    pa53: str,
) -> pd.Series:
    """4-level drinking classification (2-axis: frequency x binge).

    PA50: "Had at least one drink past month" (0/1 after binary recode).
    PA51: "How often drink past month": ordinal, reverse-coded:
          1=every day, 2=5-6 days/wk, 3=3-4 days/wk,
          4=1-2 days/wk, 5=<1 day/wk, 6=never (volunteered).
          Lower value = more frequent.  PA51=6 set to NaN (contradicts PA50=1).
    PA53: "Times had 5+ drinks same occasion past month" (0-31).

    Categories (independent axes, no priority cascade):
      - 'Past-month abstainer'  : PA50 == 0
      - 'Infrequent-non-binge'  : PA51 >= median (<=2 days/wk) & PA53 <= 1
      - 'Frequent-non-binge'    : PA51 <  median (>=3 days/wk) & PA53 <= 1
      - 'Binge'                  : PA53 > 1  (any frequency)

    Parameters
    ----------
    df : pd.DataFrame
    pa50 : str  "Had at least one drink past month" (0/1).
    pa51 : str  "Drinking frequency past month" (1-6 ordinal, lower = more frequent; 6 set to NaN).
    pa53 : str  "Binge episodes past month" (0-31 count).

    Returns
    -------
    pd.Series (object / string)
    """
    a50 = _safe_col(df, pa50)
    a51 = _safe_col(df, pa51)
    a53 = _safe_col(df, pa53)

    # PA51=6 ("never drink" volunteered) contradicts PA50=1; treat as missing
    a51 = a51.where(a51 != 6, np.nan)

    med51 = a51.median()

    out = pd.Series(np.nan, index=df.index, dtype="object")

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
    item_cols: list[str],
    threshold: int = 1,
) -> pd.Series:
    """Row-wise sum of drink exposure items if >= *threshold* are non-NA.

    Typically: PA58 + PA59 (2 items), threshold=1.

    Parameters
    ----------
    df : pd.DataFrame
    item_cols : list[str]  Column names (e.g. ['B1PA58', 'B1PA59']).
    threshold : int        Minimum non-NA count.

    Returns
    -------
    pd.Series (float64)
    """
    cnt = _row_count_nonna(df, item_cols)
    s = _row_sum(df, item_cols)
    return s.where(cnt >= threshold, other=np.nan)


def derive_drink_exposure_pa59(
    df: pd.DataFrame,
    pa59: str,
) -> pd.Series:
    """Single-item drink exposure (PA59 only).

    For W1<->MR2 comparison (MR2 lacks PA58).

    Parameters
    ----------
    df : pd.DataFrame
    pa59 : str  Column name.

    Returns
    -------
    pd.Series (float64)
    """
    return _safe_col(df, pa59)


# ACE composites (12 output variables)

def derive_ace_composite(
    df: pd.DataFrame,
    item_cols: list[str],
    min_pct: float = 0.5,
) -> pd.Series:
    """Generic ACE composite: rowSums if >= min_pct fraction of items non-NA.

    If *item_cols* is empty (construct absent in wave), returns all NaN.

    Parameters
    ----------
    df : pd.DataFrame
    item_cols : list[str]  Source item column names.
    min_pct : float        Minimum fraction of non-NA items required (default 0.5).

    Returns
    -------
    pd.Series (float64)
    """
    present = [c for c in item_cols if c in df.columns]
    if not present:
        return pd.Series(np.nan, index=df.index, dtype="float64")

    n_items = len(present)
    min_nonna = max(1, int(np.ceil(n_items * min_pct)))

    cnt = df[present].notna().sum(axis=1)
    total = df[present].astype("float64").sum(axis=1, min_count=1)

    return total.where(cnt >= min_nonna, other=np.nan)


# Physical activity (7 output variables + vigorous-only variants)

# Items must already be frequency-weighted (recode_engine.recode_six_weighted).


def derive_activity_composites(
    df: pd.DataFrame,
    vig_cols: list[str],
    mod_cols: list[str],
    light_cols: list[str],
) -> pd.DataFrame:
    """Compute all physical activity composite variables.

    After frequency-weighted recoding (done in recode_engine), the 18
    activity items are arranged as 3 intensity levels x 3 domains x 2
    season items:

        Vigorous: [A,B] = job; [C,D] = chores; [E,F] = leisure
        Moderate: same layout
        Light:    same layout

    Computes 9 intermediate pair means (vjob, mjob, ..., lleisure), then:
        - vmJobAct, vmChoresAct, vmLeisureAct       (vigorous*2 + moderate*1)
        - vmJobAct_vig, vmChoresAct_vig, vmLeisureAct_vig  (vigorous*2 only)
        - lightAct  (mean of ljob, lchores, lleisure temps)

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with recoded activity columns.
    vig_cols : list[str]
        6 vigorous activity columns [A, B, C, D, E, F].
        MIDUS2: B1SA30A-F.  MIDUS3: C1SA26A-F.  MR2: RB1SA26A-F.
    mod_cols : list[str]
        6 moderate activity columns [A, B, C, D, E, F].
        MIDUS2: B1SA31A-F.  MIDUS3: C1SA27A-F.  MR2: RB1SA27A-F.
    light_cols : list[str]
        6 light activity columns [A, B, C, D, E, F].
        MIDUS2: B1SA32A-F.  MIDUS3: C1SA28A-F.  MR2: RB1SA28A-F.

    Returns
    -------
    pd.DataFrame
        Columns vmJobAct, vmChoresAct, vmLeisureAct, their _vig variants,
        and lightAct. Indexed same as *df*.
    """
    out = pd.DataFrame(index=df.index)

    def _pair_mean(cols_pair: list[str]) -> pd.Series:
        """Mean of 2 items if >= 1 non-NA, else NaN."""
        return _row_mean_if(df, cols_pair, min_nonna=1)

    # Vigorous sub-scores (pair means)
    vjob     = _pair_mean(vig_cols[0:2])
    vchores  = _pair_mean(vig_cols[2:4])
    vleisure = _pair_mean(vig_cols[4:6])

    # Moderate sub-scores
    mjob     = _pair_mean(mod_cols[0:2])
    mchores  = _pair_mean(mod_cols[2:4])
    mleisure = _pair_mean(mod_cols[4:6])

    # Light sub-scores
    ljob     = _pair_mean(light_cols[0:2])
    lchores  = _pair_mean(light_cols[2:4])
    lleisure = _pair_mean(light_cols[4:6])

    # Combined composites: vigorous*2 + moderate*1
    out["vmJobAct"]     = vjob * 2 + mjob * 1
    out["vmChoresAct"]  = vchores * 2 + mchores * 1
    out["vmLeisureAct"] = vleisure * 2 + mleisure * 1

    # Vigorous-only composites (cross-wave comparable, including MIDUS3
    # where moderate items are absent)
    out["vmJobAct_vig"]     = vjob * 2
    out["vmChoresAct_vig"]  = vchores * 2
    out["vmLeisureAct_vig"] = vleisure * 2

    # Light activity: mean of the 3 light sub-scores where >= 1 non-NA
    light_df = pd.DataFrame({
        "ljob": ljob, "lchores": lchores, "lleisure": lleisure,
    })
    cnt = light_df.notna().sum(axis=1)
    out["lightAct"] = light_df.mean(axis=1, skipna=True).where(cnt >= 1, np.nan)

    return out


# CIRS (12-item Comorbidity Index)

def derive_cirs(
    df: pd.DataFrame,
    # Cardiac
    pa8: str,
    pa7ba: str,
    pa7bb: str,
    pangin: str | None,
    # Hypertension
    sa11s: str,
    sa12a: str,
    # Respiratory
    sa11a: str,
    sa11b: str,
    # EENT (eyes-ears-nose-throat)
    sa7d: str,
    # GI
    sa12f: str,
    sa11aa: str,
    # Gallbladder
    sa11l: str,
    # Renal
    sa11j: str,
    # Musculoskeletal
    sa11d: str,
    sa12g: str,
    # Neurological
    sa11v: str,
    pa6c: str,
    pa6d: str,
    sa11y: str,
    # Diabetes
    sa11x: str,
    sa11g: str,
    # Cancer
    pa26: str,
    pa28a: str,
    # Stroke
    sa11z: str,
    pa6a: str,
) -> pd.DataFrame:
    """Compute the 12-item CIRS comorbidity index.

    Returns a DataFrame with columns: ``cirs`` and ``cirs_no_angina``.

    Angina: ``PANGIN in {0,1,2,3} -> 1``, ``6 -> 0`` (codebook: 0-3 are angina
    classes, 6 is no angina).

    When *pangin* is None (MR2: angina battery absent), hrattack excludes
    angina and ``cirs_no_angina == cirs``.

    EENT: SA7D is a 1-5 trouble scale (1 = a lot, 5 = not at all). After the
    6 - x recode, 5 = a lot, and the flag is ``SA7D == 5``. Expects recoded data.

    Other items flag ``value == 1`` (mostly binary 0/1 after recode).

    Parameters
    ----------
    df : pd.DataFrame
        Data after the recode step (binary items are 0/1, SA7D is reverse-coded).
    pangin : str or None
        Column for angina classification.  None if absent (MR2).
    <other params> : str
        Column names for each CIRS source item.

    Returns
    -------
    pd.DataFrame
        Columns: ``cirs``, ``cirs_no_angina``.
    """
    def _flag(col: str) -> pd.Series:
        """Convert to binary flag: 1 if value == 1, else 0. NaN -> 0."""
        s = _safe_col(df, col)
        return np.where(s == 1, 1.0, 0.0)

    def _flag_sa7d(col: str) -> pd.Series:
        """EENT flag: 1 if recoded SA7D == 5 (a lot of trouble), else 0."""
        s = _safe_col(df, col)
        return np.where(s == 5, 1.0, 0.0)

    def _flag_angina(col: str | None) -> pd.Series:
        """Angina flag: 1 if PANGIN in {0,1,2,3}, 0 if PANGIN == 6 or NA.

        PANGIN 0-3 are angina severity classes; 6 = none.
        """
        if col is None or col not in df.columns:
            return pd.Series(0.0, index=df.index, dtype="float64")
        s = _safe_col(df, col)
        return np.where(s.isin([0, 1, 2, 3]), 1.0, 0.0)

    # Build 12 condition flags
    angina_flag = _flag_angina(pangin)

    hrattack    = np.where(
        (_safe_col(df, pa8) == 1) |
        (_safe_col(df, pa7ba) == 1) |
        (_safe_col(df, pa7bb) == 1) |
        (angina_flag == 1),
        1.0, 0.0,
    )
    hyper       = np.where(
        (_safe_col(df, sa11s) == 1) | (_safe_col(df, sa12a) == 1), 1.0, 0.0,
    )
    respiratory = np.where(
        (_safe_col(df, sa11a) == 1) | (_safe_col(df, sa11b) == 1), 1.0, 0.0,
    )
    eent        = _flag_sa7d(sa7d)
    gi          = np.where(
        (_safe_col(df, sa12f) == 1) | (_safe_col(df, sa11aa) == 1), 1.0, 0.0,
    )
    gallbladder = _flag(sa11l)
    renal       = _flag(sa11j)
    musculo     = np.where(
        (_safe_col(df, sa11d) == 1) | (_safe_col(df, sa12g) == 1), 1.0, 0.0,
    )
    neuro       = np.where(
        (_safe_col(df, sa11v) == 1) |
        (_safe_col(df, pa6c) == 1) |
        (_safe_col(df, pa6d) == 1) |
        (_safe_col(df, sa11y) == 1),
        1.0, 0.0,
    )
    diabete     = np.where(
        (_safe_col(df, sa11x) == 1) | (_safe_col(df, sa11g) == 1), 1.0, 0.0,
    )
    cancer      = np.where(
        (_safe_col(df, pa26) == 1) | (_safe_col(df, pa28a) == 1), 1.0, 0.0,
    )
    stroke      = np.where(
        (_safe_col(df, sa11z) == 1) | (_safe_col(df, pa6a) == 1), 1.0, 0.0,
    )

    # Totals
    all_12 = np.array([
        hrattack, hyper, respiratory, eent, gi, gallbladder,
        renal, musculo, neuro, diabete, cancer, stroke,
    ])
    cirs = all_12.sum(axis=0)

    # cirs_no_angina: same as cirs but hrattack excludes angina
    hrattack_no_angina = np.where(
        (_safe_col(df, pa8) == 1) |
        (_safe_col(df, pa7ba) == 1) |
        (_safe_col(df, pa7bb) == 1),
        1.0, 0.0,
    )
    all_12_no_angina = np.array([
        hrattack_no_angina, hyper, respiratory, eent, gi, gallbladder,
        renal, musculo, neuro, diabete, cancer, stroke,
    ])
    cirs_no_angina = all_12_no_angina.sum(axis=0)

    return pd.DataFrame({
        "cirs": cirs,
        "cirs_no_angina": cirs_no_angina,
    }, index=df.index)


# Health outcomes

def derive_health_affect_work(
    df: pd.DataFrame,
    pa4: str,
    pa5: str,
) -> pd.Series:
    """Health days affecting work: PA4 + PA5, missing days counted as 0.

    Parameters
    ----------
    df : pd.DataFrame
    pa4, pa5 : str  Column names.

    Returns
    -------
    pd.Series (float64)
    """
    a = _safe_col(df, pa4)
    b = _safe_col(df, pa5)
    # Missing days count as 0; both missing gives 0.
    return a.fillna(0) + b.fillna(0)


def derive_middle_age(
    df: pd.DataFrame,
    se_male_start: str,
    se_male_end: str,
    se_female_start: str,
    se_female_end: str,
    sex_col: str,
) -> pd.DataFrame:
    """Derive startMiddleAge and endMiddleAge (sex-dependent).

    Males use SE16/SE17 (MIDUS2) or SE18/SE19 (MIDUS3/MR2).
    Females use SE18/SE19 (MIDUS2) or SE20/SE21 (MIDUS3/MR2).

    Values outside [10, 100] are set to NaN.

    Parameters
    ----------
    df : pd.DataFrame
    se_male_start, se_male_end : str      Column names for male middle age.
    se_female_start, se_female_end : str   Column names for female middle age.
    sex_col : str  Sex column (1=male, 2=female in SPSS coding).

    Returns
    -------
    pd.DataFrame
        Columns: ``startMiddleAge``, ``endMiddleAge``.
    """
    sex = _safe_col(df, sex_col)
    ms  = _safe_col(df, se_male_start)
    me  = _safe_col(df, se_male_end)
    fs  = _safe_col(df, se_female_start)
    fe  = _safe_col(df, se_female_end)

    # Clamp to [10, 100]
    def _clamp(s):
        return s.where((s >= 10) & (s <= 100), other=np.nan)

    ms, me, fs, fe = _clamp(ms), _clamp(me), _clamp(fs), _clamp(fe)

    start = np.where(sex == 1, ms, np.where(sex == 2, fs, np.nan))
    end   = np.where(sex == 1, me, np.where(sex == 2, fe, np.nan))

    return pd.DataFrame({
        "startMiddleAge": pd.Series(start, index=df.index, dtype="float64"),
        "endMiddleAge":   pd.Series(end,   index=df.index, dtype="float64"),
    })


# Self-tests

if __name__ == "__main__":
    import sys

    passed = 0
    failed = 0
    errors: list[str] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        global passed, failed
        if condition:
            passed += 1
            print(f"  PASS  {name}")
        else:
            failed += 1
            msg = f"  FAIL  {name}"
            if detail:
                msg += f"  -- {detail}"
            print(msg)
            errors.append(name)

    print("derive_smoke_status")
    df_smoke = pd.DataFrame({
        "PA38A": [0.0,  0.0,  0.0,  1.0,  1.0,  np.nan, np.nan],
        "PA39":  [0.0,  0.0,  0.0,  0.0,  1.0,  np.nan, np.nan],
        "PA40":  [np.nan, np.nan, np.nan, np.nan, 5.0, np.nan, np.nan],
        "PA43":  [np.nan, 5.0, 20.0, 5.0, np.nan, np.nan, np.nan],
        "PA44":  [np.nan, np.nan, np.nan, np.nan, np.nan, 2.0,   1.0],
    })
    ss = derive_smoke_status(df_smoke, "PA38A", "PA39", "PA40", "PA43", "PA44")
    check("never smoker (PA38A=0)",       ss.iloc[0] == "Never smoked regular")
    check("never smoker (PA38A=NA, PA44!=1)", ss.iloc[5] == "Never smoked regular")
    check("PA38A=NA & PA44=1 -> NaN",     pd.isna(ss.iloc[6]))

    print("\nderive_smoke_duration")
    df_dur = pd.DataFrame({
        "PA38A": [0.0, 1.0, 1.0, np.nan],
        "PA38B": [np.nan, 20.0, 18.0, np.nan],
        "PA39":  [np.nan, 0.0,  1.0,  np.nan],
        "PA42":  [np.nan, 35.0, np.nan, np.nan],
        "AGE":   [50.0,  50.0,  50.0, 50.0],
        "PA44":  [np.nan, np.nan, np.nan, 2.0],
    })
    sd = derive_smoke_duration(df_dur, "PA38A", "PA38B", "PA39", "PA42", "AGE", "PA44")
    check("never smoker -> 0",            sd.iloc[0] == 0.0)
    check("former: 35 - 20 = 15",         sd.iloc[1] == 15.0)
    check("current: 50 - 18 = 32",        sd.iloc[2] == 32.0)
    check("never (PA44!=1) -> 0",          sd.iloc[3] == 0.0)

    print("\nderive_smoke_amount")
    df_amt = pd.DataFrame({
        "PA38A": [np.nan, np.nan, 1.0,  1.0,  2.0],
        "PA40":  [np.nan, np.nan, 10.0, np.nan, np.nan],
        "PA43":  [np.nan, np.nan, np.nan, 15.0, np.nan],
        "PA44":  [2.0,    1.0,   np.nan, np.nan, np.nan],
    })
    sa = derive_smoke_amount(df_amt, "PA38A", "PA40", "PA43", "PA44")
    check("PA38A=NA & PA44!=1 -> 0",      sa.iloc[0] == 0.0)
    check("PA38A=NA & PA44=1 -> NaN",     np.isnan(sa.iloc[1]))
    check("PA40 not NA -> PA40=10",        sa.iloc[2] == 10.0)
    check("PA43 not NA -> PA43=15",        sa.iloc[3] == 15.0)
    check("PA38A=2 -> 1",                  sa.iloc[4] == 1.0)

    print("\nderive_smoke_exposure")
    df_exp = pd.DataFrame({
        "PA45A": [1.0, np.nan, np.nan, np.nan],
        "PA45B": [0.0, np.nan, np.nan, np.nan],
        "PA45C": [1.0, np.nan, np.nan, np.nan],
        "PA46":  [1.0, 1.0,   np.nan, np.nan],
        "PA47":  [0.0, 0.0,   1.0,    np.nan],
        "PA48":  [1.0, 1.0,   np.nan, np.nan],
    })
    # 6-item version, threshold=4
    se6 = derive_smoke_exposure(
        df_exp, ["PA45A", "PA45B", "PA45C", "PA46", "PA47", "PA48"], threshold=4)
    check("6-item row0: 6 items -> mean",    abs(se6.iloc[0] - 4/6) < 1e-9)
    check("6-item row1: 3 items < 4 -> NaN", np.isnan(se6.iloc[1]))

    # 3-item version, threshold=2
    se3 = derive_smoke_exposure_3item(df_exp, "PA46", "PA47", "PA48", threshold=2)
    check("3-item row0: 3 items -> mean", abs(se3.iloc[0] - 2/3) < 1e-9)
    # row1: PA46=1, PA47=0, PA48=1 -> mean = 2/3
    check("3-item row1: 2 items -> mean", abs(se3.iloc[1] - 2/3) < 1e-9)
    check("3-item row2: 1 item < 2 -> NaN", np.isnan(se3.iloc[2]))

    # PA51: 1=every day .. 5=<1 day/wk (lower = more frequent). median of [5,2,5,2] = 3.5
    print("\nderive_drink_status")
    df_drink = pd.DataFrame({
        "PA50": [0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "PA51": [np.nan, 5.0, 2.0, 5.0, 2.0, 6.0],
        "PA53": [np.nan, 0.0, 0.0, 5.0, 3.0, 0.0],
    })
    ds = derive_drink_status(df_drink, "PA50", "PA51", "PA53")
    check("past-month abstainer (PA50=0)",              ds.iloc[0] == "Past-month abstainer")
    check("infrequent-non-binge (PA51>=med, PA53=0)",   ds.iloc[1] == "Infrequent-non-binge")
    check("frequent-non-binge (PA51<med, PA53=0)",      ds.iloc[2] == "Frequent-non-binge")
    check("binge (PA53=5>1, infrequent)",               ds.iloc[3] == "Binge")
    check("binge (PA53=3>1, frequent)",                 ds.iloc[4] == "Binge")
    check("PA51=6 -> NaN (contradicts PA50=1)",         pd.isna(ds.iloc[5]))

    print("\nderive_drink_exposure")
    df_de = pd.DataFrame({
        "PA58": [1.0, np.nan, np.nan],
        "PA59": [1.0, 1.0,   np.nan],
    })
    de = derive_drink_exposure(df_de, ["PA58", "PA59"], threshold=1)
    check("both present: 1+1=2",         de.iloc[0] == 2.0)
    check("one present: 0+1=1",          de.iloc[1] == 1.0)
    check("both NA -> NaN",              np.isnan(de.iloc[2]))

    dp = derive_drink_exposure_pa59(df_de, "PA59")
    check("PA59-only row0=1",            dp.iloc[0] == 1.0)
    check("PA59-only row2=NaN",          np.isnan(dp.iloc[2]))

    print("\nderive_ace_composite")
    df_ace = pd.DataFrame({
        "SE11A": [1.0, 1.0, np.nan, np.nan],
        "SE11F": [0.0, np.nan, np.nan, np.nan],
        "SE11G": [1.0, np.nan, np.nan, np.nan],
        "SE11H": [0.0, np.nan, 1.0,   np.nan],
    })
    # 4 items, min_pct=0.5 -> need >= 2 non-NA
    ace = derive_ace_composite(df_ace, ["SE11A", "SE11F", "SE11G", "SE11H"], min_pct=0.5)
    check("4 items present -> sum=2",    ace.iloc[0] == 2.0)
    check("1 item < 2 -> NaN",           np.isnan(ace.iloc[1]))
    check("1 item < 2 -> NaN (row2)",   np.isnan(ace.iloc[2]))
    check("0 items -> NaN",              np.isnan(ace.iloc[3]))

    # Empty item list (MIDUS3/MR2 absent constructs)
    ace_empty = derive_ace_composite(df_ace, [], min_pct=0.5)
    check("empty items -> all NaN",       ace_empty.isna().all())

    print("\nderive_activity_composites")
    df_act = pd.DataFrame({
        # Vigorous (already frequency-weighted)
        "SA30A": [3.0, 1.0, np.nan, np.nan],
        "SA30B": [1.0, 1.0, np.nan, np.nan],
        "SA30C": [0.75, np.nan, np.nan, np.nan],
        "SA30D": [0.25, np.nan, np.nan, np.nan],
        "SA30E": [3.0, np.nan, np.nan, np.nan],
        "SA30F": [1.0, np.nan, np.nan, np.nan],
        # Moderate
        "SA31A": [0.1, 0.1, np.nan, np.nan],
        "SA31B": [0.0, 0.0, np.nan, np.nan],
        "SA31C": [1.0, np.nan, np.nan, np.nan],
        "SA31D": [1.0, np.nan, np.nan, np.nan],
        "SA31E": [0.75, np.nan, np.nan, np.nan],
        "SA31F": [0.25, np.nan, np.nan, np.nan],
        # Light
        "SA32A": [3.0, np.nan, np.nan, np.nan],
        "SA32B": [1.0, np.nan, np.nan, np.nan],
        "SA32C": [0.75, np.nan, np.nan, np.nan],
        "SA32D": [0.25, np.nan, np.nan, np.nan],
        "SA32E": [0.1, np.nan, np.nan, np.nan],
        "SA32F": [0.0, np.nan, np.nan, np.nan],
    })
    act = derive_activity_composites(
        df_act,
        vig_cols=["SA30A", "SA30B", "SA30C", "SA30D", "SA30E", "SA30F"],
        mod_cols=["SA31A", "SA31B", "SA31C", "SA31D", "SA31E", "SA31F"],
        light_cols=["SA32A", "SA32B", "SA32C", "SA32D", "SA32E", "SA32F"],
    )
    # Row 0: vjob = mean(3,1) = 2.0; mjob = mean(0.1,0) = 0.05
    #   vmJobAct = 2*2 + 0.05 = 4.05
    check("vmJobAct row0 = 4.05",     abs(act["vmJobAct"].iloc[0] - 4.05) < 1e-9)
    # vmJobAct_vig = 2*2 = 4.0
    check("vmJobAct_vig row0 = 4.0",  abs(act["vmJobAct_vig"].iloc[0] - 4.0) < 1e-9)
    # Row 3: all NA -> NaN
    check("vmJobAct row3 = NaN",      np.isnan(act["vmJobAct"].iloc[3]))
    check("lightAct row3 = NaN",      np.isnan(act["lightAct"].iloc[3]))

    # lightAct row0: ljob=mean(3,1)=2, lchores=mean(0.75,0.25)=0.5, lleisure=mean(0.1,0)=0.05
    #   lightAct = mean(2, 0.5, 0.05) = 0.85
    check("lightAct row0 = 0.85",     abs(act["lightAct"].iloc[0] - 0.85) < 1e-9)

    print("\nderive_cirs")
    df_cirs = pd.DataFrame({
        "PA8":    [1.0, 0.0, 0.0, 0.0],
        "PA7BA":  [0.0, 0.0, 0.0, 0.0],
        "PA7BB":  [0.0, 0.0, 0.0, 0.0],
        "PANGIN": [6.0, 0.0, 1.0, np.nan],  # 6=absent, 0-3=present
        "SA11S":  [1.0, 0.0, 0.0, 0.0],
        "SA12A":  [0.0, 0.0, 0.0, 0.0],
        "SA11A":  [0.0, 0.0, 0.0, 0.0],
        "SA11B":  [0.0, 0.0, 0.0, 0.0],
        "SA7D":   [5.0, 1.0, 0.0, 0.0],  # 5 = worst pain (recoded)
        "SA12F":  [0.0, 0.0, 0.0, 0.0],
        "SA11AA": [0.0, 0.0, 0.0, 0.0],
        "SA11L":  [0.0, 0.0, 0.0, 0.0],
        "SA11J":  [0.0, 0.0, 0.0, 0.0],
        "SA11D":  [0.0, 0.0, 0.0, 0.0],
        "SA12G":  [0.0, 0.0, 0.0, 0.0],
        "SA11V":  [0.0, 0.0, 0.0, 0.0],
        "PA6C":   [0.0, 0.0, 0.0, 0.0],
        "PA6D":   [0.0, 0.0, 0.0, 0.0],
        "SA11Y":  [0.0, 0.0, 0.0, 0.0],
        "SA11X":  [0.0, 0.0, 0.0, 0.0],
        "SA11G":  [0.0, 0.0, 0.0, 0.0],
        "PA26":   [0.0, 0.0, 0.0, 0.0],
        "PA28A":  [0.0, 0.0, 0.0, 0.0],
        "SA11Z":  [0.0, 0.0, 0.0, 0.0],
        "PA6A":   [0.0, 0.0, 0.0, 0.0],
    })
    cirs_out = derive_cirs(
        df_cirs,
        pa8="PA8", pa7ba="PA7BA", pa7bb="PA7BB", pangin="PANGIN",
        sa11s="SA11S", sa12a="SA12A",
        sa11a="SA11A", sa11b="SA11B",
        sa7d="SA7D",
        sa12f="SA12F", sa11aa="SA11AA",
        sa11l="SA11L",
        sa11j="SA11J",
        sa11d="SA11D", sa12g="SA12G",
        sa11v="SA11V", pa6c="PA6C", pa6d="PA6D", sa11y="SA11Y",
        sa11x="SA11X", sa11g="SA11G",
        pa26="PA26", pa28a="PA28A",
        sa11z="SA11Z", pa6a="PA6A",
    )
    # Row 0: PA8=1 (hrattack), PANGIN=6 (absent -> angina=0), SA11S=1 (hyper), SA7D=5 (EENT) = 3
    check("cirs row0 = 3",              cirs_out["cirs"].iloc[0] == 3.0)
    check("cirs_no_angina row0 = 3",    cirs_out["cirs_no_angina"].iloc[0] == 3.0)

    # Row 1: PANGIN=0 counts as angina -> hrattack=1 only from angina = 1 total
    check("cirs row1 = 1 (PANGIN=0)",   cirs_out["cirs"].iloc[1] == 1.0)
    check("cirs_no_angina row1 = 0",    cirs_out["cirs_no_angina"].iloc[1] == 0.0)

    # Row 2: PANGIN=1 (present) -> hrattack=1
    check("cirs row2 = 1 (PANGIN=1)",   cirs_out["cirs"].iloc[2] == 1.0)

    # Row 3: PANGIN=NaN -> angina absent -> hrattack=0
    check("cirs row3 = 0 (PANGIN=NaN)", cirs_out["cirs"].iloc[3] == 0.0)

    # Test with pangin=None (MR2 scenario)
    cirs_mr2 = derive_cirs(
        df_cirs,
        pa8="PA8", pa7ba="PA7BA", pa7bb="PA7BB", pangin=None,
        sa11s="SA11S", sa12a="SA12A",
        sa11a="SA11A", sa11b="SA11B",
        sa7d="SA7D",
        sa12f="SA12F", sa11aa="SA11AA",
        sa11l="SA11L",
        sa11j="SA11J",
        sa11d="SA11D", sa12g="SA12G",
        sa11v="SA11V", pa6c="PA6C", pa6d="PA6D", sa11y="SA11Y",
        sa11x="SA11X", sa11g="SA11G",
        pa26="PA26", pa28a="PA28A",
        sa11z="SA11Z", pa6a="PA6A",
    )
    check("MR2 cirs == cirs_no_angina",
          (cirs_mr2["cirs"] == cirs_mr2["cirs_no_angina"]).all())
    # Row 0: PA8=1 -> hrattack=1, hyper=1, EENT=1 = 3; no angina component
    check("MR2 cirs row0 = 3",          cirs_mr2["cirs"].iloc[0] == 3.0)
    # Row 1: no conditions (PANGIN ignored) -> 0
    check("MR2 cirs row1 = 0",          cirs_mr2["cirs"].iloc[1] == 0.0)

    print("\nderive_health_affect_work")
    df_haw = pd.DataFrame({
        "PA4": [5.0, np.nan, 3.0],
        "PA5": [10.0, 7.0, np.nan],
    })
    haw = derive_health_affect_work(df_haw, "PA4", "PA5")
    check("5 + 10 = 15",                haw.iloc[0] == 15.0)
    check("NA + 7 = 7",                 haw.iloc[1] == 7.0)
    check("3 + NA = 3",                 haw.iloc[2] == 3.0)

    print("\nderive_middle_age")
    df_ma = pd.DataFrame({
        "SEX":   [1.0, 2.0, 1.0, 1.0],
        "SE16":  [40.0, 45.0, 5.0, 50.0],   # male start
        "SE17":  [55.0, 60.0, 55.0, 105.0],  # male end
        "SE18":  [42.0, 43.0, 42.0, 43.0],   # female start
        "SE19":  [58.0, 59.0, 58.0, 59.0],   # female end
    })
    ma = derive_middle_age(df_ma, "SE16", "SE17", "SE18", "SE19", "SEX")
    check("male start = 40",            ma["startMiddleAge"].iloc[0] == 40.0)
    check("female start = 43",          ma["startMiddleAge"].iloc[1] == 43.0)
    check("male start < 10 -> NaN",     np.isnan(ma["startMiddleAge"].iloc[2]))
    check("male end > 100 -> NaN",      np.isnan(ma["endMiddleAge"].iloc[3]))
    check("female end = 59",            ma["endMiddleAge"].iloc[1] == 59.0)

    print("\nmissing column handling")
    df_empty = pd.DataFrame({"X": [1.0, 2.0, 3.0]})
    # Smoke exposure with missing columns
    se_miss = derive_smoke_exposure(df_empty, ["MISSING1", "MISSING2"], threshold=1)
    check("missing cols -> all NaN",     se_miss.isna().all())
    # ACE with missing columns
    ace_miss = derive_ace_composite(df_empty, ["MISSING1", "MISSING2"], min_pct=0.5)
    check("ACE missing cols -> all NaN", ace_miss.isna().all())
    # Activity with missing columns
    act_miss = derive_activity_composites(
        df_empty,
        vig_cols=["M1", "M2", "M3", "M4", "M5", "M6"],
        mod_cols=["M7", "M8", "M9", "M10", "M11", "M12"],
        light_cols=["M13", "M14", "M15", "M16", "M17", "M18"],
    )
    check("activity missing -> all NaN",
          act_miss[["vmJobAct", "lightAct"]].isna().all().all())

    print(f"\nResults: {passed} passed, {failed} failed")
    if errors:
        print(f"Failed tests: {', '.join(errors)}")
    sys.exit(1 if failed else 0)
