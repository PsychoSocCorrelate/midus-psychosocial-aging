"""
Recode functions for SPSS-derived survey frames: 8 recode types plus sentinel cleanup.

Binary recodes map 1 -> 1 and 2 -> 0; everything else, including DK (7),
RF (8) and out-of-range codes, becomes NaN. Some binary-listed columns need
other handling: ``*SALCOH`` (alcohProb) is already 0/1 in the SAV and ``*PANGIN`` is an angina
class (0-3), so both go in ``skip_binary_recode``; ``*PC4`` is a count (0-4)
and goes in ``count_to_binary`` (0 -> 0, >=1 -> 1).

Usage
-----
>>> import pandas as pd
>>> from preprocess.survey.recode_engine import apply_recodes, cleanup_sentinels
>>>
>>> config = {
...     "binary_vars": ["B1PB6", "B1PB7", "B1SC1"],
...     "three_vars": ["B1SG6"],
...     "skip_binary_recode": ["B1SALCOH"],
...     "count_to_binary": ["B1PC4"],
... }
>>> df = apply_recodes(df, config)
>>> df = cleanup_sentinels(df)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "apply_recodes",
    "cleanup_sentinels",
    "recode_binary",
    "recode_reverse",
    "recode_three_alt",
    "recode_six_weighted",
    "recode_sqrt",
    "recode_count_to_binary",
]

# Lookup table for six_weighted_vars
# Frequency-weighted mapping: 1=3.0, 2=1.0, 3=0.75, 4=0.25, 5=0.1, 6=0.0
SIX_WEIGHTED_MAP: dict[float, float] = {
    1.0: 3.0,
    2.0: 1.0,
    3.0: 0.75,
    4.0: 0.25,
    5.0: 0.1,
    6.0: 0.0,
}


# Recode functions (Series in, Series out)


def recode_binary(s: pd.Series) -> pd.Series:
    """Recode a binary SPSS variable: 1->1, 2->0, 3->NaN, else->NaN.

    DK (7), RF (8) and out-of-range codes map to NaN.

    Parameters
    ----------
    s : pd.Series
        Numeric series with values typically in {1, 2, 3} plus NaN.

    Returns
    -------
    pd.Series
        Recoded series with values in {0.0, 1.0, NaN}.
    """
    result = np.where(s == 1, 1.0, np.where(s == 2, 0.0, np.where(s == 3, np.nan, np.nan)))
    return pd.Series(result, index=s.index, dtype="float64")


def recode_count_to_binary(s: pd.Series) -> pd.Series:
    """Recode a count variable to binary: 0->0, >=1->1, NaN stays NaN.

    Used for variables like ``*PC4`` (child loss count 0-4).

    Parameters
    ----------
    s : pd.Series
        Numeric series with non-negative integer values.

    Returns
    -------
    pd.Series
        Binary series: 0.0 or 1.0.
    """
    result = np.where(s.isna(), np.nan, np.where(s >= 1, 1.0, 0.0))
    return pd.Series(result, index=s.index, dtype="float64")


def recode_reverse(s: pd.Series, scale_max_plus_one: int) -> pd.Series:
    """Reverse-code an ordinal scale: ``scale_max_plus_one - x``.

    For a 3-point scale (1-3): pass ``scale_max_plus_one=4``  -> ``4 - x``
    For a 4-point scale (1-4): pass ``scale_max_plus_one=5``  -> ``5 - x``
    For a 5-point scale (1-5): pass ``scale_max_plus_one=6``  -> ``6 - x``
    For a 6-point scale (1-6): pass ``scale_max_plus_one=6``  -> ``6 - x``
        (6-point uses 6 - x, so the range is 0-5)

    NaN values propagate naturally through subtraction.

    Parameters
    ----------
    s : pd.Series
        Numeric ordinal values.
    scale_max_plus_one : int
        The constant to subtract from.

    Returns
    -------
    pd.Series
        Reverse-coded values (float64 to preserve NaN).
    """
    return (scale_max_plus_one - s).astype("float64")


def recode_three_alt(s: pd.Series) -> pd.Series:
    """Recode the 3-point alternative scale (weight change items).

    Mapping:
    - 4  -> NaN  (not applicable / didn't gain-lose weight)
    - >=2 -> 0
    - 1  -> 1
    - NaN -> NaN

    Parameters
    ----------
    s : pd.Series
        Numeric series with values in {1, 2, 4} plus NaN.

    Returns
    -------
    pd.Series
        Recoded series with values in {0.0, 1.0, NaN}.
    """
    result = np.where(
        s.isna(),
        np.nan,
        np.where(s == 4, np.nan, np.where(s >= 2, 0.0, s)),
    )
    return pd.Series(result, index=s.index, dtype="float64")


def recode_six_weighted(s: pd.Series) -> pd.Series:
    """Apply frequency-weighted mapping to a 6-point activity scale.

    Mapping: {1: 3.0, 2: 1.0, 3: 0.75, 4: 0.25, 5: 0.1, 6: 0.0}.
    Values not in the mapping (including DK/RF codes) become NaN.

    Parameters
    ----------
    s : pd.Series
        Numeric series with values in {1, 2, 3, 4, 5, 6} plus NaN.

    Returns
    -------
    pd.Series
        Mapped continuous values.
    """
    return s.map(SIX_WEIGHTED_MAP).astype("float64")


def recode_sqrt(s: pd.Series) -> pd.Series:
    """Square-root transform.  Negative values become NaN.

    Parameters
    ----------
    s : pd.Series
        Numeric series (non-negative expected).

    Returns
    -------
    pd.Series
        Square-root of the input; negatives -> NaN.
    """
    return np.sqrt(s.astype("float64"))


# Orchestrator


def apply_recodes(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Apply all recode types to *df* based on *config*, returning a new DataFrame.

    Types are applied in a fixed order: binary, three, three_alt, four, five,
    six, six_weighted, sqrt. Columns absent from *df* are skipped.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (typically read from an SPSS SAV file via
        ``pyreadstat`` or ``pandas``).
    config : dict
        Keys correspond to recode types.  Each value is a list of column
        names.  Recognised keys:

        - ``binary_vars``       : 1->1, 2->0, 3->NaN, else->NaN
        - ``three_vars``        : 4 - x   (reverse 3-point)
        - ``three_vars_alt``    : 4->NaN, >=2->0, else->1  (weight change)
        - ``four_vars``         : 5 - x   (reverse 4-point)
        - ``five_vars``         : 6 - x   (reverse 5-point)
        - ``six_vars``          : 6 - x   (reverse 6-point, not 7 - x)
        - ``six_weighted_vars`` : frequency-weight lookup
        - ``sqrt_vars``         : sqrt(x)
        - ``skip_binary_recode``: columns that appear in ``binary_vars`` but
          must not be recoded (e.g. ``*SALCOH``, ``*PANGIN``).
        - ``count_to_binary``   : columns where 0->0, >=1->1 (e.g. ``*PC4``).

    Returns
    -------
    pd.DataFrame
        A copy of *df* with recodes applied.  The original *df* is not
        modified.
    """
    d = df.copy()

    # Sets of columns to exclude from the standard binary recode
    skip_set = set(config.get("skip_binary_recode", []))
    count_set = set(config.get("count_to_binary", []))

    # binary_vars
    for col in config.get("binary_vars", []):
        if col not in d.columns:
            continue
        if col in skip_set:
            # Leave as-is (already coded correctly in the SAV)
            continue
        if col in count_set:
            d[col] = recode_count_to_binary(d[col])
        else:
            d[col] = recode_binary(d[col])

    # Also apply count_to_binary for vars listed only there (not in binary_vars)
    for col in config.get("count_to_binary", []):
        if col in d.columns and col not in set(config.get("binary_vars", [])):
            d[col] = recode_count_to_binary(d[col])

    # three_vars (4 - x)
    for col in config.get("three_vars", []):
        if col in d.columns:
            d[col] = recode_reverse(d[col], 4)

    # three_vars_alt
    for col in config.get("three_vars_alt", []):
        if col in d.columns:
            d[col] = recode_three_alt(d[col])

    # four_vars (5 - x)
    for col in config.get("four_vars", []):
        if col in d.columns:
            d[col] = recode_reverse(d[col], 5)

    # five_vars (6 - x)
    for col in config.get("five_vars", []):
        if col in d.columns:
            d[col] = recode_reverse(d[col], 6)

    # six_vars (6 - x)
    for col in config.get("six_vars", []):
        if col in d.columns:
            d[col] = recode_reverse(d[col], 6)

    # six_weighted_vars
    for col in config.get("six_weighted_vars", []):
        if col in d.columns:
            d[col] = recode_six_weighted(d[col])

    # sqrt_vars
    for col in config.get("sqrt_vars", []):
        if col in d.columns:
            d[col] = recode_sqrt(d[col])

    return d


def cleanup_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    """Set all negative numeric cell values to NaN.

    SPSS DK (7) and RF (8) codes that are not formally declared as
    user-missing survive ``read_sav()`` / ``pyreadstat.read_sav()`` as
    numeric.  After reverse coding (e.g. ``6 - 7 = -1``) they become
    negative.  This function zeroes them out.

    Non-numeric columns are left untouched.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that may contain negative sentinel values.

    Returns
    -------
    pd.DataFrame
        A copy with negatives in numeric columns replaced by NaN.
    """
    d = df.copy()
    num_cols = d.select_dtypes(include="number").columns
    d[num_cols] = d[num_cols].where(d[num_cols] >= 0, other=np.nan)
    return d


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

    print("recode_binary")
    s = pd.Series([1, 2, 3, np.nan, 0, 4, 7, 8])
    r = recode_binary(s)
    check("1 -> 1",          r.iloc[0] == 1.0)
    check("2 -> 0",          r.iloc[1] == 0.0)
    check("3 -> NaN",        np.isnan(r.iloc[2]))
    check("NaN stays NaN",   np.isnan(r.iloc[3]))
    check("0 -> NaN", np.isnan(r.iloc[4]))
    check("4 -> NaN", np.isnan(r.iloc[5]))
    check("7 (DK) -> NaN", np.isnan(r.iloc[6]))
    check("8 (RF) -> NaN", np.isnan(r.iloc[7]))

    print("\nrecode_count_to_binary")
    s = pd.Series([0, 1, 2, 3, 4, np.nan])
    r = recode_count_to_binary(s)
    check("0 -> 0",        r.iloc[0] == 0.0)
    check("1 -> 1",        r.iloc[1] == 1.0)
    check("4 -> 1",        r.iloc[4] == 1.0)
    check("NaN -> NaN",    np.isnan(r.iloc[5]))

    print("\nrecode_reverse")
    s = pd.Series([1, 2, 3, np.nan])
    # 3-point: 4 - x
    r3 = recode_reverse(s, 4)
    check("3pt: 1 -> 3",   r3.iloc[0] == 3.0)
    check("3pt: 3 -> 1",   r3.iloc[2] == 1.0)
    check("3pt: NaN",      np.isnan(r3.iloc[3]))
    # 5-point: 6 - x
    s5 = pd.Series([1, 3, 5, np.nan])
    r5 = recode_reverse(s5, 6)
    check("5pt: 1 -> 5",   r5.iloc[0] == 5.0)
    check("5pt: 5 -> 1",   r5.iloc[2] == 1.0)
    # 6-point: 6 - x
    s6 = pd.Series([1, 6, np.nan])
    r6 = recode_reverse(s6, 6)
    check("6pt: 1 -> 5",   r6.iloc[0] == 5.0)
    check("6pt: 6 -> 0",   r6.iloc[1] == 0.0)

    print("\nrecode_three_alt (weight change)")
    s = pd.Series([1, 2, 3, 4, np.nan])
    r = recode_three_alt(s)
    check("1 -> 1",         r.iloc[0] == 1.0)
    check("2 -> 0 (>=2)",   r.iloc[1] == 0.0)
    check("3 -> 0 (>=2)",   r.iloc[2] == 0.0)
    check("4 -> NaN (N/A)", np.isnan(r.iloc[3]))
    check("NaN -> NaN",     np.isnan(r.iloc[4]))

    print("\nrecode_six_weighted")
    s = pd.Series([1, 2, 3, 4, 5, 6, np.nan, 7])
    r = recode_six_weighted(s)
    check("1 -> 3.0",       r.iloc[0] == 3.0)
    check("2 -> 1.0",       r.iloc[1] == 1.0)
    check("3 -> 0.75",      r.iloc[2] == 0.75)
    check("4 -> 0.25",      r.iloc[3] == 0.25)
    check("5 -> 0.1",       r.iloc[4] == 0.1)
    check("6 -> 0.0",       r.iloc[5] == 0.0)
    check("NaN -> NaN",     np.isnan(r.iloc[6]))
    check("7 (DK) -> NaN",  np.isnan(r.iloc[7]))

    print("\nrecode_sqrt")
    s = pd.Series([0, 4, 9, 16, np.nan])
    r = recode_sqrt(s)
    check("sqrt(0)  = 0",   r.iloc[0] == 0.0)
    check("sqrt(4)  = 2",   r.iloc[1] == 2.0)
    check("sqrt(9)  = 3",   r.iloc[2] == 3.0)
    check("sqrt(16) = 4",   r.iloc[3] == 4.0)
    check("sqrt(NaN)= NaN", np.isnan(r.iloc[4]))

    print("\napply_recodes (integration)")
    df = pd.DataFrame({
        "B1PB6":    [1, 2, 3, np.nan],
        "B1SALCOH": [0, 1, 0, np.nan],   # already 0/1, must skip
        "B1PANGIN": [0, 1, 2, 3],         # 0-3 angina class, must skip
        "B1PC4":    [0, 1, 3, np.nan],    # count 0-4, must go to count_to_binary
        "B1SG6":    [1, 2, 3, np.nan],    # three_vars: 4-x
        "B1SA43A":  [1, 2, 4, np.nan],    # three_vars_alt
        "B1SM9":    [1, 2, 3, 4],         # four_vars: 5-x
        "B1PA1":    [1, 3, 5, np.nan],    # five_vars: 6-x
        "B1SA67":   [1, 3, 6, np.nan],    # six_vars: 6-x
        "B1SA30A":  [1, 3, 6, np.nan],    # six_weighted
        "B1SH9A":   [0, 4, 9, np.nan],    # sqrt
    })
    cfg = {
        "binary_vars":       ["B1PB6", "B1SALCOH", "B1PANGIN", "B1PC4"],
        "three_vars":        ["B1SG6"],
        "three_vars_alt":    ["B1SA43A"],
        "four_vars":         ["B1SM9"],
        "five_vars":         ["B1PA1"],
        "six_vars":          ["B1SA67"],
        "six_weighted_vars": ["B1SA30A"],
        "sqrt_vars":         ["B1SH9A"],
        "skip_binary_recode": ["B1SALCOH", "B1PANGIN"],
        "count_to_binary":   ["B1PC4"],
    }
    out = apply_recodes(df, cfg)

    check("binary B1PB6[0]=1",         out["B1PB6"].iloc[0] == 1.0)
    check("binary B1PB6[1]=0",         out["B1PB6"].iloc[1] == 0.0)
    check("binary B1PB6[2]=NaN",       np.isnan(out["B1PB6"].iloc[2]))
    check("skip SALCOH unchanged[0]=0", out["B1SALCOH"].iloc[0] == 0.0)
    check("skip SALCOH unchanged[1]=1", out["B1SALCOH"].iloc[1] == 1.0)
    check("skip PANGIN unchanged[2]=2", out["B1PANGIN"].iloc[2] == 2.0)
    check("count_to_binary PC4[0]=0",  out["B1PC4"].iloc[0] == 0.0)
    check("count_to_binary PC4[1]=1",  out["B1PC4"].iloc[1] == 1.0)
    check("count_to_binary PC4[2]=1",  out["B1PC4"].iloc[2] == 1.0)
    check("three_vars SG6[0]=3",       out["B1SG6"].iloc[0] == 3.0)
    check("three_alt SA43A[0]=1",      out["B1SA43A"].iloc[0] == 1.0)
    check("three_alt SA43A[2]=NaN",    np.isnan(out["B1SA43A"].iloc[2]))
    check("four_vars SM9[0]=4",        out["B1SM9"].iloc[0] == 4.0)
    check("five_vars PA1[0]=5",        out["B1PA1"].iloc[0] == 5.0)
    check("six_vars SA67[0]=5",        out["B1SA67"].iloc[0] == 5.0)
    check("six_vars SA67[2]=0",        out["B1SA67"].iloc[2] == 0.0)
    check("six_weighted SA30A[0]=3.0", out["B1SA30A"].iloc[0] == 3.0)
    check("sqrt SH9A[1]=2.0",         out["B1SH9A"].iloc[1] == 2.0)

    # Verify original df not mutated
    check("original df unchanged",     df["B1PB6"].iloc[0] == 1)

    print("\ncleanup_sentinels")
    df2 = pd.DataFrame({
        "num": [1.0, -1.0, 0.0, -7.0, np.nan, 5.0],
        "txt": ["a", "b", "c", "d", "e", "f"],
    })
    out2 = cleanup_sentinels(df2)
    check("positive kept",        out2["num"].iloc[0] == 1.0)
    check("-1 -> NaN",            np.isnan(out2["num"].iloc[1]))
    check("0 kept",               out2["num"].iloc[2] == 0.0)
    check("-7 -> NaN",            np.isnan(out2["num"].iloc[3]))
    check("NaN stays NaN",        np.isnan(out2["num"].iloc[4]))
    check("text col untouched",   out2["txt"].iloc[1] == "b")
    check("original df2 unchanged", df2["num"].iloc[1] == -1.0)

    print("\nmissing columns skipped")
    df3 = pd.DataFrame({"X": [1, 2, 3]})
    cfg3 = {"binary_vars": ["NONEXISTENT"], "sqrt_vars": ["ALSO_MISSING"]}
    out3 = apply_recodes(df3, cfg3)
    check("no error on missing cols", True)
    check("data unchanged",           (out3["X"] == df3["X"]).all())

    print(f"\nResults: {passed} passed, {failed} failed")
    if errors:
        print(f"Failed tests: {', '.join(errors)}")
    sys.exit(1 if failed else 0)
