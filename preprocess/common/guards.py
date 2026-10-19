"""Row-count and key-uniqueness guards for cohort concats and merges.

M2ID and MRID are both renamed to M2MRID without an offset, so every cross-cohort
join relies on the two id ranges (M2 10001-19193, MR 30000-39985) never overlapping.
pandas also matches null keys to null keys in a merge, and M2MRID comes from
pd.to_numeric(..., errors="coerce"), so a malformed id becomes a null that can fan
out rows. These checks raise instead of warning.
"""
from __future__ import annotations

import pandas as pd


def assert_unique_key(df: pd.DataFrame, key: str, where: str) -> pd.DataFrame:
    """Raise if `key` repeats. Nulls are reported separately, not counted as dupes."""
    n_null = int(df[key].isna().sum())
    dupes = df[key].dropna()
    dupes = dupes[dupes.duplicated(keep=False)]
    if len(dupes):
        sample = sorted(dupes.unique())[:10]
        raise ValueError(
            f"{where}: {dupes.nunique()} duplicate values of {key!r} "
            f"across {len(dupes)} rows (e.g. {sample}). "
            f"Either the M and R id ranges overlap or a cohort frame repeats a person."
        )
    if n_null:
        raise ValueError(
            f"{where}: {n_null} rows have a null {key!r}. Nulls match nulls in a "
            f"pandas merge."
        )
    return df


def concat_cohorts(frames: list[pd.DataFrame], key: str, where: str) -> pd.DataFrame:
    """Concatenate per-cohort frames and check the id key is still unique."""
    out = pd.concat(frames, ignore_index=True)
    return assert_unique_key(out, key, where)


def merge_no_fanout(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: str,
    where: str,
    how: str = "left",
) -> pd.DataFrame:
    """Left-join that refuses to change the row count.

    Drops null keys on the right, checks the right key is unique, and checks the
    row count is unchanged.
    """
    n_before = len(left)
    n_null_right = int(right[on].isna().sum())
    right = right[right[on].notna()]
    assert_unique_key(right, on, f"{where} (right side)")
    out = left.merge(right, on=on, how=how)
    if len(out) != n_before:
        raise ValueError(
            f"{where}: merge changed the row count {n_before} -> {len(out)}. "
            f"Check for null keys on the left."
        )
    if n_null_right:
        print(f"  [guard] {where}: dropped {n_null_right} right-side rows with a null "
              f"{on} before merging")
    return out
