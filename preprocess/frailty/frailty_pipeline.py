"""
Frailty scoring and imputation for MIDUS waves 1 and 2.

The pipeline:
  1. Loads a frailty key xlsx and filters to non-psychological items (40 items).
  2. Renames raw SPSS columns to standardised recode_var names (with _1, _2, ... suffixes
     for multi-flag items separated by "/" in the key).
  3. Applies frailty-specific recodes:
       - binary (2-x), binary_alt (_3 suffix), three-point, four-point, five-point,
         five-reverse, five-alt, six-point maps
       - combine_2 logic (and/or/rx modes) for multi-flag items
       - limitRecreation three-flag max-of-three
       - multiCancer multi-flag (>=2 of 11)
       - heartTrouble any-flag
       - chronicIllness continuous rescale
       - _99 sentinel override for binary_vars_alt
  4. Drops rows with >20% missing across frailty vars.
  5. Imputes with sklearn IterativeImputer (XGBoost estimator, one run,
     max_iter=10). The imputer does not reach its stopping criterion within 10
     iterations on any cohort (nor within 50; on midus2 the 10- and 50-iteration
     solutions differ in nearly every imputed cell, by about 0.18 on a 0-1 scale),
     so imputed values are truncated rather than converged. Cell missingness is
     1.1-1.8% per cohort and 62-74% of participants have no imputed cells. One
     run is used because the estimator is deterministic without subsampling, so
     repeated runs are identical.
  6. Computes FInoPsy = row mean of the 40 frailty items, each scored 0-1.

Usage:
  uv run python -m preprocess.frailty.frailty_pipeline --wave 1
  uv run python -m preprocess.frailty.frailty_pipeline --wave 2
  uv run python -m preprocess.frailty.frailty_pipeline --wave 1 --cohort midus2
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import pyreadstat
from scipy.stats import mode as scipy_mode
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from preprocess.common.paths import ROOT as PROJECT_ROOT
from preprocess.survey.missing_reasons import write_reason_artifacts

logger = logging.getLogger(__name__)


# Estimator resolution
# XGBoost with device="cuda" falls back to CPU when no GPU is visible, and CPU and
# GPU fits differ (on a 2000x39 fit with the same seed, 358 of 2000 predictions
# differed by more than 1e-6). A missing GPU is therefore an error unless
# allow_cpu_fallback=True.


def _cuda_device_available() -> bool:
    """True if a CUDA device is visible."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        pass
    exe = shutil.which("nvidia-smi")
    if not exe:
        return False
    try:
        proc = subprocess.run([exe, "-L"], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0 and proc.stdout.strip().startswith("GPU")


def resolve_estimator_spec(allow_cpu_fallback: bool = False) -> dict[str, Any]:
    """Choose the imputation estimator.

    Raises
    ------
    RuntimeError
        If xgboost is missing or no GPU is visible and the caller has not opted
        into the CPU fallback. Both conditions change imputed values.
    """
    if _HAS_XGB and _cuda_device_available():
        return {"kind": "xgboost", "device": "cuda", "xgboost": xgb.__version__}
    if not allow_cpu_fallback:
        reason = (
            "xgboost is not installed"
            if not _HAS_XGB
            else "no CUDA device is visible"
        )
        raise RuntimeError(
            f"Frailty imputation needs a GPU: {reason}. CPU fits differ from GPU "
            "fits. Pass --allow-cpu-fallback to run on CPU anyway."
        )
    if _HAS_XGB:
        return {"kind": "xgboost", "device": "cpu", "xgboost": xgb.__version__}
    return {"kind": "randomforest", "device": "cpu", "xgboost": None}


def _build_estimator(spec: dict[str, Any], seed: int, n_estimators: int) -> Any:
    """Instantiate the estimator named by *spec* for one imputation run."""
    if spec["kind"] == "xgboost":
        return xgb.XGBRegressor(
            n_estimators=n_estimators,
            device=spec["device"],
            tree_method="hist",
            random_state=seed,
            verbosity=1,  # show the GPU-fallback warning
        )
    return RandomForestRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=-1)


# Constants: wave -> cohort config

CANONICAL_FRAILTY_KEY = (
    PROJECT_ROOT / "data/key_map/frailty/frailty_key_MIDUS_canonical.xlsx"
)

WAVE_CONFIG: dict[int, dict[str, Any]] = {
    1: {
        "key_path": CANONICAL_FRAILTY_KEY,
        "cohorts": {
            "midus2": {
                "sav_path": PROJECT_ROOT / "data/raw/wave1/M2_P1_SURVEY_N4963_20200720.sav",
                "id_col": "M2ID",
                "status_col": "B1STATUS",
                "status_val": 2,
                "matched_col": "matched_MIDUS2",
                "id_prefix": None,
            },
            "midusr": {
                "sav_path": PROJECT_ROOT / "data/raw/wave1/MR1_P1_SURVEY_N3577_20250407.sav",
                "id_col": "MRID",
                "status_col": "RA1STATUS",
                "status_val": None,  # filter != 1
                "status_negate": True,
                "status_negate_val": 1,
                "matched_col": "matched_MIDUSR",
                "id_prefix": None,  # M2ID and MRID ranges don't collide
            },
        },
        "output_dir": PROJECT_ROOT / "data/processed/wave1",
        "pooled_csv": "pooled_frailty_M2R.csv",
    },
    2: {
        "key_path": CANONICAL_FRAILTY_KEY,
        "cohorts": {
            "midus3": {
                "sav_path": PROJECT_ROOT / "data/raw/wave2/M3_P1_SURVEY_N3294_20251029.sav",
                "id_col": "M2ID",
                "status_col": "C1STATUS",
                "status_val": 4,
                "matched_col": "matched_MIDUS3",
                "id_prefix": None,
            },
            "mr2": {
                "sav_path": PROJECT_ROOT / "data/raw/wave2/MR2_P1_SURVEY_N2154_20251003.sav",
                "id_col": "MRID",
                "status_col": "RB1STATUS",
                "status_val": 4,
                "matched_col": "matched_MR2",
                "id_prefix": None,
            },
        },
        "output_dir": PROJECT_ROOT / "data/processed/wave2",
        "pooled_csv": "pooled_frailty_M3MR2.csv",
    },
}

# 1. Load frailty key


def load_frailty_key(path: str | Path) -> pd.DataFrame:
    """Load frailty key xlsx, filter to non-psychological included items.

    Returns DataFrame with columns: recode_var, matched_variable (per cohort),
    variable_labels, Response, etc.  Only the 40 non-psychological items are kept.
    """
    path = Path(path)
    key = pd.read_excel(path, engine="openpyxl")
    key = key[key["frailty_include"] == "y"]
    key = key[key["PsychoFlag"].isna()]
    logger.info("Frailty key loaded: %d non-psychological items from %s", len(key), path.name)
    return key.reset_index(drop=True)


# 2. Rename SPSS columns to recode_var names


def _rename_frailty_vars(
    data: pd.DataFrame,
    id_col: str,
    lookup: pd.DataFrame,
) -> pd.DataFrame:
    """Rename raw SPSS columns to standardised recode_var names with _1, _2, ... suffixes.

    The lookup must have columns 'matched_variable' and 'recode_var'.
    Multi-source variables are separated by "/" in matched_variable (e.g. "B1PC7/B1PC7A").

    Parameters
    ----------
    data : DataFrame with raw SPSS columns
    id_col : name of ID column to preserve
    lookup : DataFrame with matched_variable and recode_var columns

    Returns
    -------
    DataFrame with id_col + renamed frailty columns
    """
    if id_col not in data.columns:
        raise ValueError(f"ID column '{id_col}' not found in data")

    result = data[[id_col]].copy()

    for _, row in lookup.iterrows():
        raw_match = str(row["matched_variable"])
        new_base = row["recode_var"]

        parts = [p.strip() for p in raw_match.split("/")]
        for j, orig_var in enumerate(parts, start=1):
            new_name = f"{new_base}_{j}"
            if orig_var not in data.columns or orig_var in ("", "nan"):
                warnings.warn(
                    f"Variable '{orig_var}' not found in data; filling '{new_name}' with NA"
                )
                result[new_name] = np.nan
            else:
                result[new_name] = pd.to_numeric(data[orig_var], errors="coerce")

    return result


# 3. Frailty-specific recodes

# Recode maps (raw response -> frailty score)
_MAP3 = {1: 1.00, 2: 0.00, 3: 0.50}
_MAP4 = {1: 1.00, 2: 0.66, 3: 0.33, 4: 0.00}
_MAP5 = {1: 0.00, 2: 0.25, 3: 0.50, 4: 0.75, 5: 1.00}
_MAP5_REV = {1: 1.00, 2: 0.75, 3: 0.50, 4: 0.25, 5: 0.00}
_MAP5_ALT = {0: 0.00, 1: 0.33, 2: 0.66, 3: 1.00, 6: 0.00}
_MAP6 = {1: 1.00, 2: 0.80, 3: 0.60, 4: 0.40, 5: 0.20, 6: 0.00}

# Variable groups
_BINARY_VARS = [
    "migraine", "dental", "insomnia", "chestPain", "lungProb", "asthma",
    "allergic", "skinProb", "cancer", "stomachPain", "sciatica", "hernia",
    "thyroidDisease", "gallbladderProb", "painDiagnosed",
]
_BINARY_VARS_ALT = ["heartAttack", "irregularHeart"]
_THREE_VARS = ["highBP"]
_FOUR_VARS = ["limitShower", "limitWalking", "limitGrocery", "limitTravel", "limitModAct"]
_FIVE_VARS = ["SRH"]
_FIVE_REVERSE_VARS = ["lonely"]
_FIVE_VARS_ALT = ["angina"]
_SIX_VARS = ["jointPain", "backPain", "headPain", "bladderProb"]

_TWO_AND = ["heartAttack", "irregularHeart", "hipPain", "kneePain"]
_TWO_OR = ["stroke"]
_TWO_RX = ["cholesterol", "arthritis", "diabetes"]


def _apply_recode_map(
    data: pd.DataFrame,
    out: pd.DataFrame,
    var: str,
    mapping: dict[int | float, float] | None,
    fn: Any | None = None,
    suffix: str = "_1",
) -> None:
    """Apply a recode map or function to a single variable.

    An all-NA source column gives an all-NaN output.
    """
    in_col = f"{var}{suffix}"
    if in_col not in data.columns:
        warnings.warn(f"Missing variable: {in_col}")
        return

    if data[in_col].isna().all():
        out[var] = np.nan
        return

    if fn is not None:
        out[var] = fn(data[in_col])
    elif mapping is not None:
        out[var] = data[in_col].map(mapping)
    else:
        raise ValueError("Must provide either mapping or fn")


def _combine_2(
    data: pd.DataFrame,
    var: str,
    mode: Literal["and", "or", "rx"],
) -> pd.Series:
    """Combine two-flag variables.

    If v2 is absent or all-NA, recode v1 alone (1->1, 2->0, else NaN).
    """
    v1_col = f"{var}_1"
    v2_col = f"{var}_2"

    n = len(data)
    result = pd.Series(np.nan, index=data.index)

    v1_exists = v1_col in data.columns and not data[v1_col].isna().all()
    v2_exists = v2_col in data.columns and not data[v2_col].isna().all()

    if not v2_exists:
        # Fall back to v1-only recode
        if v1_exists:
            v1 = data[v1_col]
            if mode == "rx" and v1.dropna().nunique() >= 6:
                # Single ordinal variable (e.g. MIDUS3 merged binary+frequency)
                result = np.select(
                    [v1 == 1, v1 == 2, v1 == 3, v1 == 4, v1 == 5, v1 == 6],
                    [1.0, 0.8, 0.6, 0.4, 0.2, 0.0],
                    default=np.nan,
                )
            else:
                result = np.where(v1 == 2, 0.0, np.where(v1 == 1, 1.0, np.nan))
            return pd.Series(result, index=data.index)
        else:
            return pd.Series(np.nan, index=data.index)

    v1 = data[v1_col]
    v2 = data[v2_col]

    if mode == "rx":
        n_unique = v1.dropna().nunique()
        if n_unique < 6:
            # Binary v1 + ordinal v2
            result = np.select(
                [
                    v1 == 2,
                    v2 == 1,
                    v2 == 2,
                    v2 == 3,
                    v2 == 4,
                    v2 == 5,
                    v2 == 6,
                ],
                [0.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0],
                default=np.nan,
            )
        else:
            # Ordinal v1 only
            result = np.select(
                [v1 == 1, v1 == 2, v1 == 3, v1 == 4, v1 == 5, v1 == 6],
                [1.0, 0.8, 0.6, 0.4, 0.2, 0.0],
                default=np.nan,
            )
    elif mode == "and":
        result = np.where(
            (v1 == 1) & (v2 == 1), 1.0,
            np.where((v1 == 2) | (v2 == 2), 0.0, np.nan),
        )
    elif mode == "or":
        result = np.where(
            (v1 == 1) | (v2 == 1), 1.0,
            np.where((v1 == 2) & (v2 == 2), 0.0, np.nan),
        )

    return pd.Series(result, index=data.index, dtype=float)


def apply_frailty_recodes(df: pd.DataFrame, key: pd.DataFrame) -> pd.DataFrame:
    """Apply all frailty-specific recodes to a renamed DataFrame.

    Parameters
    ----------
    df : DataFrame from _rename_frailty_vars (has id_col + recode_var_N columns)
    key : frailty key (used to identify recode_var list)

    Returns
    -------
    DataFrame with id_col + recoded frailty variables (one per recode_var)
    """
    out = df.copy()

    # Binary: 2-x (suffix _1)
    for var in _BINARY_VARS:
        _apply_recode_map(df, out, var, mapping=None, fn=lambda x: 2 - x, suffix="_1")

    # Binary alt: 2-x (suffix _3) -> stored as var_99
    for var in _BINARY_VARS_ALT:
        col_3 = f"{var}_3"
        col_99 = f"{var}_99"
        if col_3 in df.columns and not df[col_3].isna().all():
            out[col_99] = 2 - df[col_3]
        else:
            out[col_99] = np.nan

    for var in _THREE_VARS:
        _apply_recode_map(df, out, var, mapping=_MAP3, suffix="_1")

    for var in _FOUR_VARS:
        _apply_recode_map(df, out, var, mapping=_MAP4, suffix="_1")

    for var in _FIVE_VARS:
        _apply_recode_map(df, out, var, mapping=_MAP5, suffix="_1")

    for var in _FIVE_REVERSE_VARS:
        _apply_recode_map(df, out, var, mapping=_MAP5_REV, suffix="_1")

    for var in _FIVE_VARS_ALT:
        _apply_recode_map(df, out, var, mapping=_MAP5_ALT, suffix="_1")

    for var in _SIX_VARS:
        _apply_recode_map(df, out, var, mapping=_MAP6, suffix="_1")

    # Combine two-flag variables
    for var in _TWO_AND:
        out[var] = _combine_2(out, var, "and")
    for var in _TWO_OR:
        out[var] = _combine_2(out, var, "or")
    for var in _TWO_RX:
        out[var] = _combine_2(out, var, "rx")

    # limitRecreation: three-flag max-of-three
    var = "limitRecreation"
    col1, col2, col3 = f"{var}_1", f"{var}_2", f"{var}_3"
    if not all(c in out.columns for c in [col1, col2, col3]):
        warnings.warn(f"Missing 3-flag columns for {var}; skipping three-flag logic")
        out[var] = np.nan
    else:
        s2 = out[col2] / 10.0
        s3 = out[col3] / 10.0
        v1 = out[col1]
        max_23 = np.fmax(s2.values, s3.values)  # NaN-aware max

        out[var] = np.where(
            (v1 == 1) & (s2 > 0) & (s3 > 0), 0.0,
            np.where(v1 == 1, max_23,
                     np.where(v1 == 2, 0.0, np.nan)),
        )

    # multiCancer: multi-flag (>=2 of 11)
    mc_cols = [c for c in out.columns if c.startswith("multiCancer_")]
    for c in mc_cols:
        out[c] = np.where(out[c] == 2, 0, np.where(out[c].notna(), 1, np.nan))

    mc1 = "multiCancer_1"
    mc_aux = [c for c in mc_cols if c != mc1 and c != "multiCancer_1"]
    # multiCancer_2 through multiCancer_11
    mc_2_to_11 = [f"multiCancer_{i}" for i in range(2, 12) if f"multiCancer_{i}" in out.columns]
    if mc1 in out.columns:
        aux_sum = out[mc_2_to_11].sum(axis=1, min_count=1) if mc_2_to_11 else pd.Series(0.0, index=out.index)
        out["multiCancer"] = np.where(
            out[mc1] == 0, 0.0,
            np.where(out[mc1] == 1, (aux_sum >= 2).astype(float), np.nan),
        )
    else:
        out["multiCancer"] = np.nan

    # _99 sentinel override for binary_vars_alt
    for var in _BINARY_VARS_ALT:
        col_99 = f"{var}_99"
        if col_99 not in out.columns:
            out[col_99] = np.nan

        # Override: if _99 == 1, force var = 1; elif var == 1, keep 1; else keep var
        cur = out.get(var, pd.Series(np.nan, index=out.index))
        s99 = out[col_99]
        out[var] = np.where(s99 == 1, 1.0, np.where(cur == 1, 1.0, cur))

    # heartTrouble: any-flag
    ht_cols = [c for c in out.columns if c.startswith("heartTrouble") and c != "heartTrouble_1"]
    ht1 = "heartTrouble_1"
    if ht1 in out.columns:
        any_ht = out[ht_cols].eq(1).any(axis=1) if ht_cols else pd.Series(False, index=out.index)
        out["heartTrouble"] = np.where(
            any_ht, 1.0,
            np.where(out[ht1] == 2, 0.0, np.nan),
        )
    else:
        out["heartTrouble"] = np.nan

    # chronicIllness: continuous rescale
    ci1 = "chronicIllness_1"
    if ci1 in out.columns:
        ci = out[ci1].astype(float)
        out["chronicIllness"] = np.where(ci >= 18, 1.0, np.where(ci.notna(), ci / 18.0, np.nan))
    else:
        out["chronicIllness"] = np.nan

    # Drop numbered suffix columns (keep only final recode_var names)
    frailty_vars = list(key["recode_var"])
    id_cols = [c for c in out.columns if c not in frailty_vars and not c.endswith(
        tuple(f"_{i}" for i in range(1, 100))
    )]
    keep_cols = []
    for c in out.columns:
        if c in frailty_vars:
            keep_cols.append(c)
        elif not any(c.endswith(f"_{i}") for i in range(1, 100)):
            keep_cols.append(c)

    seen = set()
    final_cols = []
    for c in keep_cols:
        if c not in seen:
            seen.add(c)
            final_cols.append(c)

    out = out[final_cols]
    return out


# 4. Scoring (rename + recode + filter)


def _scoring_frailty(
    data: pd.DataFrame,
    id_col: str,
    lookup: pd.DataFrame,
    key: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Full scoring: rename -> recode -> missingness filter.

    Parameters
    ----------
    data : raw SAV DataFrame
    id_col : ID column name
    lookup : DataFrame with matched_variable and recode_var columns
    key : full frailty key (for recode_var list)

    Returns
    -------
    (filtered_df, missing_pct_dict)
    """
    renamed = _rename_frailty_vars(data, id_col, lookup)
    recoded = apply_frailty_recodes(renamed, key)

    frailty_vars = list(key["recode_var"])
    present_fvars = [v for v in frailty_vars if v in recoded.columns]

    missing_pct = {}
    for v in present_fvars:
        missing_pct[v] = float(recoded[v].isna().mean()) if v in recoded.columns else np.nan

    # Filter rows with >20% missing frailty vars
    n_vars = len(frailty_vars)
    miss_count = recoded[present_fvars].isna().sum(axis=1)
    mask = miss_count <= (n_vars * 0.2)
    filtered = recoded[mask].copy()

    logger.info("Scoring: %d -> %d rows after 20%% missingness filter", len(recoded), len(filtered))
    return filtered, missing_pct


# 5. Imputation


def _clamp01(arr: np.ndarray) -> np.ndarray:
    """Clamp numeric values to [0, 1]."""
    return np.clip(arr, 0.0, 1.0)


def _get_mode(values: np.ndarray) -> float:
    """Return the mode of a 1D array, ignoring NaN."""
    vals = values[~np.isnan(values)]
    if len(vals) == 0:
        return np.nan
    counter = Counter(vals)
    return counter.most_common(1)[0][0]



def _round_to_levels(values: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Snap each value to the nearest level the column actually takes."""
    return levels[np.abs(values[..., None] - levels).argmin(axis=-1)]


def _pool_categorical(col_runs: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Pool M imputation runs of one categorical column to a modal level.

    `col_runs` is (M, n_rows). The imputer returns continuous values, so each run is
    snapped to an observed level before the mode is taken. Ties go to the level
    nearest the mean of the unrounded runs, then to the lower level.
    """
    n_rows = col_runs.shape[1]
    out = np.full(n_rows, np.nan)
    for i in range(n_rows):
        runs = col_runs[:, i]
        runs = runs[~np.isnan(runs)]
        if runs.size == 0:
            continue
        snapped = _round_to_levels(runs, levels)
        vals, counts = np.unique(snapped, return_counts=True)
        top = vals[counts == counts.max()]
        if top.size == 1:
            out[i] = top[0]
        else:
            # Compare distances with a tolerance so run order (float summation order)
            # cannot break an exact tie; np.unique sorts, so the fallback is the
            # lowest level.
            d = np.abs(top - runs.mean())
            out[i] = top[np.flatnonzero(d <= d.min() + 1e-12)[0]]
    return out


def impute_frailty(
    df: pd.DataFrame,
    id_col: str,
    frailty_cols: list[str],
    n_imputations: int = 1,
    seed: int = 100,
    max_iter: int = 10,
    n_estimators: int = 200,
    max_unique_for_categorical: int = 4,
    allow_cpu_fallback: bool = False,
) -> tuple[pd.DataFrame, list[float]]:
    """Impute with IterativeImputer (estimator from resolve_estimator_spec).

    Each run uses a different seed. Numeric columns with <= max_unique_for_categorical
    unique values are treated as categorical: each run is snapped to an observed
    level and the modal level is taken. Others are mean-pooled.

    Returns
    -------
    (pooled_df, oob_errors_placeholder)
        pooled_df has id_col + frailty_cols (imputed, clamped to [0,1])
        oob_errors_placeholder is a list of NaN; IterativeImputer has no OOB error.
    """
    ids = df[[id_col]].copy()
    imp_data = df[frailty_cols].copy().astype(float)

    # Drop all-NA columns
    all_na_cols = [c for c in imp_data.columns if imp_data[c].isna().all()]
    if all_na_cols:
        logger.info("Dropping all-NA frailty vars: %s", ", ".join(all_na_cols))
    present_cols = [c for c in imp_data.columns if c not in all_na_cols]
    imp_data = imp_data[present_cols]

    if imp_data.shape[1] == 0:
        raise ValueError("No columns left for imputation after dropping all-NA columns.")

    # Identify categorical columns and their levels, taken from the pre-imputation
    # data so imputed fractions never become levels.
    is_categorical: dict[str, bool] = {}
    levels: dict[str, np.ndarray] = {}
    for col in imp_data.columns:
        observed = imp_data[col].dropna().unique()
        is_categorical[col] = len(observed) <= max_unique_for_categorical
        if is_categorical[col]:
            levels[col] = np.sort(observed.astype(float))
            logger.debug("Column '%s' -> categorical, levels %s", col, levels[col].tolist())

    spec = resolve_estimator_spec(allow_cpu_fallback=allow_cpu_fallback)
    logger.info(
        "Imputation estimator: %s on %s (xgboost=%s)",
        spec["kind"], spec["device"], spec["xgboost"],
    )
    if spec["device"] != "cuda":
        logger.warning(
            "Running on CPU; imputed values differ from GPU runs."
        )

    imputed_arrays: list[np.ndarray] = []
    for m in range(n_imputations):
        this_seed = seed + m + 1
        logger.info("Imputation %d/%d (seed=%d)", m + 1, n_imputations, this_seed)

        estimator = _build_estimator(spec, this_seed, n_estimators)

        imputer = IterativeImputer(
            estimator=estimator,
            max_iter=max_iter,
            random_state=this_seed,
            sample_posterior=False,
        )
        imputed = imputer.fit_transform(imp_data.values)
        imputed = _clamp01(imputed)
        imputed_arrays.append(imputed)

    stacked = np.stack(imputed_arrays, axis=0)  # (M, n_rows, n_cols)
    pooled = np.empty_like(imputed_arrays[0])

    n_moved_by_snapping = 0
    for j, col in enumerate(present_cols):
        col_stack = stacked[:, :, j]  # (M, n_rows)
        if is_categorical[col]:
            pooled[:, j] = _pool_categorical(col_stack, levels[col])
            old = col_stack[0, :]
            moved = int(np.sum(~np.isnan(old) & (np.abs(pooled[:, j] - old) > 1e-12)))
            n_moved_by_snapping += moved
            if moved:
                logger.info("  pooled %s: %d cell(s) changed by snapping", col, moved)
        else:
            pooled[:, j] = np.nanmean(col_stack, axis=0)
    logger.info(
        "Categorical pooling: %d column(s) snapped to observed levels, "
        "%d cell(s) changed by snapping",
        sum(1 for c in present_cols if is_categorical[c]), n_moved_by_snapping,
    )

    pooled = _clamp01(pooled)

    # Per-person dispersion: mean over imputed cells of the between-run SD.
    # Observed cells are excluded (identical across runs). Undefined (NaN) with a
    # single run.
    was_missing = imp_data.isna().to_numpy()
    with np.errstate(invalid="ignore"):
        per_cell_sd = np.nanstd(stacked, axis=0)          # (n_rows, n_cols)
    per_cell_sd = np.where(was_missing, per_cell_sd, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-observed rows are all-NaN
        dispersion = np.nanmean(per_cell_sd, axis=1)
    if n_imputations < 2:
        dispersion = np.full(len(imp_data), np.nan)
    n_imputed_per_row = was_missing.sum(axis=1)

    pooled_df = pd.DataFrame(pooled, columns=present_cols, index=imp_data.index)
    # Dispersion is not written (undefined with one run). FInoPsy_n_imputed records
    # how many cells were imputed per person.
    pooled_df["FInoPsy_n_imputed"] = n_imputed_per_row
    _d = dispersion[~np.isnan(dispersion)]
    n_rows_imputed = int((n_imputed_per_row > 0).sum())
    logger.info(
        "Imputed %d cells across %d/%d rows",
        int(n_imputed_per_row.sum()), n_rows_imputed, len(imp_data),
    )
    if n_imputations < 2:
        logger.info("Dispersion undefined with n_imputations=%d", n_imputations)
    else:
        logger.info(
            "Dispersion: defined for %d/%d rows; between-run SD mean=%.5f max=%.5f",
            len(_d), len(dispersion), _d.mean() if len(_d) else 0.0,
            _d.max() if len(_d) else 0.0,
        )

    # Add back all-NA columns as NaN
    for col in all_na_cols:
        pooled_df[col] = np.nan

    pooled_df = pd.concat([ids.reset_index(drop=True), pooled_df.reset_index(drop=True)], axis=1)

    oob_placeholder = [np.nan] * n_imputations

    return pooled_df, oob_placeholder


# 6. FInoPsy computation


def compute_finopsy(df: pd.DataFrame, frailty_cols: list[str]) -> pd.Series:
    """Compute FInoPsy = row mean of the frailty items, each scored 0-1.

    Parameters
    ----------
    df : DataFrame with imputed frailty columns
    frailty_cols : list of column names to average

    Returns
    -------
    Series of FInoPsy scores
    """
    present = [c for c in frailty_cols if c in df.columns]
    return df[present].mean(axis=1)


# 7. Full pipeline


def _emit_frailty_missing_reasons(
    sav_path: str | Path,
    cohort: str,
    key: pd.DataFrame,
    matched_col: str,
    id_col: str,
    kept_ids: set,
) -> None:
    """Write the missing-reason mask for the frailty items.

    These are the cells the imputer fills, so the mask separates an item a cohort
    was not asked from one a respondent declined. It covers the rows that pass the
    20% missingness filter. Items built from two SPSS variables (``B1PC7/B1PC7A``)
    are masked per source under one ``analysis_name``.
    """
    rename: dict[str, str] = {}
    multi: list[str] = []
    for matched, recode in zip(key[matched_col], key["recode_var"]):
        if pd.isna(matched):
            continue
        sources = [t.strip() for t in str(matched).split("/") if t.strip()]
        if len(sources) > 1:
            multi.append(str(recode))
        for src in sources:
            rename[src] = str(recode)
    if not rename:
        logger.warning("[%s] no frailty source variables resolved; skipping reason mask", cohort)
        return
    out_dir = PROJECT_ROOT / "data" / "processed" / "missing_reasons"
    write_reason_artifacts(
        sav_path=sav_path,
        columns=sorted(rename),
        id_col=id_col,
        out_dir=out_dir,
        stem=f"{cohort}_frailty",
        rename=rename,
        derived_columns=sorted(multi),
        keep_ids=kept_ids,
    )


def run_frailty_pipeline(
    sav_path: str | Path,
    key_path: str | Path,
    cohort: str,
    id_col: str,
    matched_col: str,
    status_col: str | None = None,
    status_val: int | None = None,
    status_negate: bool = False,
    status_negate_val: int | None = None,
    id_prefix: str | None = None,
    n_imputations: int = 1,
    seed: int = 100,
    allow_cpu_fallback: bool = False,
) -> pd.DataFrame:
    """Run the full frailty pipeline for a single cohort.

    Parameters
    ----------
    sav_path : path to the raw SPSS SAV file
    key_path : path to frailty key xlsx
    cohort : cohort name (for logging)
    id_col : name of ID column in the SAV file
    matched_col : name of the column in the frailty key mapping to this cohort
    status_col : column for status filter (e.g. B1STATUS)
    status_val : value to keep (or negate)
    status_negate : if True, filter != status_negate_val instead of == status_val
    status_negate_val : value to exclude when status_negate=True
    id_prefix : prefix to prepend to IDs (unused; M2ID and MRID do not overlap)
    n_imputations : number of imputation runs
    seed : base random seed

    Returns
    -------
    DataFrame with id_col + frailty vars + FInoPsy
    """
    logger.info("Frailty pipeline: %s", cohort)

    logger.info("Loading %s", sav_path)
    raw, meta = pyreadstat.read_sav(str(sav_path))
    logger.info("Raw data: %d rows x %d cols", raw.shape[0], raw.shape[1])

    if status_col is not None:
        if status_negate:
            raw = raw[raw[status_col] != status_negate_val]
        elif status_val is not None:
            raw = raw[raw[status_col] == status_val]
        logger.info("After status filter: %d rows", len(raw))

    if id_prefix:
        raw[id_col] = raw[id_col].apply(lambda x: int(f"{id_prefix}{int(x)}") if pd.notna(x) else x)

    key = load_frailty_key(key_path)

    # Build lookup for this cohort
    lookup = key[[matched_col, "recode_var"]].rename(columns={matched_col: "matched_variable"})

    filtered, missing_pct = _scoring_frailty(raw, id_col, lookup, key)
    logger.info("After scoring + filtering: %d rows", len(filtered))

    # Missing-reason mask, written separately; the imputer input is unchanged.
    _emit_frailty_missing_reasons(
        sav_path, cohort, key, matched_col, id_col, set(filtered[id_col].dropna()),
    )

    # Report high missingness
    high_miss = {k: v for k, v in missing_pct.items() if v > 0.20}
    if high_miss:
        logger.warning("High missingness (>20%%): %s", high_miss)

    frailty_vars = list(key["recode_var"])
    present_vars = [v for v in frailty_vars if v in filtered.columns and not filtered[v].isna().all()]

    pooled, oob = impute_frailty(
        filtered,
        id_col=id_col,
        frailty_cols=present_vars,
        n_imputations=n_imputations,
        seed=seed,
        allow_cpu_fallback=allow_cpu_fallback,
    )

    # Compute FInoPsy
    pooled["FInoPsy"] = compute_finopsy(pooled, present_vars)
    logger.info("FInoPsy computed. Mean=%.4f, SD=%.4f", pooled["FInoPsy"].mean(), pooled["FInoPsy"].std())

    return pooled


def run_wave_pipeline(
    wave: int,
    cohort: str | None = None,
    n_imputations: int = 1,
    seed: int = 100,
    allow_cpu_fallback: bool = False,
) -> None:
    """Run the full frailty pipeline for a wave (1 or 2), all cohorts or one.

    Writes per-cohort CSVs and a pooled CSV to the output directory.
    """
    cfg = WAVE_CONFIG[wave]
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    cohorts_to_run = cfg["cohorts"]
    if cohort:
        if cohort not in cohorts_to_run:
            raise ValueError(f"Cohort '{cohort}' not found in wave {wave}. Available: {list(cohorts_to_run)}")
        cohorts_to_run = {cohort: cohorts_to_run[cohort]}

    results: dict[str, pd.DataFrame] = {}
    for cname, ccfg in cohorts_to_run.items():
        result = run_frailty_pipeline(
            sav_path=ccfg["sav_path"],
            key_path=cfg["key_path"],
            cohort=cname,
            id_col=ccfg["id_col"],
            matched_col=ccfg["matched_col"],
            status_col=ccfg["status_col"],
            status_val=ccfg.get("status_val"),
            status_negate=ccfg.get("status_negate", False),
            status_negate_val=ccfg.get("status_negate_val"),
            id_prefix=ccfg.get("id_prefix"),
            n_imputations=n_imputations,
            seed=seed,
            allow_cpu_fallback=allow_cpu_fallback,
        )

        per_cohort_path = output_dir / f"{cname}_frailty.csv"
        result.to_csv(per_cohort_path, index=False)
        logger.info("Written: %s (n=%d)", per_cohort_path.name, len(result))
        results[cname] = result

    # Pool cohorts if running all
    if len(results) > 1:
        dfs = []
        for cname, df in results.items():
            id_col = cfg["cohorts"][cname]["id_col"]
            dfs.append(df.rename(columns={id_col: "id"}))

        pooled = pd.concat(dfs, ignore_index=True)

        # Recompute FInoPsy from shared present vars
        key = load_frailty_key(cfg["key_path"])
        frailty_vars = list(key["recode_var"])
        all_present = [set(df.columns) for df in dfs]
        shared_vars = set.intersection(*all_present) & set(frailty_vars)
        shared_vars = sorted(shared_vars)

        pooled["FInoPsy"] = compute_finopsy(pooled, shared_vars)
        # FInoPsy_n_imputed comes through the concat.
        if "FInoPsy_n_imputed" not in pooled.columns:
            logger.warning("FInoPsy_n_imputed missing from the pooled frame")

        pooled_path = output_dir / cfg["pooled_csv"]
        pooled.to_csv(pooled_path, index=False)
        logger.info(
            "Written: %s (n=%d, shared frailty vars=%d)",
            pooled_path.name, len(pooled), len(shared_vars),
        )


# CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Frailty scoring and imputation pipeline",
    )
    parser.add_argument("--wave", type=int, required=True, choices=[1, 2], help="Wave (1 or 2)")
    parser.add_argument("--cohort", type=str, default=None, help="Cohort (e.g. midus2, midusr, midus3, mr2)")
    parser.add_argument("--n-imputations", type=int, default=1, help="Number of imputation runs (default: 1)")
    parser.add_argument("--seed", type=int, default=100, help="Base random seed (default: 100)")
    parser.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="Run on CPU when no GPU is visible (results differ from GPU runs).",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    run_wave_pipeline(
        wave=args.wave,
        cohort=args.cohort,
        n_imputations=args.n_imputations,
        seed=args.seed,
        allow_cpu_fallback=args.allow_cpu_fallback,
    )
