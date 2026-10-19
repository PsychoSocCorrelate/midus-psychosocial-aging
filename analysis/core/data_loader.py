"""
Master dataset loader for XGBoost analyses.

Reads data/processed/master_analysis.csv and config/variable_config.yaml,
returning (X, y, feature_names) with NaN preserved (XGBoost native handling).

Cross-sectional usage:
    X, y, features = load_master("FInoPsy_W1", tiers=[1, 2])

Longitudinal usage (A4 spec: features = T1 + T2 + Y_w1, target = Y_w2):
    X, y, features = load_master("FInoPsy_W2", tiers=[1, 2], longitudinal=True)
    # Adds FInoPsy_W1 as a feature column
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from config.biomarker_constants import AL_COMPOSITE_COLUMNS
import pandas as pd

from analysis.core import age_method
from config.variable_config import get_tier_variables

ROOT = Path(__file__).resolve().parents[2]

MASTER_CSV = ROOT / "data/processed/master_analysis.csv"
CONCURRENT_CSV = ROOT / "data/processed/master_analysis_concurrent.csv"

# Categorical columns, passed to XGBoost as category dtype
CATEGORICALS = {
    "race", "sexualOrientation", "familyHeartAttack", "familyCancer",
    "relatStatus", "careGivStatus", "smokeStatus", "drinkStatus",
    "workStatus", "occGroup", "workIndustry", "spouseWork",
}

# Metadata / identifier columns never used as predictors
NON_PREDICTOR_COLS = {
    "person_id", "cohort", "wave_baseline",
    "age_W1", "age_W2",
    "sample_weight",
    "has_dnam", "has_biomarker_W1", "has_biomarker_W2",
    "has_frailty_W1", "has_frailty_W2", "has_cog_W1", "has_cog_W2",
    "family_id", "sib_type", "pair_id", "is_full_pair", "within_pair_age_diff",
}

# Allostatic load, both waves. It is exempt from sibling exclusion: the biomarker
# subsample is small enough that dropping twins costs more power than the
# dependence it removes. Outputs label AL sibling-only denominators separately
# from frailty/cognition.
AL_OUTCOME_COLS = {n for pair in AL_COMPOSITE_COLUMNS.values() for n in pair}

SIBLING_EXCLUSION_EXEMPT = set(AL_OUTCOME_COLS)

# W1 outcome to W2 outcome (for longitudinal mode)
W1_TO_W2 = {
    "FInoPsy_W1": "FInoPsy_W2",
    "btactComp_W1": "btactComp_W2",
    "ALcomp_W1": "ALcomp_W2",
    "episoMem_W1": "episoMem_W2",
    "efFactor_W1": "efFactor_W2",
}
W2_TO_W1 = {v: k for k, v in W1_TO_W2.items()}


def deduplicate_longitudinal_outcomes(names: list[str]) -> list[str]:
    """Normalize outcome names to W2 and deduplicate.

    Longitudinal scripts target W2 outcomes; passing both 'FInoPsy_W1' and
    'FInoPsy_W2' resolves to the same model. This function normalizes all names to their W2
    form and removes duplicates while preserving order.
    """
    seen: set[str] = set()
    result: list[str] = []
    for name in names:
        canonical = W1_TO_W2.get(name, name)  # map W1 to W2; leave W2/other as-is
        if canonical not in seen:
            seen.add(canonical)
            result.append(canonical)
    return result

# All outcome column names (excluded from predictor set)
ALL_OUTCOME_COLS = {
    "phenoAge_res", "grimAgeN_res", "dunedinPace_res",
    *AL_OUTCOME_COLS,
    "FInoPsy_W1", "FInoPsy_W2",
    "btactComp_W1", "btactComp_W2",
    "episoMem_W1", "episoMem_W2",
    "efFactor_W1", "efFactor_W2",
}


def load_master(
    outcome: str,
    tiers: list[int],
    *,
    longitudinal: bool = False,
    concurrent: bool = False,
    exclude_siblings: bool = True,
    exclude_high_missing: bool = True,
    age_residualize: bool = True,
    data_path: Path | str | None = None,
    cohort_filter: str | None = None,
    force_exclude_siblings: bool = False,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load master_analysis.csv and return (X, y, feature_names).

    Cross-sectional mode (longitudinal=False):
        - Features: Tier columns requested
        - Target: `outcome` column (W1 outcomes only)
        - Rows: persons with non-NA outcome

    Longitudinal mode (longitudinal=True):
        - Target: W2 outcome (e.g. FInoPsy_W2)
        - Features: Tier columns + W1 outcome (Y_w1 as autoregressive feature)
        - Rows: persons with non-NA W1 AND W2 outcomes
        - `outcome` argument should be the W2 name (e.g. "FInoPsy_W2") OR
          the W1 name (e.g. "FInoPsy_W1"); either works.

    Args:
        outcome: Column name for the target outcome.
        tiers: Tier numbers to include (e.g. [1, 2]).
        longitudinal: If True, add Y_w1 as feature and use Y_w2 as target.
        exclude_siblings: If True (default), exclude complete-pair sibling/twin
            rows from training data. Sibling rows are held out for independent
            validation via load_sibling_holdout(). Disabled for AL
            outcomes because the biomarker longitudinal sample is constrained;
            AL sibling-only results must be labeled with AL-specific denominators.
        exclude_high_missing: Drop variables with >50% missingness.
        age_residualize: Regress out age_W1 from the outcome before returning.
        data_path: Override path to master CSV (default: data/processed/master_analysis.csv).
        cohort_filter: If set ("MIDUS2" or "MIDUSR"), restrict to that cohort only.

    Returns:
        X: DataFrame of features (NaN preserved, categoricals as category dtype)
        y: Series of outcome values (age-residualized if requested)
        feature_names: List of column names in X
    """
    # Concurrent mode: use W2 predictor values from concurrent CSV.
    # Caller controls longitudinal flag independently.
    if concurrent:
        if not data_path:
            data_path = CONCURRENT_CSV

    csv_path = Path(data_path) if data_path else MASTER_CSV
    df = pd.read_csv(csv_path, low_memory=False)

    if cohort_filter:
        df = df[df["cohort"] == cohort_filter].reset_index(drop=True)

    # Sibling exclusion: remove paired individuals from training set.
    # Exempted for AL outcomes, whose biomarker sample is small; al_twins_heldout
    # refits them with the pairs held out.
    # Sibling rows are available via load_sibling_holdout() for independent validation.
    _outcome_for_exempt_check = outcome
    _is_exempt = _outcome_for_exempt_check in SIBLING_EXCLUSION_EXEMPT
    # Sensitivity: apply sibling exclusion to AL outcomes as well, to re-estimate
    # AL increments and feature selection on a sibling-independent sample.
    if force_exclude_siblings:
        _is_exempt = False
    if exclude_siblings and not _is_exempt and "pair_id" in df.columns:
        n_before = len(df)
        # Only exclude COMPLETE pairs (is_full_pair=True). Orphan sibs (pair_id notna
        # but co-sib absent) are retained because they do not inflate within-family
        # estimates and still contribute valid training signal.
        if "is_full_pair" in df.columns:
            df = df[~(df["is_full_pair"] == True)].reset_index(drop=True)
        else:
            df = df[df["pair_id"].isna()].reset_index(drop=True)
        n_removed = n_before - len(df)
        pct = n_removed / n_before * 100 if n_before > 0 else 0
        print(f"  [sibling exclusion] removed {n_removed:,} complete-pair rows "
              f"({n_removed}/{n_before} = "
              f"{pct:.1f}%); training N={len(df):,}")
    elif _is_exempt and exclude_siblings:
        print(f"  [sibling exclusion] EXEMPT for {_outcome_for_exempt_check} "
              f"(AL biomarker sample-size constraint; full N={len(df):,} retained; "
              "label AL denominators)")

    # Normalise outcome name: accept W1 or W2 for longitudinal
    outcome_target = outcome
    if longitudinal:
        if outcome in W1_TO_W2:
            outcome_target = W1_TO_W2[outcome]
        elif outcome in W2_TO_W1:
            outcome_target = outcome
        else:
            raise ValueError(
                f"Longitudinal mode requires a frailty/cog/AL outcome. "
                f"'{outcome}' has no W2 counterpart in W1_TO_W2 mapping."
            )
        outcome_w1 = W2_TO_W1[outcome_target]
    else:
        outcome_target = outcome
        outcome_w1 = None

    if outcome_target not in df.columns:
        raise ValueError(f"Outcome column '{outcome_target}' not found in master CSV.")

    predictor_vars = get_tier_variables(tiers, exclude_high_missing=exclude_high_missing)

    # Keep only variables that actually exist in the CSV
    available = set(df.columns)
    missing_vars = [v for v in predictor_vars if v not in available]
    if missing_vars:
        print(f"  [WARN] {len(missing_vars)} predictor vars not in CSV (skipped): {missing_vars[:5]}...")
    predictor_vars = [v for v in predictor_vars if v in available]

    # For longitudinal: add Y_w1 as feature
    feature_cols = list(predictor_vars)
    if longitudinal and outcome_w1 and outcome_w1 in df.columns:
        if outcome_w1 not in feature_cols:
            feature_cols.append(outcome_w1)

    # Filter rows: require non-NA outcome
    if longitudinal and outcome_w1:
        mask = df[outcome_target].notna() & df[outcome_w1].notna()
    else:
        mask = df[outcome_target].notna()
    df_sub = df[mask].reset_index(drop=True)

    # W2 rows have no age_W1; recover it from the W1 row by M2MRID, else fall
    # back to age_W2 (MIDUS Refresher has no W1).
    if longitudinal and "age_W1" in df_sub.columns:
        null_mask = df_sub["age_W1"].isna()
        if null_mask.any():
            w1_age_map = df[["M2MRID", "age_W1"]].dropna(subset=["age_W1"]).drop_duplicates("M2MRID")
            w1_age_map = w1_age_map.set_index("M2MRID")["age_W1"].to_dict()
            backfill = df_sub.loc[null_mask, "M2MRID"].map(w1_age_map)
            df_sub.loc[null_mask, "age_W1"] = backfill.values
            still_null = df_sub["age_W1"].isna().sum()
            if still_null > 0:
                df_sub.loc[df_sub["age_W1"].isna(), "age_W1"] = \
                    df_sub.loc[df_sub["age_W1"].isna(), "age_W2"]
            print(f"  Backfilled age_W1 for {null_mask.sum()} w2 rows "
                  f"({still_null} still null, used age_W2)")

    X = df_sub[feature_cols].copy()

    # Encode categorical columns as pd.Categorical for native XGBoost support.
    # XGBoost 1.6+ with enable_categorical=True learns optimal category
    # partitions (not ordinal splits). NaN preserved as missing.
    for col in X.columns:
        if X[col].dtype == object or col in CATEGORICALS:
            # XGBoost 3.2 rejects float, nullable-int, and empty-category columns.
            # Cast to string, remove unused categories (e.g. after subsetting).
            X[col] = X[col].astype(str).replace("nan", pd.NA).replace("<NA>", pd.NA)
            X[col] = pd.Categorical(X[col]).remove_unused_categories()

    # Drop all-NaN categorical columns (XGBoost rejects zero categories)
    empty_cat_cols = [c for c in X.columns
                      if hasattr(X[c], 'cat') and len(X[c].cat.categories) == 0]
    if empty_cat_cols:
        print(f"  [data_loader] dropping {len(empty_cat_cols)} all-NaN categorical(s): {empty_cat_cols}")
        X = X.drop(columns=empty_cat_cols)
        feature_cols = [c for c in feature_cols if c not in empty_cat_cols]

    y = df_sub[outcome_target].copy().astype(float)

    # Longitudinal: residualize the Y_W1 feature on age_W1 so it is on the same
    # age-adjusted scale as the target; otherwise Y_W1 carries age variance into
    # the psychosocial SHAP attributions.
    if longitudinal and outcome_w1 and outcome_w1 in X.columns:
        if "age_W1" in df_sub.columns:
            from sklearn.linear_model import LinearRegression as _LR
            _age = df_sub["age_W1"].values.astype(float)
            _yw1 = X[outcome_w1].values.astype(float)
            _valid = ~np.isnan(_age) & ~np.isnan(_yw1)
            if _valid.sum() > 10:
                _reg = _LR()
                _reg.fit(_age[_valid].reshape(-1, 1), _yw1[_valid])
                _yw1_resid = _yw1 - _reg.predict(_age.reshape(-1, 1)).ravel()
                _yw1_resid[~_valid] = np.nan
                X[outcome_w1] = _yw1_resid
                print(f"  age-residualized {outcome_w1} feature "
                      f"(r_before={np.corrcoef(_age[_valid], _yw1[_valid])[0,1]:.3f})")

    # Residualize on contemporaneous age: age_W2 for W2 outcomes, age_W1 otherwise.
    if longitudinal or outcome_target.endswith("_W2"):
        _age_col = "age_W2"
    else:
        _age_col = "age_W1"

    # Global residualization (fit on the full sample) leaks slightly, negligibly
    # for large n. For within-fold
    # residualization, pass age_residualize=False and use
    # XGBoostPipeline.run_cv(X, y, features, age=age_arr) instead.
    _applied_global = False
    if age_residualize and _age_col in df_sub.columns:
        age = df_sub[_age_col].values.astype(float)
        valid = ~np.isnan(age) & ~np.isnan(y.values)
        if valid.sum() > 10:
            from sklearn.linear_model import LinearRegression
            reg = LinearRegression()
            reg.fit(age[valid].reshape(-1, 1), y.values[valid])
            y_resid = np.full_like(y.values, np.nan)
            y_resid[valid] = y.values[valid] - reg.predict(age[valid].reshape(-1, 1)).ravel()
            y = pd.Series(y_resid, index=y.index, name=outcome_target)
            _applied_global = True
            print(f"  outcome residualized with {_age_col}")

    # Extract raw age for within-fold residualization in XGBoostPipeline.
    # For longitudinal: use age_W2 (contemporaneous with target).
    age_arr: np.ndarray | None = None
    if _age_col in df_sub.columns:
        age_arr = df_sub[_age_col].values.astype(float)

    # Record the age-adjustment method; run_cv completes it if it residualizes
    # within folds.
    age_method.record_load(
        outcome=outcome_target,
        flag_passed=age_residualize,
        age_column=_age_col if _age_col in df_sub.columns else None,
        applied_global=_applied_global,
        longitudinal=longitudinal,
        concurrent=concurrent,
        cohort_filter=cohort_filter,
        n_rows=len(df_sub),
    )

    n_total = len(df_sub)
    n_outcome = int(y.notna().sum())
    print(
        f"  load_master({outcome_target!r}, tiers={tiers}, longitudinal={longitudinal}): "
        f"n={n_total:,}, outcome_nonNA={n_outcome:,}, features={len(feature_cols)}"
    )

    return X, y, feature_cols, age_arr


def load_sibling_holdout(
    outcome: str,
    tiers: list[int],
    *,
    longitudinal: bool = False,
    exclude_high_missing: bool = True,
    age_residualize: bool = True,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Return complete sibling/twin pairs only, as a holdout for a model trained on non-siblings.

    Use after training a model with load_master(..., exclude_siblings=True).
    Apply trained model to this holdout to assess generalizability.

    Not available for AL outcomes because sibling exclusion is intentionally not
    applied to AL training; report AL sibling-only denominators explicitly in
    A6/SI outputs instead.

    Returns same (X, y, feature_names, age) tuple as load_master().
    """
    if outcome in SIBLING_EXCLUSION_EXEMPT:
        raise ValueError(
            f"load_sibling_holdout() not available for {outcome}: "
            f"sibling exclusion was not applied to AL training data; "
            f"use AL-specific sibling denominators in A6/SI outputs."
        )

    csv_path = MASTER_CSV
    df = pd.read_csv(csv_path, low_memory=False)

    # Keep only complete-pair sibling rows (both members present in data).
    # Orphan sibs are retained in training via load_master; only complete pairs
    # form a valid holdout (both members excluded from training, so independent).
    if "pair_id" not in df.columns:
        raise ValueError("pair_id column not found in master CSV.")
    n_before = len(df)
    if "is_full_pair" in df.columns:
        df = df[df["is_full_pair"] == True].reset_index(drop=True)
    else:
        df = df[df["pair_id"].notna()].reset_index(drop=True)
    print(f"  [sibling holdout] {len(df):,} complete-pair rows (from {n_before:,} total)")

    # load_master reads from a path, so pass the filtered rows through a temporary CSV.
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        tmp_path = f.name
    try:
        df.to_csv(tmp_path, index=False)
        X, y, features, age = load_master(
            outcome, tiers,
            longitudinal=longitudinal,
            exclude_siblings=False,   # already filtered to siblings only
            exclude_high_missing=exclude_high_missing,
            age_residualize=age_residualize,
            data_path=tmp_path,
        )
    finally:
        os.unlink(tmp_path)

    return X, y, features, age


if __name__ == "__main__":
    print("data_loader sanity check")
    X, y, features, age = load_master("FInoPsy_W1", tiers=[1, 2], age_residualize=False)
    print(f"  X shape: {X.shape}")
    print(f"  y non-NA: {y.notna().sum()}")
    print(f"  age non-NA: {np.sum(~np.isnan(age)):,}")
    print()
    X2, y2, features2, age2 = load_master("FInoPsy_W2", tiers=[1, 2], longitudinal=True, age_residualize=False)
    print(f"  Longitudinal X shape: {X2.shape}")
    print(f"  Longitudinal y non-NA: {y2.notna().sum()}")
    print(f"  'FInoPsy_W1' in features: {'FInoPsy_W1' in features2}")
