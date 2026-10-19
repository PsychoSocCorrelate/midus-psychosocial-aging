#!/usr/bin/env python3
"""
A6: Sibling/Twin Between/Within Decomposition

Two parallel Mundlak between-within analyses on twin/sibling pairs:

  1. Cross-sectional at W1 (concurrent state)
       Y_w1_ij = β_B · X̄_w1_j + β_W · (X_w1_ij − X̄_w1_j)
               + age_w1_between + age_w1_within + sex
               + (1|pair_id) + ε_ij
     Focal predictors: top-20 ∩ stable (top20_stability ≥ 0.60) from
     CS-1 bootstrap SHAP output.

  2. Longitudinal, W1 to W2 (change)
       Y_w2_ij = β_B · X̄_j       + β_W · (X_ij − X̄_j)
               + α_B · Ȳ_w1_j     + α_W · (Y_w1_ij − Ȳ_w1_j)
               + age_w2_between   + age_w2_within + sex
               + (1|pair_id) + ε_ij
     Focal predictors: stable vars from A4 longitudinal SHAP (excluding
     the autoregressive W1 outcome).

Also regenerates frailty_item_icc.csv (MZ-pair item-level ICC diagnostic).

Usage:
    uv run python analysis/pipelines/a6_sibling_decomposition.py
    uv run python analysis/pipelines/a6_sibling_decomposition.py --top_k 10 --n_boot 500
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.core.output_log import write_output_log
from analysis.core import age_method
from analysis.core import provenance
from analysis.core import artifact_paths as ap
A4_DIR = ROOT / "output/a4_longitudinal"
OUTPUT_DIR = ROOT / "output/a6_sibling"
MASTER_CSV = ROOT / "data/processed/master_analysis.csv"
FRAILTY_W1_CSV = ROOT / "data/processed/wave1/pooled_frailty_M2R.csv"
FRAILTY_W2_CSV = ROOT / "data/processed/wave2/pooled_frailty_M3MR2.csv"

OUTCOMES = {
    "FInoPsy": ("FInoPsy_W1", "FInoPsy_W2"),
    "btactComp": ("btactComp_W1", "btactComp_W2"),
    # Focal predictors come from cs1 and a4 SHAP, so both must exist for an outcome.
    "ALcomp": ("ALcomp_W1", "ALcomp_W2"),
}

# All sib types considered for All_paired
ALL_SIB_TYPES = ["MZ", "DZ_SS", "DZ_OS", "sibling"]

# Minimum discordant pairs for the within-pair slope to be estimable (below 2 it
# has no residual df). This is an estimability floor rather than a power
# threshold; per-predictor power is in analysis/pipelines/within_family_mde.py,
# and a6_discordant_pairs.csv records the count for every dummy.
MIN_DISCORDANT_PAIRS = 2


# Categorical variable configuration
# To add a new categorical: add an entry to CATEGORICAL_CONFIG with:
#   "groups": dict mapping raw values to group labels
#   "reference": the group label to use as reference (omitted from dummies)
#   "description": report description
#
# Raw values are matched as strings (after str() conversion) to handle
# both float-coded (1.0, 2.0) and string-coded ("Working now") variables.

CATEGORICAL_CONFIG = {
    "workStatus": {
        "groups": {
            "Working now": "employed",
            "Self-employed": "employed",
            "Retired": "retired",
            "Homemaker": "not_working",
            "Disabled": "not_working",
            "Laid-off/Seeking-work": "not_working",
            "Other work": "not_working",
            "Student": "not_working",
        },
        "reference": "employed",
        "description": "Work status: retired vs employed, not-working vs employed",
    },
    "occGroup": {
        "groups": {
            "1.0": "professional_mgmt",   # Management
            "2.0": "professional_mgmt",   # Professional
            "3.0": "service_sales_office", # Service
            "4.0": "service_sales_office", # Sales
            "5.0": "service_sales_office", # Office/Admin
            "6.0": "blue_collar",          # Farming/Fishing/Forestry
            "7.0": "blue_collar",          # Construction/Extraction
            "8.0": "blue_collar",          # Installation/Maintenance
            "9.0": "blue_collar",          # Production
            "10.0": "blue_collar",         # Transportation
            "11.0": "blue_collar",         # Armed Forces
            # 96-99 are NA-like: not in labor force, unknown, missing, INAPP
        },
        "reference": "service_sales_office",
        "description": (
            "Occupational group: professional/mgmt (1-2) vs "
            "service/sales/office (3-5, ref) vs blue-collar (6-11)"
        ),
    },
    "workIndustry": {
        "groups": {
            "1.0": "goods_production",     # Agriculture/Mining
            "2.0": "goods_production",     # Construction
            "3.0": "goods_production",     # Manufacturing
            "4.0": "trade_transport",      # Wholesale
            "5.0": "trade_transport",      # Retail
            "6.0": "trade_transport",      # Transportation/Warehousing/Utilities
            "7.0": "knowledge_govt",       # Information
            "8.0": "knowledge_govt",       # Finance/Insurance/Real Estate
            "9.0": "knowledge_govt",       # Professional/Scientific/Management
            "10.0": "knowledge_govt",      # Education/Health Care/Social Assistance
            "11.0": "service_arts",        # Arts/Entertainment/Food
            "12.0": "service_arts",        # Other Services
            "13.0": "knowledge_govt",      # Public Administration
            "14.0": "knowledge_govt",      # Military
        },
        "reference": "service_arts",
        "description": (
            "Industry: goods production (1-3) vs trade/transport (4-6) vs "
            "knowledge/govt (7-10,13-14) vs service/arts (11-12, ref)"
        ),
    },
}


# Categorical expansion and discordant pair counting

def expand_categorical(
    df: pd.DataFrame,
    var: str,
    config: dict,
) -> list[str]:
    """Create k-1 dummy columns from a categorical variable using config.

    Returns list of dummy column names (excluding reference category).
    The dummies are named "{var}__{group}" (double underscore separator).
    """
    mapping = config["groups"]
    reference = config["reference"]

    # Map raw values to groups
    grouped = df[var].astype(str).map(mapping)
    # Values not in mapping become NaN (e.g. 96.0, 97.0, nan)

    # Get all non-reference groups
    all_groups = sorted(set(mapping.values()) - {reference})

    dummy_cols = []
    for group in all_groups:
        col_name = f"{var}__{group}"
        df[col_name] = np.where(
            grouped.isna(), np.nan,
            (grouped == group).astype(float),
        )
        dummy_cols.append(col_name)

    return dummy_cols


def count_discordant_pairs(
    data: pd.DataFrame,
    dummy_col: str,
) -> int:
    """Count pairs where siblings differ on a binary dummy variable.

    A pair is discordant if one sibling has dummy=1 and the other has dummy=0
    (both must be non-NaN).
    """
    pair_stats = data.groupby("pair_id")[dummy_col].agg(["sum", "count"])
    # Both valid (count==2) and exactly one has the trait (sum==1)
    discordant = ((pair_stats["count"] == 2) & (pair_stats["sum"] == 1)).sum()
    return int(discordant)


def expand_and_filter_categoricals(
    focal_vars: list[str],
    paired_data: pd.DataFrame,
) -> tuple[list[str], list[dict]]:
    """Replace categorical focal vars with pre-expanded dummies, filter by discordant pairs.

    Assumes expand_categorical() was already called on the full DataFrame
    before prepare_pair_data() created paired_data (so dummy columns exist).

    Args:
        focal_vars: Original focal variable names from A4.
        paired_data: Paired subset (dummy columns already present).

    Returns:
        expanded_focals: New focal list with categoricals replaced by dummies.
        discordant_report: List of dicts for reporting.
    """
    expanded = []
    report = []

    for var in focal_vars:
        if var in CATEGORICAL_CONFIG:
            config = CATEGORICAL_CONFIG[var]
            reference = config["reference"]
            all_groups = sorted(set(config["groups"].values()) - {reference})
            dummy_cols = [f"{var}__{group}" for group in all_groups]

            for dcol in dummy_cols:
                n_disc = count_discordant_pairs(paired_data, dcol)
                group_label = dcol.split("__", 1)[1]
                passes = n_disc >= MIN_DISCORDANT_PAIRS

                report.append({
                    "original_var": var,
                    "dummy": dcol,
                    "group": group_label,
                    "reference": config["reference"],
                    "discordant_pairs": n_disc,
                    "included": passes,
                })

                if passes:
                    expanded.append(dcol)
                else:
                    print(f"    [SKIP] {dcol}: only {n_disc} discordant pairs "
                          f"(< {MIN_DISCORDANT_PAIRS})")
        else:
            expanded.append(var)

    return expanded, report


# Data preparation

def parse_args():
    p = argparse.ArgumentParser(description="A6 sibling/twin decomposition")
    p.add_argument("--top_k", type=int, default=15)
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--outcomes", nargs="+", default=["all"])
    return p.parse_args()


def load_focal_variables(outcome_w2: str, top_k: int) -> list[str]:
    """Load stable top-20 variable names from A4 bootstrap SHAP output.

    Only includes variables marked as stable (top-20 stability >= 60%),
    excluding the autoregressive W1 outcome feature.
    """
    safe = outcome_w2.replace("/", "_")
    path = ap.A4_DIR / "shap" / f"{safe}_long_bootstrap_shap.csv"
    if not path.exists():
        raise FileNotFoundError(f"A4 SHAP not found: {path}. Run A4 first.")
    df = pd.read_csv(path, index_col=0)
    w1_col = outcome_w2.replace("_W2", "_W1")
    # Filter to stable variables only, excluding autoregressive feature
    stable_df = df[(df["stable"] == True) & (df.index != w1_col)]
    return list(stable_df.index)


def load_focal_variables_cs(outcome_w1: str, top_k: int = 20) -> list[str]:
    """Load top-20-∩-stable variable names from CS-1 bootstrap SHAP output.

    Intersection rule: variables must be both in the top-`top_k` by
    mean|SHAP| AND have top20_stability ≥ 0.60. Source files live in
    output/cs1_cross_sectional/shap/ and are keyed by the W1 outcome
    name (e.g. FInoPsy_W1_T1T2_bootstrap_shap.csv).
    """
    safe = outcome_w1.replace("/", "_")
    path = ap.CS1_DIR / "shap" / f"{safe}_T1T2_bootstrap_shap.csv"
    if not path.exists():
        raise FileNotFoundError(f"CS-1 SHAP not found: {path}. Run CS-1 first.")
    df = pd.read_csv(path, index_col=0)
    # Rank by mean|SHAP| and take top-k
    ranked = df.sort_values("mean_abs_shap", ascending=False)
    top = ranked.head(top_k)
    # Intersect with stable
    selected = top[top["stable"] == True]
    return list(selected.index)


def _decompose_between_within(
    sub: pd.DataFrame,
    var: str,
) -> bool:
    """Add {var}_between and {var}_within columns.

    For numeric variables (including 0/1 dummies).
    Returns False if variable doesn't exist or is non-numeric.
    """
    if var not in sub.columns:
        sub[f"{var}_between"] = np.nan
        sub[f"{var}_within"] = np.nan
        return False

    # Ensure numeric
    col = sub[var]
    if col.dtype.name == 'category':
        col = col.cat.codes.replace(-1, np.nan).astype(float)
        sub[var] = col
    elif col.dtype == object:
        col = pd.to_numeric(col, errors='coerce')
        sub[var] = col

    # Only compute pair mean when both members are non-NaN
    both_valid = sub.groupby("pair_id")[var].transform(
        lambda x: x.notna().all()
    )
    pair_mean = sub.groupby("pair_id")[var].transform("mean")

    sub[f"{var}_between"] = np.where(both_valid, pair_mean, np.nan)
    sub[f"{var}_within"] = np.where(both_valid, sub[var] - pair_mean, np.nan)
    return True


def prepare_pair_data_cs(
    df: pd.DataFrame,
    outcome_w1: str,
    focal_vars: list[str],
    sib_types: list[str],
) -> pd.DataFrame:
    """Prepare paired data for the cross-sectional (W1-only) model.

    Decomposes each focal var and age_W1 into between/within. No Y_w1
    lag is included (outcome is Y_w1 itself).
    """
    mask = (
        df["sib_type"].isin(sib_types)
        & df["pair_id"].notna()
        & df[outcome_w1].notna()
    )
    sub = df[mask].copy()

    pair_counts = sub.groupby("pair_id").size()
    full_pairs = pair_counts[pair_counts >= 2].index
    sub = sub[sub["pair_id"].isin(full_pairs)].copy()

    if len(sub) < 10:
        return pd.DataFrame()

    all_decompose = [v for v in focal_vars if v not in CATEGORICAL_CONFIG]
    if "age_W1" in sub.columns:
        all_decompose.append("age_W1")

    for var in all_decompose:
        _decompose_between_within(sub, var)

    return sub


def prepare_pair_data(
    df: pd.DataFrame,
    outcome_w1: str,
    outcome_w2: str,
    focal_vars: list[str],
    sib_types: list[str],
) -> pd.DataFrame:
    """Prepare paired data with full Mundlak decomposition.

    Decomposes: each focal var, Y_w1, and age_W2 into between/within.
    """
    mask = (
        df["sib_type"].isin(sib_types)
        & df["pair_id"].notna()
        & df[outcome_w1].notna()
        & df[outcome_w2].notna()
    )
    sub = df[mask].copy()

    # Keep only complete pairs
    pair_counts = sub.groupby("pair_id").size()
    full_pairs = pair_counts[pair_counts >= 2].index
    sub = sub[sub["pair_id"].isin(full_pairs)].copy()

    if len(sub) < 10:
        return pd.DataFrame()

    # Decompose ALL Level-1 variables: focal X, Y_w1, age_W2
    # Skip raw categoricals; their dummies are decomposed later
    all_decompose = [v for v in focal_vars if v not in CATEGORICAL_CONFIG]
    all_decompose.append(outcome_w1)
    if "age_W2" in sub.columns:
        all_decompose.append("age_W2")
    elif "age_W1" in sub.columns:
        # Fallback if age_W2 not available
        all_decompose.append("age_W1")

    for var in all_decompose:
        _decompose_between_within(sub, var)

    return sub


# Model fitting

def _build_formula_and_cols(
    outcome_w2: str,
    outcome_w1: str,
    focal_var: str,
    data_columns: set[str],
) -> tuple[str, list[str], list[str]]:
    """Build formula with full Mundlak decomposition."""
    covariates = [
        f"{focal_var}_between",
        f"{focal_var}_within",
        f"{outcome_w1}_between",
        f"{outcome_w1}_within",
    ]

    # Age_W2 decomposed (preferred), fallback to age_W1
    if "age_W2_between" in data_columns:
        covariates.append("age_W2_between")
    if "age_W2_within" in data_columns:
        covariates.append("age_W2_within")

    if "age_W2_between" not in data_columns and "age_W1_between" in data_columns:
        covariates.append("age_W1_between")
    if "age_W2_within" not in data_columns and "age_W1_within" in data_columns:
        covariates.append("age_W1_within")

    # Sex (Level-2 for MZ/DZ_SS, no decomposition)
    if "sex" in data_columns:
        covariates.append("sex")

    formula = f"{outcome_w2} ~ " + " + ".join(covariates)
    model_cols = [outcome_w2] + covariates + ["pair_id"]
    return formula, model_cols, covariates


def _fit_single_model(
    data: pd.DataFrame,
    formula: str,
    model_cols: list[str],
    focal_var: str,
) -> dict | None:
    """Fit one MixedLM. Returns dict with β_B, β_W, variance components."""
    import statsmodels.formula.api as smf

    between_col = f"{focal_var}_between"
    within_col = f"{focal_var}_within"

    sub = data[model_cols].dropna()

    # Remove singletons created by dropna
    pair_counts = sub["pair_id"].value_counts()
    complete_pairs = pair_counts[pair_counts == 2].index
    sub = sub[sub["pair_id"].isin(complete_pairs)]

    if sub["pair_id"].nunique() < 10:
        return None

    # Check for zero variance
    for c in [between_col, within_col]:
        if c in sub.columns and sub[c].std() < 1e-10:
            return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = smf.mixedlm(formula, sub, groups=sub["pair_id"])
            result = model.fit(reml=True, maxiter=200)
    except Exception as e:
        return None

    # Extract variance components
    sigma2_u = (float(result.cov_re.iloc[0, 0])
                if hasattr(result.cov_re, 'iloc') else float(result.cov_re))
    sigma2_e = float(result.scale)
    icc = sigma2_u / (sigma2_u + sigma2_e) if (sigma2_u + sigma2_e) > 0 else np.nan

    out = {
        "focal_var": focal_var,
        "n_persons": len(sub),
        "n_pairs": sub["pair_id"].nunique(),
        "beta_B": result.params.get(between_col, np.nan),
        "beta_B_se": result.bse.get(between_col, np.nan),
        "beta_B_p": result.pvalues.get(between_col, np.nan),
        "beta_W": result.params.get(within_col, np.nan),
        "beta_W_se": result.bse.get(within_col, np.nan),
        "beta_W_p": result.pvalues.get(within_col, np.nan),
        "sigma2_u": sigma2_u,
        "sigma2_e": sigma2_e,
        "icc": icc,
    }

    # Attenuation ratio (only if β_B is meaningfully nonzero)
    if abs(out["beta_B"]) > out["beta_B_se"] * 0.5:
        out["attenuation_ratio"] = out["beta_W"] / out["beta_B"]
    else:
        out["attenuation_ratio"] = np.nan

    return out


def fit_between_within(
    data: pd.DataFrame,
    outcome_w2: str,
    outcome_w1: str,
    focal_var: str,
) -> dict | None:
    """Fit full Mundlak MixedLM for one focal variable. Returns raw β dict."""
    formula, model_cols, covariates = _build_formula_and_cols(
        outcome_w2, outcome_w1, focal_var, set(data.columns),
    )
    return _fit_single_model(data, formula, model_cols, focal_var)


# Post-hoc standardization

def compute_level_sds(
    data: pd.DataFrame,
    focal_var: str,
    outcome_w2: str,
) -> dict:
    """Compute level-specific SDs for Snijders & Bosker standardization."""
    between_col = f"{focal_var}_between"
    within_col = f"{focal_var}_within"

    y_pair_mean = data.groupby("pair_id")[outcome_w2].transform("mean")
    y_within = data[outcome_w2] - y_pair_mean

    valid = data[[between_col, within_col, outcome_w2]].notna().all(axis=1)
    sub = data[valid]

    return {
        "sd_x_within": sub[within_col].std() if len(sub) > 2 else np.nan,
        "sd_x_between": sub[between_col].std() if len(sub) > 2 else np.nan,
        "sd_y_within": y_within[valid].std() if valid.sum() > 2 else np.nan,
        "sd_y_between": y_pair_mean[valid].std() if valid.sum() > 2 else np.nan,
    }


def add_standardized_effects(result: dict, sds: dict) -> dict:
    """Add level-specific standardized β to result dict."""
    sd_xw, sd_xb = sds["sd_x_within"], sds["sd_x_between"]
    sd_yw, sd_yb = sds["sd_y_within"], sds["sd_y_between"]

    if pd.notna(sd_xw) and pd.notna(sd_yw) and sd_yw > 1e-10:
        result["beta_W_std"] = result["beta_W"] * sd_xw / sd_yw
    else:
        result["beta_W_std"] = np.nan

    if pd.notna(sd_xb) and pd.notna(sd_yb) and sd_yb > 1e-10:
        result["beta_B_std"] = result["beta_B"] * sd_xb / sd_yb
    else:
        result["beta_B_std"] = np.nan

    result["sd_x_within"] = sd_xw
    result["sd_x_between"] = sd_xb
    return result


# CS-mode model fitting / bootstrap / standardization (W1 only)

def _build_formula_and_cols_cs(
    outcome_w1: str,
    focal_var: str,
    data_columns: set[str],
) -> tuple[str, list[str], list[str]]:
    """Build formula for the cross-sectional (W1) Mundlak model.

    No lagged outcome. Age is age_W1 (decomposed). Sex kept as-is.
    """
    covariates = [
        f"{focal_var}_between",
        f"{focal_var}_within",
    ]
    if "age_W1_between" in data_columns:
        covariates.append("age_W1_between")
    if "age_W1_within" in data_columns:
        covariates.append("age_W1_within")
    if "sex" in data_columns:
        covariates.append("sex")

    formula = f"{outcome_w1} ~ " + " + ".join(covariates)
    model_cols = [outcome_w1] + covariates + ["pair_id"]
    return formula, model_cols, covariates


def fit_between_within_cs(
    data: pd.DataFrame,
    outcome_w1: str,
    focal_var: str,
) -> dict | None:
    """Cross-sectional Mundlak MixedLM for one focal variable.

    Returns a dict with the same schema as the longitudinal
    `_fit_single_model` for consistent downstream handling.
    """
    import statsmodels.formula.api as smf

    between_col = f"{focal_var}_between"
    within_col = f"{focal_var}_within"

    formula, model_cols, _covariates = _build_formula_and_cols_cs(
        outcome_w1, focal_var, set(data.columns),
    )

    sub = data[model_cols].dropna()
    pair_counts = sub["pair_id"].value_counts()
    complete_pairs = pair_counts[pair_counts == 2].index
    sub = sub[sub["pair_id"].isin(complete_pairs)]

    if sub["pair_id"].nunique() < 10:
        return None

    for c in [between_col, within_col]:
        if c in sub.columns and sub[c].std() < 1e-10:
            return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = smf.mixedlm(formula, sub, groups=sub["pair_id"])
            result = model.fit(reml=True, maxiter=200)
    except Exception:
        return None

    sigma2_u = (float(result.cov_re.iloc[0, 0])
                if hasattr(result.cov_re, 'iloc') else float(result.cov_re))
    sigma2_e = float(result.scale)
    icc = sigma2_u / (sigma2_u + sigma2_e) if (sigma2_u + sigma2_e) > 0 else np.nan

    out = {
        "focal_var": focal_var,
        "n_persons": len(sub),
        "n_pairs": sub["pair_id"].nunique(),
        "beta_B": result.params.get(between_col, np.nan),
        "beta_B_se": result.bse.get(between_col, np.nan),
        "beta_B_p": result.pvalues.get(between_col, np.nan),
        "beta_W": result.params.get(within_col, np.nan),
        "beta_W_se": result.bse.get(within_col, np.nan),
        "beta_W_p": result.pvalues.get(within_col, np.nan),
        "sigma2_u": sigma2_u,
        "sigma2_e": sigma2_e,
        "icc": icc,
    }

    if abs(out["beta_B"]) > out["beta_B_se"] * 0.5:
        out["attenuation_ratio"] = out["beta_W"] / out["beta_B"]
    else:
        out["attenuation_ratio"] = np.nan

    return out


def compute_level_sds_cs(
    data: pd.DataFrame,
    focal_var: str,
    outcome_w1: str,
) -> dict:
    """Level-specific SDs for Snijders & Bosker standardization (CS)."""
    between_col = f"{focal_var}_between"
    within_col = f"{focal_var}_within"

    y_pair_mean = data.groupby("pair_id")[outcome_w1].transform("mean")
    y_within = data[outcome_w1] - y_pair_mean

    valid = data[[between_col, within_col, outcome_w1]].notna().all(axis=1)
    sub = data[valid]

    return {
        "sd_x_within": sub[within_col].std() if len(sub) > 2 else np.nan,
        "sd_x_between": sub[between_col].std() if len(sub) > 2 else np.nan,
        "sd_y_within": y_within[valid].std() if valid.sum() > 2 else np.nan,
        "sd_y_between": y_pair_mean[valid].std() if valid.sum() > 2 else np.nan,
    }


def _prepare_ols_matrices_cs(
    data: pd.DataFrame,
    outcome_w1: str,
    focal_var: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Build (X, y, pair_indices) for the CS bootstrap OLS."""
    between_col = f"{focal_var}_between"
    within_col = f"{focal_var}_within"

    _, model_cols, covariates = _build_formula_and_cols_cs(
        outcome_w1, focal_var, set(data.columns),
    )

    sub = data[model_cols].dropna()
    pair_counts = sub["pair_id"].value_counts()
    complete_pairs = pair_counts[pair_counts == 2].index
    sub = sub[sub["pair_id"].isin(complete_pairs)]

    if len(sub) < 20:
        return None, None, None, -1, -1

    y = sub[outcome_w1].values.astype(np.float32)
    X = sub[covariates].values.astype(np.float32)
    X = np.column_stack([np.ones(len(y), dtype=np.float32), X])

    pair_labels = sub["pair_id"].values
    _, pair_indices = np.unique(pair_labels, return_inverse=True)

    idx_B = covariates.index(between_col) + 1
    idx_W = covariates.index(within_col) + 1

    return X, y, pair_indices, idx_B, idx_W


def bootstrap_between_within_cs(
    data: pd.DataFrame,
    outcome_w1: str,
    focal_var: str,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """Pairs-level bootstrap for CS (W1) β_B and β_W CIs."""
    X_np, y_np, pair_indices, idx_B, idx_W = _prepare_ols_matrices_cs(
        data, outcome_w1, focal_var,
    )
    if X_np is None:
        return {"boot_n": 0, "beta_B_ci": (np.nan, np.nan),
                "beta_W_ci": (np.nan, np.nan)}

    try:
        import cupy as cp
        xp = cp
        X_dev = cp.asarray(X_np)
        y_dev = cp.asarray(y_np)
        use_gpu = True
    except ImportError:
        xp = np
        X_dev = X_np
        y_dev = y_np
        use_gpu = False

    unique_pairs = np.unique(pair_indices)
    n_pairs = len(unique_pairs)
    pair_row_map = {p: np.where(pair_indices == p)[0] for p in unique_pairs}

    rng = np.random.default_rng(seed)
    beta_B_boot = np.empty(n_boot, dtype=np.float32)
    beta_W_boot = np.empty(n_boot, dtype=np.float32)
    n_success = 0

    for i in range(n_boot):
        boot_pair_idx = rng.integers(0, n_pairs, size=n_pairs)
        row_idx = np.concatenate(
            [pair_row_map[unique_pairs[p]] for p in boot_pair_idx]
        )
        if use_gpu:
            row_idx_dev = cp.asarray(row_idx)
            X_boot = X_dev[row_idx_dev]
            y_boot = y_dev[row_idx_dev]
        else:
            X_boot = X_dev[row_idx]
            y_boot = y_dev[row_idx]

        try:
            XtX = X_boot.T @ X_boot
            Xty = X_boot.T @ y_boot
            beta = xp.linalg.solve(XtX, Xty)
            if use_gpu:
                beta_B_boot[n_success] = float(beta[idx_B].get())
                beta_W_boot[n_success] = float(beta[idx_W].get())
            else:
                beta_B_boot[n_success] = float(beta[idx_B])
                beta_W_boot[n_success] = float(beta[idx_W])
            n_success += 1
        except Exception:
            continue

    if n_success < n_boot * 0.5:
        return {"boot_n": n_success, "beta_B_ci": (np.nan, np.nan),
                "beta_W_ci": (np.nan, np.nan)}

    beta_B_valid = beta_B_boot[:n_success]
    beta_W_valid = beta_W_boot[:n_success]

    return {
        "boot_n": n_success,
        "beta_B_ci": (float(np.percentile(beta_B_valid, 2.5)),
                      float(np.percentile(beta_B_valid, 97.5))),
        "beta_W_ci": (float(np.percentile(beta_W_valid, 2.5)),
                      float(np.percentile(beta_W_valid, 97.5))),
        "beta_B_boot_se": float(np.std(beta_B_valid)),
        "beta_W_boot_se": float(np.std(beta_W_valid)),
    }


def compute_icc_by_zygosity_cs(
    df: pd.DataFrame,
    outcome_w1: str,
    sib_types: list[str],
    label: str,
) -> dict | None:
    """Null-model ICC at W1: Y_w1 ~ age_W1_B + age_W1_W + sex + (1|pair)."""
    import statsmodels.formula.api as smf

    mask = (
        df["sib_type"].isin(sib_types) & df["pair_id"].notna()
        & df[outcome_w1].notna()
    )
    sub = df[mask].copy()
    pair_counts = sub.groupby("pair_id").size()
    sub = sub[sub["pair_id"].isin(pair_counts[pair_counts >= 2].index)].copy()

    if len(sub) < 20:
        return None

    if "age_W1" in sub.columns:
        _decompose_between_within(sub, "age_W1")

    covs = []
    if "age_W1_between" in sub.columns:
        covs += ["age_W1_between", "age_W1_within"]
    if "sex" in sub.columns:
        covs.append("sex")

    formula = f"{outcome_w1} ~ " + (" + ".join(covs) if covs else "1")
    model_cols = [outcome_w1] + covs + ["pair_id"]
    sub2 = sub[model_cols].dropna()

    pc = sub2["pair_id"].value_counts()
    sub2 = sub2[sub2["pair_id"].isin(pc[pc == 2].index)]

    if len(sub2) < 20:
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = smf.mixedlm(formula, sub2, groups=sub2["pair_id"])
            result = model.fit(reml=True, maxiter=200)
    except Exception:
        return None

    sigma2_u = (float(result.cov_re.iloc[0, 0])
                if hasattr(result.cov_re, 'iloc') else float(result.cov_re))
    sigma2_e = float(result.scale)
    icc = sigma2_u / (sigma2_u + sigma2_e) if (sigma2_u + sigma2_e) > 0 else np.nan

    return {
        "group": label,
        "n_persons": len(sub2),
        "n_pairs": sub2["pair_id"].nunique(),
        "sigma2_u": sigma2_u,
        "sigma2_e": sigma2_e,
        "icc": icc,
    }


# GPU-accelerated pairs-level bootstrap

def _prepare_ols_matrices(
    data: pd.DataFrame,
    outcome_w2: str,
    outcome_w1: str,
    focal_var: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Build design matrix X, response y, pair_indices for OLS bootstrap.

    Uses OLS (not MixedLM) for bootstrap speed. For balanced twin data with
    random intercept, OLS with cluster-robust SEs gives equivalent fixed-effect
    inference (Wooldridge 2010, §7.8).
    """
    between_col = f"{focal_var}_between"
    within_col = f"{focal_var}_within"

    _, model_cols, covariates = _build_formula_and_cols(
        outcome_w2, outcome_w1, focal_var, set(data.columns),
    )

    sub = data[model_cols].dropna()

    # Remove singletons
    pair_counts = sub["pair_id"].value_counts()
    complete_pairs = pair_counts[pair_counts == 2].index
    sub = sub[sub["pair_id"].isin(complete_pairs)]

    if len(sub) < 20:
        return None, None, None, -1, -1

    y = sub[outcome_w2].values.astype(np.float32)
    X = sub[covariates].values.astype(np.float32)
    X = np.column_stack([np.ones(len(y), dtype=np.float32), X])

    pair_labels = sub["pair_id"].values
    _, pair_indices = np.unique(pair_labels, return_inverse=True)

    idx_B = covariates.index(between_col) + 1
    idx_W = covariates.index(within_col) + 1

    return X, y, pair_indices, idx_B, idx_W


def bootstrap_between_within(
    data: pd.DataFrame,
    outcome_w2: str,
    outcome_w1: str,
    focal_var: str,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """GPU-accelerated pairs-level bootstrap for β_B and β_W CIs."""
    result = _prepare_ols_matrices(data, outcome_w2, outcome_w1, focal_var)
    X_np, y_np, pair_indices, idx_B, idx_W = result

    if X_np is None:
        return {"boot_n": 0, "beta_B_ci": (np.nan, np.nan),
                "beta_W_ci": (np.nan, np.nan)}

    try:
        import cupy as cp
        xp = cp
        X_dev = cp.asarray(X_np)
        y_dev = cp.asarray(y_np)
        use_gpu = True
    except ImportError:
        xp = np
        X_dev = X_np
        y_dev = y_np
        use_gpu = False

    unique_pairs = np.unique(pair_indices)
    n_pairs = len(unique_pairs)

    pair_row_map = {}
    for p in unique_pairs:
        pair_row_map[p] = np.where(pair_indices == p)[0]

    rng = np.random.default_rng(seed)
    beta_B_boot = np.empty(n_boot, dtype=np.float32)
    beta_W_boot = np.empty(n_boot, dtype=np.float32)
    n_success = 0

    for i in range(n_boot):
        boot_pair_idx = rng.integers(0, n_pairs, size=n_pairs)
        row_idx = np.concatenate(
            [pair_row_map[unique_pairs[p]] for p in boot_pair_idx]
        )

        if use_gpu:
            row_idx_dev = cp.asarray(row_idx)
            X_boot = X_dev[row_idx_dev]
            y_boot = y_dev[row_idx_dev]
        else:
            X_boot = X_dev[row_idx]
            y_boot = y_dev[row_idx]

        try:
            XtX = X_boot.T @ X_boot
            Xty = X_boot.T @ y_boot
            beta = xp.linalg.solve(XtX, Xty)

            if use_gpu:
                beta_B_boot[n_success] = float(beta[idx_B].get())
                beta_W_boot[n_success] = float(beta[idx_W].get())
            else:
                beta_B_boot[n_success] = float(beta[idx_B])
                beta_W_boot[n_success] = float(beta[idx_W])
            n_success += 1
        except Exception:
            continue

    if n_success < n_boot * 0.5:
        return {"boot_n": n_success, "beta_B_ci": (np.nan, np.nan),
                "beta_W_ci": (np.nan, np.nan)}

    beta_B_valid = beta_B_boot[:n_success]
    beta_W_valid = beta_W_boot[:n_success]

    return {
        "boot_n": n_success,
        "beta_B_ci": (float(np.percentile(beta_B_valid, 2.5)),
                      float(np.percentile(beta_B_valid, 97.5))),
        "beta_W_ci": (float(np.percentile(beta_W_valid, 2.5)),
                      float(np.percentile(beta_W_valid, 97.5))),
        "beta_B_boot_se": float(np.std(beta_B_valid)),
        "beta_W_boot_se": float(np.std(beta_W_valid)),
    }


# ICC by zygosity

def compute_icc_by_zygosity(
    df: pd.DataFrame,
    outcome_w1: str,
    outcome_w2: str,
    sib_types: list[str],
    label: str,
) -> dict | None:
    """Fit null model Y_w2 ~ Y_w1_B + Y_w1_W + age_W2_B + age_W2_W + sex + (1|pair).
    Returns ICC = σ²_u / (σ²_u + σ²_e)."""
    import statsmodels.formula.api as smf

    mask = (
        df["sib_type"].isin(sib_types) & df["pair_id"].notna()
        & df[outcome_w1].notna() & df[outcome_w2].notna()
    )
    sub = df[mask].copy()
    pair_counts = sub.groupby("pair_id").size()
    sub = sub[sub["pair_id"].isin(pair_counts[pair_counts >= 2].index)].copy()

    if len(sub) < 20:
        return None

    _decompose_between_within(sub, outcome_w1)

    # age_W2 preferred, fallback age_W1
    age_var = "age_W2" if "age_W2" in sub.columns else "age_W1"
    _decompose_between_within(sub, age_var)

    covs = [f"{outcome_w1}_between", f"{outcome_w1}_within"]
    if f"{age_var}_between" in sub.columns:
        covs += [f"{age_var}_between", f"{age_var}_within"]
    if "sex" in sub.columns:
        covs.append("sex")

    formula = f"{outcome_w2} ~ " + " + ".join(covs)
    model_cols = [outcome_w2] + covs + ["pair_id"]
    sub2 = sub[model_cols].dropna()

    pc = sub2["pair_id"].value_counts()
    sub2 = sub2[sub2["pair_id"].isin(pc[pc == 2].index)]

    if len(sub2) < 20:
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = smf.mixedlm(formula, sub2, groups=sub2["pair_id"])
            result = model.fit(reml=True, maxiter=200)
    except Exception:
        return None

    sigma2_u = (float(result.cov_re.iloc[0, 0])
                if hasattr(result.cov_re, 'iloc') else float(result.cov_re))
    sigma2_e = float(result.scale)
    icc = sigma2_u / (sigma2_u + sigma2_e) if (sigma2_u + sigma2_e) > 0 else np.nan

    return {
        "group": label,
        "n_persons": len(sub2),
        "n_pairs": sub2["pair_id"].nunique(),
        "sigma2_u": sigma2_u,
        "sigma2_e": sigma2_e,
        "icc": icc,
    }


# Cross-twin cross-trait correlation

def compute_ctct(
    df: pd.DataFrame,
    outcome_w2: str,
    focal_vars: list[str],
    sib_types: list[str],
    label: str,
) -> list[dict]:
    """Cross-twin cross-trait: r(Twin1_X, Twin2_Y) for each focal var."""
    mask = (
        df["sib_type"].isin(sib_types) & df["pair_id"].notna()
        & df[outcome_w2].notna()
    )
    sub = df[mask].copy()
    pair_counts = sub.groupby("pair_id").size()
    sub = sub[sub["pair_id"].isin(pair_counts[pair_counts == 2].index)]

    rows = []
    for var in focal_vars:
        if var not in sub.columns:
            continue

        pairs = sub.groupby("pair_id").apply(
            lambda g: pd.Series({
                "x1": g[var].iloc[0], "y1": g[outcome_w2].iloc[0],
                "x2": g[var].iloc[1], "y2": g[outcome_w2].iloc[1],
            }) if len(g) == 2 else pd.Series(dtype=float),
            include_groups=False,
        ).dropna()

        if len(pairs) < 10:
            continue

        x_cross = pd.to_numeric(
            np.concatenate([pairs["x1"].values, pairs["x2"].values]),
            errors='coerce',
        ).astype(float)
        y_cross = pd.to_numeric(
            np.concatenate([pairs["y2"].values, pairs["y1"].values]),
            errors='coerce',
        ).astype(float)
        valid = ~(np.isnan(x_cross) | np.isnan(y_cross))
        if valid.sum() < 10:
            continue

        r, p = spearmanr(x_cross[valid], y_cross[valid])
        rows.append({
            "group": label,
            "focal_var": var,
            "ctct_r": r,
            "ctct_p": p,
            "n_pairs": len(pairs),
        })

    return rows


# BH-FDR correction

def bh_fdr(pvals: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted p-values."""
    n = len(pvals)
    if n == 0:
        return np.array([])
    order = np.argsort(pvals)
    adjusted = np.empty(n)
    adjusted[order[-1]] = pvals[order[-1]]
    for i in range(n - 2, -1, -1):
        adjusted[order[i]] = min(
            adjusted[order[i + 1]],
            pvals[order[i]] * n / (i + 1),
        )
    return np.clip(adjusted, 0, 1)


# ICC diagnostic: FInoPsy change heritability

def _simple_icc(groups: np.ndarray) -> float:
    """One-way random ICC(1,1) from n×2 array of paired values."""
    if len(groups) < 5:
        return np.nan
    k = groups.shape[1]  # 2
    n = groups.shape[0]
    grand_mean = groups.mean()
    ms_between = k * np.sum((groups.mean(axis=1) - grand_mean) ** 2) / (n - 1)
    ms_within = np.sum((groups - groups.mean(axis=1, keepdims=True)) ** 2) / (n * (k - 1))
    if (ms_between + (k - 1) * ms_within) == 0:
        return np.nan
    return (ms_between - ms_within) / (ms_between + (k - 1) * ms_within)


def _get_pair_values(sub: pd.DataFrame, col: str) -> np.ndarray:
    """Extract n×2 array of paired values for ICC computation."""
    mask = sub[col].notna()
    s = sub[mask]
    pc = s.groupby("pair_id").size()
    s = s[s["pair_id"].isin(pc[pc == 2].index)]
    groups = []
    for _, grp in s.groupby("pair_id"):
        vals = grp[col].values
        if len(vals) == 2:
            groups.append(vals)
    return np.array(groups) if groups else np.empty((0, 2))


def icc_diagnostic(df: pd.DataFrame) -> dict:
    """Comprehensive ICC diagnostic for FInoPsy change.

    Returns a dict of ICC diagnostics.
    """
    from scipy import stats

    # helpers
    def _get_sib_pairs(df, sib_type):
        mask = (df["sib_type"] == sib_type) & (df["is_full_pair"] == True) & df["pair_id"].notna()
        sub = df[mask].copy()
        pc = sub.groupby("pair_id").size()
        return sub[sub["pair_id"].isin(pc[pc == 2].index)].copy()

    mz = _get_sib_pairs(df, "MZ")
    dz = _get_sib_pairs(df, "DZ_SS")

    result = {
        "sample_size": {},
        "test_retest": {},
        "simple_icc": {},
        "within_pair_diff": {},
        "w1_icc": {},
    }

    # 1. Sample size & completeness
    result["n_mz_pairs"] = mz["pair_id"].nunique()
    for name, (w1, w2) in OUTCOMES.items():
        both = mz[w1].notna() & mz[w2].notna()
        pair_both = mz[both].groupby("pair_id").size()
        n_pairs = (pair_both == 2).sum()
        result["sample_size"][name] = n_pairs

    # 2. Test-retest r & residual variance (among MZ)
    for name, (w1, w2) in OUTCOMES.items():
        mask = mz[w1].notna() & mz[w2].notna()
        sub = mz[mask]
        r, _ = stats.pearsonr(sub[w1], sub[w2])
        var_w2 = sub[w2].var()
        slope, intercept = np.polyfit(sub[w1], sub[w2], 1)
        resid = sub[w2] - (slope * sub[w1] + intercept)
        var_resid = resid.var()
        result["test_retest"][name] = {
            "r": r,
            "pct_residual": 100 * var_resid / var_w2 if var_w2 > 0 else np.nan,
        }

    # 3. Simple ICC (no covariates) on W2, change, and residualized change
    for name, (w1, w2) in OUTCOMES.items():
        mask = mz[w1].notna() & mz[w2].notna()
        sub = mz[mask].copy()
        pc = sub.groupby("pair_id").size()
        sub = sub[sub["pair_id"].isin(pc[pc == 2].index)]

        icc_w2 = _simple_icc(_get_pair_values(sub, w2))
        sub["_change"] = sub[w2] - sub[w1]
        icc_change = _simple_icc(_get_pair_values(sub, "_change"))
        slope, intercept = np.polyfit(sub[w1], sub[w2], 1)
        sub["_resid"] = sub[w2] - (slope * sub[w1] + intercept)
        icc_resid = _simple_icc(_get_pair_values(sub, "_resid"))

        result["simple_icc"][name] = {
            "w2_raw": icc_w2,
            "change": icc_change,
            "resid": icc_resid,
        }

    # 4. Within-pair |difference| on change: MZ vs DZ
    for name, (w1, w2) in OUTCOMES.items():
        result["within_pair_diff"][name] = {}
        for label, sub in [("MZ", mz), ("DZ_SS", dz)]:
            mask = sub[w1].notna() & sub[w2].notna()
            s = sub[mask].copy()
            s["change"] = s[w2] - s[w1]
            pc = s.groupby("pair_id").size()
            s = s[s["pair_id"].isin(pc[pc == 2].index)]
            diffs = []
            for _, grp in s.groupby("pair_id"):
                vals = grp["change"].values
                if len(vals) == 2:
                    diffs.append(abs(vals[0] - vals[1]))
            diffs = np.array(diffs) if diffs else np.array([])
            result["within_pair_diff"][name][label] = {
                "n_pairs": len(diffs),
                "mean": float(diffs.mean()) if len(diffs) > 0 else np.nan,
                "median": float(np.median(diffs)) if len(diffs) > 0 else np.nan,
            }

    # 5. W1 cross-sectional ICC (baseline heritability)
    for name, (w1, w2) in OUTCOMES.items():
        result["w1_icc"][name] = {}
        for label, sub in [("MZ", mz), ("DZ_SS", dz)]:
            pairs = _get_pair_values(sub, w1)
            result["w1_icc"][name][label] = _simple_icc(pairs)

    return result


# Frailty item-level ICC (MZ)
# 37 items; limitWalking, limitRecreation and limitModAct are excluded as
# effectively fully imputed at W2 in the pooled sample.

FRAILTY_ITEMS = [
    "cholesterol", "gallbladderProb", "limitGrocery", "limitShower", "lonely",
    "backPain", "insomnia", "dental", "sciatica", "highBP", "SRH", "cancer",
    "kneePain", "stomachPain", "thyroidDisease", "limitTravel", "headPain",
    "bladderProb", "allergic", "migraine", "lungProb", "skinProb", "chestPain",
    "asthma", "stroke", "hernia", "irregularHeart", "hipPain", "arthritis",
    "diabetes", "multiCancer", "heartAttack", "heartTrouble", "jointPain",
    "painDiagnosed", "angina", "chronicIllness",
]


def compute_frailty_item_icc(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Per-item MZ test-retest r, change variance, and change ICC(1,1).

    Merges the item-level W1/W2 frailty files onto the master MZ-pair subset,
    then for each item computes retest correlation and MZ change ICC.
    Writes output_dir/frailty_item_icc.csv.
    """
    from scipy import stats

    if not FRAILTY_W1_CSV.exists() or not FRAILTY_W2_CSV.exists():
        print(f"  [SKIP] frailty item files not found "
              f"({FRAILTY_W1_CSV.name}, {FRAILTY_W2_CSV.name})")
        return pd.DataFrame()

    f1 = pd.read_csv(FRAILTY_W1_CSV)
    f2 = pd.read_csv(FRAILTY_W2_CSV)

    # High-missingness flag comes from the pre-imputation missingness table.
    miss_path = output_dir / "frailty_item_missingness.csv"
    if miss_path.exists():
        miss = pd.read_csv(miss_path).set_index("item")
        high_miss_set = set(
            miss.index[
                (miss["W1_miss_pct"] > 20) | (miss["W2_miss_pct"] > 20)
            ]
        )
    else:
        high_miss_set = set()

    # Restrict to MZ complete pairs.
    mz_mask = (
        (df["sib_type"] == "MZ")
        & (df["is_full_pair"] == True)
        & df["pair_id"].notna()
    )
    # Eligibility uses FInoPsy from the master: the pooled wave-2 file holds the
    # 38-item intersection, so its FInoPsy differs for MIDUS3. Item columns still
    # come from the pooled files, since the master has no item-level data.
    mz = df[mz_mask][["M2MRID", "pair_id", "FInoPsy_W1", "FInoPsy_W2"]].copy()
    pc = mz.groupby("pair_id").size()
    mz = mz[mz["pair_id"].isin(pc[pc == 2].index)]

    # Merge W1 and W2 item values via M2MRID <-> id.
    # Rename the pooled composite so it does not shadow the master's.
    f1_r = f1.rename(columns={c: f"{c}_W1" for c in f1.columns if c != "id"}) \
             .rename(columns={"FInoPsy_W1": "FInoPsy_W1_pooled"})
    f2_r = f2.rename(columns={c: f"{c}_W2" for c in f2.columns if c != "id"}) \
             .rename(columns={"FInoPsy_W2": "FInoPsy_W2_pooled"})
    merged = (
        mz.merge(f1_r, left_on="M2MRID", right_on="id", how="left")
          .drop(columns=["id"])
          .merge(f2_r, left_on="M2MRID", right_on="id", how="left")
          .drop(columns=["id"])
    )

    # Pair-count denominator: pairs where both members have both-wave records
    # for the FInoPsy composite (stable reference across items).
    has_both = merged["FInoPsy_W1"].notna() & merged["FInoPsy_W2"].notna()
    eligible = merged[has_both]
    pc2 = eligible.groupby("pair_id").size()
    eligible_pairs = pc2[pc2 == 2].index
    sample = merged[merged["pair_id"].isin(eligible_pairs)].copy()
    n_pairs_global = len(eligible_pairs)

    rows = []
    for item in FRAILTY_ITEMS:
        w1_col = f"{item}_W1"
        w2_col = f"{item}_W2"
        if w1_col not in sample.columns or w2_col not in sample.columns:
            continue

        valid = sample[w1_col].notna() & sample[w2_col].notna()
        s = sample[valid].copy()

        if len(s) < 5:
            continue

        x1 = s[w1_col].astype(float)
        x2 = s[w2_col].astype(float)

        try:
            retest_r, _ = stats.pearsonr(x1, x2)
        except Exception:
            retest_r = np.nan

        s["_change"] = x2 - x1
        pc_s = s.groupby("pair_id").size()
        s = s[s["pair_id"].isin(pc_s[pc_s == 2].index)]

        change_icc_mz = _simple_icc(_get_pair_values(s, "_change"))
        w1_var = float(x1.var())
        change_var = float(s["_change"].var()) if len(s) >= 2 else np.nan

        rows.append({
            "item": item,
            "high_miss": bool(item in high_miss_set),
            "retest_r": round(float(retest_r), 3) if pd.notna(retest_r) else np.nan,
            "change_icc_mz": (round(float(change_icc_mz), 3)
                              if pd.notna(change_icc_mz) else np.nan),
            "n_pairs": n_pairs_global,
            "w1_var": round(w1_var, 4),
            "change_var": round(change_var, 4) if pd.notna(change_var) else np.nan,
        })

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out

    # Sort by change_icc_mz, descending.
    out = out.sort_values("change_icc_mz", ascending=False, na_position="last")
    path = output_dir / "frailty_item_icc.csv"
    out.to_csv(path, index=False)
    print(f"  Wrote {path.name}: {len(out)} items, n_pairs={n_pairs_global}")
    return out


# Structured output bookkeeping

def write_denominator_outputs(
    results_by_outcome: dict,
    icc_results: dict,
    output_dir: Path,
    cs_results_by_outcome: dict | None = None,
    cs_icc_results: dict | None = None,
) -> None:
    """Write CSV denominator bridges for A6 and SI builders."""
    icc_rows = []
    for analysis, source in [
        ("cs", cs_icc_results or {}),
        ("change", icc_results or {}),
    ]:
        for outcome, icc_list in source.items():
            for icc in icc_list:
                if not icc:
                    continue
                icc_rows.append({
                    "analysis": analysis,
                    "outcome": outcome,
                    "group": icc["group"],
                    "denominator_type": "overview/ICC outcome pairs",
                    "n_pairs": icc["n_pairs"],
                    "icc": icc["icc"],
                    "sigma2_u": icc["sigma2_u"],
                    "sigma2_e": icc["sigma2_e"],
                    "note": (
                        "ALcomp biomarker-specific denominator"
                        if outcome == ap.AL_STEM else ""
                    ),
                })
    if icc_rows:
        pd.DataFrame(icc_rows).to_csv(
            output_dir / "a6_icc_denominators.csv", index=False,
        )

    model_rows = []
    for analysis, source in [
        ("cs", cs_results_by_outcome or {}),
        ("change", results_by_outcome or {}),
    ]:
        for outcome, groups in source.items():
            for group_label, df_result in groups.items():
                if df_result.empty:
                    continue
                for _, row in df_result.iterrows():
                    model_rows.append({
                        "analysis": analysis,
                        "outcome": outcome,
                        "group": group_label,
                        "variable": row["focal_var"],
                        "denominator_type": "model-complete coefficient row",
                        "n_pairs": row["n_pairs"],
                        "note": (
                            "ALcomp biomarker-specific denominator"
                            if outcome == ap.AL_STEM else ""
                        ),
                    })
    if model_rows:
        pd.DataFrame(model_rows).to_csv(
            output_dir / "a6_model_denominators.csv", index=False,
        )


def write_pipeline_log(
    args: argparse.Namespace,
    outcomes: list[str],
    results_by_outcome: dict,
    cs_results_by_outcome: dict,
    ctct_results: dict,
    discordant_reports: list,
) -> None:
    outputs = [
        {"path": "frailty_item_missingness.csv", "kind": "CSV"},
        {"path": "frailty_item_icc.csv", "kind": "CSV"},
        {"path": "a6_icc_denominators.csv", "kind": "CSV"},
        {"path": "a6_model_denominators.csv", "kind": "CSV"},
        {"path": "a6_discordant_pairs.csv", "kind": "CSV", "status": "written" if discordant_reports else "not written"},
    ]
    for outcome in outcomes:
        for group_label in ("All_paired", "MZ_sensitivity"):
            outputs.append({"path": f"a6_cs_{group_label}_{outcome}.csv", "kind": "CSV"})
            outputs.append({"path": f"a6_{group_label}_{outcome}.csv", "kind": "CSV"})
        outputs.append({
            "path": f"a6_ctct_{outcome}.csv",
            "kind": "CSV",
            "status": "written" if ctct_results.get(outcome) else "not written",
        })

    run_args = vars(args).copy()
    run_args["resolved_outcomes"] = outcomes
    write_output_log(OUTPUT_DIR, "a6_output_log.md", Path(__file__).name, outputs, run_args)


# Main

def main():
    # Record input provenance and the age-adjustment method beside the outputs.
    provenance.stamp_at_exit(OUTPUT_DIR, "a6_family_decomposition", ["data/processed/master_analysis.csv", "data/processed/wave1/pooled_frailty_M2R.csv", "data/processed/wave2/pooled_frailty_M3MR2.csv", "output/a4_longitudinal", "output/cs1_cross_sectional"])
    age_method.emit_at_exit(OUTPUT_DIR, "a6_family_decomposition")
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    outcomes = (list(OUTCOMES.keys()) if "all" in args.outcomes
                else args.outcomes)

    print("A6: SIBLING/TWIN BETWEEN/WITHIN DECOMPOSITION")
    print(f"Bootstrap: {args.n_boot}× pairs-level")
    print(f"Model: Full Mundlak (Y_w1, age_W2 decomposed for change; "
          f"age_W1 for CS)")
    print(f"Categoricals: k-1 dummies, min {MIN_DISCORDANT_PAIRS} "
          f"discordant pairs")
    print(f"Primary: All_paired | Sensitivity: MZ-only")
    print()

    df = pd.read_csv(MASTER_CSV, low_memory=False)

    # Encode sex numerically
    if "sex" in df.columns and df["sex"].dtype == object:
        df["sex"] = pd.factorize(df["sex"])[0].astype(float)

    # Pre-expand all categoricals on the full dataframe
    # (dummy columns persist across outcomes)
    for var in CATEGORICAL_CONFIG:
        if var in df.columns:
            expand_categorical(df, var, CATEGORICAL_CONFIG[var])

    results_by_outcome = {}
    icc_results = {}
    ctct_results = {}
    discordant_reports = []

    # Cross-sectional (W1) outputs
    cs_results_by_outcome = {}
    cs_icc_results = {}
    cs_variable_sets = {}

    ANALYSIS_GROUPS = {
        "All_paired": ALL_SIB_TYPES,
        "MZ_sensitivity": ["MZ"],
    }

    for outcome_label in outcomes:
        if outcome_label not in OUTCOMES:
            print(f"  [SKIP] Unknown outcome: {outcome_label}")
            continue

        outcome_w1, outcome_w2 = OUTCOMES[outcome_label]
        print(f"\nOutcome: {outcome_label} ({outcome_w1} to {outcome_w2})")

        # Load focal variables from A4
        try:
            raw_focal_vars = load_focal_variables(outcome_w2, args.top_k)
        except FileNotFoundError as e:
            print(f"  {e}")
            continue
        print(f"  A4 focal variables ({len(raw_focal_vars)}): "
              f"{raw_focal_vars[:5]}...")

        # ICC by zygosity
        icc_list = []
        for label, sib_types in [
            ("MZ", ["MZ"]),
            ("DZ_SS", ["DZ_SS"]),
            ("All_paired", ALL_SIB_TYPES),
        ]:
            icc = compute_icc_by_zygosity(
                df, outcome_w1, outcome_w2, sib_types, label,
            )
            if icc:
                icc_list.append(icc)
                print(f"  ICC ({label}): {icc['icc']:.3f} "
                      f"(n_pairs={icc['n_pairs']})")
        icc_results[outcome_label] = icc_list

        # CS (W1) ICC by zygosity
        cs_icc_list = []
        for label, sib_types in [
            ("MZ", ["MZ"]),
            ("DZ_SS", ["DZ_SS"]),
            ("All_paired", ALL_SIB_TYPES),
        ]:
            icc_cs = compute_icc_by_zygosity_cs(
                df, outcome_w1, sib_types, label,
            )
            if icc_cs:
                cs_icc_list.append(icc_cs)
                print(f"  ICC_W1 ({label}): {icc_cs['icc']:.3f} "
                      f"(n_pairs={icc_cs['n_pairs']})")
        cs_icc_results[outcome_label] = cs_icc_list

        # CS (W1) Between/Within models
        try:
            cs_focal_vars = load_focal_variables_cs(outcome_w1, top_k=20)
        except FileNotFoundError as e:
            print(f"  [CS] {e}")
            cs_focal_vars = []
        cs_variable_sets[outcome_label] = list(cs_focal_vars)
        print(f"  CS-1 focal vars ({len(cs_focal_vars)}): "
              f"{cs_focal_vars[:5]}...")

        cs_groups = {}
        for group_label, sib_types in ANALYSIS_GROUPS.items():
            print(f"\n  --- CS / {group_label} ---")
            data_cs = prepare_pair_data_cs(
                df, outcome_w1, cs_focal_vars, sib_types,
            )
            if len(data_cs) == 0 or not cs_focal_vars:
                print("    Insufficient paired data")
                cs_groups[group_label] = pd.DataFrame()
                continue

            n_pairs_cs = data_cs["pair_id"].nunique()
            print(f"    n={len(data_cs)}, pairs={n_pairs_cs}")

            expanded_cs, disc_cs = expand_and_filter_categoricals(
                cs_focal_vars, data_cs,
            )
            for r in disc_cs:
                discordant_reports.append(
                    {**r, "outcome": outcome_label,
                     "group": group_label, "layer": "cross_sectional"},
                )
            for var in expanded_cs:
                if f"{var}_between" not in data_cs.columns:
                    _decompose_between_within(data_cs, var)

            rows_cs = []
            for var in expanded_cs:
                r = fit_between_within_cs(data_cs, outcome_w1, var)
                if r is None:
                    continue

                sds = compute_level_sds_cs(data_cs, var, outcome_w1)
                add_standardized_effects(r, sds)

                boot = bootstrap_between_within_cs(
                    data_cs, outcome_w1, var, n_boot=args.n_boot,
                )
                r["beta_W_ci_lo"] = boot["beta_W_ci"][0]
                r["beta_W_ci_hi"] = boot["beta_W_ci"][1]
                r["beta_B_ci_lo"] = boot["beta_B_ci"][0]
                r["beta_B_ci_hi"] = boot["beta_B_ci"][1]
                r["boot_n"] = boot["boot_n"]
                if "beta_W_boot_se" in boot:
                    r["beta_W_boot_se"] = boot["beta_W_boot_se"]
                    r["beta_B_boot_se"] = boot["beta_B_boot_se"]
                rows_cs.append(r)

                bw_std_str = (f"{r.get('beta_W_std', np.nan):.3f}"
                              if pd.notna(r.get('beta_W_std')) else "NA")
                print(f"    {var}: β_W={r['beta_W']:.4f} "
                      f"[{boot['beta_W_ci'][0]:.4f}, "
                      f"{boot['beta_W_ci'][1]:.4f}], "
                      f"β_W_std={bw_std_str}, β_B={r['beta_B']:.4f}")

            if rows_cs:
                cs_df = pd.DataFrame(rows_cs)
                p_vals = cs_df["beta_W_p"].values
                valid_p = ~np.isnan(p_vals)
                q_vals = np.full(len(p_vals), np.nan)
                if valid_p.sum() > 0:
                    q_vals[valid_p] = bh_fdr(p_vals[valid_p])
                cs_df["beta_W_q"] = q_vals

                cs_groups[group_label] = cs_df
                cs_df.to_csv(
                    OUTPUT_DIR / f"a6_cs_{group_label}_{outcome_label}.csv",
                    index=False,
                )
            else:
                cs_groups[group_label] = pd.DataFrame()

        cs_results_by_outcome[outcome_label] = cs_groups

        # CTCT correlations (MZ and DZ only)
        ctct_list = []
        for label, sib_types in [("MZ", ["MZ"]), ("DZ_SS", ["DZ_SS"])]:
            ctct_list.extend(
                compute_ctct(df, outcome_w2, raw_focal_vars, sib_types, label)
            )
        ctct_results[outcome_label] = ctct_list

        # Between/Within models
        groups = {}
        for group_label, sib_types in ANALYSIS_GROUPS.items():
            print(f"\n  --- {group_label} ---")

            # Prepare paired data (with B/W decomposition of raw focals)
            data = prepare_pair_data(
                df, outcome_w1, outcome_w2, raw_focal_vars, sib_types,
            )
            if len(data) == 0:
                print(f"    Insufficient paired data")
                groups[group_label] = pd.DataFrame()
                continue

            n_pairs = data["pair_id"].nunique()
            print(f"    n={len(data)}, pairs={n_pairs}")

            # Expand categoricals and filter by discordant pairs
            focal_vars, disc_report = expand_and_filter_categoricals(
                raw_focal_vars, data,
            )

            # Keep discordant counts for every stratum, since the floor is applied per stratum.
            for r in disc_report:
                discordant_reports.append(
                    {**r, "outcome": outcome_label,
                     "group": group_label, "layer": "change"},
                )

            print(f"    Expanded focal vars ({len(focal_vars)}): "
                  f"{focal_vars[:8]}...")

            # Ensure all expanded dummies are decomposed
            for var in focal_vars:
                if f"{var}_between" not in data.columns:
                    _decompose_between_within(data, var)

            rows = []
            for var in focal_vars:
                result = fit_between_within(data, outcome_w2, outcome_w1, var)
                if result is None:
                    continue

                sds = compute_level_sds(data, var, outcome_w2)
                add_standardized_effects(result, sds)

                boot = bootstrap_between_within(
                    data, outcome_w2, outcome_w1, var,
                    n_boot=args.n_boot,
                )
                result["beta_W_ci_lo"] = boot["beta_W_ci"][0]
                result["beta_W_ci_hi"] = boot["beta_W_ci"][1]
                result["beta_B_ci_lo"] = boot["beta_B_ci"][0]
                result["beta_B_ci_hi"] = boot["beta_B_ci"][1]
                result["boot_n"] = boot["boot_n"]
                if "beta_W_boot_se" in boot:
                    result["beta_W_boot_se"] = boot["beta_W_boot_se"]
                    result["beta_B_boot_se"] = boot["beta_B_boot_se"]

                rows.append(result)

                bw_std_str = (f"{result.get('beta_W_std', np.nan):.3f}"
                              if pd.notna(result.get('beta_W_std')) else "NA")
                print(f"    {var}: β_W={result['beta_W']:.4f} "
                      f"[{boot['beta_W_ci'][0]:.4f}, "
                      f"{boot['beta_W_ci'][1]:.4f}], "
                      f"β_W_std={bw_std_str}, "
                      f"β_B={result['beta_B']:.4f}")

            if rows:
                result_df = pd.DataFrame(rows)

                # BH-FDR correction on β_W p-values
                p_vals = result_df["beta_W_p"].values
                valid_p = ~np.isnan(p_vals)
                q_vals = np.full(len(p_vals), np.nan)
                if valid_p.sum() > 0:
                    q_vals[valid_p] = bh_fdr(p_vals[valid_p])
                result_df["beta_W_q"] = q_vals

                groups[group_label] = result_df
                result_df.to_csv(
                    OUTPUT_DIR / f"a6_{group_label}_{outcome_label}.csv",
                    index=False,
                )
            else:
                groups[group_label] = pd.DataFrame()

        results_by_outcome[outcome_label] = groups

    # Save CTCT
    for outcome_label, ctct_list in ctct_results.items():
        if ctct_list:
            pd.DataFrame(ctct_list).to_csv(
                OUTPUT_DIR / f"a6_ctct_{outcome_label}.csv", index=False,
            )

    # Save discordant report
    if discordant_reports:
        pd.DataFrame(discordant_reports).to_csv(
            OUTPUT_DIR / "a6_discordant_pairs.csv", index=False,
        )

    # ICC diagnostic
    print("\nICC Diagnostic: FInoPsy change heritability")
    icc_diag = icc_diagnostic(df)
    for name, d in icc_diag["simple_icc"].items():
        print(f"  {name}: ICC_MZ W2={d['w2_raw']:.3f}, "
              f"change={d['change']:.3f}, resid={d['resid']:.3f}")

    # Frailty-item MZ ICC
    print("\nFrailty-item MZ ICC")
    try:
        compute_frailty_item_icc(df, OUTPUT_DIR)
    except Exception as e:
        print(f"  [WARN] frailty_item_icc not written: {e}")

    # Summary
    write_denominator_outputs(
        results_by_outcome, icc_results, OUTPUT_DIR,
        cs_results_by_outcome=cs_results_by_outcome,
        cs_icc_results=cs_icc_results,
    )

    write_pipeline_log(
        args, outcomes, results_by_outcome, cs_results_by_outcome,
        ctct_results, discordant_reports,
    )
    print(f"\nDone. Results in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
