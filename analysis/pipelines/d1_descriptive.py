#!/usr/bin/env python3
"""
D1: Descriptive Analysis (Supplemental Materials)

Generates:
  1. Sample descriptive table (demographics + outcome means/SDs by wave)
  2. Attrition analysis (W1-to-W2 completers vs dropouts)
  3. Outcome & change-score distributions (density plots)
  4. Test-retest correlations (W1-W2) for outcomes
  5. Zero-order outcome correlations (age-residualized)
  6. Predictor test-retest reliability (T1+T2 variables, W1-W2), and its median by domain

Usage:
    uv run python analysis/pipelines/d1_descriptive.py

Output: output/d1_descriptive/
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import yaml
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import cohen_kappa_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.core import provenance

from analysis.core.data_loader import get_tier_variables, CATEGORICALS
from analysis.core.pub_style import display_name, get_subdomain

OUTPUT_DIR = ROOT / "output" / "d1_descriptive"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MASTER_CSV = ROOT / "data" / "processed" / "master_analysis.csv"
CONCURRENT_CSV = ROOT / "data" / "processed" / "master_analysis_concurrent.csv"
CATEGORY_LABELS_YAML = ROOT / "config" / "category_labels.yaml"

# Outcome pairs: (W1, W2, display_name)
LONGITUDINAL_OUTCOMES = [
    ("FInoPsy_W1", "FInoPsy_W2", "Frailty Index"),
    ("btactComp_W1", "btactComp_W2", "Cognitive Composite"),
    # 16-marker allostatic-load composite.
    ("ALcomp_W1", "ALcomp_W2", "Allostatic Load"),
]

# Cross-sectional only outcomes (supplement)
CS_OUTCOMES = [
    ("phenoAge_res", "PhenoAge (residualized)"),
]

# Demographics for Table 1
DEMO_CONTINUOUS = [
    ("age_W1", "Age (W1)"),
]
DEMO_BINARY = [
    ("female", "Female"),
    ("hispanic", "Hispanic"),
]

FAMILY_STRATA = {
    "MZ": ["MZ"],
    "DZ_SS": ["DZ_SS"],
    "DZ_OS": ["DZ_OS"],
    "All_paired": ["MZ", "DZ_SS", "DZ_OS", "sibling"],
}


def load_category_labels() -> dict[str, dict]:
    """Load category labels used to classify discrete predictor measurement types."""
    with open(CATEGORY_LABELS_YAML, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return {str(k): v for k, v in raw.items() if isinstance(v, dict)}


CATEGORY_LABELS = load_category_labels()


def predictor_measurement_type(var: str, df_ref: pd.DataFrame) -> str:
    """Classify predictors using category_labels.yaml as the authoritative source."""
    label_map = CATEGORY_LABELS.get(var)
    if label_map:
        if len(label_map) == 2:
            return "binary"
        if var in CATEGORICALS:
            return "nominal"
        return "ordinal"
    if var in CATEGORICALS:
        return "nominal"
    if var in df_ref.columns and int(df_ref[var].dropna().nunique()) == 2:
        return "binary"
    return "continuous"


def load_data() -> pd.DataFrame:
    """Load master dataset."""
    df = pd.read_csv(MASTER_CSV, low_memory=False)
    print(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
    return df


def age_residualize(values: np.ndarray, age: np.ndarray) -> np.ndarray:
    """Residualize values against age via OLS. Returns residuals; NaN preserved."""
    valid = ~np.isnan(values) & ~np.isnan(age)
    if valid.sum() < 10:
        return values
    reg = LinearRegression()
    reg.fit(age[valid].reshape(-1, 1), values[valid])
    resid = np.full_like(values, np.nan)
    resid[valid] = values[valid] - reg.predict(age[valid].reshape(-1, 1)).ravel()
    return resid


# 1. Sample descriptive table
def sample_descriptive_table(df: pd.DataFrame) -> pd.DataFrame:
    """Sample demographics and outcome descriptives (SI Table S11)."""
    rows = []

    # Total N
    rows.append({"Variable": "N (total)", "Statistic": f"{len(df):,}"})

    # Cohort breakdown
    if "cohort" in df.columns:
        for cohort, n in df["cohort"].value_counts().sort_index().items():
            rows.append({"Variable": f"  {cohort}", "Statistic": f"{n:,} ({n/len(df)*100:.1f}%)"})

    # Continuous demographics
    for col, label in DEMO_CONTINUOUS:
        if col in df.columns:
            vals = df[col].dropna()
            rows.append({
                "Variable": label,
                "Statistic": f"{vals.mean():.1f} ({vals.std():.1f}), N={len(vals):,}",
            })

    # Binary demographics
    for col, label in DEMO_BINARY:
        if col in df.columns:
            vals = df[col].dropna()
            n_pos = int(vals.sum())
            rows.append({
                "Variable": label,
                "Statistic": f"{n_pos:,} ({n_pos/len(vals)*100:.1f}%), N={len(vals):,}",
            })

    # Race distribution
    if "race" in df.columns:
        rows.append({"Variable": "Race", "Statistic": ""})
        for race, n in df["race"].value_counts().sort_index().items():
            rows.append({"Variable": f"  {race}", "Statistic": f"{n:,} ({n/len(df)*100:.1f}%)"})

    # Education
    if "education" in df.columns:
        vals = df["education"].dropna()
        rows.append({
            "Variable": "Education (years)",
            "Statistic": f"{vals.mean():.1f} ({vals.std():.1f}), N={len(vals):,}",
        })

    rows.append({"Variable": "", "Statistic": ""})
    rows.append({"Variable": "--- Outcomes ---", "Statistic": ""})

    # Longitudinal outcomes: W1 and W2
    for w1, w2, label in LONGITUDINAL_OUTCOMES:
        for col, wave in [(w1, "W1"), (w2, "W2")]:
            if col in df.columns:
                vals = df[col].dropna()
                rows.append({
                    "Variable": f"{label} ({wave})",
                    "Statistic": f"{vals.mean():.3f} ({vals.std():.3f}), N={len(vals):,}",
                })

    # Cross-sectional outcomes
    for col, label in CS_OUTCOMES:
        if col in df.columns:
            vals = df[col].dropna()
            rows.append({
                "Variable": label,
                "Statistic": f"{vals.mean():.3f} ({vals.std():.3f}), N={len(vals):,}",
            })

    table = pd.DataFrame(rows)
    return table


# 2. Attrition analysis
def attrition_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Compare W1-to-W2 completers vs dropouts on key baseline variables."""
    results = []

    for w1, w2, label in LONGITUDINAL_OUTCOMES:
        if w1 not in df.columns or w2 not in df.columns:
            continue

        has_w1 = df[w1].notna()
        has_w2 = df[w2].notna()

        # Among those with W1 data, compare W2-completers vs W2-dropouts
        w1_sample = df[has_w1].copy()
        w1_sample["completer"] = w1_sample[w2].notna()
        n_comp = int(w1_sample["completer"].sum())
        n_drop = int((~w1_sample["completer"]).sum())

        row = {
            "Outcome Domain": label,
            "N_W1": int(has_w1.sum()),
            "N_W2": int((has_w1 & has_w2).sum()),
            "N_dropout": n_drop,
            "Retention%": f"{n_comp / len(w1_sample) * 100:.1f}",
        }

        # Compare baseline variables between completers and dropouts
        compare_vars = [
            ("age_W1", "Age"),
            ("female", "Female"),
            ("education", "Education"),
            (w1, f"{label}_W1"),
        ]

        for cvar, clabel in compare_vars:
            if cvar not in w1_sample.columns:
                continue
            comp_vals = w1_sample.loc[w1_sample["completer"], cvar].dropna()
            drop_vals = w1_sample.loc[~w1_sample["completer"], cvar].dropna()
            if len(comp_vals) < 5 or len(drop_vals) < 5:
                continue

            # Cohen's d
            pooled_sd = np.sqrt(
                ((len(comp_vals) - 1) * comp_vals.std()**2 + (len(drop_vals) - 1) * drop_vals.std()**2)
                / (len(comp_vals) + len(drop_vals) - 2)
            )
            d = (comp_vals.mean() - drop_vals.mean()) / pooled_sd if pooled_sd > 0 else 0

            # t-test
            t_stat, p_val = stats.ttest_ind(comp_vals, drop_vals, equal_var=False)

            row[f"{clabel}_d"] = f"{d:+.3f}"
            row[f"{clabel}_p"] = f"{p_val:.1e}" if p_val < 0.001 else f"{p_val:.3f}"

        results.append(row)

    return pd.DataFrame(results)


# 3. Outcome & change-score distributions
def plot_outcome_distributions(df: pd.DataFrame):
    """Density plots for W1 outcomes, W2 outcomes, and change scores."""

    # Panel A: W1 outcome distributions
    w1_outcomes = [(w1, label) for w1, _, label in LONGITUDINAL_OUTCOMES]
    w1_outcomes += [(col, label) for col, label in CS_OUTCOMES]

    n_out = len(w1_outcomes)
    fig, axes = plt.subplots(1, n_out, figsize=(4 * n_out, 4))
    if n_out == 1:
        axes = [axes]
    for i, (col, label) in enumerate(w1_outcomes):
        ax = axes[i]
        if col not in df.columns:
            ax.set_visible(False)
            continue
        vals = df[col].dropna()
        ax.hist(vals, bins=50, density=True, alpha=0.7, color="steelblue", edgecolor="none")
        ax.set_title(f"{label}\n(N={len(vals):,}, skew={vals.skew():.2f})", fontsize=10)
        ax.set_ylabel("Density")
    fig.suptitle("W1 / Cross-Sectional Outcome Distributions", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_outcome_distributions_w1.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_outcome_distributions_w1.png")

    # Panel B: Change score distributions
    reported = LONGITUDINAL_OUTCOMES
    n_long = len(reported)
    fig, axes = plt.subplots(1, n_long, figsize=(4 * n_long, 4))
    if n_long == 1:
        axes = [axes]
    for i, (w1, w2, label) in enumerate(reported):
        ax = axes[i]
        if w1 not in df.columns or w2 not in df.columns:
            ax.set_visible(False)
            continue
        mask = df[w1].notna() & df[w2].notna()
        change = df.loc[mask, w2].values - df.loc[mask, w1].values
        n = len(change)
        ax.hist(change, bins=50, density=True, alpha=0.7, color="coral", edgecolor="none")
        ax.axvline(0, color="black", ls="--", lw=0.8)
        ax.set_title(f"Δ{label}\n(N={n:,}, M={np.mean(change):.3f}, SD={np.std(change):.3f})",
                     fontsize=10)
        ax.set_ylabel("Density")
        ax.set_xlabel("W2 − W1")
    fig.suptitle("Change Score Distributions (W2 − W1)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_change_score_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_change_score_distributions.png")


# 4. Test-retest correlations
def test_retest_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """W1-W2 reliability for longitudinal outcomes."""
    rows = []
    for w1, w2, label in LONGITUDINAL_OUTCOMES:
        if w1 not in df.columns or w2 not in df.columns:
            continue
        mask = df[w1].notna() & df[w2].notna()
        v1 = df.loc[mask, w1].values
        v2 = df.loc[mask, w2].values
        n = int(mask.sum())
        r_pearson, p_pearson = stats.pearsonr(v1, v2)
        r_spearman, p_spearman = stats.spearmanr(v1, v2)

        # Also compute age-residualized test-retest
        if "age_W1" in df.columns and "age_W2" in df.columns:
            age1 = df.loc[mask, "age_W1"].values.astype(float)
            age2 = df.loc[mask, "age_W2"].values.astype(float)
            v1_res = age_residualize(v1.astype(float), age1)
            v2_res = age_residualize(v2.astype(float), age2)
            valid_res = ~np.isnan(v1_res) & ~np.isnan(v2_res)
            r_res, p_res = stats.pearsonr(v1_res[valid_res], v2_res[valid_res])
        else:
            r_res, p_res = np.nan, np.nan

        # SD of the raw change score on the same paired sample.
        change = v2.astype(float) - v1.astype(float)

        rows.append({
            "Outcome": label,
            "N": n,
            "r_Pearson": f"{r_pearson:.3f}",
            "r_Spearman": f"{r_spearman:.3f}",
            "r_age_residualized": f"{r_res:.3f}",
            "sd_change": f"{np.std(change, ddof=1):.3f}",
        })
    return pd.DataFrame(rows)


# 5. Zero-order outcome correlations (age-residualized)
def outcome_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Pearson correlations among age-residualized outcomes."""
    all_outcomes = [w1 for w1, _, _ in LONGITUDINAL_OUTCOMES] + [c for c, _ in CS_OUTCOMES]
    labels = [label for _, _, label in LONGITUDINAL_OUTCOMES] + [label for _, label in CS_OUTCOMES]

    # Residualize each against age_W1
    residualized = {}
    for col, label in zip(all_outcomes, labels):
        if col not in df.columns:
            continue
        vals = df[col].values.astype(float)
        if "age_W1" in df.columns:
            age = df["age_W1"].values.astype(float)
            residualized[label] = age_residualize(vals, age)
        else:
            residualized[label] = vals

    res_df = pd.DataFrame(residualized)

    # Pairwise correlations with N
    n_vars = len(res_df.columns)
    corr_vals = np.full((n_vars, n_vars), np.nan)
    n_vals = np.full((n_vars, n_vars), 0, dtype=int)
    cols = list(res_df.columns)

    for i in range(n_vars):
        for j in range(n_vars):
            mask = res_df[cols[i]].notna() & res_df[cols[j]].notna()
            n_pair = int(mask.sum())
            n_vals[i, j] = n_pair
            if n_pair > 5:
                r, _ = stats.pearsonr(res_df.loc[mask, cols[i]], res_df.loc[mask, cols[j]])
                corr_vals[i, j] = r

    corr_df = pd.DataFrame(corr_vals, index=cols, columns=cols)

    # Also save N matrix
    n_df = pd.DataFrame(n_vals, index=cols, columns=cols)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(9, 7))
    mask_upper = np.triu(np.ones_like(corr_vals, dtype=bool), k=1)

    # Plot lower triangle with values
    im = ax.imshow(np.where(~mask_upper, corr_vals, np.nan),
                   cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n_vars))
    ax.set_yticks(range(n_vars))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(cols, fontsize=9)

    # Annotate cells with r (lower triangle) and N (upper triangle)
    for i in range(n_vars):
        for j in range(n_vars):
            if i == j:
                ax.text(j, i, "1.0", ha="center", va="center", fontsize=8, color="black")
            elif i > j:
                val = corr_vals[i, j]
                if not np.isnan(val):
                    color = "white" if abs(val) > 0.5 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)
            else:
                ax.text(j, i, f"{n_vals[i, j]}", ha="center", va="center", fontsize=7,
                        color="gray")

    fig.colorbar(im, ax=ax, label="Pearson r", shrink=0.8)
    ax.set_title("Age-Residualized Outcome Correlations\n(lower: r, upper: N)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_outcome_correlations.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_outcome_correlations.png")

    return corr_df, n_df


# 6. Predictor test-retest reliability (T1+T2)
def _round_metric(value: float, digits: int = 3) -> float:
    """Round finite metric values and preserve missing values as NaN."""
    return round(float(value), digits) if pd.notna(value) and np.isfinite(value) else np.nan


def _pearson_value(x: pd.Series, y: pd.Series) -> float:
    """Pearson correlation for paired numeric series with constant-input guard."""
    a = pd.to_numeric(x, errors="coerce")
    b = pd.to_numeric(y, errors="coerce")
    valid = ~a.isna() & ~b.isna()
    if int(valid.sum()) < 10 or a[valid].nunique() < 2 or b[valid].nunique() < 2:
        return np.nan
    value, _ = stats.pearsonr(a[valid], b[valid])
    return value


def _spearman_value(x: pd.Series, y: pd.Series) -> float:
    """Spearman correlation for paired numeric series with constant-input guard."""
    a = pd.to_numeric(x, errors="coerce")
    b = pd.to_numeric(y, errors="coerce")
    valid = ~a.isna() & ~b.isna()
    if int(valid.sum()) < 10 or a[valid].nunique() < 2 or b[valid].nunique() < 2:
        return np.nan
    value, _ = stats.spearmanr(a[valid], b[valid])
    return value


def _cohens_kappa_value(x: pd.Series, y: pd.Series) -> float:
    """Cohen's kappa for paired categorical series with undefined-case guard."""
    s1 = x.astype(str)
    s2 = y.astype(str)
    labels = sorted(set(s1.unique()) | set(s2.unique()))
    if len(labels) < 2:
        return np.nan
    return cohen_kappa_score(s1, s2, labels=labels)


def _unique_label(values: pd.Series) -> str | float:
    """Compact unique-value label for source-table diagnostics."""
    nuniq = int(values.dropna().nunique())
    return ">5" if nuniq > 5 else str(nuniq)


def _complete_pair_values(
    df: pd.DataFrame,
    var: str,
    sib_types: list[str],
) -> pd.DataFrame:
    """Return one complete Wave 1 value pair per pair_id for a variable."""
    required = {"pair_id", "sib_type", var}
    if not required.issubset(df.columns):
        return pd.DataFrame(columns=["pair_id", "member_1", "member_2"])

    cols = ["pair_id", "sib_type", var]
    if "M2MRID" in df.columns:
        cols.append("M2MRID")
    mask = df["sib_type"].isin(sib_types) & df["pair_id"].notna()
    if "is_full_pair" in df.columns:
        mask &= df["is_full_pair"].fillna(False).astype(bool)

    sub = df.loc[mask, cols].dropna(subset=[var]).copy()
    if sub.empty:
        return pd.DataFrame(columns=["pair_id", "member_1", "member_2"])

    pair_counts = sub.groupby("pair_id").size()
    complete_pair_ids = pair_counts[pair_counts == 2].index
    sub = sub[sub["pair_id"].isin(complete_pair_ids)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["pair_id", "member_1", "member_2"])

    sort_cols = ["pair_id"] + (["M2MRID"] if "M2MRID" in sub.columns else [])
    sub = sub.sort_values(sort_cols)
    pairs = (
        sub.groupby("pair_id", sort=True)[var]
        .agg(member_1="first", member_2="last")
        .reset_index()
    )
    return pairs


def _family_resemblance_value(
    pairs: pd.DataFrame,
    measurement_type: str,
) -> tuple[str, float, float]:
    """Compute type-appropriate pair resemblance and exact agreement."""
    if pairs.empty:
        metric = "Pearson r" if measurement_type == "continuous" else (
            "Spearman rho" if measurement_type == "ordinal" else "Cohen's kappa"
        )
        return metric, np.nan, np.nan

    x1 = pairs["member_1"]
    x2 = pairs["member_2"]

    if measurement_type == "continuous":
        value = _pearson_value(x1, x2)
        return "Pearson r", _round_metric(value), np.nan

    if measurement_type == "ordinal":
        value = _spearman_value(x1, x2)
        agreement = (x1.astype(str).values == x2.astype(str).values).mean() * 100
        return "Spearman rho", _round_metric(value), round(float(agreement), 1)

    s1 = x1.astype(str)
    s2 = x2.astype(str)
    value = _cohens_kappa_value(s1, s2)
    agreement = (s1.values == s2.values).mean() * 100
    return "Cohen's kappa", _round_metric(value), round(float(agreement), 1)


def _family_resemblance_rows(
    df: pd.DataFrame,
    variables: list[dict[str, str]],
) -> pd.DataFrame:
    """Build Wave 1 sibling/twin resemblance rows for outcome or predictor variables."""
    rows = []
    for spec in variables:
        var = spec["Variable"]
        if var not in df.columns:
            continue
        measurement_type = spec["Type"]
        unique_label = _unique_label(df[var]) if var in df.columns else np.nan

        for stratum, sib_types in FAMILY_STRATA.items():
            pairs = _complete_pair_values(df, var, sib_types)
            n_pairs = len(pairs)
            if n_pairs < 10:
                metric, value, agreement = _family_resemblance_value(
                    pairs.iloc[0:0],
                    measurement_type,
                )
            else:
                metric, value, agreement = _family_resemblance_value(
                    pairs,
                    measurement_type,
                )

            row = {
                "Variable": var,
                "Display label": spec.get("Display label", display_name(var)),
                "Wave": spec.get("Wave", "Wave 1"),
                "Type": measurement_type,
                "Sibling stratum": stratum,
                "N_pairs": n_pairs,
                "Metric": metric,
                "Value": value,
                "Pct_Agreement": agreement,
                "Unique_var_n": unique_label,
            }
            if "Outcome" in spec:
                row = {"Outcome": spec["Outcome"], **row}
            rows.append(row)

    return pd.DataFrame(rows)


def outcome_family_resemblance_w1(df: pd.DataFrame) -> pd.DataFrame:
    """Wave 1 twin/sibling resemblance for outcome variables."""
    variables = [
        {
            "Outcome": label,
            "Variable": w1,
            "Display label": label,
            "Wave": "Wave 1",
            "Type": predictor_measurement_type(w1, df),
        }
        for w1, _, label in LONGITUDINAL_OUTCOMES
    ]
    variables.extend(
        {
            "Outcome": label,
            "Variable": col,
            "Display label": label,
            "Wave": "Cross-sectional",
            "Type": predictor_measurement_type(col, df),
        }
        for col, label in CS_OUTCOMES
    )
    return _family_resemblance_rows(df, variables)


def predictor_family_resemblance_w1(df: pd.DataFrame) -> pd.DataFrame:
    """Wave 1 twin/sibling resemblance for Tier 1 + Tier 2 predictors."""
    predictor_vars = [v for v in get_tier_variables([1, 2], exclude_high_missing=True) if v in df.columns]
    variables = [
        {
            "Variable": var,
            "Display label": display_name(var),
            "Wave": "Wave 1",
            "Type": predictor_measurement_type(var, df),
        }
        for var in predictor_vars
    ]
    return _family_resemblance_rows(df, variables)


def predictor_test_retest(df_w1: pd.DataFrame, df_w2: pd.DataFrame) -> pd.DataFrame:
    """Compute W1-W2 test-retest reliability for all T1+T2 predictor variables.

    Continuous variables: Pearson r.
    Ordinal variables: Spearman rho + percent agreement.
    Binary/nominal variables: Cohen's kappa + percent agreement.
    """
    predictor_vars = get_tier_variables([1, 2], exclude_high_missing=True)

    # Merge on M2MRID; suffixes mark the wave
    merged = df_w1[["M2MRID"] + [v for v in predictor_vars if v in df_w1.columns]].merge(
        df_w2[["M2MRID"] + [v for v in predictor_vars if v in df_w2.columns]],
        on="M2MRID", suffixes=("_w1", "_w2"),
    )
    print(f"  Predictor reliability: {len(merged):,} persons with both waves")

    rows = []
    for var in predictor_vars:
        c1, c2 = f"{var}_w1", f"{var}_w2"
        if c1 not in merged.columns or c2 not in merged.columns:
            continue

        # Unique value count from overall W2 data
        if var in df_w2.columns:
            nuniq = int(df_w2[var].dropna().nunique())
            uniq_label = ">5" if nuniq > 5 else str(nuniq)
        else:
            uniq_label = np.nan

        type_label = predictor_measurement_type(var, df_w2)

        if type_label in ("binary", "nominal"):
            # Drop rows where either wave is NaN
            valid = merged[[c1, c2]].dropna()
            n = len(valid)
            if n < 10:
                rows.append({"Variable": var, "Type": type_label, "N": n,
                             "Metric": "Cohen's kappa", "Value": np.nan,
                             "Pct_Agreement": np.nan,
                             "Unique_var_n": uniq_label})
                continue
            s1 = valid[c1].astype(str)
            s2 = valid[c2].astype(str)
            kappa = _cohens_kappa_value(s1, s2)
            pct_agree = (s1.values == s2.values).mean() * 100
            rows.append({"Variable": var, "Type": type_label, "N": n,
                         "Metric": "Cohen's kappa", "Value": _round_metric(kappa),
                         "Pct_Agreement": round(pct_agree, 1),
                         "Unique_var_n": uniq_label})
        elif type_label == "ordinal":
            valid = merged[[c1, c2]].dropna()
            n = len(valid)
            if n < 10:
                rows.append({"Variable": var, "Type": type_label, "N": n,
                             "Metric": "Spearman rho", "Value": np.nan,
                             "Pct_Agreement": np.nan,
                             "Unique_var_n": uniq_label})
                continue
            rho = _spearman_value(valid[c1], valid[c2])
            s1 = valid[c1].astype(str)
            s2 = valid[c2].astype(str)
            pct_agree = (s1.values == s2.values).mean() * 100
            rows.append({"Variable": var, "Type": type_label, "N": n,
                         "Metric": "Spearman rho", "Value": _round_metric(rho),
                         "Pct_Agreement": round(pct_agree, 1),
                         "Unique_var_n": uniq_label})
        else:
            v1 = pd.to_numeric(merged[c1], errors="coerce")
            v2 = pd.to_numeric(merged[c2], errors="coerce")
            valid = ~v1.isna() & ~v2.isna()
            n = int(valid.sum())
            if n < 10:
                rows.append({"Variable": var, "Type": "continuous", "N": n,
                             "Metric": "Pearson r", "Value": np.nan,
                             "Unique_var_n": uniq_label})
                continue
            r_val = _pearson_value(v1[valid], v2[valid])
            rows.append({"Variable": var, "Type": "continuous", "N": n,
                         "Metric": "Pearson r", "Value": _round_metric(r_val),
                         "Unique_var_n": uniq_label})

    result = pd.DataFrame(rows)
    return result


# 6b. Reliability diagnostics: flag preprocess/coding vs floor/ceiling issues
def diagnose_reliability(
    pred_retest: pd.DataFrame,
    df_w1: pd.DataFrame,
    df_w2: pd.DataFrame,
) -> pd.DataFrame:
    """Add flag and diagnosis columns to predictor reliability table.

    flag = 1.0: probable preprocess / cross-wave coding issue
    flag = 0.5: floor/ceiling artifact (reliability mechanically depressed)
    flag = NaN: no issue detected (or variable has adequate reliability)

    Runs diagnostics on all variables, including those with adequate reliability.
    """
    predictor_vars = get_tier_variables([1, 2], exclude_high_missing=True)

    merged = df_w1[["M2MRID"] + [v for v in predictor_vars if v in df_w1.columns]].merge(
        df_w2[["M2MRID"] + [v for v in predictor_vars if v in df_w2.columns]],
        on="M2MRID", suffixes=("_w1", "_w2"),
    )

    flags = {}  # var: (flag, diagnosis)

    for _, row in pred_retest.iterrows():
        var = row["Variable"]
        vtype = row["Type"]
        c1, c2 = f"{var}_w1", f"{var}_w2"
        if c1 not in merged.columns or c2 not in merged.columns:
            continue

        # Discrete diagnostics
        if vtype in ("ordinal", "nominal", "binary"):
            w1_vals = df_w1[var].dropna() if var in df_w1.columns else pd.Series(dtype=float)
            w2_vals = df_w2[var].dropna() if var in df_w2.columns else pd.Series(dtype=float)
            if len(w1_vals) < 10 or len(w2_vals) < 10:
                continue

            u1 = set(w1_vals.astype(str).unique())
            u2 = set(w2_vals.astype(str).unique())
            n_u1, n_u2 = len(u1), len(u2)

            # Category count change
            lost = u1 - u2
            gained = u2 - u1

            # Distribution shift: for each shared category, compare proportions
            vc1 = w1_vals.astype(str).value_counts(normalize=True)
            vc2 = w2_vals.astype(str).value_counts(normalize=True)
            shared = u1 & u2
            max_prop_shift = 0.0
            shift_detail = ""
            for cat in shared:
                p1 = vc1.get(cat, 0)
                p2 = vc2.get(cat, 0)
                shift = abs(p2 - p1)
                if shift > max_prop_shift:
                    max_prop_shift = shift
                    shift_detail = f"cat {cat}: {p1:.0%}→{p2:.0%}"

            # Dominant category (floor equivalent)
            max_p1 = vc1.max() if len(vc1) > 0 else 0
            max_p2 = vc2.max() if len(vc2) > 0 else 0

            # For binary vars, check if shift is predominantly one-directional
            # (plausible state transition, e.g. aging into Medicare) vs bidirectional
            is_unidirectional_shift = False
            if vtype == "binary" and max_prop_shift > 0.20:
                valid_pair = merged[[c1, c2]].dropna()
                s1p = valid_pair[c1].astype(str)
                s2p = valid_pair[c2].astype(str)
                cats = sorted(set(s1p.unique()) | set(s2p.unique()))
                if len(cats) == 2:
                    a, b_ = cats  # two category labels
                    n_ab = int(((s1p == a) & (s2p == b_)).sum())  # a to b
                    n_ba = int(((s1p == b_) & (s2p == a)).sum())  # b to a
                    total_change = n_ab + n_ba
                    if total_change > 0:
                        minority_dir = min(n_ab, n_ba) / total_change
                        is_unidirectional_shift = minority_dir < 0.10

            # For multi-category vars: detect systematic category remapping
            # (large fraction of one W1 category maps to a single different W2 category)
            remap_detail = ""
            has_remap = False
            if vtype in ("ordinal", "nominal") and n_u1 > 4:
                valid_pair = merged[[c1, c2]].dropna()
                sp1 = valid_pair[c1].astype(str)
                sp2 = valid_pair[c2].astype(str)
                for cat_from in sorted(shared):
                    mask_from = sp1 == cat_from
                    n_from = int(mask_from.sum())
                    if n_from < 100:  # skip small categories (noisy %)
                        continue
                    dest = sp2[mask_from]
                    dest_counts = dest.value_counts()
                    # Exclude staying in same category
                    off_diag = dest_counts.drop(cat_from, errors="ignore")
                    if len(off_diag) == 0:
                        continue
                    top_dest = off_diag.index[0]
                    top_n = int(off_diag.iloc[0])
                    # >50% of a sizeable category remapped to one other category
                    if top_n / n_from > 0.50:
                        has_remap = True
                        remap_detail = (f"cat {cat_from}→{top_dest}: "
                                        f"{top_n}/{n_from} ({top_n/n_from:.0%})")
                        break  # one example suffices

            # Decision logic
            if lost or gained or (n_u1 != n_u2):
                parts = []
                # Sorted so the string is stable across runs.
                if lost:
                    parts.append(f"W1-only cats: {{{', '.join(repr(c) for c in sorted(lost, key=str))}}}")
                if gained:
                    parts.append(f"W2-only cats: {{{', '.join(repr(c) for c in sorted(gained, key=str))}}}")
                if max_prop_shift > 0.20:
                    parts.append(shift_detail)
                flags[var] = (1.0, f"category set changed ({n_u1}→{n_u2}); {'; '.join(parts)}")
            elif max_prop_shift > 0.20 and not is_unidirectional_shift:
                flags[var] = (1.0, f"large distribution shift; {shift_detail}")
            elif has_remap:
                flags[var] = (1.0, f"systematic category remap; {remap_detail}")
            elif max_p1 > 0.90 or max_p2 > 0.90:
                dom_cat = vc1.idxmax() if max_p1 >= max_p2 else vc2.idxmax()
                flags[var] = (0.5, f"floor/ceiling: dominant cat '{dom_cat}' ≥90%")

        # Continuous diagnostics
        else:
            v1 = pd.to_numeric(merged[c1], errors="coerce")
            v2 = pd.to_numeric(merged[c2], errors="coerce")
            valid = ~v1.isna() & ~v2.isna()
            n = int(valid.sum())
            if n < 20:
                continue
            a, b = v1[valid].values, v2[valid].values

            n_u1 = len(np.unique(a))
            n_u2 = len(np.unique(b))
            sd1, sd2 = a.std(), b.std()
            sd_ratio = sd2 / sd1 if sd1 > 0 else np.nan

            # Range
            rng1 = (a.min(), a.max())
            rng2 = (b.min(), b.max())

            # Floor / ceiling
            floor1 = np.mean(a == a.min()) * 100
            floor2 = np.mean(b == b.min()) * 100
            ceil1 = np.mean(a == a.max()) * 100
            ceil2 = np.mean(b == b.max()) * 100

            # Decision logic
            reasons = []

            # Unique-value collapse (strong preprocess signal)
            if n_u1 >= 5 and n_u2 <= 2:
                reasons.append(f"unique vals collapsed ({n_u1}→{n_u2})")
            elif n_u2 >= 5 and n_u1 <= 2:
                reasons.append(f"unique vals collapsed ({n_u1}→{n_u2})")
            elif min(n_u1, n_u2) > 2 and max(n_u1, n_u2) / min(n_u1, n_u2) > 3:
                reasons.append(f"unique count ratio {n_u1}→{n_u2}")

            # SD ratio far from 1
            if sd_ratio is not np.nan and (sd_ratio < 0.5 or sd_ratio > 2.0):
                reasons.append(f"SD ratio={sd_ratio:.2f}")

            # Range change
            rng_span1 = rng1[1] - rng1[0]
            rng_span2 = rng2[1] - rng2[0]
            if rng_span1 > 0 and rng_span2 > 0:
                span_ratio = rng_span2 / rng_span1
                if span_ratio < 0.33 or span_ratio > 3.0:
                    reasons.append(
                        f"range [{rng1[0]:.1f},{rng1[1]:.1f}]→[{rng2[0]:.1f},{rng2[1]:.1f}]"
                    )

            if reasons:
                flags[var] = (1.0, "; ".join(reasons))
                continue

            # Floor/ceiling check (no preprocess issue found above)
            fc_reasons = []
            if floor1 > 80 or floor2 > 80:
                fc_reasons.append(f"floor {floor1:.0f}%/{floor2:.0f}%")
            if ceil1 > 80 or ceil2 > 80:
                fc_reasons.append(f"ceiling {ceil1:.0f}%/{ceil2:.0f}%")
            if fc_reasons:
                flags[var] = (0.5, "; ".join(fc_reasons))

    # Merge into table
    pred_retest = pred_retest.copy()
    pred_retest["Flag"] = pred_retest["Variable"].map(
        lambda v: flags[v][0] if v in flags else np.nan
    )
    pred_retest["Diagnosis"] = pred_retest["Variable"].map(
        lambda v: flags[v][1] if v in flags and flags[v][0] == 1.0 else ""
    )
    return pred_retest


def reliability_by_domain(pred_retest: pd.DataFrame) -> pd.DataFrame:
    """Median predictor test-retest reliability per domain (Table S1's subdomain).

    Metric-mixing rule: `Value` is the metric fitted to each predictor's
    measurement type, so a domain median pools metrics. Continuous predictors
    contribute Pearson r, ordinal ones Spearman rho, and binary and nominal ones
    Cohen's kappa; none is converted to another. The per-metric counts say how
    much of each domain's median rests on which metric. Tier 1 predictors form
    their own domain (`tier_1`).
    """
    df = pred_retest.assign(Domain=pred_retest["Variable"].map(get_subdomain))
    missing = sorted(df.loc[df["Domain"].eq(""), "Variable"])
    if missing:
        raise ValueError(f"predictors without a subdomain in variable_config.yaml: {missing}")
    counts = pd.crosstab(df["Domain"], df["Metric"])
    out = (
        df.groupby("Domain")["Value"]
        .agg(n_predictors="size", median_reliability="median")
        .join(pd.DataFrame({
            "n_pearson_r": counts.get("Pearson r", 0),
            "n_spearman_rho": counts.get("Spearman rho", 0),
            "n_cohens_kappa": counts.get("Cohen's kappa", 0),
        }))
        .reset_index()
        .rename(columns={"Domain": "domain"})
        .sort_values("median_reliability", ascending=False, ignore_index=True)
    )
    out["median_reliability"] = out["median_reliability"].round(4)
    return out


# Main
def main():
    # Record the inputs this output was built from.
    provenance.stamp_at_exit(OUTPUT_DIR, "d1_descriptive", ["data/processed/master_analysis.csv", "data/processed/master_analysis_concurrent.csv"])
    print("D1: Descriptive Analysis")

    df = load_data()

    # 1. Sample descriptive table
    print("\n--- 1. Sample Descriptive Table ---")
    table1 = sample_descriptive_table(df)
    table1.to_csv(OUTPUT_DIR / "sample_descriptives.csv", index=False)
    print(table1.to_string(index=False))

    # 2. Attrition analysis
    print("\n--- 2. Attrition Analysis ---")
    attrition = attrition_analysis(df)
    attrition.to_csv(OUTPUT_DIR / "attrition.csv", index=False)
    print(attrition.to_string(index=False))

    # 3. Distributions
    print("\n--- 3. Outcome & Change-Score Distributions ---")
    plot_outcome_distributions(df)

    # 4. Test-retest
    print("\n--- 4. Test-Retest Correlations ---")
    retest = test_retest_correlations(df)
    retest.to_csv(OUTPUT_DIR / "outcome_test_retest.csv", index=False)
    print(retest.to_string(index=False))

    # 5. Zero-order correlations
    print("\n--- 5. Outcome Correlations (age-residualized) ---")
    corr_df, n_df = outcome_correlation_matrix(df)
    corr_df.to_csv(OUTPUT_DIR / "outcome_correlations.csv")
    n_df.to_csv(OUTPUT_DIR / "outcome_correlations_n.csv")
    print(corr_df.round(3).to_string())

    # 6. Predictor test-retest reliability
    print("\n--- 6. Predictor Test-Retest Reliability (T1+T2) ---")
    df_w2 = pd.read_csv(CONCURRENT_CSV, low_memory=False)
    pred_retest = predictor_test_retest(df, df_w2)

    # 6b. Diagnose reliability issues
    print("\n--- 6b. Reliability Diagnostics ---")
    pred_retest = diagnose_reliability(pred_retest, df, df_w2)
    pred_retest.to_csv(OUTPUT_DIR / "predictor_reliability.csv", index=False)
    by_domain = reliability_by_domain(pred_retest)
    by_domain.to_csv(OUTPUT_DIR / "predictor_reliability_by_domain.csv", index=False)
    print(by_domain.to_string(index=False))

    n_flag1 = int((pred_retest["Flag"] == 1.0).sum())
    n_flag05 = int((pred_retest["Flag"] == 0.5).sum())
    print(f"  Flag=1 (preprocess/coding): {n_flag1}")
    if n_flag1 > 0:
        for _, r in pred_retest[pred_retest["Flag"] == 1.0].iterrows():
            print(f"    {r['Variable']}: {r['Diagnosis']}")
    print(f"  Flag=0.5 (floor/ceiling): {n_flag05}")
    if n_flag05 > 0:
        for _, r in pred_retest[pred_retest["Flag"] == 0.5].iterrows():
            print(f"    {r['Variable']}")

    # Summary stats
    cont = pred_retest[pred_retest["Type"] == "continuous"]
    ordinal = pred_retest[pred_retest["Type"] == "ordinal"]
    cat = pred_retest[pred_retest["Type"].isin(["binary", "nominal"])]
    print(f"  Continuous variables: {len(cont)}")
    if len(cont) > 0:
        print(f"    Pearson r: median={cont['Value'].median():.3f}, "
              f"range=[{cont['Value'].min():.3f}, {cont['Value'].max():.3f}]")
    print(f"  Ordinal variables: {len(ordinal)}")
    if len(ordinal) > 0:
        print(f"    Spearman rho: median={ordinal['Value'].median():.3f}, "
              f"range=[{ordinal['Value'].min():.3f}, {ordinal['Value'].max():.3f}]")
        print(f"    % Agreement: median={ordinal['Pct_Agreement'].median():.1f}, "
              f"range=[{ordinal['Pct_Agreement'].min():.1f}, {ordinal['Pct_Agreement'].max():.1f}]")
    print(f"  Binary/nominal variables: {len(cat)}")
    if len(cat) > 0:
        print(f"    Cohen's kappa: median={cat['Value'].median():.3f}, "
              f"range=[{cat['Value'].min():.3f}, {cat['Value'].max():.3f}]")
        print(f"    % Agreement: median={cat['Pct_Agreement'].median():.1f}, "
              f"range=[{cat['Pct_Agreement'].min():.1f}, {cat['Pct_Agreement'].max():.1f}]")
    # Flag low-reliability variables
    low_r = cont[cont["Value"] < 0.5]
    if len(low_r) > 0:
        print(f"  Low reliability (r < .50): {len(low_r)} variables")
        for _, row in low_r.iterrows():
            print(f"    {row['Variable']}: r={row['Value']}, N={row['N']}")

    # 7. Wave 1 sibling/twin family resemblance
    print("\n--- 7. Wave 1 Family Resemblance ---")
    outcome_family = outcome_family_resemblance_w1(df)
    outcome_family.to_csv(OUTPUT_DIR / "outcome_family_resemblance_w1.csv", index=False)
    predictor_family = predictor_family_resemblance_w1(df)
    predictor_family.to_csv(OUTPUT_DIR / "predictor_family_resemblance_w1.csv", index=False)
    print(f"  Outcome family rows: {len(outcome_family):,}")
    print(f"  Predictor family rows: {len(predictor_family):,}")

    print(f"\nAll outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
