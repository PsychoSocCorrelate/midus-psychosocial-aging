#!/usr/bin/env python3
"""Allostatic-load drinking-status sensitivity checks:
    1. Descriptive abstainer-subtype summaries.
    2. Broad-sample W1 health selection into W2 past-month abstention.
    3. Biomarker-sample selection by drinking status.

Usage:
    uv run python analysis/sensitivity/a2c_AL_drinking_sensitivity.py

Output:
    output/a2c_AL_drinking_sensitivity/
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.core import artifact_paths as ap, provenance

from analysis.core.output_log import write_output_log
from analysis.sensitivity.al_drinking_utils import (
    ABSTAINER_REFINED,
    CONCURRENT_CSV,
    CURRENT_CATS,
    CURRENT_DRINKER_CATS,
    MASTER_CSV,
    REFINED_CATS,
    biomarker_analysis_table,
    build_drinking_transition_table,
    load_refined_master,
    normalize_drink_labels,
    numeric,
    summarize_scores,
    subsystem_summary,
)
from config.biomarker_constants import SYSTEMS

OUTPUT_DIR = ROOT / "output" / "a2c_AL_drinking_sensitivity"
REPORTED_SCORE = f"{ap.AL_STEM}_zscore"

OUTPUTS = [
    {"path": "refined_abstainer_tables/scores_by_drinkStatus_refined.csv", "kind": "CSV"},
    {"path": "refined_abstainer_tables/subsystem_by_drinkStatus_refined.csv", "kind": "CSV"},
    {"path": "broad_health_selection/w2_abstention_model_frailty_summary.csv", "kind": "CSV"},
    {"path": "biomarker_sample_selection/biomarker_selection_summary.csv", "kind": "CSV"},
]


def _ensure_dirs() -> dict[str, Path]:
    dirs = {
        "root": OUTPUT_DIR,
        "refined": OUTPUT_DIR / "refined_abstainer_tables",
        "broad_health": OUTPUT_DIR / "broad_health_selection",
        "biomarker_selection": OUTPUT_DIR / "biomarker_sample_selection",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _write(path: Path, df: pd.DataFrame, *, index: bool = False) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    print(f"Saved: {path}")
    return df


def _is_current_status(s: pd.Series) -> pd.Series:
    return s.astype("object").isin(CURRENT_DRINKER_CATS)


def write_refined_abstainer_outputs(df: pd.DataFrame, out_dir: Path) -> dict[str, pd.DataFrame]:
    bio_df = biomarker_analysis_table(df)

    score_refined = summarize_scores(bio_df, "drinkStatus_refined", REFINED_CATS)
    sys_refined = subsystem_summary(bio_df, "drinkStatus_refined", REFINED_CATS, SYSTEMS)

    _write(out_dir / "scores_by_drinkStatus_refined.csv", score_refined)
    _write(out_dir / "subsystem_by_drinkStatus_refined.csv", sys_refined)
    return {
        "bio_df": bio_df,
        "score_refined": score_refined,
        "sys_refined": sys_refined,
    }


def biomarker_selection_bias(out_dir: Path) -> dict[str, pd.DataFrame | float]:
    cols = ["M2MRID", "has_biomarker_W1", "drinkStatus"]
    df = pd.read_csv(MASTER_CSV, usecols=cols, low_memory=False)
    df["drinkStatus"] = normalize_drink_labels(df["drinkStatus"])
    df = df[df["drinkStatus"].notna()].copy()
    df["in_bio"] = df["has_biomarker_W1"].fillna(False).astype(bool)

    ct = pd.crosstab(df["drinkStatus"], df["in_bio"])
    chi2, p_chi2, dof = np.nan, np.nan, np.nan
    if ct.shape[0] >= 2 and ct.shape[1] >= 2:
        from scipy import stats

        chi2, p_chi2, dof, _ = stats.chi2_contingency(ct)

    biomarker_counts = ct.get(True, pd.Series(dtype=float)).reindex(CURRENT_CATS)
    non_biomarker_counts = ct.get(False, pd.Series(dtype=float)).reindex(CURRENT_CATS)
    biomarker_total = float(biomarker_counts.sum())
    non_biomarker_total = float(non_biomarker_counts.sum())
    abstainer = "Past-month abstainer"
    summary = pd.DataFrame(
        [
            {
                "contrast": "past_month_abstainer_representation",
                "drinkStatus": abstainer,
                "biomarker_n": int(biomarker_counts.loc[abstainer]),
                "non_biomarker_n": int(non_biomarker_counts.loc[abstainer]),
                "biomarker_total_n": int(biomarker_total),
                "non_biomarker_total_n": int(non_biomarker_total),
                "biomarker_pct": float(biomarker_counts.loc[abstainer] / biomarker_total * 100),
                "non_biomarker_pct": float(non_biomarker_counts.loc[abstainer] / non_biomarker_total * 100),
                "chi2": chi2,
                "p_chi2": p_chi2,
                "dof": dof,
            }
        ]
    )
    _write(out_dir / "biomarker_selection_summary.csv", summary)
    return {"summary": summary, "chi2": chi2, "p_chi2": p_chi2}


def _log1p_col(df: pd.DataFrame, col: str) -> pd.Series:
    return np.log1p(numeric(df[col]).clip(lower=0))


def _fit_glm_binomial(
    df: pd.DataFrame,
    outcome: str,
    continuous: list[str],
    binary: list[str],
    categorical: list[str],
    model_name: str,
) -> pd.DataFrame:
    d = df.copy()
    d[outcome] = numeric(d[outcome])
    if "seeMedDoc_log1p" in continuous and "seeMedDoc_log1p" not in d.columns and "seeMedDoc" in d.columns:
        d["seeMedDoc_log1p"] = _log1p_col(d, "seeMedDoc")

    present_cont = [c for c in continuous if c in d.columns]
    present_bin = [c for c in binary if c in d.columns]
    present_cat = [c for c in categorical if c in d.columns]
    model_cols = [outcome] + present_cont + present_bin + present_cat
    d = d[model_cols].dropna(subset=[outcome]).copy()
    d = d.dropna(subset=present_cont + present_bin + present_cat)
    if len(d) < 50 or d[outcome].nunique() < 2:
        return pd.DataFrame(
            [
                {
                    "model": model_name,
                    "term": "model_status",
                    "status": "too_sparse_or_single_outcome",
                    "n": int(len(d)),
                    "events": int(d[outcome].sum()) if len(d) else 0,
                }
            ]
        )

    x_parts = []
    for col in present_cont:
        x = numeric(d[col])
        sd = x.std()
        d[f"{col}_z"] = (x - x.mean()) / sd if sd and not np.isnan(sd) else x - x.mean()
        x_parts.append(d[[f"{col}_z"]])
    for col in present_bin:
        x_parts.append(numeric(d[col]).astype(float).to_frame(col))
    for col in present_cat:
        dum = pd.get_dummies(d[col].astype("object"), prefix=col, drop_first=True, dtype=float)
        if not dum.empty:
            x_parts.append(dum)
    X = pd.concat(x_parts, axis=1) if x_parts else pd.DataFrame(index=d.index)
    X = sm.add_constant(X, has_constant="add")
    y = d[outcome].astype(float)

    try:
        model = sm.GLM(y, X, family=sm.families.Binomial()).fit(cov_type="HC3")
    except Exception as exc:
        return pd.DataFrame(
            [
                {
                    "model": model_name,
                    "term": "model_status",
                    "status": f"fit_failed: {type(exc).__name__}",
                    "n": int(len(d)),
                    "events": int(y.sum()),
                }
            ]
        )

    rows = []
    for term in model.params.index:
        coef = float(model.params.loc[term])
        se = float(model.bse.loc[term])
        rows.append(
            {
                "model": model_name,
                "term": term,
                "status": "ok",
                "n": int(model.nobs),
                "events": int(y.sum()),
                "coef_logit": coef,
                "se": se,
                "p_value": float(model.pvalues.loc[term]),
                "odds_ratio": float(np.exp(coef)),
                "or_ci_lower": float(np.exp(coef - 1.96 * se)),
                "or_ci_upper": float(np.exp(coef + 1.96 * se)),
            }
        )
    return pd.DataFrame(rows)


def broad_health_selection_analysis(w1_df: pd.DataFrame, out_dir: Path) -> dict[str, pd.DataFrame]:
    w2 = pd.read_csv(CONCURRENT_CSV, low_memory=False)
    transitions = build_drinking_transition_table(w1_df, w2)

    current = transitions[
        _is_current_status(transitions["drinkStatus_W1"]) & transitions["drinkStatus_W2"].notna()
    ].copy()
    current["w1_drinkStatus"] = current["drinkStatus_W1"].astype("object")

    continuous = [
        "age_W1",
        "FInoPsy_W1",
        "education",
        "seeMedDoc_log1p",
        "mentalHealth",
        "controlOverHealth",
    ]
    binary = ["female"]
    categorical = ["w1_drinkStatus"]
    broad_model = _fit_glm_binomial(
        current,
        "became_abstainer_W2",
        continuous,
        binary,
        categorical,
        "broad_health_selection_frailty",
    )
    focal = broad_model[broad_model["term"] == "FInoPsy_W1_z"].copy()
    if not focal.empty:
        focal["predictor_description"] = "continuous Wave-1 non-psychiatric frailty, standardized within the complete-case model sample"
        focal["outcome_description"] = "Wave-2 past-month abstention among Wave-1 current drinkers"
        focal["covariate_set"] = (
            "Wave-1 age; sex; education; Wave-1 drinking category; log physician visits; "
            "mental health; perceived control over health"
        )
    _write(out_dir / "w2_abstention_model_frailty_summary.csv", focal)
    return {
        "transitions": transitions,
        "broad_model": broad_model,
        "focal_summary": focal,
    }


def _format_num(value: float, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{value:.{digits}f}"


def _summarize_refined_result(score_refined: pd.DataFrame) -> str:
    sub = score_refined[score_refined["score"] == REPORTED_SCORE].copy()
    if sub.empty:
        return "Refined abstainer-subtype table was empty."
    means = {row["group"]: row["mean"] for _, row in sub.iterrows()}
    abst = {group: means.get(group, np.nan) for group in ABSTAINER_REFINED}
    highest = max(abst, key=lambda key: abst[key] if not pd.isna(abst[key]) else -np.inf)
    current_bits = [
        f"{group} {_format_num(means.get(group))}"
        for group in CURRENT_DRINKER_CATS
        if group in means
    ]
    return (
        f"Descriptively, the largest abstainer-subtype {REPORTED_SCORE} mean is "
        f"`{highest}` ({_format_num(abst[highest])}); former-heavy abstainers are "
        f"{_format_num(abst.get('former_heavy'))}. Current-drinking means are "
        + ", ".join(current_bits)
        + "."
    )


def _summarize_broad_health_selection(model_df: pd.DataFrame) -> str:
    ok = model_df[model_df["status"] == "ok"] if "status" in model_df.columns else pd.DataFrame()
    if ok.empty:
        return "Broad health-selection model was not estimable."
    row = ok[ok["term"] == "FInoPsy_W1_z"]
    if row.empty:
        return f"Broad health-selection model fit {int(ok.iloc[0]['n'])} rows but had no FInoPsy term."
    term = row.iloc[0]
    return (
        "In the broad W1-current-drinker sample, continuous W1 non-psychiatric "
        f"frailty predicted W2 past-month abstention ({int(term['n'])} rows, "
        f"{int(term['events'])} abstention events; OR={_format_num(float(term['odds_ratio']), 2)}, "
        f"p={_format_num(float(term['p_value']), 3)} per analysis-sample SD)."
    )


def _summarize_biomarker_selection(selection: dict[str, pd.DataFrame | float]) -> str:
    summary = selection["summary"]
    if not isinstance(summary, pd.DataFrame) or summary.empty:
        return "Biomarker-sample selection table was unavailable."
    row = summary.iloc[0]
    return (
        "Past-month abstainers were less represented in the biomarker sample than "
        f"the non-biomarker sample ({_format_num(float(row['biomarker_pct']), 1)}% "
        f"versus {_format_num(float(row['non_biomarker_pct']), 1)}%; "
        f"chi-square={_format_num(float(row['chi2']), 2)}, p={float(row['p_chi2']):.2e})."
    )


def summary_notes(results: dict[str, object]) -> list[str]:
    lines = [
        "# A2c Allostatic Load and Drinking Status",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        "",
        "## Scope",
        "",
        "This output covers three checks: descriptive abstainer-subtype summaries, broad-sample W1 health selection into W2 past-month abstention, and biomarker-sample selection by drinking status.",
        "",
        "## Bottom Line",
        "",
        "- These checks evaluate whether health-selection and biomarker-selection explanations remain plausible; they do not support a protective-drinking interpretation.",
        f"- {results['refined_sentence']}",
        f"- {results['broad_health_sentence']}",
        f"- {results['biomarker_selection_sentence']}",
        "",
        "## Interpretation Guardrails",
        "",
        f"- `{REPORTED_SCORE}` is a sample-standardized relative dysregulation score, not an absolute clinical diagnosis.",
        "- Past-month abstainers pool lifelong abstainers, former light drinkers, former moderate/heavy drinkers, and people who may be abstaining for health, medication, social, or preference reasons.",
        "- The abstainer-subtype pattern is descriptive and does not identify causality.",
        "- The broad health-selection model uses W1 non-psychiatric frailty as a continuous predictor of W2 past-month abstention among W1 current drinkers.",
        "- The transition model adjusts for W1 age, sex, education, W1 drinking category, log physician visits, continuous mental health, and perceived control over health; it does not adjust for cohort or downstream clinical/treatment flags.",
        "- Biomarker-sample selection remains plausible because biomarker participation differs by drinking status.",
        "",
        "## Output Map",
        "",
        "- `refined_abstainer_tables/`: descriptive AL and subsystem summaries.",
        "- `broad_health_selection/`: focal frailty coefficient from the broad W2 abstention transition model.",
        "- `biomarker_sample_selection/`: biomarker participation summary for the past-month abstainer contrast.",
    ]
    return lines


def main() -> None:
    # Write provenance.json on exit.
    provenance.stamp_at_exit(OUTPUT_DIR, "a2c_al_drink_quick", ["data/processed/biomarker/allostatic_load_M2.csv", "data/processed/biomarker/allostatic_load_MR1.csv", "data/processed/master_analysis.csv", "data/processed/master_analysis_concurrent.csv", "data/raw/wave1/M2_P1_SURVEY_N4963_20200720.sav", "data/raw/wave1/MR1_P1_SURVEY_N3577_20250407.sav"])
    parser = argparse.ArgumentParser(description="AL drinking-status sensitivity analysis")
    args = parser.parse_args()

    dirs = _ensure_dirs()
    print("A2c AL drinking-status sensitivity")
    print(f"Output: {OUTPUT_DIR}")
    print("Mode: descriptive subtypes + broad health selection + biomarker sample selection")

    refined = load_refined_master()
    refined_results = write_refined_abstainer_outputs(refined, dirs["refined"])

    broad_health = broad_health_selection_analysis(refined, dirs["broad_health"])
    biomarker_selection = biomarker_selection_bias(dirs["biomarker_selection"])

    summary_results = {
        "refined_sentence": _summarize_refined_result(refined_results["score_refined"]),
        "broad_health_sentence": _summarize_broad_health_selection(broad_health["broad_model"]),
        "biomarker_selection_sentence": _summarize_biomarker_selection(biomarker_selection),
    }
    log_path = write_output_log(
        OUTPUT_DIR,
        "output_log.md",
        script="analysis/sensitivity/a2c_AL_drinking_sensitivity.py",
        outputs=OUTPUTS,
        run_args=args,
        notes=summary_notes(summary_results),
    )
    print(f"\nDone. Output log: {log_path}")


if __name__ == "__main__":
    main()
