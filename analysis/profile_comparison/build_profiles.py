"""Reproducible profile specification.

Generates config/profile_specification.yaml for the
favorable / neutral / unfavorable focal-predictor profiles used by the SI profile
comparison (Table S16, Fig. S7).

Protocol:
  * selection (independent per outcome): rank predictors by bootstrap mean|SHAP|
    (output/cs1_cross_sectional/shap/<outcome>_T1T2_bootstrap_shap.csv), keep those
    with top-20 stability >= STABILITY, drop predictors flagged as direct/proximal
    construct-overlap for that outcome (frailty only, from config/frailty_overlap_audit.csv),
    and take the top K.
  * values (continuous): neutral = sample median (P50); favorable/unfavorable = the
    P_LO / P_HI empirical percentiles, oriented by the sign of Spearman corr(feature,
    signed SHAP) so the healthier-predicted end is "favorable". Near-zero |corr|
    (|rho| < MONO_TOL) is flagged as non-monotone for expert review.
  * values (categorical): neutral = modal category; favorable / unfavorable = the
    levels (with n >= MIN_LEVEL_N) whose mean signed SHAP is in the healthier / less
    healthy direction.
  * Empirical-override: documented expert values for predictors where the empirical percentile
    is clinically misleading or the dependence is non-monotone.

"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from analysis.core.data_loader import CATEGORICALS, load_master
from analysis.core import artifact_paths as ap

ROOT = Path(__file__).resolve().parents[2]
CONFIG_OUT = ROOT / "config" / "profile_specification.yaml"

OUTCOMES = {
    "FInoPsy_W1": ("Frailty", "lower"),
    "btactComp_W1": ("Cognition", "higher"),
    f"{ap.AL_STEM}_W1": ("Allostatic load", "lower"),
}
K = 6
STABILITY = 0.6
P_LO, P_HI = 10, 90
MONO_TOL = 0.10
MIN_LEVEL_N = 30

# Eligibility filter (modifiable-exposures-only): focal predictors must be modifiable
# psychosocial / behavioral / social exposures. Excluded categories:
#   * demographic identity: not modifiable, inappropriate as a favorable/unfavorable lever
#   * health-service utilization / clinical-care contact: SHAP direction mainly reflects
#     reverse causation (being treated because already ill)
EXCLUDE_IDENTITY = {"race", "female", "hispanic", "sexualOrientation"}
EXCLUDE_CARE_UTILIZATION = {
    "physicianDiet", "physicalPTFreq", "chiropractorFreq", "usePainkiller", "useDepression",
    "privateClinic", "HMOClinic", "publicClinic", "outPatient", "emergencyCare", "urgentCare",
    "needMedNotGetting", "haveHealthInsurance", "medicare", "medicaid", "weightChangeHealthIssue",
}
INELIGIBLE = EXCLUDE_IDENTITY | EXCLUDE_CARE_UTILIZATION

# Documented overrides based on strong empirical evidence (predictor -> {favorable, unfavorable, rationale}).
OVERRIDES = {
    "sleepAvg": {"favorable": 7.5, "unfavorable": 5.0,
                 "rationale": "U-shaped dependence; favorable = ~optimal sleep, not a distribution tail."},
    "bmi": {"favorable": 22.0, "unfavorable": 34.0,
            "rationale": "Clinical bands (normal vs class-1 obesity) preferred over raw percentiles."},
}


def _overlap_excluded(outcome: str) -> set[str]:
    if outcome != "FInoPsy_W1":
        return set()
    audit = pd.read_csv(ROOT / "config" / "frailty_overlap_audit.csv")
    return set(audit.loc[audit["overlap_code"].isin(["a", "b"]), "predictor"].astype(str))


def _aligned(outcome: str):
    X, y, features, _ = load_master(outcome, tiers=[1, 2])
    Xv = X[y.notna().values].reset_index(drop=True)
    robust = pd.read_csv(ap.CS1_DIR / "shap_raw" / f"{outcome}_T1T2_robust_shap.csv")
    shared = [c for c in robust.columns if c in Xv.columns]
    assert len(Xv) == len(robust), f"{outcome}: row mismatch {len(Xv)} vs {len(robust)}"
    return Xv[shared], robust[shared]


def _select(outcome: str) -> list[tuple[str, float, float]]:
    boot = pd.read_csv(ap.CS1_DIR / "shap" / f"{outcome}_T1T2_bootstrap_shap.csv", index_col=0)
    stab_col = next((c for c in boot.columns if c.endswith("stability")), None)
    excl = _overlap_excluded(outcome)
    boot = boot.sort_values("mean_abs_shap", ascending=False)
    out = []
    for name, r in boot.iterrows():
        name = str(name)
        stab = float(r[stab_col]) if stab_col else 1.0
        if stab < STABILITY or name in excl or name in INELIGIBLE:
            continue
        out.append((name, float(r["mean_abs_shap"]), stab))  # all eligible, ranked
    return out


def _continuous_spec(vals, shap, healthier):
    vals = np.asarray(vals, dtype=float)
    shap = np.asarray(shap, dtype=float)
    ok = ~np.isnan(vals) & ~np.isnan(shap)
    rho = float(spearmanr(vals[ok], shap[ok]).correlation)
    p10, p50, p90 = (round(float(np.percentile(vals[ok], p)), 4) for p in (P_LO, 50, P_HI))
    higher_is_worse = (rho > 0) if healthier == "lower" else (rho < 0)
    fav, unf = (p10, p90) if higher_is_worse else (p90, p10)
    return {
        "favorable": {"value": fav, "source": f"P{P_LO if higher_is_worse else P_HI}"},
        "neutral": {"value": p50, "source": "median"},
        "unfavorable": {"value": unf, "source": f"P{P_HI if higher_is_worse else P_LO}"},
        "shap_feature_spearman": round(rho, 3),
        "monotone": bool(abs(rho) >= MONO_TOL),
    }


def _categorical_spec(level_vals, shap, healthier):
    df = pd.DataFrame({"lvl": pd.Series(level_vals).astype(str).values, "shap": np.asarray(shap, float)})
    df = df[~df["lvl"].isin(["nan", "None", ""])]
    g = df.groupby("lvl")["shap"].agg(["mean", "count"])
    g = g[g["count"] >= MIN_LEVEL_N]
    means = g["mean"].sort_values()
    healthier_level = str(means.index[0] if healthier == "lower" else means.index[-1])
    worse_level = str(means.index[-1] if healthier == "lower" else means.index[0])
    modal = str(df["lvl"].mode().iloc[0])
    return {
        "favorable": {"value": healthier_level, "source": "min/max mean-SHAP level"},
        "neutral": {"value": modal, "source": "modal category"},
        "unfavorable": {"value": worse_level, "source": "min/max mean-SHAP level"},
        "level_mean_shap": {str(k): round(float(v), 4) for k, v in g["mean"].items()},
    }


def build():
    spec = {
        "meta": {
            "procedure": "Hybrid: reproducible top-K stable mean|SHAP| selection (independent per "
                         "outcome) + empirical-percentile values oriented by SHAP-dependence "
                         "direction, with documented expert overrides.",
            "K": K, "stability_threshold": STABILITY,
            "value_percentiles": {"favorable_or_unfavorable": [P_LO, P_HI], "neutral": 50},
            "selection_source": "output/cs1_cross_sectional/shap/<outcome>_T1T2_bootstrap_shap.csv",
            "direction_source": "output/cs1_cross_sectional/shap_raw/<outcome>_T1T2_robust_shap.csv",
            "eligibility": "modifiable exposures only; excludes demographic identity and "
                           "health-service-utilization/clinical-care items (reverse-causation prone)",
            "excluded_identity": sorted(EXCLUDE_IDENTITY),
            "excluded_care_utilization": sorted(EXCLUDE_CARE_UTILIZATION),
            "exclusions": {"FInoPsy_W1": "construct-overlap codes a,b (config/frailty_overlap_audit.csv)"},
            "non_focal_baseline": "sample median (numeric) / modal category (categorical)",
            "generated_by": "analysis/profile_comparison/build_profiles.py",
        },
        "outcomes": {},
    }
    for okey, (label, healthier) in OUTCOMES.items():
        Xv, robust = _aligned(okey)
        entry = {"label": label, "healthier_direction": healthier, "focal": [], "dropped_non_monotone": []}
        for name, mshap, stab in _select(okey):
            if len(entry["focal"]) == K:
                break
            is_cat = name in CATEGORICALS or Xv[name].dtype.name == "category"
            rec = {"name": name, "mean_abs_shap": round(mshap, 4), "top20_stability": round(stab, 3),
                   "type": "categorical" if is_cat else "continuous"}
            if is_cat:
                rec.update(_categorical_spec(Xv[name], robust[name], healthier))
            else:
                rec.update(_continuous_spec(Xv[name], robust[name], healthier))
                if name in OVERRIDES:
                    ov = OVERRIDES[name]
                    rec["favorable"] = {"value": ov["favorable"], "source": "override"}
                    rec["unfavorable"] = {"value": ov["unfavorable"], "source": "override"}
                    rec["override_rationale"] = ov["rationale"]
                elif not rec["monotone"]:
                    # No determinable favorable direction -> unsuitable as a lever; backfill next.
                    entry["dropped_non_monotone"].append(name)
                    continue
            entry["focal"].append(rec)
        spec["outcomes"][okey] = entry

    CONFIG_OUT.write_text(yaml.safe_dump(_py(spec), sort_keys=False, allow_unicode=True, width=100))
    return spec


def _py(obj):
    """Recursively coerce numpy scalar types to native types for YAML."""
    if isinstance(obj, dict):
        return {str(k): _py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_py(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.str_):
        return str(obj)
    return obj


def _summary(spec):
    for okey, e in spec["outcomes"].items():
        print(f"\n== {e['label']} ({okey}; healthier={e['healthier_direction']}) ==")
        for f in e["focal"]:
            fav, neu, unf = f["favorable"]["value"], f["neutral"]["value"], f["unfavorable"]["value"]
            extra = f" rho={f.get('shap_feature_spearman')}" + ("" if f.get("monotone", True) else " NON-MONOTONE")
            extra += "  [OVERRIDE]" if f.get("override_rationale") else ""
            print(f"  {f['name']:20s} {f['type'][:4]:4s} |SHAP|={f['mean_abs_shap']:.3f} stab={f['top20_stability']:.2f}"
                  f" | fav={fav} neu={neu} unf={unf}{extra}")


if __name__ == "__main__":
    s = build()
    _summary(s)
    print(f"\nWrote {CONFIG_OUT.relative_to(ROOT)}")
