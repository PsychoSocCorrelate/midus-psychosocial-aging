#!/usr/bin/env python3
"""
Post-processing: apply exclusions, drop constituent items, join composites.

Run from repo root:
    uv run python -m preprocess.qa.build_analysis_ready

Input:
    data/processed/wave1/all_M2MR_py.csv
    data/processed/wave2/all_M3MR2_py.csv
    data/processed/shared/composite_cfa_scores.csv (10 composites)

Output:
    data/processed/wave1/all_M2MR_analysis.csv
    data/processed/wave2/all_M3MR2_analysis.csv
"""

import pandas as pd
import numpy as np

from preprocess.common.paths import PROCESSED_DIR, ROOT

# 1. Load data

w1 = pd.read_csv(PROCESSED_DIR / "wave1/all_M2MR_py.csv")
w2 = pd.read_csv(PROCESSED_DIR / "wave2/all_M3MR2_py.csv")
comp = pd.read_csv(PROCESSED_DIR / "shared/composite_cfa_scores.csv")

print(f"W1: {w1.shape[0]} x {w1.shape[1]}")
print(f"W2: {w2.shape[0]} x {w2.shape[1]}")
print(f"Composites: {comp.shape[0]} x {comp.shape[1]}")

# 2. Define exclusions

# Category A: Construct overlap with outcomes (frailty/cog/epigenetic)
cat_a = [
    "cirs_no_angina",       # r=0.645 with FInoPsy (circular)
    "hrattack", "diabetes", "stroke", "lungDisease",  # CIRS sub-items
    "anemia", "thyroid", "osteoporosis", "arthritis",  # CIRS sub-items
    "cancer", "cancerType",
    "limitActivity",        # ADL/frailty overlap
    "limitModAct", "limitWalking", "limitRecreation",  # frailty sub-items
    "limitClimb1", "limitClimb2", "limitBendKneel",    # frailty sub-items
]

# Category B: near-constant (>95% one value)
cat_b = [
    "useStimulant",         # >97% zero
    "useSedative",          # >96% zero
    "useTranquilizer",      # >96% zero
    "liveInstitute",        # >99% zero
    "noPhone",              # >99% zero
    "noUsualCare",          # >94% zero
]

# Category C: Structurally absent in W2 (100% NA, not measurement error)
cat_c = [
    "legalEvent",
    "relationship",
    "ecoCurr",
    "famLossACE",
    "famLossCurr",
    "abuseEvent",
]

# Category D: Low validity / incomparable constructs
cat_d = [
    "sexIntimacy",          # construct changed across waves (B1SM10 intimacy vs C1SM10 importance)
    "workForPay",           # near-ceiling (r=0.025)
    "healthAffectWork",
]

# Category E: metadata and ids to drop (keep M2MRID, familyId, sampleWeight,
#             sampleId, flag_edu_change, spouseWork, female)
cat_e_drop = [
    "sex",                  # redundant with female
    "birthYear",            # derivable from age; not a predictor
    "age_at_wave",          # redundant with age (verified identical in W2)
    # "cohort" is kept for per-cohort splits
    "M3RE_FILTER",          # SPSS filter variable (all 0 for MIDUS3, NA for MR2)
]

all_drop = cat_a + cat_b + cat_c + cat_d + cat_e_drop

# 3. Define composite constituent items to drop
# These individual items are replaced by their composite score.

composite_constituents = {
    # psychWellbeing (1-factor, 9 items)
    "psychWellbeing": [
        "autonomy", "envMastery", "persGrowth", "posRelation",
        "lifePurpose", "selfAccept", "persMastery", "percConstraints", "selfEsteem",
    ],
    "giveSupport": [
        "givePartnerEmoSupp", "giveParentEmoSupp", "giveInLawEmoSupp",
        "giveChildEmoSupp", "giveFriendEmoSupp",
    ],
    "recSupport": [
        "recPartnerEmoSupp", "recParentEmoSupp", "recInLawEmoSupp",
        "recChildEmoSupp", "recFriendEmoSupp",
    ],
    "religiosity": [
        "spirituality", "religiousId", "attendReligious",
        "religiousCopingA", "religiousCopingB",
    ],
    "activeCoping": [
        "copingGrowth", "copingActive", "planning",
        "primaryControl", "posReappraisal",
    ],
    "spouseQuality": [
        "spouseSupport", "spouseStrain", "spouseSolidarity",
        "spouseDisagree", "rateCurrentRelation", "maritalRisk",
    ],
    "familyQuality": [
        "famSupport", "famStrain", "famSolidarity", "supportForFam",
    ],
    "friendQuality": [
        "friendSupport", "friendStrain", "friendSolidarity",
    ],
    "lifeSatisfaction": [
        "lifeSat", "lifeOverall", "rateToday", "rateCurrentFin",
    ],
    "sleepAvg": [
        "sleepWorkday", "sleepWknd",
    ],
    # supplements: items dropped, no composite
    "supplements_drop": [
        "multiVitamins", "vitaminC", "fishOil",
    ],
}

items_to_drop = []
for items in composite_constituents.values():
    items_to_drop.extend(items)

# 4. Apply exclusions and drops

def process_wave(df, wave_label):
    """Apply exclusions, drop constituent items, join composites."""
    n_before = df.shape[1]

    all_remove = set(all_drop + items_to_drop)
    actually_removed = [c for c in all_remove if c in df.columns]
    df = df.drop(columns=actually_removed)

    print(f"\n{wave_label}:")
    print(f"  Dropped {len(actually_removed)} columns ({n_before} -> {df.shape[1]})")

    comp_wave = comp[comp["wave"] == ("W1" if "M2MR" in wave_label else "W2")]
    composite_cols = [c for c in comp_wave.columns if c not in ("M2MRID", "wave", "group")]
    comp_join = comp_wave[["M2MRID"] + composite_cols]

    assert comp_join["M2MRID"].is_unique, f"Duplicate M2MRID in composites for {wave_label}"

    # Drop pre-existing composite columns to avoid merge suffix conflicts
    overlap = [c for c in composite_cols if c in df.columns]
    if overlap:
        df = df.drop(columns=overlap)

    df = df.merge(comp_join, on="M2MRID", how="left")
    print(f"  Joined {len(composite_cols)} composite columns -> {df.shape[1]} total")

    for c in composite_cols:
        n_valid = df[c].notna().sum()
        print(f"    {c:25s}  non-NA: {n_valid:5d} ({100*n_valid/len(df):.1f}%)")

    return df


w1_out = process_wave(w1, "W1 (M2MR)")
w2_out = process_wave(w2, "W2 (M3MR2)")

# 5. Verify common columns

w1_cols = set(w1_out.columns)
w2_cols = set(w2_out.columns)

only_w1 = w1_cols - w2_cols
only_w2 = w2_cols - w1_cols
common = w1_cols & w2_cols

print(f"\nColumn comparison:")
print(f"  Common: {len(common)}")
if only_w1:
    print(f"  W1 only ({len(only_w1)}): {sorted(only_w1)}")
if only_w2:
    print(f"  W2 only ({len(only_w2)}): {sorted(only_w2)}")

# 6. Save

w1_path = PROCESSED_DIR / "wave1/all_M2MR_analysis.csv"
w2_path = PROCESSED_DIR / "wave2/all_M3MR2_analysis.csv"
w1_out.to_csv(w1_path, index=False)
w2_out.to_csv(w2_path, index=False)

print(f"\nSaved: {w1_path.relative_to(ROOT)}  ({w1_out.shape[0]} x {w1_out.shape[1]})")
print(f"Saved: {w2_path.relative_to(ROOT)}  ({w2_out.shape[0]} x {w2_out.shape[1]})")

# 7. Summary table

print("\nAnalysis-ready summary")
print(f"  W1: {w1_out.shape[0]:,} participants x {w1_out.shape[1]} variables")
print(f"  W2: {w2_out.shape[0]:,} participants x {w2_out.shape[1]} variables")
print(f"  Excluded: {len(all_drop)} individual vars (cat A-E)")
print(f"  Composited: {len(items_to_drop)} items -> {len(composite_constituents)} composites")
# Methods come from the fit tables, so an alignment-to-mean-of-z fallback shows here.
print(f"  Composite method (read from the fit tables):")
_align_path = PROCESSED_DIR / "shared/composite_align_fit.csv"
_methods: dict[str, str] = {}
if _align_path.exists():
    _af = pd.read_csv(_align_path)
    for _, _r in _af.iterrows():
        _methods[_r["composite"]] = str(_r["method"])
_cfa_path = PROCESSED_DIR / "shared/composite_cfa_fit.csv"
if _cfa_path.exists():
    _cf = pd.read_csv(_cfa_path)
    for _name in _cf["composite"].unique():
        _base = str(_name).split("_")[0]
        _methods.setdefault(_base, "CFA factor scores")
for _c in sorted(composite_constituents):
    if _c == "supplements_drop":
        continue
    print(f"    {_c:<18} {_methods.get(_c, 'not reported in any fit table')}")
print(f"    Dropped (no composite):          multiVitamins, vitaminC, fishOil")
