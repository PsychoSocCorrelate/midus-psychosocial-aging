"""
Survey QC pipeline: per-cohort recoding and derivation, then pooling by wave.

Uses:
  - recode_engine.py     (apply_recodes, cleanup_sentinels)
  - variable_maps.py     (get_cohort_map, get_recode_config, load_variable_key,
                           COHORT_PREFIXES, ABSENT_VARIABLES)
  - derived_demographics.py  (demographics, work, relationships, caregiving,
                               smoking, drinking, ACE, physical activity)
  - derived_health.py    (CIRS, activity_composites, health_affect_work,
                           middle_age)

Usage:
    uv run python -m preprocess.survey.qc_pipeline --wave 1
    uv run python -m preprocess.survey.qc_pipeline --wave 2
    uv run python -m preprocess.survey.qc_pipeline --wave 1 --cohort midus2
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pyreadstat

from preprocess.survey.missing_reasons import write_reason_artifacts
from sklearn.decomposition import PCA

from preprocess.common.paths import ROOT as PROJECT_ROOT
from preprocess.survey.recode_engine import apply_recodes, cleanup_sentinels
from preprocess.survey.variable_maps import (
    ABSENT_VARIABLES,
    COHORT_PREFIXES,
    get_cohort_map,
    get_recode_config,
)
from preprocess.survey.derived_demographics import (
    derive_ace_composite,
    derive_caregiv_intensity,
    derive_caregiv_status,
    derive_drink_exposure,
    derive_drink_exposure_pa59,
    derive_drink_status,
    derive_family_cancer,
    derive_family_heart_attack,
    derive_female,
    derive_health_affect_work,
    derive_hispanic,
    derive_manage_role,
    derive_marr_duration,
    derive_occ_group,
    derive_partner_age_gap,
    derive_partner_edu_gap,
    derive_race,
    derive_relat_cohab_length,
    derive_relat_status,
    derive_sex_partner,
    derive_smoke_amount,
    derive_smoke_duration,
    derive_smoke_exposure,
    derive_smoke_exposure_3item,
    derive_smoke_status,
    derive_start_middle_age,
    derive_end_middle_age,
    derive_sub_achievement,
    derive_times_married,
    derive_work_for_pay,
    derive_work_industry,
    derive_work_status,
)
from preprocess.survey.derived_health import (
    derive_activity_composites,
    derive_cirs,
)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("qc_pipeline")


# SAV file paths

SAV_PATHS: dict[str, Path] = {
    "midus2": PROJECT_ROOT / "data" / "raw" / "wave1" / "M2_P1_SURVEY_N4963_20200720.sav",
    "midusr": PROJECT_ROOT / "data" / "raw" / "wave1" / "MR1_P1_SURVEY_N3577_20250407.sav",
    "midus3": PROJECT_ROOT / "data" / "raw" / "wave2" / "M3_P1_SURVEY_N3294_20251029.sav",
    "mr2":    PROJECT_ROOT / "data" / "raw" / "wave2" / "MR2_P1_SURVEY_N2154_20251003.sav",
}

# Status column and filter per cohort
# MIDUS2:  B1STATUS == 2
# MIDUSR:  RA1STATUS != 1
# MIDUS3:  C1STATUS == 4
# MR2:     RB1STATUS == 4
STATUS_FILTERS: dict[str, tuple[str, str, object]] = {
    #            (column,       operator, value)
    "midus2": ("B1STATUS",   "==", 2),
    "midusr": ("RA1STATUS",  "!=", 1),
    "midus3": ("C1STATUS",   "==", 4),
    "mr2":    ("RB1STATUS",  "==", 4),
}

# Wave membership
WAVE_COHORTS: dict[int, list[str]] = {
    1: ["midus2", "midusr"],
    2: ["midus3", "mr2"],
}

# Labels for the 'cohort' column. sampleId is a different variable: it comes from
# the key map as SAMPLMAJ, whose values are sample strata (1 Main RDD, 2 Sibling,
# 3 Twin, 4 City oversample, 20+ refresher), not cohorts.
COHORT_LABELS: dict[str, str] = {
    "midus2": "MIDUS2",
    "midusr": "MIDUSR",
    "midus3": "MIDUS3",
    "mr2":    "MR2",
}


# Industry code lookup for MIDUS2 (raw census code -> industry_code)
_INDUSTRY_MAP_PATH = PROJECT_ROOT / "data" / "key_map" / "midus2_industry_label_map_key.csv"
if _INDUSTRY_MAP_PATH.exists():
    _ind_df = pd.read_csv(_INDUSTRY_MAP_PATH)
    _INDUSTRY_MAP: dict = dict(zip(_ind_df["code"], _ind_df["industry_code"]))
else:
    _INDUSTRY_MAP = {}
    log.warning("Industry map not found at %s", _INDUSTRY_MAP_PATH)


# Cohort-specific column name look-ups for derived variables
# Each cohort has different SPSS prefixes and sometimes different variable
# numbering.  This dict maps (cohort, semantic_name) -> SPSS column name.

def _p(cohort: str) -> str:
    """Return the SPSS prefix for a cohort (e.g. 'B1' for midus2)."""
    return COHORT_PREFIXES[cohort]


def _col(cohort: str, suffix: str) -> str:
    """Build a fully-prefixed column name: prefix + suffix."""
    return f"{_p(cohort)}{suffix}"


# Cohort-specific configurations for derived variables

def _get_cohort_cols(cohort: str) -> dict:
    """Return a dict of semantic column names for a given cohort.

    This centralises all cohort-specific SPSS variable numbering so the
    process_cohort function can call derive_* functions generically.
    """
    p = _p(cohort)

    # Shared across all cohorts (most variables have stable numbering)
    cols: dict = {
        # demographics
        "prsex":    f"{p}PRSEX",
        "pf1":      f"{p}PF1",
        "sxornt":   f"{p}SXORNT" if cohort in ("midus2",) else None,
        # work
        "pb4n":     f"{p}PB4N",
        "pb6":      f"{p}PB6",
        "pbwork":   f"{p}PBWORK",
        "pb7":      f"{p}PB7",
        "pocmaj":   f"{p}POCMAJ",
        "pocpmj":   f"{p}POCPMJ" if cohort not in ("mr2",) else None,
        "pindmj":   f"{p}PINDMJ" if cohort not in ("mr2",) else None,
        "ppinmj":   f"{p}PPINMJ" if cohort not in ("mr2",) else None,
        # relationships
        "pb19":     f"{p}PB19",
        "pb30":     f"{p}PB30",
        "pb20":     f"{p}PB20",
        "pb31m":    f"{p}PB31M",
        "pb31n":    f"{p}PB31N",
        "pb1":      f"{p}PB1",     # education
        "pb33":     f"{p}PB33",    # partner education
        "psage":    f"{p}PSAGE",   # partner age
        # sex partner columns set below (cohort-specific gate+count)
        # caregiving
        "pd1":      f"{p}PD1",
        "pd7":      f"{p}PD7",
        "pd14":     f"{p}PD14",
        "pd15":     f"{p}PD15",
        "pd18":     f"{p}PD18",
        # smoking
        "pa38a":    f"{p}PA38A",
        "pa38b":    f"{p}PA38B",
        "pa39":     f"{p}PA39",
        "pa40":     f"{p}PA40",
        "pa42":     f"{p}PA42",
        "pa43":     f"{p}PA43",
        "pa44":     f"{p}PA44",
        # smoke exposure (6-item W1)
        "pa45a":    f"{p}PA45A",
        "pa45b":    f"{p}PA45B",
        "pa45c":    f"{p}PA45C",
        "pa46":     f"{p}PA46",
        "pa47":     f"{p}PA47",
        "pa48":     f"{p}PA48",
        # drinking
        "pa50":     f"{p}PA50",
        "pa51":     f"{p}PA51",
        "pa53":     f"{p}PA53",
        "pa58":     f"{p}PA58",
        "pa59":     f"{p}PA59",
        # health
        "pa49":     f"{p}PA49",
        "pa4":      f"{p}PA4",
        "pa5":      f"{p}PA5",
        "sbmi":     f"{p}SBMI",
        "smarrs":   f"{p}SMARRS",
        # family history
        "pa10a": f"{p}PA10A", "pa10b": f"{p}PA10B", "pa10c": f"{p}PA10C",
        "pa10d": f"{p}PA10D", "pa10e": f"{p}PA10E", "pa10f": f"{p}PA10F",
        "pa10g": f"{p}PA10G", "pa10h": f"{p}PA10H",
        "pa30a": f"{p}PA30A", "pa30b": f"{p}PA30B", "pa30c": f"{p}PA30C",
        "pa30d": f"{p}PA30D", "pa30e": f"{p}PA30E", "pa30f": f"{p}PA30F",
        # psychological
        "pg5":  f"{p}PG5",
        "pg5a": f"{p}PG5A",
        # CIRS items
        "pa8":    f"{p}PA8",
        "pa7ba":  f"{p}PA7BA",
        "pa7bb":  f"{p}PA7BB",
        "pangin": f"{p}PANGIN" if cohort not in ("mr2",) else None,
        "sa11s":  f"{p}SA11S",
        "sa12a":  f"{p}SA12A",
        "sa11a":  f"{p}SA11A",
        "sa11b":  f"{p}SA11B",
        "sa7d":   f"{p}SA7D",
        "sa12f":  f"{p}SA12F",
        "sa11aa": f"{p}SA11AA",
        "sa11l":  f"{p}SA11L",
        "sa11j":  f"{p}SA11J",
        "sa11d":  f"{p}SA11D",
        "sa12g":  f"{p}SA12G",
        "sa11v":  f"{p}SA11V",
        "pa6c":   f"{p}PA6C",
        "pa6d":   f"{p}PA6D",
        "sa11y":  f"{p}SA11Y",
        "sa11x":  f"{p}SA11X",
        "sa11g":  f"{p}SA11G",
        "pa26":   f"{p}PA26",
        "pa28a":  f"{p}PA28A",
        "sa11z":  f"{p}SA11Z",
        "pa6a":   f"{p}PA6A",
        # friend contact (reverse-coded 9-x)
        "friend_contact": None,  # set per cohort below
    }

    # Age column
    if cohort == "midus2":
        cols["age"] = "B1PRAGE_2019"
        cols["pbyear"] = "B1PBYEAR_2019"
    elif cohort == "midusr":
        cols["age"] = "RA1PRAGE"
        cols["pbyear"] = "RA1PBYEAR"
    elif cohort == "midus3":
        cols["age"] = "C1PRAGE"
        cols["pbyear"] = "C1PBYEAR"
        cols["pbwork"] = "C1PB2AWK"   # MIDUS3 renamed PBWORK -> PB2AWK
    elif cohort == "mr2":
        cols["age"] = "RB1PRAGE"
        cols["pbyear"] = "RB1PBYEAR"
        cols["pbwork"] = "RB1PB2AWK"  # MR2 renamed PBWORK -> PB2AWK

    # Age-at-event outlier columns
    if cohort == "midus2":
        cols["age_event_cols"] = ["B1PA37", "B1PA49", "B1PB2"]
    elif cohort == "midusr":
        cols["age_event_cols"] = ["RA1PA37", "RA1PA49", "RA1PB2"]
    elif cohort == "midus3":
        cols["age_event_cols"] = ["C1PA37", "C1PA49", "C1PB2"]
    elif cohort == "mr2":
        cols["age_event_cols"] = ["RB1PA36", "RB1PA49", "RB1PB2"]

    # Friend-contact reverse code (9 - x, 8-pt ordinal)
    if cohort == "midus2":
        cols["friend_contact"] = "B1SJ3"
    elif cohort == "midusr":
        cols["friend_contact"] = "RA1SK5"
    elif cohort == "midus3":
        cols["friend_contact"] = "C1SJ5"
    elif cohort == "mr2":
        cols["friend_contact"] = "RB1SJ5"

    # Nap frequency column
    if cohort == "midus2":
        cols["nap_col"] = "B1SA60"
    elif cohort == "midusr":
        cols["nap_col"] = "RA1SA56"
    elif cohort in ("midus3", "mr2"):
        # MIDUS3: SA60->SA56 renumbered; MR2 uses same numbering as MIDUSR
        cols["nap_col"] = f"{p}SA56"

    # MR2 smoking renumbering: PA38A -> PA37
    if cohort == "mr2":
        cols["pa38a"] = "RB1PA37"
        cols["pa38b"] = "RB1PA38"  # PA38B -> PA38

    # Activity columns (vigorous / moderate / light)
    # MIDUS2: SA30/31/32; MIDUSR/MIDUS3/MR2: SA26/27/28
    if cohort == "midus2":
        cols["vig_cols"]   = [f"B1SA30{c}" for c in "ABCDEF"]
        cols["mod_cols"]   = [f"B1SA31{c}" for c in "ABCDEF"]
        cols["light_cols"] = [f"B1SA32{c}" for c in "ABCDEF"]
    else:
        cols["vig_cols"]   = [f"{p}SA26{c}" for c in "ABCDEF"]
        cols["mod_cols"]   = [f"{p}SA27{c}" for c in "ABCDEF"]
        cols["light_cols"] = [f"{p}SA28{c}" for c in "ABCDEF"]

    # ACE items (cohort-specific)
    if cohort == "midus2":
        cols["educationACE_items"]   = ["B1SE11A", "B1SE11F", "B1SE11G", "B1SE11H"]
        cols["educationACE_3_items"] = ["B1SE11A", "B1SE11F", "B1SE11G"]
        cols["famDysfuncACE_items"]  = ["B1SE11B", "B1SE11D", "B1SE11E", "B1SE11L"]
        cols["famDysfuncACE_3_items"]= ["B1SE11B", "B1SE11D", "B1SE11E"]
        cols["ecoACE_item"]          = "B1SE11C"
        cols["ecoCurr_items"]        = []  # computed via age-split logic
        cols["famLossACE_items"]     = []  # computed via age-split logic
        cols["famLossCurr_items"]    = []  # computed via age-split logic
        cols["legalEvent_items"]     = ["B1SE11U", "B1SE11V"]
        cols["abuseEvent_items"]     = ["B1SE11S", "B1SE11T"]
        cols["relationship_items"]   = ["B1SE11M", "B1SE11N"]
    elif cohort == "midusr":
        cols["educationACE_items"]   = ["RA1SF11A", "RA1SF11F", "RA1SF11G"]
        cols["educationACE_3_items"] = ["RA1SF11A", "RA1SF11F", "RA1SF11G"]
        cols["famDysfuncACE_items"]  = ["RA1SF11B", "RA1SF11D", "RA1SF11E"]
        cols["famDysfuncACE_3_items"]= ["RA1SF11B", "RA1SF11D", "RA1SF11E"]
        cols["ecoACE_item"]          = "RA1SF11C"
        cols["ecoCurr_items"]        = []
        cols["famLossACE_items"]     = []
        cols["famLossCurr_items"]    = []
        cols["legalEvent_items"]     = []  # absent in MIDUSR
        cols["abuseEvent_items"]     = []
        cols["relationship_items"]   = []
    elif cohort == "midus3":
        cols["educationACE_items"]   = ["C1SE11A", "C1SE11F", "C1SE11G"]
        cols["educationACE_3_items"] = ["C1SE11A", "C1SE11F", "C1SE11G"]
        cols["famDysfuncACE_items"]  = ["C1SE11B", "C1SE11D", "C1SE11E"]
        cols["famDysfuncACE_3_items"]= ["C1SE11B", "C1SE11D", "C1SE11E"]
        cols["ecoACE_item"]          = "C1SE11C"
        cols["ecoCurr_items"]        = []  # absent in MIDUS3
        cols["famLossACE_items"]     = []
        cols["famLossCurr_items"]    = []
        cols["legalEvent_items"]     = []  # absent
        cols["abuseEvent_items"]     = []
        cols["relationship_items"]   = []
    elif cohort == "mr2":
        cols["educationACE_items"]   = ["RB1SE11A", "RB1SE11F", "RB1SE11G"]
        cols["educationACE_3_items"] = ["RB1SE11A", "RB1SE11F", "RB1SE11G"]
        cols["famDysfuncACE_items"]  = ["RB1SE11B", "RB1SE11D", "RB1SE11E"]
        cols["famDysfuncACE_3_items"]= ["RB1SE11B", "RB1SE11D", "RB1SE11E"]
        cols["ecoACE_item"]          = "RB1SE11C"
        cols["ecoCurr_items"]        = []  # absent
        cols["famLossACE_items"]     = []
        cols["famLossCurr_items"]    = []
        cols["legalEvent_items"]     = []  # absent
        cols["abuseEvent_items"]     = []
        cols["relationship_items"]   = []

    # Smoke exposure item list (6-item for W1, 3-item for W2)
    if cohort in ("midus2", "midusr"):
        cols["smoke_exposure_items"] = [
            cols["pa45a"], cols["pa45b"], cols["pa45c"],
            cols["pa46"], cols["pa47"], cols["pa48"],
        ]
        cols["smoke_exposure_min"] = 4
    else:
        # W2: PA45A/B/C absent; 3-item version
        cols["smoke_exposure_items"] = [cols["pa46"], cols["pa47"], cols["pa48"]]
        cols["smoke_exposure_min"] = 2

    # Middle age columns (sex-dependent SE items)
    if cohort == "midus2":
        cols["se_male_start"]   = "B1SE16"
        cols["se_male_end"]     = "B1SE17"
        cols["se_female_start"] = "B1SE18"
        cols["se_female_end"]   = "B1SE19"
    elif cohort == "midusr":
        cols["se_male_start"]   = "RA1SE16"
        cols["se_male_end"]     = "RA1SE17"
        cols["se_female_start"] = "RA1SE18"
        cols["se_female_end"]   = "RA1SE19"
    elif cohort == "midus3":
        # MIDUS3 and MR2 renumbered these to SE18-SE21
        cols["se_male_start"]   = "C1SE18"
        cols["se_male_end"]     = "C1SE19"
        cols["se_female_start"] = "C1SE20"
        cols["se_female_end"]   = "C1SE21"
    elif cohort == "mr2":
        cols["se_male_start"]   = "RB1SE18"
        cols["se_male_end"]     = "RB1SE19"
        cols["se_female_start"] = "RB1SE20"
        cols["se_female_end"]   = "RB1SE21"

    # Interview date (for marriage duration)
    if cohort == "midus2":
        cols["interview_mo"] = "B1PIDATE_MO"
        cols["interview_yr"] = "B1PIDATE_YR"
    elif cohort == "midusr":
        cols["interview_mo"] = "RA1PIDATE_MO"
        cols["interview_yr"] = "RA1PIDATE_YR"
    elif cohort == "midus3":
        cols["interview_mo"] = "C1PIDATE_MO"
        cols["interview_yr"] = "C1PIDATE_YR"
    elif cohort == "mr2":
        cols["interview_mo"] = "RB1PIDATE_MO"
        cols["interview_yr"] = "RB1PIDATE_YR"

    # Marriage date columns
    _marr_suffixes = [
        ("first_marr_mo", "PB21M"), ("first_marr_yr", "PB21Y"),
        ("first_died_mo", "PB23M"), ("first_died_yr", "PB23Y"),
        ("first_sep_mo", "PB24M"),  ("first_sep_yr", "PB24Y"),
        ("first_div_mo", "PB25M"),  ("first_div_yr", "PB25Y"),
        ("recent_marr_mo", "PB26M"),("recent_marr_yr", "PB26Y"),
        ("recent_sep_mo", "PB27M"), ("recent_sep_yr", "PB27Y"),
        ("recent_div_mo", "PB28M"), ("recent_div_yr", "PB28Y"),
        ("recent_died_mo", "PB29M"),("recent_died_yr", "PB29Y"),
    ]
    for key_name, suffix in _marr_suffixes:
        cols[key_name] = f"{p}{suffix}"

    # Household income (for MR2 winsorization)
    # MIDUS2 uses B1STINC1, MIDUSR uses RA1STINC1, MR2 uses RB1STINC
    if cohort in ("midus2", "midusr"):
        cols["stinc"] = f"{p}STINC1"
    else:
        cols["stinc"] = f"{p}STINC"

    # Sample weight (MIDUSR special case)
    cols["sample_weight"] = "MRCOMBWT" if cohort == "midusr" else None

    # Race column (PF7A per cohort, recode 4/5 -> 3, 6 -> 4 in derive_race)
    cols["race_col"] = f"{p}PF7A"

    # Sex partner columns (gate + count composite for non-midus2)
    if cohort == "midus2":
        cols["sm7_gate"] = None
        cols["sm7_count"] = "B1SM7"
    elif cohort == "midusr":
        cols["sm7_gate"] = "RA1SN7"
        cols["sm7_count"] = "RA1SN8"
    elif cohort == "midus3":
        cols["sm7_gate"] = "C1SM7"
        cols["sm7_count"] = "C1SM8"
    elif cohort == "mr2":
        cols["sm7_gate"] = "RB1SM7"
        cols["sm7_count"] = "RB1SM8"

    # Occupation: PSOC column (midus2 only, for '55-' -> occGroup=10)
    cols["psoc"] = f"{p}PSOC" if cohort == "midus2" else None

    # Work industry: raw census codes (midus2 only)
    if cohort == "midus2":
        cols["pind_raw"] = "B1PIND"
        cols["pindp_raw"] = "B1PINDP"
    else:
        cols["pind_raw"] = None
        cols["pindp_raw"] = None

    return cols


# Step 1-9: Per-cohort processing

def _emit_missing_reasons(
    sav_path: Path,
    cohort: str,
    out_df: pd.DataFrame,
    rename_existing: dict[str, str],
) -> None:
    """Write the missing-reason mask for this cohort.

    Rereads the SAV with ``user_missing=True``; the analysis frame is unchanged.
    Only columns taken directly from one SPSS variable get a reason. Derived
    columns (``workForPay``, ``smokeStatus``, the CIRS score) combine several
    sources, so they are listed in the coverage file instead.
    """
    id_src = next((k for k, v in rename_existing.items() if v == "M2MRID"), None)
    if id_src is None:
        log.warning("[%s] no id column in the rename map; skipping reason mask", cohort)
        return
    renamed_out = {v for v in rename_existing.values()}
    derived = [c for c in out_df.columns if c not in renamed_out]
    keep_ids = set(out_df["M2MRID"].dropna()) if "M2MRID" in out_df.columns else None
    out_dir = PROJECT_ROOT / "data" / "processed" / "missing_reasons"
    write_reason_artifacts(
        sav_path=sav_path,
        columns=sorted(rename_existing),
        id_col=id_src,
        out_dir=out_dir,
        stem=cohort,
        rename=rename_existing,
        derived_columns=derived,
        keep_ids=keep_ids,
        id_name="M2MRID",
    )


def process_cohort(sav_path: Path, cohort: str) -> pd.DataFrame:
    """Run the full QC pipeline for one cohort (steps 1-9).

    Parameters
    ----------
    sav_path : Path
        Path to the SAV file.
    cohort : str
        One of 'midus2', 'midusr', 'midus3', 'mr2'.

    Returns
    -------
    pd.DataFrame
        QC-processed data with standardised (analysis) column names.
    """
    t0 = time.time()
    cols = _get_cohort_cols(cohort)
    p = _p(cohort)

    # Step 1: Load SAV
    log.info("[%s] Step 1  Loading SAV: %s", cohort, sav_path)
    df, meta = pyreadstat.read_sav(str(sav_path), apply_value_formats=False)
    log.info("[%s]   Raw: %d rows x %d cols", cohort, len(df), len(df.columns))

    # Step 2: Filter status
    status_col, op, val = STATUS_FILTERS[cohort]
    if op == "==":
        df = df[df[status_col] == val].copy()
    elif op == "!=":
        df = df[df[status_col] != val].copy()
    df.reset_index(drop=True, inplace=True)
    log.info("[%s] Step 2  After status filter (%s %s %s): %d rows",
             cohort, status_col, op, val, len(df))

    # Step 3: Apply recodes
    log.info("[%s] Step 3  Applying recodes", cohort)
    recode_cfg = get_recode_config(cohort)
    df = apply_recodes(df, recode_cfg)

    # Step 4: Cleanup sentinels (negative values -> NaN)
    log.info("[%s] Step 4  Cleanup sentinels", cohort)
    df = cleanup_sentinels(df)

    # Step 5: Individual reverse codes
    log.info("[%s] Step 5  Individual reverse codes", cohort)

    # Friend contact: 9 - x (8-pt ordinal)
    fc = cols["friend_contact"]
    if fc and fc in df.columns:
        df[fc] = 9 - df[fc]

    # Nap frequency: cap at 45
    nap = cols["nap_col"]
    if nap and nap in df.columns:
        df[nap] = df[nap].where(df[nap] <= 45, np.nan)

    # BMI: > 60 -> NaN
    bmi_col = cols["sbmi"]
    if bmi_col in df.columns:
        df[bmi_col] = df[bmi_col].where(df[bmi_col] <= 60, np.nan)

    # Age-at-event: clamp to [10, current_age]
    age_col = cols["age"]
    if age_col in df.columns:
        for ae_col in cols.get("age_event_cols", []):
            if ae_col in df.columns:
                df[ae_col] = df[ae_col].where(
                    (df[ae_col] >= 10) & (df[ae_col] <= df[age_col]), np.nan
                )

    # MR2: maritalRisk cap at 9
    if cohort == "mr2":
        mrs = cols["smarrs"]
        if mrs in df.columns:
            df[mrs] = df[mrs].where(df[mrs] <= 9, np.nan)

    # MR2: householdIncome winsorize at $300K
    if cohort == "mr2":
        stinc = cols["stinc"]
        if stinc in df.columns:
            df[stinc] = df[stinc].clip(upper=300000)

    # Step 5b: lossChild gate for MR2
    # After binary recode, PC3 is 1/0. If PC2 != 0 (no child deaths), force 0.
    if cohort == "mr2":
        pc2 = f"{p}PC2"
        pc3 = f"{p}PC3"
        if pc2 in df.columns and pc3 in df.columns:
            df.loc[df[pc2] != 0, pc3] = 0
            log.info("[%s] Step 5b  lossChild gate: set PC3=0 where PC2!=0", cohort)

    # Step 6: Derive variables
    log.info("[%s] Step 6  Deriving variables", cohort)

    # Demographics
    df["female"] = derive_female(df, cols["prsex"])
    df["hispanic"] = derive_hispanic(df, cols["pf1"])
    df["race"] = derive_race(df, cols["race_col"])

    # Work
    df["workForPay"] = derive_work_for_pay(df, cols["pb4n"], cols["pb6"], cols["pbwork"])
    df["workStatus"] = derive_work_status(df, cols["pbwork"])
    df["manageRole"] = derive_manage_role(df, cols["pb7"])
    df["occGroup"] = derive_occ_group(
        df, cols["pocmaj"],
        cols.get("pocpmj"),
        cohort=cohort,
        psoc_col=cols.get("psoc"),
    )
    if cols.get("pindmj") or cols.get("pind_raw"):
        df["workIndustry"] = derive_work_industry(
            df, cols.get("pindmj", ""), cols.get("ppinmj"),
            cohort=cohort,
            pind_raw_col=cols.get("pind_raw"),
            pindp_raw_col=cols.get("pindp_raw"),
            industry_map=_INDUSTRY_MAP if cohort == "midus2" else None,
        )
    else:
        df["workIndustry"] = np.nan

    # Relationships
    df["relatStatus"] = derive_relat_status(df, cols["pb19"], cols["pb30"])
    df["marrDuration"] = derive_marr_duration(
        df,
        interview_mo_col=cols["interview_mo"],
        interview_yr_col=cols["interview_yr"],
        first_marr_mo_col=cols["first_marr_mo"],
        first_marr_yr_col=cols["first_marr_yr"],
        first_died_mo_col=cols["first_died_mo"],
        first_died_yr_col=cols["first_died_yr"],
        first_sep_mo_col=cols["first_sep_mo"],
        first_sep_yr_col=cols["first_sep_yr"],
        first_div_mo_col=cols["first_div_mo"],
        first_div_yr_col=cols["first_div_yr"],
        recent_marr_mo_col=cols["recent_marr_mo"],
        recent_marr_yr_col=cols["recent_marr_yr"],
        recent_sep_mo_col=cols["recent_sep_mo"],
        recent_sep_yr_col=cols["recent_sep_yr"],
        recent_div_mo_col=cols["recent_div_mo"],
        recent_div_yr_col=cols["recent_div_yr"],
        recent_died_mo_col=cols["recent_died_mo"],
        recent_died_yr_col=cols["recent_died_yr"],
        age_col=cols["age"],
        relat_status=df["relatStatus"],
    )
    df["timesMarried"] = derive_times_married(df, cols["pb20"], cols["pb19"])
    df["relatCohabLength"] = derive_relat_cohab_length(
        df, cols["pb31m"], cols["pb31n"], df["marrDuration"],
    )
    df["partnerAgeGap"] = derive_partner_age_gap(df, cols["age"], cols["psage"])
    df["partnerEduGap"] = derive_partner_edu_gap(df, cols["pb1"], cols["pb33"])
    df["sexPartner"] = derive_sex_partner(df, cols["sm7_gate"], cols["sm7_count"])

    # Caregiving
    df["careGivStatus"] = derive_caregiv_status(
        df, cols["pd1"], cols["pd7"], cols["pd18"],
    )
    df["careGivIntensity"] = derive_caregiv_intensity(
        df, cols["pd14"], cols["pd15"], df["careGivStatus"],
    )

    # Smoking
    df["smokeStatus"] = derive_smoke_status(
        df, cols["pa38a"], cols["pa39"], cols["pa40"], cols["pa43"], cols["pa44"],
    )
    df["smokeDuration"] = derive_smoke_duration(
        df, cols["pa38a"], cols["pa38b"], cols["pa39"],
        cols["pa42"], cols["pa44"], cols["age"],
    )
    df["smokeAmount"] = derive_smoke_amount(
        df, cols["pa38a"], cols["pa40"], cols["pa43"], cols["pa44"],
    )
    df["smokeExposure"] = derive_smoke_exposure(
        df, cols["smoke_exposure_items"], cols["smoke_exposure_min"],
    )
    df["smokeExposure_3item"] = derive_smoke_exposure_3item(
        df, cols["pa46"], cols["pa47"], cols["pa48"],
    )

    # Drinking
    df["drinkStatus"] = derive_drink_status(df, cols["pa50"], cols["pa51"], cols["pa53"])
    df["drinkExposure"] = derive_drink_exposure(df, cols["pa58"], cols["pa59"])
    df["drinkExposure_PA59"] = derive_drink_exposure_pa59(df, cols["pa59"])

    # Family history
    df["familyHeartAttack"] = derive_family_heart_attack(
        df,
        cols["pa10a"], cols["pa10b"], cols["pa10c"],
        cols["pa10d"], cols["pa10e"], cols["pa10f"],
        cols["pa10g"], cols["pa10h"],
    )
    df["familyCancer"] = derive_family_cancer(
        df,
        cols["pa30a"], cols["pa30b"], cols["pa30c"],
        cols["pa30d"], cols["pa30e"], cols["pa30f"],
    )

    # Psychological
    df["subAchievement"] = derive_sub_achievement(df, cols["pg5"], cols["pg5a"])

    # ACE composites
    df["educationACE"] = derive_ace_composite(df, cols["educationACE_items"])
    df["educationACE_3item"] = derive_ace_composite(df, cols["educationACE_3_items"])
    df["famDysfuncACE"] = derive_ace_composite(df, cols["famDysfuncACE_items"])
    df["famDysfuncACE_3item"] = derive_ace_composite(df, cols["famDysfuncACE_3_items"])

    # ecoACE: single item
    eco_col = cols["ecoACE_item"]
    if eco_col in df.columns:
        df["ecoACE"] = df[eco_col].astype("float64")
        df["ecoACE_1item"] = df[eco_col].astype("float64")
    else:
        df["ecoACE"] = np.nan
        df["ecoACE_1item"] = np.nan

    # Absent constructs -> NA
    for var_name in ABSENT_VARIABLES.get(cohort, []):
        if var_name not in df.columns:
            df[var_name] = np.nan

    # ecoCurr, famLossACE, famLossCurr, legalEvent, abuseEvent, relationship
    # For W1 cohorts these are derived from age-split SE11 items (complex);
    # for W2 cohorts they are intentionally NA.
    for composite in ["ecoCurr", "famLossACE", "famLossCurr",
                      "legalEvent", "abuseEvent", "relationship"]:
        item_key = f"{composite}_items"
        items = cols.get(item_key, [])
        if items:
            df[composite] = derive_ace_composite(df, items)
        elif composite not in df.columns:
            df[composite] = np.nan

    # Physical activity
    act_df = derive_activity_composites(
        df, cols["vig_cols"], cols["mod_cols"], cols["light_cols"],
    )
    for c in act_df.columns:
        df[c] = act_df[c]

    # Health affect work
    df["healthAffectWork"] = derive_health_affect_work(df, cols["pa4"], cols["pa5"])

    # Step 7: CIRS
    log.info("[%s] Step 7  Computing CIRS", cohort)
    cirs_df = derive_cirs(
        df,
        pa8=cols["pa8"], pa7ba=cols["pa7ba"], pa7bb=cols["pa7bb"],
        pangin=cols["pangin"],
        sa11s=cols["sa11s"], sa12a=cols["sa12a"],
        sa11a=cols["sa11a"], sa11b=cols["sa11b"],
        sa7d=cols["sa7d"],
        sa12f=cols["sa12f"], sa11aa=cols["sa11aa"],
        sa11l=cols["sa11l"],
        sa11j=cols["sa11j"],
        sa11d=cols["sa11d"], sa12g=cols["sa12g"],
        sa11v=cols["sa11v"], pa6c=cols["pa6c"], pa6d=cols["pa6d"], sa11y=cols["sa11y"],
        sa11x=cols["sa11x"], sa11g=cols["sa11g"],
        pa26=cols["pa26"], pa28a=cols["pa28a"],
        sa11z=cols["sa11z"], pa6a=cols["pa6a"],
    )
    df["cirs"] = cirs_df["cirs"]
    df["cirs_no_angina"] = cirs_df["cirs_no_angina"]

    # Middle age (sex-dependent)
    # Clamp SE items to [10, 100] before deriving
    for se_col in [cols["se_male_start"], cols["se_male_end"],
                   cols["se_female_start"], cols["se_female_end"]]:
        if se_col in df.columns:
            df[se_col] = df[se_col].where(
                (df[se_col] >= 10) & (df[se_col] <= 100), np.nan
            )

    df["startMiddleAge"] = derive_start_middle_age(
        df, cols["prsex"], cols["se_male_start"], cols["se_female_start"],
    )
    df["endMiddleAge"] = derive_end_middle_age(
        df, cols["prsex"], cols["se_male_end"], cols["se_female_end"],
    )

    # Step 8: Rename columns (SPSS -> analysis names via xlsx key)
    log.info("[%s] Step 8  Renaming columns", cohort)
    rename_map = get_cohort_map(cohort)
    rename_existing = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=rename_existing)

    # Drop columns that still have the SPSS prefix (not in the rename map
    # and not derived variables we just created)
    derived_cols = {
        "female", "hispanic", "race", "workForPay", "workStatus", "manageRole",
        "occGroup", "workIndustry", "relatStatus", "marrDuration", "timesMarried",
        "relatCohabLength", "partnerAgeGap", "partnerEduGap", "sexPartner",
        "careGivStatus", "careGivIntensity", "smokeStatus", "smokeDuration",
        "smokeAmount", "smokeExposure", "smokeExposure_3item",
        "drinkStatus", "drinkExposure", "drinkExposure_PA59",
        "familyHeartAttack", "familyCancer", "subAchievement",
        "educationACE", "educationACE_3item", "famDysfuncACE", "famDysfuncACE_3item",
        "ecoACE", "ecoACE_1item", "ecoCurr", "famLossACE", "famLossCurr",
        "legalEvent", "abuseEvent", "relationship",
        "vmJobAct", "vmChoresAct", "vmLeisureAct",
        "vmJobAct_vig", "vmChoresAct_vig", "vmLeisureAct_vig", "lightAct",
        "healthAffectWork", "cirs", "cirs_no_angina",
        "startMiddleAge", "endMiddleAge",
        "cohort", "sampleId",
    }
    # Also keep any column that was renamed (now has analysis name)
    renamed_cols = set(rename_existing.values())

    # Drop SPSS-prefix columns, work variables, temp columns
    to_drop = []
    for c in df.columns:
        if c in derived_cols or c in renamed_cols:
            continue
        # Keep M2MRID / M2ID / MRID (already renamed)
        if c in ("M2MRID", "M2ID", "MRID"):
            continue
        # Drop anything with SPSS prefix, _workvar, _temp, _FLAG, or
        # count_nonmissing patterns
        if (c.startswith(p) or
            "_workvar" in c or
            "_temp" in c or
            "_FLAG" in c or
            "nonmissing" in c or
            c == "interviewDate"):
            to_drop.append(c)
    if to_drop:
        df = df.drop(columns=to_drop, errors="ignore")

    # Step 9: MIDUSR sampleWeight rename
    if cohort == "midusr" and "MRCOMBWT" in df.columns:
        df = df.rename(columns={"MRCOMBWT": "sampleWeight"})

    # Deduplicate columns (derive step may create cols that rename also creates)
    df = df.loc[:, ~df.columns.duplicated(keep="last")]

    # Step 9b: Missing-reason mask (written separately, not merged)
    log.info("[%s] Step 9b  Missing-reason mask", cohort)
    _emit_missing_reasons(sav_path, cohort, df, rename_existing)

    elapsed = time.time() - t0
    log.info("[%s] Steps 1-9 complete: %d rows x %d cols (%.1fs)",
             cohort, len(df), len(df.columns), elapsed)
    return df


# Cross-cohort helpers (steps 10-15)

def apply_mipca_override(
    df: pd.DataFrame,
    item_cols: list[str],
    output_col: str,
) -> pd.DataFrame:
    """Column-mean imputation + sklearn PCA (n_components=1).

    Column means fill missing values, then PC1 scores from a one-component
    PCA are stored in *output_col*.

    Parameters
    ----------
    df : pd.DataFrame
    item_cols : list[str]  Source item columns.
    output_col : str       Name for the PC1 score column.

    Returns
    -------
    pd.DataFrame  with *output_col* added/replaced.
    """
    present = [c for c in item_cols if c in df.columns]
    if not present:
        log.warning("MIPCA: no item columns found for %s; setting to NaN", output_col)
        df[output_col] = np.nan
        return df

    sub = df[present].astype("float64")
    # Column-mean imputation
    col_means = sub.mean()
    sub_imputed = sub.fillna(col_means)

    # Drop rows still containing NaN after imputation (a column that is
    # entirely NaN has a NaN mean, so fillna does nothing)
    valid_mask = sub_imputed.notna().all(axis=1)
    if valid_mask.sum() < 2:
        log.warning("MIPCA: fewer than 2 valid rows for %s; setting to NaN", output_col)
        df[output_col] = np.nan
        return df

    pca = PCA(n_components=1)
    scores = np.full(len(df), np.nan)
    scores[valid_mask.values] = pca.fit_transform(sub_imputed.loc[valid_mask]).ravel()
    df[output_col] = scores

    log.info("  MIPCA %s: %d items, %d valid rows, explained_var=%.3f",
             output_col, len(present), valid_mask.sum(),
             pca.explained_variance_ratio_[0])
    return df


# Non-comparable variable lists

# Wave 1: these are dropped after pooling (replaced by _3item / _vig / _PA59
# / _no_angina alternatives)
NONCOMPARABLE_W1: list[str] = [
    "sexIntimacy", "billPayDifficulty",
    "demandAtHome", "controlTimeHome", "timeGetThingDone", "interruptionHome",
    "prideWorkHome", "respectWorkHome", "workPayMoreReward",
    "feltAge", "desiredAge",
    "educationACE", "famDysfuncACE", "ecoACE",
    "smokeExposure", "drinkExposure", "cirs",
    "vmJobAct", "vmLeisureAct", "vmChoresAct",
]

# Wave 2: rename composites to _3item / _1item names, then drop originals
NONCOMPARABLE_W2_DROP: list[str] = [
    "drinkExposure", "cirs",
    "vmJobAct", "vmLeisureAct", "vmChoresAct",
]

RENAME_W2: dict[str, str] = {
    "educationACE":  "educationACE_3item",
    "famDysfuncACE": "famDysfuncACE_3item",
    "ecoACE":        "ecoACE_1item",
    "smokeExposure": "smokeExposure_3item",
}


# Pool and finalize (steps 10-15)

def pool_and_finalize(
    dfs: dict[str, pd.DataFrame],
    wave: int,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Pool cohort DataFrames and apply cross-cohort steps (10-15).

    Parameters
    ----------
    dfs : dict[str, pd.DataFrame]
        {cohort_name: processed_df} from process_cohort().
    wave : int
        1 or 2.
    output_path : Path, optional
        Where to save the CSV. If None, uses default location.

    Returns
    -------
    pd.DataFrame  The pooled, finalized DataFrame.
    """
    t0 = time.time()

    # Step 10: Pool cohorts
    log.info("Step 10  Pooling %d cohort(s)", len(dfs))
    for name, df in dfs.items():
        df["cohort"] = COHORT_LABELS[name]
    pooled = pd.concat(list(dfs.values()), ignore_index=True)
    log.info("  Pooled: %d rows x %d cols", len(pooled), len(pooled.columns))

    # Outcomes and CFA composites are not joined here; they enter in the master
    # builds and build_analysis_ready.py.

    # Step 13: Drop non-comparable variables
    log.info("Step 13  Dropping non-comparable variables")
    if wave == 1:
        n_drop = sum(1 for v in NONCOMPARABLE_W1 if v in pooled.columns)
        pooled = pooled.drop(columns=[v for v in NONCOMPARABLE_W1 if v in pooled.columns])
        log.info("  Dropped %d non-comparable W1 composites", n_drop)
    else:
        # W2: the originals equal the _3item/_1item columns created in step 6
        # (same source items), so drop the originals.
        w2_originals_to_drop = [k for k in RENAME_W2 if k in pooled.columns]
        if w2_originals_to_drop:
            pooled = pooled.drop(columns=w2_originals_to_drop)
            log.info("  Dropped %d W2 originals (kept _3item/_1item versions): %s",
                     len(w2_originals_to_drop), w2_originals_to_drop)

        n_drop = sum(1 for v in NONCOMPARABLE_W2_DROP if v in pooled.columns)
        pooled = pooled.drop(
            columns=[v for v in NONCOMPARABLE_W2_DROP if v in pooled.columns],
        )
        log.info("  Dropped %d non-comparable W2 composites", n_drop)

    # Step 13b: Drop intermediate columns not in final analysis set
    # Keep only: analysis variables (from variable_config.yaml) + outcomes + metadata + composites
    log.info("Step 13b  Dropping intermediate columns")
    import yaml
    config_path = PROJECT_ROOT / "config" / "variable_config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            vcfg = yaml.safe_load(f)
        keep_vars = set(vcfg.get("tier1", []) + vcfg.get("tier2", []) + vcfg.get("tier3", []))
        # Add outcomes, metadata, composites, join columns
        keep_vars.update([
            "M2MRID", "age", "sampleWeight", "sampleId", "cohort", "wave",
            "cirs", "cirs_no_angina",
            "flag_edu_change",
        ])
        # Constituent items needed by cfa_composites.R (dropped later by build_analysis_ready.py)
        keep_vars.update([
            # psychWellbeing (9 items)
            "autonomy", "envMastery", "persGrowth", "posRelation",
            "lifePurpose", "selfAccept", "persMastery", "percConstraints", "selfEsteem",
            # giveSupport (5)
            "givePartnerEmoSupp", "giveParentEmoSupp", "giveInLawEmoSupp",
            "giveChildEmoSupp", "giveFriendEmoSupp",
            # recSupport (5)
            "recPartnerEmoSupp", "recParentEmoSupp", "recInLawEmoSupp",
            "recChildEmoSupp", "recFriendEmoSupp",
            # religiosity (5)
            "spirituality", "religiousId", "attendReligious",
            "religiousCopingA", "religiousCopingB",
            # activeCoping (5)
            "copingGrowth", "copingActive", "planning",
            "primaryControl", "posReappraisal",
            # spouseQuality (6)
            "spouseSupport", "spouseStrain", "spouseSolidarity",
            "spouseDisagree", "rateCurrentRelation", "maritalRisk",
            # familyQuality (4)
            "famSupport", "famStrain", "famSolidarity", "supportForFam",
            # friendQuality (3)
            "friendSupport", "friendStrain", "friendSolidarity",
            # lifeSatisfaction (4)
            "lifeSat", "lifeOverall", "rateToday", "rateCurrentFin",
            # sleepAvg (2)
            "sleepWorkday", "sleepWknd",
        ])
        before = len(pooled.columns)
        pooled = pooled[[c for c in pooled.columns if c in keep_vars]]
        log.info("  Kept %d/%d columns (dropped %d intermediate)",
                 len(pooled.columns), before, before - len(pooled.columns))

    # Step 14: Add metadata
    log.info("Step 14  Adding metadata")
    pooled["wave"] = wave

    # Step 15: Save CSV
    if output_path is None:
        if wave == 1:
            out_dir = PROJECT_ROOT / "data" / "processed" / "wave1"
            output_path = out_dir / "all_M2MR_py.csv"
        else:
            out_dir = PROJECT_ROOT / "data" / "processed" / "wave2"
            output_path = out_dir / "all_M3MR2_py.csv"
    else:
        out_dir = output_path.parent

    out_dir.mkdir(parents=True, exist_ok=True)
    pooled.to_csv(output_path, index=False)

    elapsed = time.time() - t0
    log.info("Step 15  Saved: %s", output_path)
    log.info("  Final: %d rows x %d cols (%.1fs)", len(pooled), len(pooled.columns), elapsed)

    # Summary
    for label in pooled["cohort"].unique():
        n = (pooled["cohort"] == label).sum()
        log.info("  %s: %d rows", label, n)

    return pooled


# Main entry point

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Survey QC pipeline",
    )
    parser.add_argument(
        "--wave", type=int, required=True, choices=[1, 2],
        help="Wave to process (1 = MIDUS2+MIDUSR, 2 = MIDUS3+MR2)",
    )
    parser.add_argument(
        "--cohort", type=str, default=None,
        choices=["midus2", "midusr", "midus3", "mr2"],
        help="Process a single cohort only (skip pooling)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Override output CSV path",
    )
    args = parser.parse_args()

    wave = args.wave
    cohorts = WAVE_COHORTS[wave]

    if args.cohort:
        if args.cohort not in cohorts:
            log.error("Cohort %s is not in wave %d (%s)", args.cohort, wave, cohorts)
            sys.exit(1)
        cohorts = [args.cohort]

    log.info("QC pipeline, wave %d", wave)
    log.info("Cohorts: %s", cohorts)

    processed: dict[str, pd.DataFrame] = {}
    for cohort in cohorts:
        sav = SAV_PATHS[cohort]
        if not sav.exists():
            log.error("SAV file not found: %s", sav)
            sys.exit(1)
        processed[cohort] = process_cohort(sav, cohort)

    # If only one cohort requested, save it directly without pooling
    if args.cohort:
        out = args.output or str(
            PROJECT_ROOT / "data" / "processed"
            / f"wave{wave}" / f"{args.cohort}_qc.csv"
        )
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df = processed[args.cohort]
        df["cohort"] = COHORT_LABELS[args.cohort]
        df.to_csv(out_path, index=False)
        log.info("Single-cohort output: %s (%d rows x %d cols)",
                 out_path, len(df), len(df.columns))
    else:
        out_path = Path(args.output) if args.output else None
        pool_and_finalize(processed, wave, out_path)

    log.info("Done.")


if __name__ == "__main__":
    main()
