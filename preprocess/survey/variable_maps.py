"""
SPSS -> analysis variable name mappings for all 4 MIDUS cohorts.
Source of truth: data/key_map/midus2R_key_extended.xlsx

Each cohort has a dict mapping SPSS column names to standardised analysis names.
The xlsx key has columns: value, midus2, midusr, midus3, mr2, category, remove,
plus Note_for_longitudinal (col 11).

Cohorts: midus2 (wave 1, prefix B1), midusr (wave 1, RA1), midus3 (wave 2, C1),
mr2 (wave 2, RB1).

Usage
-----
>>> from preprocess.survey.variable_maps import get_cohort_map, get_recode_config
>>> m2_map = get_cohort_map("midus2")       # {SPSS_name: analysis_name}
>>> m2_cfg = get_recode_config("midus2")     # recode type lists for apply_recodes()
"""

from __future__ import annotations

import string
from pathlib import Path
import pandas as pd

from preprocess.common.paths import KEY_MAP_DIR

__all__ = [
    "COHORT_PREFIXES",
    "ABSENT_VARIABLES",
    "VARIABLE_RENUMBERING_MIDUS3",
    "MR2_SECTION_SHIFTS",
    "load_variable_key",
    "get_cohort_map",
    "get_recode_config",
]

# Constants

#: SPSS column prefix for each cohort
COHORT_PREFIXES: dict[str, str] = {
    "midus2": "B1",
    "midusr": "RA1",
    "midus3": "C1",
    "mr2": "RB1",
}

#: Variables that are absent (NA) in certain cohorts.
#: These are genuinely absent constructs, not coding errors.
ABSENT_VARIABLES: dict[str, list[str]] = {
    "midus2": [],
    "midusr": [],
    "midus3": [
        # ACE life-event composites (SE11 has only A-G; H+ absent)
        "ecoCurr",         # SE11 Y,W,X,J,I all absent
        "famLossACE",      # SE11 K,O,Q all absent
        "famLossCurr",     # SE11 K,O,P all absent
        "legalEvent",      # SE11 U,V absent
        "abuseEvent",      # SE11 S,T absent
        "relationship",    # SE11 M,N absent
        # Demand-at-home items (SF33/SF34 repurposed in MIDUS3)
        "demandAtHome",
        "demandAtHomeSub",
        # Subjective age items (SE14/SE15 repurposed in MIDUS3)
        "feltAge",
        "desiredAge",
        # PA45A/B/C (childhood smoke exposure) absent/renumbered
    ],
    "mr2": [
        # Same ACE gaps as MIDUS3
        "ecoCurr",         # SF12B is ordinal 1-3, not binary ecoCurr
        "famLossACE",
        "famLossCurr",
        "legalEvent",
        "abuseEvent",
        "relationship",
        # MR2-specific absences
        "familyId",        # not collected in MR2
        # Angina (PANGIN) absent -> cirs_no_angina = cirs
        # PA45A/B/C absent (childhood smoke exposure)
        # PA38A (binary smoke ever) -> MR2 renumbered to PA37
    ],
}

#: Variable renumbering between MIDUS2/MIDUSR and MIDUS3.
#: MIDUS3 renumbered several SAQ sections.  The keys are the MIDUS2 suffix
#: patterns and the values are the MIDUS3 equivalents.
VARIABLE_RENUMBERING_MIDUS3: dict[str, str] = {
    # Activity section: MIDUS2 SA30/31/32 -> MIDUS3 SA26/27/28
    "SA30": "SA26",  # vigorous activity
    "SA31": "SA27",  # moderate activity
    "SA32": "SA28",  # light activity
    # Leisure / reading activities: SA33 -> SA29
    "SA33": "SA29",
    # Weight change: SA43 -> SA39
    "SA43": "SA39",
    # Medication items: SA48 -> SA44
    "SA48": "SA44",
    # Sleep quality: SA52 -> SA48
    "SA52": "SA48",
    # Alt-medicine: SA56 -> SA52
    "SA56": "SA52",
    # Nap frequency: SA60 -> SA56
    "SA60": "SA56",
    # Word games: SA33B -> SA29B  (sub-item of SA33)
    # Spirituality: SA67 -> SA65
    "SA67": "SA65",
}

#: MR2 section shifts relative to MIDUSR.
#: In MIDUSR: work=SG, financial=SH.
#: In MR2:    work=SF, financial=SG, contribution-to-others=SH.
MR2_SECTION_SHIFTS: dict[str, dict[str, str]] = {
    "work": {
        "midusr_section": "SG",
        "mr2_section": "SF",
        "example": "rateCurrentWork: RA1SG1 -> RB1SF1",
    },
    "financial": {
        "midusr_section": "SH",
        "mr2_section": "SG",
        "example": "rateCurrentFin: RA1SH1 -> RB1SG1",
    },
    "contribution_to_others": {
        "midusr_section": "SI",
        "mr2_section": "SH",
        "example": "emotional support items: RA1SI9/10 -> RB1SH9/10",
    },
}


# Key file path

def _key_path() -> Path:
    """Resolve the canonical xlsx key file under data/key_map."""
    return KEY_MAP_DIR / "midus2R_key_extended.xlsx"


# Public API


def load_variable_key(path: str | Path | None = None) -> pd.DataFrame:
    """Load the extended variable key from the xlsx file.

    Parameters
    ----------
    path : str or Path, optional
        Override path to the xlsx file. Defaults to the canonical
        ``data/key_map/midus2R_key_extended.xlsx``.

    Returns
    -------
    pd.DataFrame
        Full key with columns: value, midus2, midusr, midus3, mr2,
        category, remove, plus any additional columns (e.g.
        Note_for_longitudinal).  Rows with ``remove == 1`` are
        already filtered out.
    """
    p = Path(path) if path is not None else _key_path()
    df = pd.read_excel(p, sheet_name=0)
    # Drop rows with remove == 1
    if "remove" in df.columns:
        df = df[df["remove"] != 1].reset_index(drop=True)
    return df


def get_cohort_map(cohort: str, path: str | Path | None = None) -> dict[str, str]:
    """Return {spss_name: analysis_name} for a single cohort.

    Parameters
    ----------
    cohort : str
        One of ``"midus2"``, ``"midusr"``, ``"midus3"``, ``"mr2"``.
    path : str or Path, optional
        Override xlsx path.

    Returns
    -------
    dict[str, str]
        Mapping from SPSS column name to the standardised analysis name.
        Entries where the SPSS name is NA (variable absent in that cohort)
        are excluded.

    Raises
    ------
    ValueError
        If *cohort* is not one of the four recognised cohorts.
    """
    if cohort not in COHORT_PREFIXES:
        raise ValueError(
            f"Unknown cohort {cohort!r}; choose from {list(COHORT_PREFIXES)}"
        )
    key = load_variable_key(path)
    # The xlsx has a column named after each cohort (midus2, midusr, midus3, mr2)
    # and a 'value' column with the analysis name. Map input, output and cirs rows.
    mask = key["category"].isin(["input", "output", "cirs"])
    sub = key.loc[mask, [cohort, "value"]].dropna(subset=[cohort, "value"])
    return dict(zip(sub[cohort], sub["value"]))


def get_recode_config(cohort: str) -> dict[str, list[str]]:
    """Return the full recode configuration for a cohort.

    The returned dict has 10 keys matching the recode engine's expectations:
    ``binary_vars``, ``three_vars``, ``three_vars_alt``, ``four_vars``,
    ``five_vars``, ``six_vars``, ``six_weighted_vars``, ``sqrt_vars``,
    ``skip_binary_recode``, ``count_to_binary``.

    All SPSS column names use the correct cohort prefix and any
    wave-specific renumbering.

    Parameters
    ----------
    cohort : str
        One of ``"midus2"``, ``"midusr"``, ``"midus3"``, ``"mr2"``.

    Returns
    -------
    dict[str, list[str]]
        Recode configuration suitable for passing to
        ``recode_engine.apply_recodes()``.

    Raises
    ------
    ValueError
        If *cohort* is not recognised.
    """
    if cohort not in COHORT_PREFIXES:
        raise ValueError(
            f"Unknown cohort {cohort!r}; choose from {list(COHORT_PREFIXES)}"
        )
    return _RECODE_CONFIGS[cohort]


# Recode configurations per cohort
# Each list holds the exact SPSS variable names used for that cohort.

_LETTERS = list(string.ascii_uppercase)  # A-Z


def _expand(prefix: str, base: str, suffixes: list[str]) -> list[str]:
    """Build ``[prefix+base+s for s in suffixes]``."""
    return [f"{prefix}{base}{s}" for s in suffixes]


# MIDUS2 (B1)

_M2_binary_vars: list[str] = [
    "B1PB6", "B1PB7", "B1PB30",
    "B1PC3",
    "B1PD1", "B1PD7", "B1PD10", "B1PD11", "B1PD12", "B1PD13",
    "B1PE1", "B1PE2", "B1PE3",
    "B1PA38A", "B1PA39", "B1PA41", "B1PA45A", "B1PA45B", "B1PA45C",
    "B1PA46", "B1PA47", "B1PA48",
    "B1PA50", "B1PA58", "B1PA59",
    "B1SG17",
    "B1SA14A", "B1SA14B", "B1SA14N",
    "B1SA48A", "B1SA48B", "B1SA48C", "B1SA48D", "B1SA48E", "B1SA48F",
    "B1SA48H", "B1SA52",
    "B1SA62A", "B1SA62B", "B1SA62C", "B1SA62D", "B1SA62E",
    "B1SALCOH", "B1SC1", "B1SC8A", "B1SC8B",
    *_expand("B1SE2", "", _LETTERS[:16]),        # B1SE2A..P
    *_expand("B1SE11", "", _LETTERS + ["AA"]),   # B1SE11A..Z, B1SE11AA
]

_M2_recode: dict[str, list[str]] = {
    "binary_vars": _M2_binary_vars,
    "three_vars": ["B1SG6"],
    "three_vars_alt": ["B1SA43A", "B1SA43B"],
    "four_vars": [
        "B1SG7", "B1SM9", "B1SM10",
        "B1PG4", "B1PG6", "B1PG7", "B1PG8", "B1PG9",
    ],
    "five_vars": [
        "B1PA1", "B1PA2", "B1PA3",
        "B1SA6A", "B1SA6B", "B1SA6C", "B1SA6D", "B1SA6E",
        "B1SA7A", "B1SA7B", "B1SA7C", "B1SA7D",
        "B1SA56C", "B1SA56E", "B1SA56F", "B1SA56G", "B1SA56H",
        "B1SA56L", "B1SA56M", "B1SA56N", "B1SA56O", "B1SA56P",
        "B1SF33A", "B1SF33B", "B1SF33C", "B1SF33D",
        "B1SF34A", "B1SF34B", "B1SF34C",
    ],
    "six_vars": [
        "B1SA33A", "B1SA33B", "B1SA33C", "B1SA33D", "B1SA33E", "B1SA33F",
        "B1SM8", "B1SA67", "B1SI1", "B1SI2", "B1SN3D",
    ],
    "six_weighted_vars": [
        *_expand("B1SA30", "", [f"{c}" for c in _LETTERS[:6]]),   # B1SA30A..F
        *_expand("B1SA31", "", [f"{c}" for c in _LETTERS[:6]]),   # B1SA31A..F
        *_expand("B1SA32", "", [f"{c}" for c in _LETTERS[:6]]),   # B1SA32A..F
    ],
    "sqrt_vars": [
        "B1SH9A", "B1SH9B", "B1SH9C", "B1SH9D", "B1SH9E",
        "B1SH10A", "B1SH10B", "B1SH10C", "B1SH10D", "B1SH10E",
    ],
    # skip_binary_recode: these appear in binary_vars but must not be
    # run through the standard 1->1, 2->0, 3->NA recode.
    # - SALCOH: already coded 0/1 in SAV
    # - PANGIN: coded 0-3 (angina class), recoded separately in derive_cirs() (derived_health.py)
    "skip_binary_recode": ["B1SALCOH", "B1PANGIN"],
    # count_to_binary: count variable where 0->0, >=1->1
    # - PC4: child loss count (0-4)
    "count_to_binary": ["B1PC4"],
}


# MIDUSR (RA1)

_MR_binary_vars: list[str] = [
    "RA1PB6", "RA1PB7", "RA1PB30",
    "RA1PD1", "RA1PD7", "RA1PD10", "RA1PD11", "RA1PD12", "RA1PD13",
    "RA1PE1", "RA1PE2", "RA1PE3",
    "RA1PA38A", "RA1PA39", "RA1PA41", "RA1PA45A", "RA1PA45B", "RA1PA45C",
    "RA1PA46", "RA1PA47", "RA1PA48",
    "RA1PA50", "RA1PA58", "RA1PA59",
    "RA1SH38",
    "RA1SF2A", "RA1SF2B", "RA1SF2C", "RA1SF2D", "RA1SF2E", "RA1SF2F",
    "RA1SF2G", "RA1SF2H", "RA1SF2I", "RA1SG1J",
    "RA1SF2K", "RA1SF2L", "RA1SF2M", "RA1SF2N", "RA1SF2O", "RA1SF2P",
    "RA1PC3", "RA1SA14A", "RA1SA14B", "RA1SA14M",
    "RA1SA44A", "RA1SA44B", "RA1SA44C", "RA1SA44D",
    "RA1SA44E", "RA1SA44F", "RA1SA44H", "RA1SA48",
    "RA1SA58A", "RA1SA58B", "RA1SA58C", "RA1SA58D", "RA1SA58E",
    "RA1SALCOH", "RA1SC1", "RA1SC8A", "RA1SC8B",
    *_expand("RA1SF11", "", _LETTERS[:7]),        # RA1SF11A..G
    *_expand("RA1SF12", "", _LETTERS[:19]),        # RA1SF12A..S
]

_MR_recode: dict[str, list[str]] = {
    "binary_vars": _MR_binary_vars,
    "three_vars": ["RA1SH7"],
    "three_vars_alt": ["RA1SA39AA", "RA1SA39AB"],
    "four_vars": [
        "RA1SN10", "RA1SN11",
        "RA1PG4", "RA1PG6", "RA1PG7", "RA1PG8", "RA1PG9",
    ],
    "five_vars": [
        "RA1PA1", "RA1PA2", "RA1PA3",
        "RA1SA6A", "RA1SA6B", "RA1SA6C", "RA1SA6D", "RA1SA6E",
        "RA1SA7A", "RA1SA7B", "RA1SA7C", "RA1SA7D",
        "RA1SA52C", "RA1SA52E", "RA1SA52F", "RA1SA52G", "RA1SA52H",
        "RA1SA52L", "RA1SA52M", "RA1SA52N", "RA1SA52O", "RA1SA52P",
        "RA1SG44A", "RA1SG44B", "RA1SG44C", "RA1SG44D",
        "RA1SG45A", "RA1SG45B", "RA1SG45C",
    ],
    "six_vars": [
        "RA1SA29A", "RA1SA29B", "RA1SA29C", "RA1SA29D", "RA1SA29E", "RA1SA29F",
        "RA1SA65", "RA1SJ1", "RA1SJ2", "RA1SP3D",
    ],
    "six_weighted_vars": [
        *_expand("RA1SA26", "", [c for c in _LETTERS[:6]]),
        *_expand("RA1SA27", "", [c for c in _LETTERS[:6]]),
        *_expand("RA1SA28", "", [c for c in _LETTERS[:6]]),
    ],
    "sqrt_vars": [
        "RA1SI9A", "RA1SI9B", "RA1SI9C", "RA1SI9D", "RA1SI9E",
        "RA1SI10A", "RA1SI10B", "RA1SI10C", "RA1SI10D", "RA1SI10E",
    ],
    "skip_binary_recode": ["RA1SALCOH", "RA1PANGIN"],
    "count_to_binary": ["RA1PC4"],
}


# MIDUS3 (C1)
# Key renumbering vs MIDUS2:
#   Activity  SA30/31/32 -> SA26/27/28
#   Leisure   SA33       -> SA29
#   WtChange  SA43       -> SA39
#   Medication SA48      -> SA44
#   Sleep     SA52       -> SA48
#   AltMed    SA56       -> SA52
#   Nap       SA60       -> SA56
#   Spirit    SA67       -> SA65
#
# MIDUS3 financial section: inserted "before recession" item (C1SG2),
# pushing ago/future/control/effort down by one (SG2->SG3, SG3->SG4, etc.).
# moneyForNeeds is C1SG7.

_M3_binary_vars: list[str] = [
    "C1PB6", "C1PB7", "C1PB30",
    "C1PC3",
    "C1PD1", "C1PD7", "C1PD10", "C1PD11", "C1PD12", "C1PD13",
    "C1PE1", "C1PE2", "C1PE3",
    "C1PA38A", "C1PA39", "C1PA41", "C1PA45A", "C1PA45B", "C1PA45C",
    "C1PA46", "C1PA47", "C1PA48",
    "C1PA50", "C1PA58", "C1PA59",
    "C1SG38",            # housingStatus (own/rent)
    "C1SA14A", "C1SA14B", "C1SA14N",
    "C1SA44A", "C1SA44B", "C1SA44C", "C1SA44D", "C1SA44E", "C1SA44F",
    "C1SA44H", "C1SA48",
    "C1SA62A", "C1SA62B", "C1SA62C", "C1SA62D", "C1SA62E",
    "C1SALCOH", "C1SC1", "C1SC8A", "C1SC8B",
    *_expand("C1SE2", "", _LETTERS[:16]),         # C1SE2A..P
    *_expand("C1SE11", "", _LETTERS[:7]),          # C1SE11A..G (only A-G in MIDUS3)
]

_M3_recode: dict[str, list[str]] = {
    "binary_vars": _M3_binary_vars,
    "three_vars": ["C1SG7"],
    "three_vars_alt": ["C1SA39AA", "C1SA39AB"],   # renumbered from SA43
    "four_vars": [
        "C1SM10",   # sexImportance (construct changed from SM9+SM10 in MIDUS2)
        "C1PG4", "C1PG6", "C1PG7", "C1PG8", "C1PG9",
    ],
    "five_vars": [
        "C1PA1", "C1PA2", "C1PA3",
        "C1SA6A", "C1SA6B", "C1SA6C", "C1SA6D", "C1SA6E",
        "C1SA7A", "C1SA7B", "C1SA7C", "C1SA7D",
        # Alt-medicine: SA56->SA52 in MIDUS3
        "C1SA52C", "C1SA52E", "C1SA52F", "C1SA52G", "C1SA52H",
        "C1SA52L", "C1SA52M", "C1SA52N", "C1SA52O", "C1SA52P",
        # Demand-at-home: SF33/SF34 absent in MIDUS3 (skipped if missing)
        "C1SF33A", "C1SF33B", "C1SF33C", "C1SF33D",
        "C1SF34A", "C1SF34B", "C1SF34C",
    ],
    "six_vars": [
        # Leisure reading: SA33->SA29 in MIDUS3
        *_expand("C1SA29", "", _LETTERS[:6]),
        "C1SM9", "C1SA65", "C1SI1", "C1SI2", "C1SN3D",
    ],
    "six_weighted_vars": [
        # Activity: SA30->SA26, SA31->SA27, SA32->SA28 in MIDUS3
        *_expand("C1SA26", "", [c for c in _LETTERS[:6]]),   # vigorous
        *_expand("C1SA27", "", [c for c in _LETTERS[:6]]),   # moderate
        *_expand("C1SA28", "", [c for c in _LETTERS[:6]]),   # light
    ],
    "sqrt_vars": [
        # SH9/SH10 with 6 sub-items each (A-F, not A-E like MIDUS2)
        *_expand("C1SH9", "", _LETTERS[:6]),
        *_expand("C1SH10", "", _LETTERS[:6]),
    ],
    "skip_binary_recode": ["C1SALCOH"],
    # PANGIN exists in MIDUS3 but is not in binary_vars -- recoded in derive_cirs() (derived_health.py)
    "count_to_binary": ["C1PC4"],
}


# MR2 (RB1)
# MR2 section shifts vs MIDUSR:
#   work:       SG -> SF    (e.g. rateCurrentWork: RA1SG1 -> RB1SF1)
#   financial:  SH -> SG    (e.g. rateCurrentFin:  RA1SH1 -> RB1SG1)
#   support:    SI -> SH    (e.g. emotional support: RA1SI9 -> RB1SH9)
#
# MR2 renumbered smoking: PA38A->PA37, PA38B->PA38, PA37->PA36
# MR2 lacks: PA45A/B/C, PA58, POCMAJ/PMJ, PINDMJ, PPINMJ, PF8A, PANGIN

_MR2_binary_vars: list[str] = [
    "RB1PB6", "RB1PB7", "RB1PB30",
    "RB1PD1", "RB1PD7", "RB1PD10", "RB1PD11", "RB1PD12", "RB1PD13",
    "RB1PE1", "RB1PE2", "RB1PE3",
    "RB1PA37", "RB1PA39", "RB1PA41", "RB1PA45A", "RB1PA45B", "RB1PA45C",
    "RB1PA46", "RB1PA47", "RB1PA48",
    "RB1PA50", "RB1PA58", "RB1PA59",
    "RB1SG38",            # housingStatus
    "RB1SF2",             # ACE screening single item
    "RB1PC3", "RB1SA14A", "RB1SA14B", "RB1SA14M",
    "RB1SA44A", "RB1SA44B", "RB1SA44C", "RB1SA44D",
    "RB1SA44E", "RB1SA44F", "RB1SA44H", "RB1SA48",
    "RB1SA58A", "RB1SA58B", "RB1SA58C", "RB1SA58D",
    "RB1SA58E", "RB1SALCOH",
    "RB1SC1", "RB1SC8A", "RB1SC8B",
    *_expand("RB1SE11", "", _LETTERS[:7]),         # RB1SE11A..G
]

_MR2_recode: dict[str, list[str]] = {
    "binary_vars": _MR2_binary_vars,
    "three_vars": ["RB1SG7"],
    "three_vars_alt": ["RB1SA39AA", "RB1SA39AB"],
    "four_vars": [
        "RB1SM10",
        "RB1PG4", "RB1PG6", "RB1PG7", "RB1PG8", "RB1PG9",
    ],
    "five_vars": [
        "RB1PA1", "RB1PA2", "RB1PA3",
        "RB1SA6A", "RB1SA6B", "RB1SA6C", "RB1SA6D", "RB1SA6E",
        "RB1SA7A", "RB1SA7B", "RB1SA7C", "RB1SA7D",
        "RB1SA52C", "RB1SA52E", "RB1SA52F", "RB1SA52G", "RB1SA52H",
        "RB1SA52L", "RB1SA52M", "RB1SA52N", "RB1SA52O", "RB1SA52P",
        # MR2 demand-at-home: section SG44/SG45 (shifted from SF)
        "RB1SG44A", "RB1SG44B", "RB1SG44C", "RB1SG44D",
        "RB1SG45A", "RB1SG45B", "RB1SG45C",
    ],
    "six_vars": [
        *_expand("RB1SA29", "", _LETTERS[:6]),
        "RB1SM9", "RB1SA65", "RB1SJ1", "RB1SJ2",
        "RB1SN3D",   # attendReligious (MR2 uses SN3D, not SP3D as in MIDUSR)
    ],
    "six_weighted_vars": [
        *_expand("RB1SA26", "", [c for c in _LETTERS[:6]]),
        *_expand("RB1SA27", "", [c for c in _LETTERS[:6]]),
        *_expand("RB1SA28", "", [c for c in _LETTERS[:6]]),
    ],
    "sqrt_vars": [
        # MR2: SH9/SH10 (support section shifted from SI in MIDUSR)
        *_expand("RB1SH9", "", _LETTERS[:6]),
        *_expand("RB1SH10", "", _LETTERS[:6]),
    ],
    "skip_binary_recode": ["RB1SALCOH"],
    # PANGIN absent in MR2 entirely -- not in binary_vars, no skip needed
    "count_to_binary": ["RB1PC4"],
}

_RECODE_CONFIGS: dict[str, dict[str, list[str]]] = {
    "midus2": _M2_recode,
    "midusr": _MR_recode,
    "midus3": _M3_recode,
    "mr2": _MR2_recode,
}


# Reverse-code entries applied outside the recode engine
# Applied in qc_pipeline.py, not by apply_recodes.

#: Reverse code 8 (friend contact frequency, 9-x):
#:   midus2: B1SJ3, midusr: RA1SK5, midus3: C1SJ5, mr2: RB1SJ5
REVERSE_8_FRIEND_CONTACT: dict[str, str] = {
    "midus2": "B1SJ3",
    "midusr": "RA1SK5",
    "midus3": "C1SJ5",
    "mr2": "RB1SJ5",
}


# CLI entry point

if __name__ == "__main__":
    import sys

    print("MIDUS variable mapping summary")

    try:
        key = load_variable_key()
    except FileNotFoundError:
        print(
            f"\nERROR: Cannot find key file at {_key_path()}\n"
            "Check data/key_map/."
        )
        sys.exit(1)

    print(f"\nKey file: {_key_path()}")
    print(f"Total rows (after remove filter): {len(key)}")
    print(f"Columns: {list(key.columns)}")
    print(f"Categories: {key['category'].value_counts().to_dict()}")

    for cohort in ["midus2", "midusr", "midus3", "mr2"]:
        prefix = COHORT_PREFIXES[cohort]
        col = cohort
        print(f"\nCohort: {cohort} (prefix: {prefix})")
        if col in key.columns:
            total = key[col].notna().sum()
            n_input = key.loc[(key["category"] == "input") & key[col].notna()].shape[0]
            n_output = key.loc[(key["category"] == "output") & key[col].notna()].shape[0]
            n_cirs = key.loc[(key["category"] == "cirs") & key[col].notna()].shape[0]
            n_absent = key[col].isna().sum()
            print(f"  Mapped variables : {total} ({n_input} input, {n_output} output, {n_cirs} cirs)")
            print(f"  Absent (NA)      : {n_absent}")
        else:
            print(f"  Column '{col}' not found in key!")

        cfg = get_recode_config(cohort)
        print(f"  Recode config:")
        for rtype, vars_list in cfg.items():
            print(f"    {rtype:25s}: {len(vars_list)} vars")

        try:
            cmap = get_cohort_map(cohort)
            print(f"  Cohort map size  : {len(cmap)} entries")
        except Exception as e:
            print(f"  Cohort map ERROR : {e}")

    print("\nAbsent variables by cohort")
    for cohort, absent in ABSENT_VARIABLES.items():
        if absent:
            print(f"  {cohort}: {', '.join(absent)}")
        else:
            print(f"  {cohort}: (none)")

    print("\nMIDUS3 variable renumbering (MIDUS2 -> MIDUS3)")
    for old, new in VARIABLE_RENUMBERING_MIDUS3.items():
        print(f"  B1{old}* -> C1{new}*")

    print("\nMR2 section shifts (vs MIDUSR)")
    for domain, info in MR2_SECTION_SHIFTS.items():
        print(f"  {domain:30s}: {info['midusr_section']} -> {info['mr2_section']}  ({info['example']})")

    print()
