#!/usr/bin/env python3
"""Shared MIDUS mortality source loader and endpoint definitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MASTER_CSV = ROOT / "data/processed/master_analysis.csv"
MORTALITY_DIR = ROOT / "data/raw/mortality"
MORTALITY_CORE_SAV = MORTALITY_DIR / "MIDUS_Core_MortalityCauseData_N2822_20260511.sav"
MORTALITY_REFRESHER_SAV = MORTALITY_DIR / "MIDUS_Refresher_MortalityCauseData_N454_20260511.sav"
MORTALITY_SOURCE_FILES = {
    "core": MORTALITY_CORE_SAV,
    "refresher": MORTALITY_REFRESHER_SAV,
}
MORTALITY_SOURCE_LABEL = " + ".join(path.name for path in MORTALITY_SOURCE_FILES.values())
MIDUS2_W1_SAV = ROOT / "data/raw/wave1/M2_P1_SURVEY_N4963_20200720.sav"
MIDUSR_W1_SAV = ROOT / "data/raw/wave1/MR1_P1_SURVEY_N3577_20250407.sav"

EXPECTED_COLUMNS = ("id", "DOD_M", "DOD_Y", "UNDERCOD_ICD10")
MORTALITY_SOURCE_CONTRACT = {
    "core": {
        "path": MORTALITY_CORE_SAV,
        "columns": ("M2ID", "DOD_M", "DOD_Y", "UNDERCOD_ICD10"),
        "rename": {"M2ID": "id"},
    },
    "refresher": {
        "path": MORTALITY_REFRESHER_SAV,
        "columns": ("MRID", "DOD_M", "DOD_Y", "UNDERCOD"),
        "rename": {"MRID": "id", "UNDERCOD": "UNDERCOD_ICD10"},
    },
}
PRIMARY_CENSOR_YEAR = 2025
PRIMARY_CENSOR_DATE = pd.Timestamp(year=2025, month=12, day=31)
EXPECTED_MORTALITY_SHA256 = {
    "core": "bd0312848b8560a57779f81fdc11132c47519a6d21bb973ed1543494b50c169b",
    "refresher": "06f114566cd6dcb1c2289d39469b2187b6a2a4630bf1c35c393fcb3af7aebc06",
}


@dataclass(frozen=True)
class MortalityEndpoint:
    key: str
    label: str
    censor_year: int
    horizon_years: int | None
    primary: bool
    output_suffix: str


ENDPOINTS: dict[str, MortalityEndpoint] = {
    "through_2025": MortalityEndpoint(
        key="through_2025",
        label="all available follow-up through 2025",
        censor_year=PRIMARY_CENSOR_YEAR,
        horizon_years=None,
        primary=True,
        output_suffix="",
    ),
    "fixed_9y": MortalityEndpoint(
        key="fixed_9y",
        label="fixed 9-year mortality sensitivity",
        censor_year=PRIMARY_CENSOR_YEAR,
        horizon_years=9,
        primary=False,
        output_suffix="fixed_9y_sensitivity",
    ),
}

EXPECTED_ENDPOINT_SUMMARIES = {
    "through_2025": {
        "n": 6651,
        "n_events": 1747,
        "pre_baseline_excluded": 0,
        "mean_follow_up": 15.981,
    },
    "fixed_9y": {
        "n": 6651,
        "n_events": 608,
        "pre_baseline_excluded": 0,
        "mean_follow_up": 8.638,
    },
}


def endpoint_config(endpoint: str) -> MortalityEndpoint:
    """Return a validated mortality endpoint config."""
    if endpoint not in ENDPOINTS:
        valid = ", ".join(sorted(ENDPOINTS))
        raise ValueError(f"Unknown mortality endpoint {endpoint!r}; expected one of: {valid}")
    return ENDPOINTS[endpoint]


def output_dir_for_endpoint(base_dir: Path, endpoint: str) -> Path:
    """Use the base directory for the primary endpoint and a subdir for sensitivity."""
    cfg = endpoint_config(endpoint)
    return base_dir if cfg.primary else base_dir / cfg.output_suffix


def file_sha256(path: Path) -> str:
    """Compute sha256 without loading large files into memory at once."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def mortality_source_hashes() -> dict[str, str]:
    """Return sha256 hashes for the two canonical mortality source files."""
    return {sample: file_sha256(path) for sample, path in MORTALITY_SOURCE_FILES.items()}


def mortality_source_hash_label() -> str:
    """Compact scalar source-hash label for CSV summaries."""
    return "; ".join(f"{sample}:{sha}" for sample, sha in mortality_source_hashes().items())


def load_mortality_raw(metadata_only: bool = False) -> pd.DataFrame:
    """Read and stack the canonical core/refresher mortality SAV files."""
    import pyreadstat

    frames: list[pd.DataFrame] = []
    for sample, contract in MORTALITY_SOURCE_CONTRACT.items():
        path = contract["path"]
        source_columns = list(contract["columns"])
        if not path.exists():
            raise FileNotFoundError(f"Mortality {sample} SAV not found: {path}")

        if metadata_only:
            meta_df, _ = pyreadstat.read_sav(str(path), metadataonly=True)
            missing = set(source_columns) - set(meta_df.columns)
            if missing:
                raise ValueError(
                    f"Mortality {sample} SAV missing expected columns: {sorted(missing)}"
                )
            frames.append(pd.DataFrame(columns=EXPECTED_COLUMNS))
            continue

        frame, _ = pyreadstat.read_sav(str(path), usecols=source_columns)
        missing = set(source_columns) - set(frame.columns)
        if missing:
            raise ValueError(
                f"Mortality {sample} SAV missing expected columns: {sorted(missing)}"
            )
        frame = frame.rename(columns=contract["rename"])
        frames.append(frame[list(EXPECTED_COLUMNS)])

    return pd.concat(frames, ignore_index=True)


def _make_midmonth_date(year: pd.Series, month: pd.Series) -> pd.Series:
    """Build approximate month-level dates using day 15 for partial dates."""
    out = pd.Series(pd.NaT, index=year.index, dtype="datetime64[ns]")
    ok = year.notna() & month.notna()
    if ok.any():
        out.loc[ok] = pd.to_datetime(
            {
                "year": pd.to_numeric(year.loc[ok], errors="coerce").astype(int),
                "month": pd.to_numeric(month.loc[ok], errors="coerce").astype(int),
                "day": 15,
            },
            errors="coerce",
        )
    return out


def load_baseline_dates() -> pd.DataFrame:
    """Read W1 interview month/year for MIDUS2 and MIDUS Refresher baselines."""
    import pyreadstat

    m2, _ = pyreadstat.read_sav(
        str(MIDUS2_W1_SAV),
        usecols=["M2ID", "B1PIDATE_MO", "B1PIDATE_YR"],
    )
    m2 = m2.rename(
        columns={
            "M2ID": "person_id",
            "B1PIDATE_MO": "baseline_month",
            "B1PIDATE_YR": "baseline_year_actual",
        }
    )
    m2["baseline_date_source"] = "MIDUS2_B1PIDATE"

    mr, _ = pyreadstat.read_sav(
        str(MIDUSR_W1_SAV),
        usecols=["MRID", "RA1PIDATE_MO", "RA1PIDATE_YR"],
    )
    mr = mr.rename(
        columns={
            "MRID": "person_id",
            "RA1PIDATE_MO": "baseline_month",
            "RA1PIDATE_YR": "baseline_year_actual",
        }
    )
    mr["baseline_date_source"] = "MIDUSR_RA1PIDATE"

    dates = pd.concat([m2, mr], ignore_index=True)
    dates["person_id"] = dates["person_id"].astype(int)
    dates["baseline_date"] = _make_midmonth_date(
        dates["baseline_year_actual"],
        dates["baseline_month"],
    )
    return dates


def mortality_source_summary() -> dict:
    """Return source-file metadata used by the manifest and validation checks."""
    mort = load_mortality_raw()
    years = pd.to_numeric(mort["DOD_Y"], errors="coerce")
    return {
        "paths": {
            sample: str(path.relative_to(ROOT))
            for sample, path in MORTALITY_SOURCE_FILES.items()
        },
        "sha256": mortality_source_hashes(),
        "source_label": MORTALITY_SOURCE_LABEL,
        "rows": int(len(mort)),
        "columns": list(mort.columns),
        "unique_ids": int(mort["id"].nunique()),
        "min_dod_y": int(years.min()),
        "max_dod_y": int(years.max()),
        "n_missing_dod_y": int(years.isna().sum()),
        "matching_to_master": mortality_match_summary(),
    }


def mortality_match_summary() -> dict:
    """Summarize how raw mortality IDs overlap the analytic master frame."""
    master = pd.read_csv(MASTER_CSV, usecols=["person_id", "cohort"], low_memory=False)
    mort = load_mortality_raw().copy()
    baseline_dates = load_baseline_dates()[["person_id", "baseline_date"]]

    master["person_id"] = master["person_id"].astype(int)
    mort["person_id"] = mort["id"].astype(int)

    matched = mort[["person_id"]].merge(
        master[["person_id", "cohort"]],
        on="person_id",
        how="inner",
    )
    matched_with_dates = matched.merge(
        baseline_dates,
        on="person_id",
        how="left",
    )
    events_by_cohort = matched.groupby("cohort")["person_id"].nunique().astype(int).to_dict()

    source_ids = set(mort["person_id"])
    master_ids = set(master["person_id"])
    return {
        "source_death_records": int(len(mort)),
        "source_unique_death_ids": int(mort["person_id"].nunique()),
        "master_rows": int(len(master)),
        "master_unique_ids": int(master["person_id"].nunique()),
        "matched_death_ids": int(len(source_ids & master_ids)),
        "unmatched_death_ids": int(len(source_ids - master_ids)),
        "matched_death_ids_with_baseline_date": int(
            matched_with_dates["baseline_date"].notna().sum()
        ),
        "events_by_cohort_for_matched_ids": events_by_cohort,
    }


def summarize_survival_frame(df: pd.DataFrame) -> dict:
    """Summarize an already constructed survival frame."""
    cohort_events = (
        df.groupby("cohort")["deceased"].sum().astype(int).to_dict()
        if "cohort" in df.columns
        else {}
    )
    return {
        "endpoint": df.attrs.get("mortality_endpoint", ""),
        "endpoint_label": df.attrs.get("mortality_endpoint_label", ""),
        "source_file": MORTALITY_SOURCE_LABEL,
        "source_sha256": mortality_source_hash_label(),
        "n": int(len(df)),
        "n_events": int(df["deceased"].sum()),
        "event_rate": float(df["deceased"].mean()),
        "mean_follow_up": float(df["follow_up_years"].mean()),
        "pre_baseline_excluded": int(df.attrs.get("pre_baseline_excluded", 0)),
        "nonpositive_death_follow_up": int(df.attrs.get("nonpositive_death_follow_up", 0)),
        "censor_year": int(df.attrs.get("mortality_censor_year", PRIMARY_CENSOR_YEAR)),
        "horizon_years": df.attrs.get("mortality_horizon_years"),
        "events_by_cohort": cohort_events,
    }


def load_survival_data(endpoint: str = "through_2025", verbose: bool = True) -> pd.DataFrame:
    """Join mortality data to the master analysis frame and construct an endpoint.

    Endpoints:
      - through_2025: event if DOD_Y is observed through the 2025 release.
      - fixed_9y: event if death occurs within 9 years after cohort baseline;
        later deaths are treated as censored at 9 years.
    """
    cfg = endpoint_config(endpoint)
    master = pd.read_csv(MASTER_CSV, low_memory=False)
    mort = load_mortality_raw()
    baseline_dates = load_baseline_dates()

    mort = mort.copy()
    mort["person_id"] = mort["id"].astype(int)
    df = master.merge(
        baseline_dates[
            [
                "person_id",
                "baseline_month",
                "baseline_year_actual",
                "baseline_date",
                "baseline_date_source",
            ]
        ],
        on="person_id",
        how="left",
    ).merge(
        mort[["person_id", "DOD_M", "DOD_Y", "UNDERCOD_ICD10"]],
        on="person_id",
        how="left",
    )

    if df["baseline_date"].isna().any():
        missing = int(df["baseline_date"].isna().sum())
        raise ValueError(f"Missing W1 baseline interview date for {missing} master rows")

    df["death_year"] = pd.to_numeric(df["DOD_Y"], errors="coerce")
    df["death_date"] = _make_midmonth_date(df["DOD_Y"], df["DOD_M"])
    years_to_death = (df["death_date"] - df["baseline_date"]).dt.days / 365.25
    nonpositive = df["death_date"].notna() & (years_to_death <= 0)
    n_nonpositive = int(nonpositive.sum())
    if n_nonpositive:
        raise ValueError(
            f"Found {n_nonpositive} mortality rows with non-positive follow-up "
            "after month-level baseline matching"
        )

    if cfg.horizon_years is None:
        df["deceased"] = df["death_date"].notna().astype(int)
        stop_date = df["death_date"].where(df["deceased"].eq(1), PRIMARY_CENSOR_DATE)
        df["follow_up_years"] = np.where(
            df["deceased"].eq(1),
            years_to_death,
            (stop_date - df["baseline_date"]).dt.days / 365.25,
        )
    else:
        within_horizon = df["death_date"].notna() & (years_to_death <= cfg.horizon_years)
        df["deceased"] = within_horizon.astype(int)
        df["follow_up_years"] = np.where(
            df["deceased"].eq(1),
            years_to_death,
            cfg.horizon_years,
        )

    df["mortality_endpoint"] = cfg.key
    df["mortality_endpoint_label"] = cfg.label
    df["mortality_source_file"] = MORTALITY_SOURCE_LABEL
    df["mortality_censor_year"] = cfg.censor_year
    df["mortality_horizon_years"] = cfg.horizon_years
    df.attrs["mortality_endpoint"] = cfg.key
    df.attrs["mortality_endpoint_label"] = cfg.label
    df.attrs["mortality_censor_year"] = cfg.censor_year
    df.attrs["mortality_horizon_years"] = cfg.horizon_years
    df.attrs["pre_baseline_excluded"] = 0
    df.attrs["nonpositive_death_follow_up"] = n_nonpositive

    summary = summarize_survival_frame(df)
    df.attrs["mortality_summary"] = summary
    if verbose:
        print(
            f"  Survival data [{cfg.key}]: n={summary['n']:,}, "
            f"events={summary['n_events']:,}, "
            f"follow-up mean={summary['mean_follow_up']:.1f}y"
        )
        if summary["pre_baseline_excluded"]:
            print(f"  Excluded {summary['pre_baseline_excluded']} pre-baseline deaths")
        if n_nonpositive:
            print(f"  Non-positive death follow-up rows: {n_nonpositive}")
        print(f"  By cohort: {summary['events_by_cohort']}")
    return df
