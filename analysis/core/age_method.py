"""Record which age-adjustment method produced each estimate.

Age adjustment is a call-site argument. ``load_master`` takes ``age_residualize``
and, when true, regresses the outcome on one age column across the whole sample.
When false it returns the age array and ``XGBoostPipeline.run_cv`` residualizes
inside each fold. ``d1_descriptive`` has its own implementation.

This module records, per load, the flag passed, the age column used, and whether
residualization was global or within-fold, and writes it next to the outputs.
``load_master`` opens an entry; ``run_cv`` completes it if it residualizes within
folds, since only it knows whether the age array was passed on.

Entries are appended in call order and never overwritten, so loading the same
outcome twice under different flags gives two rows.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# Every resolution this process has made, in call order.
_LEDGER: list[dict] = []

GLOBAL = "global_whole_sample"
WITHIN_FOLD = "within_fold"
NONE = "none"

COLUMNS = ["seq", "stage", "outcome", "flag_passed", "age_column",
           "scope", "longitudinal", "concurrent", "cohort_filter", "n_rows"]


def record_load(
    outcome: str,
    flag_passed: bool,
    age_column: str | None,
    applied_global: bool,
    *,
    longitudinal: bool = False,
    concurrent: bool = False,
    cohort_filter: str | None = None,
    n_rows: int | None = None,
    stage: str = "load_master",
) -> dict:
    """Open an entry for one load_master call and return it.

    ``scope`` starts at whatever load_master itself did. If the caller goes on to
    pass the age array to run_cv, run_cv upgrades it to within-fold.
    """
    entry = {
        "seq": len(_LEDGER) + 1,
        "stage": stage,
        "outcome": outcome,
        "flag_passed": bool(flag_passed),
        "age_column": age_column,
        "scope": GLOBAL if applied_global else NONE,
        "longitudinal": bool(longitudinal),
        "concurrent": bool(concurrent),
        "cohort_filter": cohort_filter,
        "n_rows": n_rows,
    }
    _LEDGER.append(entry)
    return entry


def mark_within_fold(age_supplied: bool) -> None:
    """Upgrade the most recent entry when run_cv residualizes inside folds.

    run_cv is not told which load it belongs to, so this assumes each load is
    followed by its own fit.
    """
    if not age_supplied or not _LEDGER:
        return
    last = _LEDGER[-1]
    if last["scope"] == GLOBAL:
        # Outcome residualized twice; record both.
        last["scope"] = f"{GLOBAL}+{WITHIN_FOLD}"
    else:
        last["scope"] = WITHIN_FOLD


def entries() -> list[dict]:
    return list(_LEDGER)


def as_frame() -> pd.DataFrame:
    return pd.DataFrame(_LEDGER, columns=COLUMNS)


def resolved() -> dict | None:
    """The most recent resolved triple, for annotating a table as it is written."""
    return dict(_LEDGER[-1]) if _LEDGER else None


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    """Add the resolved triple to a results table as three columns.

    Returns the frame unchanged when nothing has been recorded, so a table written
    by a pipeline that never calls load_master (a cell-count table, say) does not
    gain empty columns claiming a method it never used.
    """
    r = resolved()
    if r is None:
        return df
    out = df.copy()
    out["age_adj_flag"] = r["flag_passed"]
    out["age_adj_column"] = r["age_column"]
    out["age_adj_scope"] = r["scope"]
    return out


def write_ledger(out_dir: Path | str, pipeline: str) -> Path | None:
    """Write the ledger beside a pipeline's own tables.

    Writes one age_adjustment_method.csv per output directory, even when nothing
    was recorded, so published tables keep their schema.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "age_adjustment_method.csv"
    if _LEDGER:
        frame = as_frame()
    else:
        # No load_master call, so no residualization: write an explicit 'none' row
        # so the file is never missing. a6, for example, decomposes age into
        # between- and within-pair components and does not residualize.
        frame = pd.DataFrame([{
            "seq": 0, "stage": "no load_master call in this run", "outcome": None,
            "flag_passed": None, "age_column": None, "scope": NONE,
            "longitudinal": None, "concurrent": None, "cohort_filter": None,
            "n_rows": None,
        }], columns=COLUMNS)
    frame.insert(1, "pipeline", pipeline)
    frame.to_csv(path, index=False)
    return path


def emit_at_exit(out_dir: Path | str, pipeline: str) -> None:
    """Write the ledger when the process ends, however it ends.

    Registered with atexit at the top of a pipeline's main(), so a run that fails
    still writes what it resolved.
    """
    import atexit
    atexit.register(write_ledger, out_dir, pipeline)


def reset() -> None:
    """Clear the ledger. For tests, and for a process that runs two pipelines."""
    _LEDGER.clear()
