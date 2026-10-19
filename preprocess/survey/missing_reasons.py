"""Missing-reason masks for the SPSS sources.

Every ``.sav`` in this project declares user-missing ranges, and the value labels
attached to those codes say why a cell is missing: refused, does not apply, no
SAQ returned, not calculated. pyreadstat's default read (``user_missing=False``)
turns all of them into ``NaN``, so missing-by-design and item nonresponse look the
same, and the frailty imputer fills both alike.

The analysis frame keeps that collapse. This module rereads each source with
``user_missing=True`` for the stage's own column list and writes the codes next to
the stage's outputs, without touching the frame the pipeline computes on.

The mask stores raw codes; the codebook maps (variable, code) to a label and a
category, so categories can be revised without rebuilding the mask.

A reason is only defined for a column that comes from exactly one source variable.
Derived columns (``workForPay``, ``smokeStatus``, the CIRS score) combine several
sources and sometimes a skip pattern, so they are listed in the coverage file with
``reason: derived`` instead.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

log = logging.getLogger(__name__)

# The main distinction is BY_DESIGN vs NONRESPONSE; the rest refine it. CONFLATED
# marks labels where the source file already merged a refusal with a blank, which
# no read can separate.
BY_DESIGN = "by_design"
NONRESPONSE = "nonresponse"
CONFLATED = "nonresponse_or_missing"
DERIVED_UNAVAILABLE = "derived_unavailable"
INVALID = "invalid"
UNSPECIFIED = "unspecified"
UNLABELED = "unlabeled"
UNCLASSIFIED = "unclassified"

CATEGORIES = (BY_DESIGN, NONRESPONSE, CONFLATED, DERIVED_UNAVAILABLE,
              INVALID, UNSPECIFIED, UNLABELED, UNCLASSIFIED)

# Rules run in order and the first match wins, so the conflated and
# "not calculated" forms must be tried before the bare REFUSED and MISSING ones
# they contain. Written against the 86 distinct label texts actually present in
# data/raw; unmatched labels become UNCLASSIFIED.
_RULES: tuple[tuple[str, str], ...] = (
    # source already merged two reasons: keep that visible
    (r"^REFUSED (OR|/) MISSING$", CONFLATED),
    (r"^INVALID OR MISSING$", CONFLATED),
    (r"^MISSING OR NOT REPORTED$", CONFLATED),
    # derived variables the depositor could not compute
    (r"^NOT CALCULATED", DERIVED_UNAVAILABLE),
    (r"^ALLOCATION REQUIRED$", DERIVED_UNAVAILABLE),
    # structurally not asked
    (r"^INAPP", BY_DESIGN),
    (r"^NOT APPLICABLE", BY_DESIGN),
    (r"^DOES NOT APPLY$", BY_DESIGN),
    (r"DOES NOT HAVE SAQ", BY_DESIGN),
    (r"^NOT IN LABOR FORCE$", BY_DESIGN),
    # skip patterns: not asked because of a substantive fact about the person
    (r"^NEVER", BY_DESIGN),
    (r"^NO CHILDREN", BY_DESIGN),
    (r"^STILL UNEMPLOYED$", BY_DESIGN),
    (r"^ALWAYS IRREGULAR$", BY_DESIGN),
    (r"^DID NOT SLEEP$", BY_DESIGN),
    (r"^CHILDHOOD$", BY_DESIGN),
    (r"^CONGENITAL/AT BIRTH$", BY_DESIGN),
    (r"HAS ?N.?T ENDED$", BY_DESIGN),
    (r"WORK\w* FROM HOME", BY_DESIGN),
    # the respondent was asked and did not give a usable answer
    (r"^REFUS", NONRESPONSE),
    (r"^DON.{0,3}KNOW", NONRESPONSE),
    (r"^UNKNOWN OR UNSURE$", NONRESPONSE),
    # answered, but the answer cannot be used
    (r"^INCORRECT$", INVALID),
    (r"^INVALID$", INVALID),
    (r"^UNABLE TO CLASSIFY$", INVALID),
    (r"^PROBLEM HEARING", INVALID),
    # declared missing with no reason given
    (r"^MISSING$", UNSPECIFIED),
    (r"^NO CAUSE OF DEATH FOUND$", UNSPECIFIED),
)

_COMPILED = tuple((re.compile(p), c) for p, c in _RULES)


def normalize_label(text: object) -> str:
    """Fold a value label to the form the rules are written against.

    The same reason is spelled several ways across the sixteen files: title case
    against upper case, a curly apostrophe against a straight one, and the typos
    DONT KNOW, DON' KNOW, DONT' KNOW and DON'TKNOW. Matching raw text would file
    those as different reasons.
    """
    s = unicodedata.normalize("NFKD", str(text))
    s = s.replace("’", "'").replace("‘", "'")
    s = re.sub(r"\s+", " ", s).strip().upper()
    return s


def classify_label(text: object) -> str:
    """Map one value label to a category, or UNCLASSIFIED if no rule matches."""
    norm = normalize_label(text)
    for rx, cat in _COMPILED:
        if rx.search(norm):
            return cat
    return UNCLASSIFIED


def _ranges(meta, var: str) -> list[tuple]:
    return [(r.get("lo"), r.get("hi")) for r in (meta.missing_ranges or {}).get(var, [])]


def _missing_mask(series: pd.Series, ranges: list[tuple]) -> np.ndarray:
    """Cells of ``series`` falling in a declared user-missing range.

    Both numeric and string ranges occur: B1PSOC is an occupation code stored as
    text whose missing values are declared as the strings '99-9998' and '99-9999'.
    """
    mask = np.zeros(len(series), dtype=bool)
    for lo, hi in ranges:
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
            if pd.api.types.is_numeric_dtype(series):
                mask |= ((series >= lo) & (series <= hi)).fillna(False).to_numpy()
        elif lo is not None and hi is not None:
            text = series.astype("string")
            hit = (text >= str(lo)) & (text <= str(hi))
            mask |= hit.fillna(False).to_numpy()
    return mask


def build_codebook(sav_path: str | Path) -> pd.DataFrame:
    """Every (variable, code) the file declares missing, with label and category."""
    sav_path = Path(sav_path)
    _, meta = pyreadstat.read_sav(str(sav_path), metadataonly=True, user_missing=True)
    labels = meta.variable_value_labels or {}
    rows = []
    for var in meta.column_names:
        rngs = _ranges(meta, var)
        if not rngs:
            continue
        seen = False
        for code, text in (labels.get(var) or {}).items():
            try:
                num = float(code)
            except (TypeError, ValueError):
                continue
            if any(isinstance(lo, (int, float)) and isinstance(hi, (int, float))
                   and lo <= num <= hi for lo, hi in rngs):
                rows.append(dict(source_file=sav_path.name, variable=var, code=code,
                                 label=str(text).strip(), category=classify_label(text)))
                seen = True
        if not seen:
            # A range with no label attached: the file says the cell is missing
            # but not why.
            for lo, hi in rngs:
                rows.append(dict(source_file=sav_path.name, variable=var,
                                 code=f"{lo}..{hi}" if lo != hi else lo,
                                 label="", category=UNLABELED))
    return pd.DataFrame(rows, columns=["source_file", "variable", "code", "label", "category"])


def build_reason_mask(
    sav_path: str | Path,
    columns: list[str],
    id_col: str,
    rename: dict[str, str] | None = None,
    keep_ids: set | None = None,
    id_name: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Long-format mask of declared-missing cells, plus a per-variable summary.

    Returns ``(mask, summary)``. ``mask`` has one row per missing cell with the id,
    the analysis-facing column name, the source variable and the raw code. Long
    format suits the sparsity (about a fifth of cells are missing) and keeps the
    source variable on every cell.

    ``rename`` maps source variable to the name the stage's output uses, so the
    mask can be joined to the output without knowing the SPSS names. ``id_name``
    does the same for the id column itself: the survey stage renames ``M2ID`` and
    ``MRID`` to ``M2MRID`` on the way out, and a mask keyed on the SPSS name would
    not join to the file it describes. Columns absent from the file are reported in
    the summary rather than raised on.
    """
    sav_path = Path(sav_path)
    rename = rename or {}
    _, meta_all = pyreadstat.read_sav(str(sav_path), metadataonly=True, user_missing=True)
    available = set(meta_all.column_names)

    wanted = [c for c in columns if c in available]
    absent = [c for c in columns if c not in available]
    need = sorted(set(wanted) | {id_col})

    frame, meta = pyreadstat.read_sav(str(sav_path), usecols=need,
                                      user_missing=True, apply_value_formats=False)
    if keep_ids is not None:
        # Restrict to the stage's kept ids, so the rows match its output exactly.
        frame = frame[frame[id_col].isin(keep_ids)].copy()

    ids = frame[id_col]
    parts, summary = [], []
    for var in wanted:
        rngs = _ranges(meta, var)
        series = frame[var]
        mask = _missing_mask(series, rngs)
        n = int(mask.sum())
        summary.append(dict(
            source_file=sav_path.name, variable=var,
            analysis_name=rename.get(var, var),
            declares_ranges=bool(rngs), n_rows=len(series),
            n_declared_missing=n,
            n_system_missing=int(series.isna().sum()),
        ))
        if n:
            parts.append(pd.DataFrame({
                id_col: ids[mask].to_numpy(),
                "variable": var,
                "analysis_name": rename.get(var, var),
                "code": series[mask].to_numpy(),
            }))
    for var in absent:
        summary.append(dict(source_file=sav_path.name, variable=var,
                            analysis_name=rename.get(var, var), declares_ranges=False,
                            n_rows=0, n_declared_missing=0, n_system_missing=0))

    cols = [id_col, "variable", "analysis_name", "code"]
    mask_df = (pd.concat(parts, ignore_index=True) if parts
               else pd.DataFrame(columns=cols))
    mask_df = mask_df.sort_values([id_col, "variable"], kind="stable").reset_index(drop=True)
    if id_name and id_name != id_col:
        mask_df = mask_df.rename(columns={id_col: id_name})
    return mask_df, pd.DataFrame(summary)


def write_reason_artifacts(
    sav_path: str | Path,
    columns: list[str],
    id_col: str,
    out_dir: str | Path,
    stem: str,
    rename: dict[str, str] | None = None,
    derived_columns: list[str] | None = None,
    keep_ids: set | None = None,
    id_name: str | None = None,
) -> dict[str, Path]:
    """Write the mask, the codebook and the coverage note for one stage.

    ``derived_columns`` names output columns that have no single source variable.
    They are listed in the coverage file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mask, summary = build_reason_mask(sav_path, columns, id_col, rename=rename,
                                      keep_ids=keep_ids, id_name=id_name)
    codebook = build_codebook(sav_path)

    key = codebook.set_index(["variable", "code"])["category"].to_dict()
    if len(mask):
        mask["category"] = [key.get((v, c), UNCLASSIFIED)
                            for v, c in zip(mask["variable"], mask["code"])]
    else:
        mask["category"] = pd.Series(dtype=object)

    coverage = pd.DataFrame(
        [dict(column=c, covered=False, reason="derived from several source variables")
         for c in sorted(derived_columns or [])]
    )

    paths = {
        "mask": out_dir / f"{stem}_missing_reason_mask.csv",
        "summary": out_dir / f"{stem}_missing_reason_summary.csv",
        "codebook": out_dir / f"{stem}_missing_reason_codebook.csv",
        "coverage": out_dir / f"{stem}_missing_reason_coverage.csv",
    }
    mask.to_csv(paths["mask"], index=False)
    summary.to_csv(paths["summary"], index=False)
    codebook.to_csv(paths["codebook"], index=False)
    coverage.to_csv(paths["coverage"], index=False)

    counts = mask["category"].value_counts().to_dict() if len(mask) else {}
    unclassified = counts.get(UNCLASSIFIED, 0)
    log.info("[%s] missing-reason mask: %d cells over %d columns; %s",
             stem, len(mask), len(summary),
             ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "none")
    if unclassified:
        log.warning("[%s] %d cells carry a label no rule matched; see the codebook",
                    stem, unclassified)
    return paths
