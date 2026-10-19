"""Fit the cross-sectional models the profile comparison evaluates.

    uv run python analysis/profile_comparison/fit_models.py

For each primary outcome, fits the Tier 1 + Tier 2 XGBoost model on the full
cross-sectional sample with the pipeline's own tuning (`XGBoostPipeline.fit_final`)
and writes to output/profile_comparison/models/<outcome>/:

  model.json    the booster
  meta.json     feature order, categorical levels, tuned parameters, training n
  context.json  the reference row (median of each numeric predictor, modal level of
                each categorical one) and the age-residualized outcome's percentiles

profile_comparison.py holds non-focal predictors at the reference row and places
each profile's prediction on the percentile grid.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.core import artifact_paths as ap  # noqa: E402
from analysis.core.data_loader import CATEGORICALS, load_master  # noqa: E402

MODELS_DIR = ROOT / "output" / "profile_comparison" / "models"
PRIMARY = ["FInoPsy_W1", "btactComp_W1", f"{ap.AL_STEM}_W1"]
TIERS = [1, 2]


def is_categorical(series: pd.Series, name: str) -> bool:
    return name in CATEGORICALS or series.dtype.name == "category"


def reference_row(X: pd.DataFrame) -> dict:
    ref = {}
    for col in X.columns:
        s = X[col]
        if is_categorical(s, col):
            mode = s.astype("category").mode(dropna=True)
            ref[col] = None if mode.empty else str(mode.iloc[0])
        else:
            v = pd.to_numeric(s, errors="coerce").dropna()
            ref[col] = None if v.empty else round(float(v.median()), 6)
    return ref


def distribution(y: pd.Series) -> dict:
    arr = np.asarray(y, dtype=float)
    arr = arr[~np.isnan(arr)]
    grid = np.arange(1, 100)
    return {
        "n": int(arr.size),
        "mean": round(float(arr.mean()), 6),
        "sd": round(float(arr.std()), 6),
        "percentile_grid": [int(g) for g in grid],
        "percentiles": [round(float(v), 6) for v in np.percentile(arr, grid)],
    }


def write(outcome: str, booster, best_params: dict) -> None:
    X, y, features, _ = load_master(outcome, tiers=TIERS, age_residualize=True)
    out = MODELS_DIR / outcome
    out.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out / "model.json"))
    meta = {
        "outcome": outcome,
        "feature_names": list(features),
        "categorical_features": {c: [str(v) for v in X[c].astype("category").cat.categories]
                                 for c in features if is_categorical(X[c], c)},
        "best_params": dict(best_params),
        "n_train": int(y.notna().sum()),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    valid = y.notna().values
    context = {"reference": reference_row(X[valid]), "distribution": distribution(y)}
    (out / "context.json").write_text(json.dumps(context, indent=1) + "\n")
    print(f"  {outcome}: {len(features)} predictors, n = {meta['n_train']}")


def fit(outcome: str) -> None:
    from analysis.core.xgboost_pipeline import XGBoostPipeline

    X, y, features, age = load_master(outcome, tiers=TIERS, age_residualize=False)
    pipe = XGBoostPipeline(n_folds=5, n_bootstrap=0, tune=True, use_gpu=True)
    model, best_params = pipe.fit_final(X, y, features, age=age)
    write(outcome, model.get_booster(), best_params)


def main() -> int:
    for outcome in PRIMARY:
        fit(outcome)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
