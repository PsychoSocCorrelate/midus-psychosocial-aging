"""
XGBoost with TreeSHAP: cross-validated R2 and bootstrap SHAP.

Implements:
  - 5-fold cross-validated XGBoost with tiered ΔR²
  - Bootstrap SHAP (200 iterations): mean |SHAP|, 95% CI, top-K stability

Usage:
    from analysis.core.xgboost_pipeline import XGBoostPipeline
    pipe = XGBoostPipeline(n_folds=5, n_bootstrap=200)
    cv_results = pipe.run_cv(X, y, feature_names)
    delta_r2 = pipe.compute_tiered_delta_r2(X_t1, X_t1t2, X_t1t2t3, y)
    shap_df = pipe.run_bootstrap_shap(X, y, feature_names)
"""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Optional
import warnings

import numpy as np
import pandas as pd

from analysis.core import age_method
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

try:
    import xgboost as xgb
except ImportError:
    raise ImportError("xgboost is required: uv run --with xgboost python ...")

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False


# Hyperparameter grid for per-fold tuning
XGB_TUNING_GRID = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.10],
    "subsample": [0.7, 1.0],
}
XGB_TUNING_GRID_SIZE = (
    len(XGB_TUNING_GRID["max_depth"])
    * len(XGB_TUNING_GRID["learning_rate"])
    * len(XGB_TUNING_GRID["subsample"])
)
XGB_TUNING_FIXED_PARAMS = {
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
}
XGB_TUNING_NUM_BOOST_ROUND = 1000
XGB_TUNING_EARLY_STOPPING_ROUNDS = 20


def _default_params(use_gpu: bool = True) -> dict:
    params = dict(
        max_depth=6,
        learning_rate=0.1,
        n_estimators=500,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        enable_categorical=True,  # native categorical splits
    )
    if use_gpu:
        try:
            import subprocess
            result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=3)
            if result.returncode == 0:
                import os
                gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
                params["device"] = f"cuda:{gpu_id}" if gpu_id.isdigit() else "cuda"
                params["tree_method"] = "hist"
                return params
        except Exception:
            pass
    params["device"] = "cpu"
    params["tree_method"] = "hist"
    return params


def _tune_params_grid(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    base_params: dict,
    n_inner_folds: int = 3,
    random_state: int = 42,
    objective: str = "reg:squarederror",
    eval_metric: str = "rmse",
) -> dict:
    """Tune XGBoost via grid search + xgb.cv early stopping within one outer fold.

    Grid over max_depth x learning_rate x subsample (18 combinations);
    n_estimators set by xgb.cv early stopping (max 1000 rounds, patience 20).

    Device/tree_method inherited from base_params.

    Returns best params dict.
    """
    device = base_params.get("device", "cpu")
    tree_method = base_params.get("tree_method", "hist")

    # Fixed (booster-level params only; enable_categorical is DMatrix/sklearn-level)
    fixed = {
        **XGB_TUNING_FIXED_PARAMS,
        "device": device,
        "tree_method": tree_method,
    }

    valid = ~np.isnan(y_train)
    X_v = X_train[valid].reset_index(drop=True)
    y_v = y_train[valid]

    dtrain = xgb.DMatrix(X_v, label=y_v, enable_categorical=True)

    best_score = -np.inf
    best_params = {}

    combos = list(product(
        XGB_TUNING_GRID["max_depth"],
        XGB_TUNING_GRID["learning_rate"],
        XGB_TUNING_GRID["subsample"],
    ))

    for md, lr, ss in combos:
        params = {
            "max_depth": md,
            "learning_rate": lr,
            "subsample": ss,
            "objective": objective,
            "eval_metric": eval_metric,
            "seed": random_state,
            **fixed,
        }

        cv_result = xgb.cv(
            params,
            dtrain,
            num_boost_round=XGB_TUNING_NUM_BOOST_ROUND,
            nfold=n_inner_folds,
            early_stopping_rounds=XGB_TUNING_EARLY_STOPPING_ROUNDS,
            verbose_eval=False,
            seed=random_state,
        )

        best_round = len(cv_result)
        # Lower-is-better metrics (rmse, logloss): negate for maximization
        best_metric = cv_result[f"test-{eval_metric}-mean"].iloc[-1]
        score = -best_metric

        if score > best_score:
            best_score = score
            best_params = {
                "max_depth": md,
                "learning_rate": lr,
                "subsample": ss,
                "n_estimators": best_round,
                **XGB_TUNING_FIXED_PARAMS,
                "random_state": random_state,
                "n_jobs": -1,
                "device": device,
                "tree_method": tree_method,
                "enable_categorical": True,
            }

    return best_params


def _fit_xgb(X: pd.DataFrame, y: np.ndarray, params: dict, task: str = "regression") -> xgb.XGBRegressor:
    """Fit an XGBRegressor/XGBClassifier, handling NaN in y by dropping those rows."""
    valid = ~np.isnan(y)
    if task == "classification":
        model = xgb.XGBClassifier(**params)
    else:
        model = xgb.XGBRegressor(**params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X[valid], y[valid])
    return model


def _fit_xgb_native(dmat: xgb.DMatrix, params: dict, num_boost_round: int = 200) -> xgb.Booster:
    """Fit via native xgb.train() on a pre-built DMatrix. Returns Booster."""
    # Convert sklearn-style params to native booster params
    native_params = {k: v for k, v in params.items()
                     if k not in ("n_estimators", "enable_categorical",
                                  "verbosity", "random_state")}
    native_params.setdefault("objective", "reg:squarederror")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        booster = xgb.train(native_params, dmat, num_boost_round=num_boost_round)
    return booster


def _shap_values(model: xgb.XGBRegressor, X: pd.DataFrame) -> np.ndarray:
    """Compute TreeSHAP values. Returns array (n_samples, n_features).

    Uses XGBoost's native pred_contribs (exact TreeSHAP, identical to
    shap.TreeExplainer) to avoid SHAP library version incompatibilities.
    The last column returned by XGBoost is the bias term; it is dropped.
    """
    booster = model.get_booster()
    # Native TreeSHAP; the booster runs on GPU and copies data as needed.
    # Device-mismatch warning from XGBoost is benign (expected when DMatrix is CPU).
    dmat = xgb.DMatrix(X, enable_categorical=True)
    sv = booster.predict(dmat, pred_contribs=True)  # shape: (n, n_features + 1)
    return sv[:, :-1]  # drop bias column


def _shap_values_native(booster: xgb.Booster, dmat: xgb.DMatrix) -> np.ndarray:
    """Compute TreeSHAP from a Booster + pre-built DMatrix. No pandas overhead."""
    sv = booster.predict(dmat, pred_contribs=True)  # shape: (n, n_features + 1)
    return sv[:, :-1]  # drop bias column


# Cross-validated R² and SHAP
class XGBoostPipeline:
    """XGBoost pipeline with SHAP importance and tiered ΔR² computation."""

    def __init__(
        self,
        n_folds: int = 5,
        n_bootstrap: int = 200,
        top_k: int = 20,
        random_state: int = 42,
        use_gpu: bool = True,
        tune: bool = True,
        n_inner_folds: int = 3,
        task: str = "regression",
    ):
        self.n_folds = n_folds
        self.n_bootstrap = n_bootstrap
        self.top_k = top_k
        self.random_state = random_state
        self.tune = tune
        self.n_inner_folds = n_inner_folds
        self.task = task
        self.params = _default_params(use_gpu=use_gpu)
        if task == "classification":
            self.params["objective"] = "binary:logistic"
            self.params["eval_metric"] = "logloss"
        print(f"  XGBoostPipeline: device={self.params.get('device','cpu')}, "
              f"n_folds={n_folds}, n_bootstrap={n_bootstrap}, "
              f"tune={tune}, task={task}"
              + (f" (grid: {XGB_TUNING_GRID_SIZE} combos × {n_inner_folds}-fold + early stop)" if tune else ""))

    def run_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: list[str],
        age: np.ndarray | None = None,
    ) -> dict:
        """5-fold CV: returns per-fold R², mean SHAP importance, and predictions.

        Args:
            X: Feature DataFrame (NaN preserved, numeric encoding for categoricals).
            y: Target Series (NaN preserved). Pass the raw outcome with `age` for
               within-fold residualization, or a pre-residualized outcome with
               age=None.
            feature_names: Column names in X.
            age: Optional 1-D array of age_W1 values (same length as X/y). When
                 provided, within-fold age residualization is performed: OLS fit on
                 train fold, residuals applied to both train and test. This avoids
                 train-test leakage.

        Returns:
            dict with keys:
              'r2_folds': list of per-fold R² on held-out test set
              'r2_mean': float
              'r2_std': float
              'shap_mean_abs': pd.Series indexed by feature_names (mean |SHAP| across folds)
              'y_pred': np.ndarray (full-sample OOB predictions, NaN where not predicted)
              'fold_shap': list of per-fold SHAP mean-abs arrays
        """
        from sklearn.linear_model import LinearRegression as _LR

        # Record whether age is residualized within folds (see age_method).
        age_method.mark_within_fold(age is not None)

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        y_arr = np.asarray(y, dtype=float)
        valid_mask = ~np.isnan(y_arr)

        r2_folds = []
        fold_shap_abs = []
        fold_models = []       # store (model, age_reg_or_None) per fold
        y_pred_full = np.full(len(y_arr), np.nan)

        X_valid = X[valid_mask].reset_index(drop=True)
        y_valid = y_arr[valid_mask]
        age_valid = age[valid_mask] if age is not None else None

        for fold_i, (train_idx, test_idx) in enumerate(kf.split(X_valid)):
            X_tr, X_te = X_valid.iloc[train_idx], X_valid.iloc[test_idx]
            y_tr, y_te = y_valid[train_idx], y_valid[test_idx]

            # Within-fold age residualization
            fold_age_reg = None
            if age_valid is not None:
                age_tr = age_valid[train_idx].reshape(-1, 1)
                age_te = age_valid[test_idx].reshape(-1, 1)
                age_train_valid = ~np.isnan(age_tr.ravel())
                age_test_valid = ~np.isnan(age_te.ravel())
                if age_train_valid.sum() > 5:
                    fold_age_reg = _LR()
                    fold_age_reg.fit(age_tr[age_train_valid], y_tr[age_train_valid])
                    # Only residualize rows with valid age (NaN rows keep raw y)
                    y_tr[age_train_valid] = y_tr[age_train_valid] - fold_age_reg.predict(age_tr[age_train_valid]).ravel()
                    y_te[age_test_valid] = y_te[age_test_valid] - fold_age_reg.predict(age_te[age_test_valid]).ravel()

            # Per-fold grid search, or fixed defaults
            if self.tune:
                fold_params = _tune_params_grid(
                    X_tr, y_tr,
                    base_params=self.params,
                    n_inner_folds=self.n_inner_folds,
                    random_state=self.random_state + fold_i,
                    objective=self.params.get("objective", "reg:squarederror"),
                    eval_metric=self.params.get("eval_metric", "rmse"),
                )
                print(f"    Fold {fold_i+1}: tuned depth={fold_params.get('max_depth')}, "
                      f"lr={fold_params.get('learning_rate'):.3f}, "
                      f"n_est={fold_params.get('n_estimators')}")
            else:
                fold_params = self.params

            model = _fit_xgb(X_tr, y_tr, fold_params, task=self.task)
            fold_models.append((model, fold_age_reg))

            if self.task == "classification":
                from sklearn.metrics import roc_auc_score
                y_prob = model.predict_proba(X_te)[:, 1]
                fold_metric = roc_auc_score(y_te, y_prob)
                y_pred_full[np.where(valid_mask)[0][test_idx]] = y_prob
            else:
                y_hat = model.predict(X_te)
                fold_metric = r2_score(y_te, y_hat)
                y_pred_full[np.where(valid_mask)[0][test_idx]] = y_hat
            r2_folds.append(fold_metric)

            # SHAP on test fold
            sv = _shap_values(model, X_te)
            fold_shap_abs.append(np.abs(sv).mean(axis=0))

            metric_name = "AUC" if self.task == "classification" else "R²"
            print(f"    Fold {fold_i+1}/{self.n_folds}: {metric_name}={fold_metric:.4f}, n_test={len(y_te)}")

        shap_mean = np.mean(fold_shap_abs, axis=0)
        shap_series = pd.Series(shap_mean, index=feature_names, name="mean_abs_shap")

        return {
            "r2_folds": r2_folds,
            "r2_mean": float(np.mean(r2_folds)),
            "r2_std": float(np.std(r2_folds)),
            "shap_mean_abs": shap_series,
            "y_pred": y_pred_full,
            "fold_shap": fold_shap_abs,
            "fold_models": fold_models,  # list of (xgb_model, age_reg_or_None)
        }

    def compute_tiered_delta_r2(
        self,
        X_t1: pd.DataFrame,
        X_t1t2: pd.DataFrame,
        X_t1t2t3: pd.DataFrame,
        y: pd.Series,
    ) -> dict:
        """Compute tiered ΔR² across three predictor sets.

        Fits three separate XGBoost models via CV and returns:
          R²_T1, R²_T1T2, R²_T1T2T3,
          ΔR²_T2 = R²_T1T2 − R²_T1,
          ΔR²_T3 = R²_T1T2T3 − R²_T1T2

        Returns dict with all R² values and deltas.
        """
        results = {}
        for label, X_tier in [("T1", X_t1), ("T1T2", X_t1t2), ("T1T2T3", X_t1t2t3)]:
            if X_tier is None or X_tier.empty:
                results[f"r2_{label}"] = np.nan
                results[f"r2_{label}_std"] = np.nan
                continue
            print(f"  Computing R² for {label} ({X_tier.shape[1]} features)...")
            cv = self.run_cv(X_tier, y, list(X_tier.columns))
            results[f"r2_{label}"] = cv["r2_mean"]
            results[f"r2_{label}_std"] = cv["r2_std"]

        results["delta_r2_T2"] = results.get("r2_T1T2", np.nan) - results.get("r2_T1", np.nan)
        results["delta_r2_T3"] = results.get("r2_T1T2T3", np.nan) - results.get("r2_T1T2", np.nan)
        return results

    def fit_final(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: list[str],
        age: np.ndarray | None = None,
    ) -> tuple[xgb.XGBRegressor, dict]:
        """Tune on full training set and fit a single final model.

        Use for sibling holdout validation: train on all non-sibling rows,
        then apply returned model to holdout via predict().

        Workflow:
            1. Grid search tunes on full training data (inner CV)
            2. Fits final model on full training data with best params
            3. Returns (model, best_params)

        Args:
            X: Full training feature matrix (non-sibling rows only)
            y: Full training target
            feature_names: Column names in X
            age: Optional age array for residualization (applied globally before tuning)

        Returns:
            (fitted_model, best_params_dict)
        """
        from sklearn.linear_model import LinearRegression as _LR

        y_arr = np.asarray(y, dtype=float)
        valid = ~np.isnan(y_arr)
        X_v = X[valid].reset_index(drop=True)
        y_v = y_arr[valid]

        # Global age residualization
        if age is not None:
            age_v = age[valid]
            age_valid = ~np.isnan(age_v)
            if age_valid.sum() > 10:
                reg = _LR()
                reg.fit(age_v[age_valid].reshape(-1, 1), y_v[age_valid])
                y_v[age_valid] = y_v[age_valid] - reg.predict(age_v[age_valid].reshape(-1, 1)).ravel()
            # Store regressor for holdout age residualization
            self._age_reg = reg
            self._age_fitted = True
        else:
            self._age_reg = None
            self._age_fitted = False

        if self.tune:
            best_params = _tune_params_grid(
                X_v, y_v,
                base_params=self.params,
                n_inner_folds=self.n_inner_folds,
                random_state=self.random_state,
                objective=self.params.get("objective", "reg:squarederror"),
                eval_metric=self.params.get("eval_metric", "rmse"),
            )
            print(f"  fit_final: tuned depth={best_params.get('max_depth')}, "
                  f"lr={best_params.get('learning_rate'):.3f}, "
                  f"n_est={best_params.get('n_estimators')}")
        else:
            best_params = self.params

        model = _fit_xgb(X_v, y_v, best_params, task=self.task)
        print(f"  fit_final: trained on N={len(y_v):,}, features={len(feature_names)}")
        return model, best_params

    def predict_holdout_ensemble(
        self,
        fold_models: list[tuple],
        X_holdout: pd.DataFrame,
        y_holdout: pd.Series,
        age_holdout: np.ndarray | None = None,
    ) -> dict:
        """Ensemble prediction on holdout using CV fold models.

        Each fold model predicts independently (with its own age regressor),
        then predictions are averaged. This is directly comparable to CV R²
        since both are out-of-sample predictions.

        Args:
            fold_models: list of (xgb_model, age_reg_or_None) from run_cv
            X_holdout: Holdout feature matrix
            y_holdout: Holdout target (raw)
            age_holdout: Holdout age array

        Returns:
            dict with 'r2', 'n', 'y_pred', 'y_true', 'per_fold_r2'
        """
        y_arr = np.asarray(y_holdout, dtype=float)
        valid = ~np.isnan(y_arr)
        X_v = X_holdout[valid].reset_index(drop=True)
        y_v = y_arr[valid]
        age_v = age_holdout[valid] if age_holdout is not None else None

        all_preds = []
        per_fold_r2 = []

        for fold_i, (model, age_reg) in enumerate(fold_models):
            # Age-residualize y using this fold's age regressor
            y_resid = y_v.copy()
            if age_reg is not None and age_v is not None:
                age_ok = ~np.isnan(age_v)
                if age_ok.sum() > 0:
                    y_resid[age_ok] = y_resid[age_ok] - age_reg.predict(
                        age_v[age_ok].reshape(-1, 1)
                    ).ravel()

            y_hat = model.predict(X_v)
            all_preds.append(y_hat)
            fold_r2 = r2_score(y_resid, y_hat)
            per_fold_r2.append(fold_r2)

        # Ensemble: average predictions across folds
        y_ensemble = np.mean(all_preds, axis=0)

        # For final R², residualize y with mean age effect across folds
        # (first fold's age_reg as representative; all are fit on ~80% of the same data)
        y_final = y_v.copy()
        first_age_reg = fold_models[0][1]
        if first_age_reg is not None and age_v is not None:
            age_ok = ~np.isnan(age_v)
            if age_ok.sum() > 0:
                y_final[age_ok] = y_final[age_ok] - first_age_reg.predict(
                    age_v[age_ok].reshape(-1, 1)
                ).ravel()

        ensemble_r2 = r2_score(y_final, y_ensemble)
        print(f"  predict_holdout_ensemble: N={len(y_v):,}, "
              f"R²={ensemble_r2:.4f} (per-fold: {[f'{r:.4f}' for r in per_fold_r2]})")

        return {
            "r2": float(ensemble_r2),
            "n": int(len(y_v)),
            "y_pred": y_ensemble,
            "y_true": y_final,
            "per_fold_r2": per_fold_r2,
        }

    def predict_holdout(
        self,
        model: xgb.XGBRegressor,
        X_holdout: pd.DataFrame,
        y_holdout: pd.Series,
        age_holdout: np.ndarray | None = None,
    ) -> dict:
        """Apply a trained model to holdout data and return R² + predictions.

        If fit_final() was called with age residualization, the same age regression
        is applied to the holdout outcome before computing R².

        Args:
            model: Fitted XGBRegressor from fit_final()
            X_holdout: Holdout feature matrix
            y_holdout: Holdout target (raw; age residualization applied internally)
            age_holdout: Holdout age array

        Returns:
            dict with 'r2', 'n', 'y_pred', 'y_true'
        """
        y_arr = np.asarray(y_holdout, dtype=float)
        valid = ~np.isnan(y_arr)
        X_v = X_holdout[valid].reset_index(drop=True)
        y_v = y_arr[valid]

        # Apply same age residualization as training
        if self._age_fitted and self._age_reg is not None and age_holdout is not None:
            age_v = age_holdout[valid]
            age_valid = ~np.isnan(age_v)
            if age_valid.sum() > 0:
                y_v[age_valid] = y_v[age_valid] - self._age_reg.predict(
                    age_v[age_valid].reshape(-1, 1)
                ).ravel()

        y_hat = model.predict(X_v)
        r2 = r2_score(y_v, y_hat)
        print(f"  predict_holdout: N={len(y_v):,}, R²={r2:.4f}")
        return {"r2": float(r2), "n": int(len(y_v)), "y_pred": y_hat, "y_true": y_v}

    def save_model(
        self,
        model: "xgb.XGBRegressor",
        best_params: dict,
        path: "Path | str",
    ) -> None:
        """Save fitted model + age regressor to disk for holdout validation."""
        import joblib
        from pathlib import Path
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        model.save_model(str(path / "model.ubj"))
        joblib.dump({
            "best_params": best_params,
            "age_reg": self._age_reg if self._age_fitted else None,
            "age_fitted": self._age_fitted,
        }, path / "meta.joblib")
        print(f"  save_model: {path}")

    @classmethod
    def load_model(cls, path: "Path | str") -> tuple["XGBoostPipeline", "xgb.XGBRegressor"]:
        """Load a saved model + age regressor. Returns (pipeline, model)."""
        import joblib
        import xgboost as xgb
        from pathlib import Path
        path = Path(path)
        meta = joblib.load(path / "meta.joblib")
        model = xgb.XGBRegressor()
        model.load_model(str(path / "model.ubj"))
        pipe = cls()
        pipe._age_reg = meta["age_reg"]
        pipe._age_fitted = meta["age_fitted"]
        return pipe, model

    def run_bootstrap_shap(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: list[str],
        n_bootstrap: Optional[int] = None,
        return_raw: bool = False,
        save_individual: bool = False,
    ) -> pd.DataFrame | tuple[pd.DataFrame, np.ndarray]:
        """Bootstrap SHAP (200 iterations).

        Each iteration trains on a bootstrap sample and computes SHAP on the
        out-of-bag rows.

        Args:
            return_raw: If True, also return the raw shap_store array
                (n_bootstrap × n_features) for downstream analyses (e.g. A3).
            save_individual: If True, accumulate per-person signed SHAP values
                across OOB appearances. Results stored on self.robust_shap_
                (n_samples × n_features) and self.oob_count_ (n_samples,).

        Returns:
            DataFrame indexed by feature_names with columns:
              mean_abs_shap, ci_lower (2.5%), ci_upper (97.5%),
              top{k}_stability (fraction in top-K), stable (bool, ≥0.60 threshold)
            If return_raw=True, returns (DataFrame, shap_store_array).
        """
        n_bs = n_bootstrap if n_bootstrap is not None else self.n_bootstrap
        y_arr = np.asarray(y, dtype=float)
        valid_mask = ~np.isnan(y_arr)
        X_v = X[valid_mask].reset_index(drop=True)
        y_v = y_arr[valid_mask]
        n = len(y_v)
        n_feat = len(feature_names)

        rng = np.random.default_rng(self.random_state)
        shap_store = np.zeros((n_bs, n_feat))
        top_k_counts = np.zeros((n_bs, n_feat), dtype=bool)

        # Per-person signed SHAP accumulators (only when save_individual)
        if save_individual:
            shap_sum = np.zeros((n, n_feat), dtype=np.float64)
            oob_count_arr = np.zeros(n, dtype=np.int32)

        # Build the DMatrix once and slice it per iteration
        full_dmat = xgb.DMatrix(
            X_v, label=y_v, enable_categorical=True,
            feature_names=feature_names,
        )
        num_boost_round = self.params.get("n_estimators", 200)

        print(f"  Bootstrap SHAP: {n_bs} iterations, n={n:,}...")
        for i in range(n_bs):
            if (i + 1) % 50 == 0:
                print(f"    Iter {i+1}/{n_bs}")

            boot_idx = rng.integers(0, n, size=n)
            oob_mask = np.ones(n, dtype=bool)
            oob_mask[np.unique(boot_idx)] = False
            if oob_mask.sum() < 5:
                continue

            boot_dmat = full_dmat.slice(boot_idx)
            oob_idx = np.where(oob_mask)[0]
            oob_dmat = full_dmat.slice(oob_idx)

            booster = _fit_xgb_native(boot_dmat, self.params, num_boost_round)
            sv = _shap_values_native(booster, oob_dmat)
            mean_abs = np.abs(sv).mean(axis=0)
            shap_store[i] = mean_abs

            # Top-K membership for this iteration
            top_k_idx = np.argsort(mean_abs)[::-1][: self.top_k]
            top_k_counts[i, top_k_idx] = True

            # Accumulate signed SHAP per person (OOB only)
            if save_individual:
                shap_sum[oob_idx] += sv
                oob_count_arr[oob_idx] += 1

        # Compute robust (OOB-averaged) signed SHAP per person
        if save_individual:
            safe_count = np.maximum(oob_count_arr, 1)
            robust_shap = shap_sum / safe_count[:, None]
            robust_shap[oob_count_arr == 0] = np.nan
            self.robust_shap_ = robust_shap
            self.oob_count_ = oob_count_arr

        mean_shap = shap_store.mean(axis=0)
        ci_lo = np.percentile(shap_store, 2.5, axis=0)
        ci_hi = np.percentile(shap_store, 97.5, axis=0)
        stability = top_k_counts.mean(axis=0)

        result = pd.DataFrame(
            {
                "mean_abs_shap": mean_shap,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
                f"top{self.top_k}_stability": stability,
                "stable": stability >= 0.60,
            },
            index=feature_names,
        ).sort_values("mean_abs_shap", ascending=False)

        if return_raw:
            return result, shap_store
        return result

    def run_bootstrap_shap_parallel(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: list[str],
        n_bootstrap: Optional[int] = None,
        n_gpus: int = 2,
        return_raw: bool = False,
        save_individual: bool = False,
    ) -> pd.DataFrame | tuple[pd.DataFrame, np.ndarray]:
        """Bootstrap SHAP with multi-GPU parallelism via ThreadPoolExecutor.

        Splits bootstrap iterations across available GPUs. Each GPU worker
        runs its share of iterations sequentially. Uses threads (not processes)
        because XGBoost releases the GIL during training and prediction.

        Optimization: builds a single DMatrix upfront and uses DMatrix.slice()
        for bootstrap/OOB splits, avoiding repeated pandas indexing and
        pd.Categorical reconstruction.

        Args:
            n_gpus: Number of CUDA devices to use (default: 2).
            return_raw: If True, also return raw shap_store array.
            save_individual: If True, accumulate per-person signed SHAP values
                across OOB appearances. Results stored on self.robust_shap_
                (n_samples × n_features) and self.oob_count_ (n_samples,).
        """
        from concurrent.futures import ThreadPoolExecutor
        import threading

        n_bs = n_bootstrap if n_bootstrap is not None else self.n_bootstrap
        y_arr = np.asarray(y, dtype=float)
        valid_mask = ~np.isnan(y_arr)
        X_v = X[valid_mask].reset_index(drop=True)
        y_v = y_arr[valid_mask]
        n = len(y_v)
        n_feat = len(feature_names)

        # Pre-generate all bootstrap indices for reproducibility
        rng = np.random.default_rng(self.random_state)
        all_boot_idx = [rng.integers(0, n, size=n) for _ in range(n_bs)]

        # Build the DMatrix once; categorical dtypes are preserved.
        # DMatrix.slice() selects rows without copying.
        full_dmat = xgb.DMatrix(
            X_v, label=y_v, enable_categorical=True,
            feature_names=feature_names,
        )

        # n_estimators from sklearn params (default 200)
        num_boost_round = self.params.get("n_estimators", 200)

        print(f"  Bootstrap SHAP (parallel, DMatrix.slice): {n_bs} iters, n={n:,}, {n_gpus} GPUs")
        progress_lock = threading.Lock()
        progress = [0]

        def worker(gpu_id: int, indices: list[int]) -> tuple:
            """Run a batch of bootstrap iterations on a specific GPU."""
            params = dict(self.params)
            params["device"] = f"cuda:{gpu_id}"

            local_shap = np.zeros((len(indices), n_feat))
            local_topk = np.zeros((len(indices), n_feat), dtype=bool)

            # Per-person signed SHAP accumulators (only when save_individual)
            local_shap_sum = np.zeros((n, n_feat), dtype=np.float64) if save_individual else None
            local_oob_count = np.zeros(n, dtype=np.int32) if save_individual else None

            for local_i, glob_i in enumerate(indices):
                boot_idx = all_boot_idx[glob_i]
                oob_mask = np.ones(n, dtype=bool)
                oob_mask[np.unique(boot_idx)] = False
                if oob_mask.sum() < 5:
                    continue

                # Row selection without a pandas copy
                boot_dmat = full_dmat.slice(boot_idx)
                oob_idx = np.where(oob_mask)[0]
                oob_dmat = full_dmat.slice(oob_idx)

                booster = _fit_xgb_native(boot_dmat, params, num_boost_round)
                sv = _shap_values_native(booster, oob_dmat)
                mean_abs = np.abs(sv).mean(axis=0)
                local_shap[local_i] = mean_abs

                top_k_idx = np.argsort(mean_abs)[::-1][: self.top_k]
                local_topk[local_i, top_k_idx] = True

                # Accumulate signed SHAP per person (OOB only)
                if save_individual:
                    local_shap_sum[oob_idx] += sv
                    local_oob_count[oob_idx] += 1

                with progress_lock:
                    progress[0] += 1
                    done = progress[0]
                    if done % 50 == 0:
                        print(f"    Progress: {done}/{n_bs}")

            return local_shap, local_topk, local_shap_sum, local_oob_count

        # Split iterations across GPUs
        chunks = [[] for _ in range(n_gpus)]
        for i in range(n_bs):
            chunks[i % n_gpus].append(i)

        # Run in parallel
        with ThreadPoolExecutor(max_workers=n_gpus) as executor:
            futures = [
                executor.submit(worker, gpu_id, chunk)
                for gpu_id, chunk in enumerate(chunks)
            ]
            results_list = [f.result() for f in futures]

        # Reassemble in original order
        shap_store = np.zeros((n_bs, n_feat))
        top_k_counts = np.zeros((n_bs, n_feat), dtype=bool)
        for gpu_id, chunk in enumerate(chunks):
            local_shap, local_topk, _, _ = results_list[gpu_id]
            for local_i, glob_i in enumerate(chunk):
                shap_store[glob_i] = local_shap[local_i]
                top_k_counts[glob_i] = local_topk[local_i]

        # Merge individual SHAP accumulators across workers
        if save_individual:
            shap_sum_total = np.zeros((n, n_feat), dtype=np.float64)
            oob_count_total = np.zeros(n, dtype=np.int32)
            for gpu_id in range(len(chunks)):
                _, _, ls, lc = results_list[gpu_id]
                shap_sum_total += ls
                oob_count_total += lc
            # Compute robust (OOB-averaged) signed SHAP per person
            safe_count = np.maximum(oob_count_total, 1)
            robust_shap = shap_sum_total / safe_count[:, None]
            robust_shap[oob_count_total == 0] = np.nan
            self.robust_shap_ = robust_shap
            self.oob_count_ = oob_count_total

        mean_shap = shap_store.mean(axis=0)
        ci_lo = np.percentile(shap_store, 2.5, axis=0)
        ci_hi = np.percentile(shap_store, 97.5, axis=0)
        stability = top_k_counts.mean(axis=0)

        result = pd.DataFrame(
            {
                "mean_abs_shap": mean_shap,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
                f"top{self.top_k}_stability": stability,
                "stable": stability >= 0.60,
            },
            index=feature_names,
        ).sort_values("mean_abs_shap", ascending=False)

        if return_raw:
            return result, shap_store
        return result
