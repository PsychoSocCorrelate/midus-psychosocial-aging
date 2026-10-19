"""
SHAP dependence plots with a smoothing spline and bootstrap 95% CI.

Produces scatter plots of robust (OOB-averaged) individual SHAP values
against feature values, with a smoothing spline and 95% confidence band
representing the estimated marginal effect.

Usage:
    from analysis.core.shap_dependence import plot_dependence, plot_dependence_panel
    fig = plot_dependence_panel(X, robust_shap, feature_names, top_k=6)
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np
import pandas as pd
import yaml
from scipy.interpolate import make_smoothing_spline

from analysis.core.data_loader import CATEGORICALS
from analysis.core.pub_style import (
    PNAS_FULL,
    display_name,
    setup_rcparams,
)

# Constants
_MIN_OBS_SMOOTH = 20


def _fit_smooth_ci(x, y, x_grid, n_boot_ci=200, seed=0):
    """Fit GCV smoothing spline + bootstrap 95 % CI.

    Aggregates to unique x-values (weighted by sqrt(count)) to satisfy
    make_smoothing_spline's strictly-increasing constraint.

    Returns (main_curve, ci_lo, ci_hi) evaluated on *x_grid*, or None.
    """
    unique_x, inverse, counts = np.unique(
        x, return_inverse=True, return_counts=True
    )
    if len(unique_x) < 4:  # cubic spline needs >= 4 points
        return None

    sum_y = np.bincount(inverse, weights=y)
    mean_y = sum_y / counts
    w = np.sqrt(counts.astype(float))

    try:
        spline = make_smoothing_spline(unique_x, mean_y, w=w)
        main_curve = spline(x_grid)
    except Exception:
        return None

    rng = np.random.default_rng(seed)
    boot_curves = np.empty((n_boot_ci, len(x_grid)))
    for b in range(n_boot_ci):
        idx = rng.choice(len(x), len(x), replace=True)
        xb, yb = x[idx], y[idx]
        ux, inv_b, cnt_b = np.unique(xb, return_inverse=True, return_counts=True)
        if len(ux) < 4:
            boot_curves[b] = main_curve
            continue
        try:
            sy = np.bincount(inv_b, weights=yb)
            sp = make_smoothing_spline(
                ux, sy / cnt_b, w=np.sqrt(cnt_b.astype(float))
            )
            boot_curves[b] = sp(x_grid)
        except Exception:
            boot_curves[b] = main_curve

    ci_lo = np.percentile(boot_curves, 2.5, axis=0)
    ci_hi = np.percentile(boot_curves, 97.5, axis=0)
    return main_curve, ci_lo, ci_hi


def _shap_tick_formatter(x, _pos=None):
    """Format SHAP tick labels: strip leading zero (0.02 to .02, -0.01 to -.01)."""
    s = f"{x:.3g}"
    if s.startswith("0."):
        return s[1:]
    if s.startswith("-0."):
        return "-" + s[2:]
    return s
_ROOT = Path(__file__).resolve().parents[2]

# Label dictionary for categorical / binary variables
def _load_category_labels() -> dict[str, dict[int, str]]:
    """Load category_labels.yaml into {var_name: {int_code: label}}."""
    with open(_ROOT / "config" / "category_labels.yaml") as f:
        raw = yaml.safe_load(f)
    # Convert keys to int where possible; keep str otherwise (e.g. cohort codes)
    def _safe_key(k):
        try:
            return int(k)
        except (ValueError, TypeError):
            return k
    return {var: {_safe_key(k): v for k, v in mapping.items()}
            for var, mapping in raw.items()}


CATEGORY_LABELS = _load_category_labels()

# Sparse short labels for ordinal features with many levels.
# Keys = anchor tick positions; values = abbreviated display labels.
_ORDINAL_ANCHORS: dict[str, dict[int, str]] = {
    "education": {
        1: "No sch",
        5: "HS grad",
        9: "BA/BS",
        12: "Doctoral",
    },
}


def _apply_ordinal_ticks(ax: plt.Axes, feature_name: str) -> bool:
    """Apply sparse short labels + minor ticks for ordinal features.

    Returns True if ticks were customised, False otherwise.
    """
    if feature_name not in _ORDINAL_ANCHORS:
        return False
    anchors = _ORDINAL_ANCHORS[feature_name]
    anchor_pos = sorted(anchors)
    ax.set_xticks(anchor_pos)
    ax.set_xticklabels([anchors[k] for k in anchor_pos], fontsize=7)
    # Minor ticks at non-anchor integer positions
    all_codes = sorted(
        k for k in CATEGORY_LABELS.get(feature_name, {}) if isinstance(k, int)
    )
    minor_pos = [c for c in all_codes if c not in anchors]
    if minor_pos:
        ax.xaxis.set_minor_locator(FixedLocator(minor_pos))
        ax.tick_params(axis="x", which="minor", length=3)
    return True


def plot_dependence(
    X: pd.DataFrame,
    robust_shap: np.ndarray,
    feature_name: str,
    feature_names: list[str],
    ax: plt.Axes | None = None,
    alpha_scatter: float = 0.25,
    scatter_size: float = 6,
    color: str = "#4682B4",
    ci_color: str = "#4682B4",
    ci_alpha: float = 0.18,
    x_override: np.ndarray | None = None,
    x_label: str | None = None,
    outcome_label: str | None = None,
) -> plt.Axes:
    """Single-feature SHAP dependence plot with smoothing spline + 95 % CI.

    Args:
        X: Feature matrix (n_samples x n_features).
        robust_shap: OOB-averaged signed SHAP (n_samples x n_features).
        feature_name: Name of the feature to plot.
        feature_names: Ordered list matching columns of robust_shap.
        ax: Matplotlib axes (created if None).
        alpha_scatter: Scatter point transparency.
        scatter_size: Scatter point size.
        color: Color for scatter and smooth line.
        ci_color: Color for CI band.
        ci_alpha: Transparency of CI band.
        x_override: If provided, use these values as x-axis instead of
            the feature values (e.g., chronological age).
        x_label: Custom x-axis label (used with x_override).
    """
    feat_idx = feature_names.index(feature_name)
    shap_vals = robust_shap[:, feat_idx]

    # Detect categorical BEFORE numeric conversion
    col = X[feature_name]
    is_cat_dtype = (col.dtype.name == "category" or feature_name in CATEGORICALS)

    if x_override is not None:
        x_vals = x_override.astype(float)
    elif is_cat_dtype:
        # For categoricals, keep raw values for label resolution;
        # numeric conversion not needed (violin plot groups by unique value)
        raw = col.values
        if col.dtype.name == "category":
            codes = col.cat.codes.values
            valid_cat = (codes >= 0) & ~np.isnan(shap_vals)
            x_vals = raw  # pass-through; _plot_categorical handles grouping
        else:
            valid_cat = ~np.isnan(shap_vals)
            try:
                x_vals = col.values.astype(float)
            except (ValueError, TypeError):
                x_vals = raw
    else:
        x_vals = col.values.astype(float)

    # Drop NaN in either
    if x_override is None and is_cat_dtype:
        valid = valid_cat
    else:
        valid = ~np.isnan(x_vals) & ~np.isnan(shap_vals)
    x_v = x_vals[valid]
    s_v = shap_vals[valid]

    if ax is None:
        _, ax = plt.subplots(figsize=(3.5, 3))

    def _n_unique(arr):
        try:
            return len(np.unique(arr[~np.isnan(arr)]))
        except (TypeError, ValueError):
            return len(set(arr))

    is_cat = (x_override is None
              and (is_cat_dtype or _n_unique(x_v) <= 5))

    _fmt = FuncFormatter(_shap_tick_formatter)

    _dname = display_name(feature_name)
    _shap_label = f"SHAP({_dname})"

    if is_cat:
        _plot_categorical(x_v, s_v, feature_name, ax, color)
        # Determine layout used (must match _plot_categorical logic)
        cat_labels = [_resolve_label(feature_name, v) for v in
                      (sorted(set(x_v), key=float)
                       if all(isinstance(v, (int, float, np.integer, np.floating))
                              for v in set(x_v))
                       else sorted(set(x_v), key=str))]
        if _is_horizontal_violin(cat_labels):
            # Horizontal violins: SHAP on x-axis
            ax.set_xlabel(_shap_label)
            ax.xaxis.set_major_formatter(_fmt)
        else:
            ax.set_xlabel(x_label if x_label else _dname)
            ax.set_ylabel(_shap_label)
            ax.yaxis.set_major_formatter(_fmt)
            ax.axhline(y=0, color="gray", ls="--", alpha=0.4, lw=0.7)
    else:
        _plot_continuous(x_v, s_v, ax,
                         alpha_scatter, scatter_size,
                         color, ci_color, ci_alpha)
        ax.set_xlabel(x_label if x_label else _dname)
        ax.set_ylabel(_shap_label)
        ax.yaxis.set_major_formatter(_fmt)
        ax.axhline(y=0, color="gray", ls="--", alpha=0.4, lw=0.7)
        _apply_ordinal_ticks(ax, feature_name)
    return ax


def _plot_continuous(
    x: np.ndarray,
    shap: np.ndarray,
    ax: plt.Axes,
    alpha_scatter: float,
    scatter_size: float,
    color: str,
    ci_color: str,
    ci_alpha: float,
    x_trim_percentile: tuple[float, float] = (0.5, 99.5),
) -> None:
    """Scatter + GCV smoothing spline + bootstrap 95 % CI for a continuous feature."""
    # Trim x-axis extremes: sparse tails cause CI explosion.
    x_lo, x_hi = np.percentile(x, x_trim_percentile)
    trim = (x >= x_lo) & (x <= x_hi)
    x_t, shap_t = x[trim], shap[trim]

    ax.scatter(x_t, shap_t, s=scatter_size, alpha=alpha_scatter, color=color,
               edgecolors="none", rasterized=True)

    if len(x_t) < _MIN_OBS_SMOOTH:
        return

    x_grid = np.linspace(x_t.min(), x_t.max(), 200)
    result = _fit_smooth_ci(x_t, shap_t, x_grid)
    if result is None:
        return

    main_curve, ci_lo, ci_hi = result
    ax.plot(x_grid, main_curve, color=color, lw=1.5)
    ax.fill_between(x_grid, ci_lo, ci_hi, color=ci_color, alpha=ci_alpha)

    # Clamp y-axis to the scatter data range so that exploding CI bands
    # at sparse tails don't crush the visible signal.
    shap_lo, shap_hi = np.percentile(shap_t, [0.5, 99.5])
    pad = (shap_hi - shap_lo) * 0.15 or 0.01
    ax.set_ylim(shap_lo - pad, shap_hi + pad)


def _wrap_label(label: str, max_width: int = 12) -> str:
    """Wrap a long label into two lines at the nearest space."""
    if len(label) <= max_width:
        return label
    # Find best split near the middle
    mid = len(label) // 2
    # Search outward from middle for a space
    best = None
    for offset in range(mid + 1):
        for pos in (mid + offset, mid - offset):
            if 0 < pos < len(label) and label[pos] == " ":
                best = pos
                break
            # Also try splitting at / or - boundaries
            if 0 < pos < len(label) and label[pos] in "/-":
                best = pos + 1
                break
        if best is not None:
            break
    if best is None:
        return label
    return label[:best].rstrip() + "\n" + label[best:].lstrip()


def _resolve_label(feature_name: str, raw_val) -> str:
    """Map a raw value to a display label using CATEGORY_LABELS."""
    label_map = CATEGORY_LABELS.get(feature_name)
    if label_map is not None:
        # Try numeric key first (values come in as float from data)
        try:
            key = int(float(raw_val))
        except (ValueError, TypeError):
            key = raw_val
        result = label_map.get(key)
        if result is not None:
            return result
        try:
            return label_map.get(float(key), str(raw_val))
        except (ValueError, TypeError):
            return str(raw_val)
    # String-labeled categoricals: already readable, just clean up
    s = str(raw_val)
    # Remove trailing ".0" from float-cast ints
    if s.endswith(".0"):
        s = s[:-2]
    return s


def _is_horizontal_violin(display_labels, threshold: int = 6, max_label_len: int = 10) -> bool:
    """Decide whether to use horizontal violin layout."""
    longest = max(len(lb) for lb in display_labels)
    return len(display_labels) > threshold or longest > max_label_len


def _plot_categorical(
    x: np.ndarray,
    shap: np.ndarray,
    feature_name: str,
    ax: plt.Axes,
    color: str,
) -> None:
    """Violin plot grouped by category level, with display labels."""
    # Numeric-safe sorting: try float, fall back to string
    try:
        unique_raw = sorted(set(x), key=float)
    except (ValueError, TypeError):
        unique_raw = sorted(set(x), key=str)

    raw_strs = np.array([str(v) for v in x])
    display_labels = [_resolve_label(feature_name, v) for v in unique_raw]
    raw_keys = [str(v) for v in unique_raw]

    groups = [shap[raw_strs == rk] for rk in raw_keys]
    # Drop empty / tiny groups
    packed = [(dl, g) for dl, g in zip(display_labels, groups) if len(g) >= 2]
    if not packed:
        return
    display_labels, groups = zip(*packed)

    positions = list(range(len(groups)))
    horizontal = _is_horizontal_violin(display_labels)

    if horizontal:
        # Horizontal violins: categories on y-axis, SHAP on x-axis
        vp = ax.violinplot(groups, positions=positions, vert=False,
                           showmedians=True, showextrema=False)
        for body in vp["bodies"]:
            body.set_facecolor(color)
            body.set_alpha(0.35)
            body.set_edgecolor(color)
        vp["cmedians"].set_color("black")
        vp["cmedians"].set_linewidth(1.2)
        ax.set_yticks(positions)
        wrapped = [_wrap_label(lb, max_width=18) for lb in display_labels]
        ax.set_yticklabels(wrapped, fontsize=6.5)
        ax.axvline(x=0, color="gray", ls="--", alpha=0.4, lw=0.7)
    else:
        vp = ax.violinplot(groups, positions=positions, showmedians=True,
                           showextrema=False)
        for body in vp["bodies"]:
            body.set_facecolor(color)
            body.set_alpha(0.35)
            body.set_edgecolor(color)
        vp["cmedians"].set_color("black")
        vp["cmedians"].set_linewidth(1.2)
        ax.set_xticks(positions)
        wrapped = [_wrap_label(lb) for lb in display_labels]
        ax.set_xticklabels(wrapped, fontsize=7, rotation=45, ha="right")


def plot_dependence_panel(
    X: pd.DataFrame,
    robust_shap: np.ndarray,
    feature_names: list[str],
    top_k: int = 12,
    ncols: int = 3,
    outcome_label: str = "",
    figsize: tuple | None = None,
    bootstrap_shap_df: pd.DataFrame | None = None,
    stability_threshold: float = 0.60,
    top_n: int | None = None,
    **kwargs,
) -> plt.Figure:
    """Multi-panel SHAP dependence for stable features.

    When bootstrap_shap_df is provided, features are first ranked by
    mean_abs_shap. If top_n is set, only the top N predictors are
    considered; among those, only features with stability >=
    stability_threshold are plotted.

    Args:
        top_k: Max features (used as fallback when bootstrap_shap_df is None).
        ncols: Columns in subplot grid.
        outcome_label: Outcome name for axis labels and title.
        bootstrap_shap_df: Bootstrap SHAP summary DataFrame (index=feature names,
            must contain 'stable' or 'top20_stability' column).
        stability_threshold: Minimum top-20 stability to include (default 0.60).
        top_n: If set, restrict to the top N predictors by mean_abs_shap
            before applying the stability filter.
        **kwargs: Passed to plot_dependence().
    """
    setup_rcparams()

    if bootstrap_shap_df is not None:
        # Determine stability column name
        stab_col = None
        for candidate in ("top20_stability", "top15_stability", "top10_stability"):
            if candidate in bootstrap_shap_df.columns:
                stab_col = candidate
                break

        # Sort by importance, optionally cap to top_n
        sorted_df = bootstrap_shap_df.sort_values("mean_abs_shap", ascending=False)
        if top_n is not None:
            sorted_df = sorted_df.head(top_n)

        if stab_col is not None:
            stable_df = sorted_df[sorted_df[stab_col] >= stability_threshold]
        else:
            # Fallback to 'stable' boolean column
            stable_df = sorted_df[sorted_df.get("stable", True)]

        # Filter to features present in feature_names
        stable_vars = [v for v in stable_df.index if v in feature_names]
        selected_features = stable_vars
        stability_map = {}
        if stab_col is not None:
            stability_map = bootstrap_shap_df[stab_col].to_dict()
    else:
        # Fallback: rank by mean |robust SHAP|, take top_k
        mean_abs = np.nanmean(np.abs(robust_shap), axis=0)
        top_idx = np.argsort(mean_abs)[::-1][:top_k]
        selected_features = [feature_names[i] for i in top_idx]
        stability_map = {}

    n_panels = len(selected_features)
    if n_panels == 0:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, "No stable features", ha="center", va="center",
                fontsize=10, transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    nrows = math.ceil(n_panels / ncols)
    if figsize is None:
        figsize = (PNAS_FULL * 2, nrows * 2.6)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = np.array(axes).flatten() if n_panels > 1 else [axes]

    for i, fname in enumerate(selected_features):
        plot_dependence(X, robust_shap, fname, feature_names,
                        ax=axes_flat[i], outcome_label=outcome_label,
                        **kwargs)
        # Title: variable name + stability %
        title_text = display_name(fname)
        if fname in stability_map:
            title_text += f"  ({stability_map[fname]:.0%})"
        axes_flat[i].set_title(title_text, fontsize=9)

    for j in range(n_panels, len(axes_flat)):
        axes_flat[j].set_visible(False)

    title = "SHAP Dependence — stable features"
    if outcome_label:
        title = f"{outcome_label} — {title}"
    fig.suptitle(title, fontsize=11, y=1.01)
    fig.tight_layout()
    return fig


# Sex-stratified dependence panel
_SEX_COLORS = {"Male": "#4682B4", "Female": "#C44E52"}
_SEX_VIOLIN_WIDTH = 0.4  # half-width for side-by-side violins


def _plot_categorical_by_sex(
    x: np.ndarray,
    shap: np.ndarray,
    sex_mask: dict[str, np.ndarray],
    feature_name: str,
    ax: plt.Axes,
) -> None:
    """Side-by-side violin plots per category level, colored by sex."""
    try:
        unique_raw = sorted(set(x), key=float)
    except (ValueError, TypeError):
        unique_raw = sorted(set(x), key=str)

    raw_strs = np.array([str(v) for v in x])
    display_labels = [_resolve_label(feature_name, v) for v in unique_raw]
    raw_keys = [str(v) for v in unique_raw]

    # Filter levels with enough data in at least one sex
    kept = []
    for rk, dl in zip(raw_keys, display_labels):
        level_mask = raw_strs == rk
        m_vals = shap[level_mask & sex_mask["Male"]]
        f_vals = shap[level_mask & sex_mask["Female"]]
        if len(m_vals) >= 2 or len(f_vals) >= 2:
            kept.append((rk, dl, m_vals, f_vals))
    if not kept:
        return

    horizontal = len(kept) > 6
    positions = list(range(len(kept)))
    w = _SEX_VIOLIN_WIDTH

    for pos, (rk, dl, m_vals, f_vals) in enumerate(kept):
        for sex_label, vals, offset in [
            ("Male", m_vals, -w / 2),
            ("Female", f_vals, w / 2),
        ]:
            if len(vals) < 2:
                continue
            color = _SEX_COLORS[sex_label]
            if horizontal:
                vp = ax.violinplot([vals], positions=[pos + offset],
                                   vert=False, showmedians=True,
                                   showextrema=False, widths=w * 0.9)
            else:
                vp = ax.violinplot([vals], positions=[pos + offset],
                                   showmedians=True, showextrema=False,
                                   widths=w * 0.9)
            for body in vp["bodies"]:
                body.set_facecolor(color)
                body.set_alpha(0.4)
                body.set_edgecolor(color)
            vp["cmedians"].set_color(color)
            vp["cmedians"].set_linewidth(1.2)

    _fmt = FuncFormatter(_shap_tick_formatter)
    labels_display = [dl for _, dl, _, _ in kept]
    if horizontal:
        ax.set_yticks(positions)
        ax.set_yticklabels(labels_display, fontsize=6.5)
        ax.axvline(x=0, color="gray", ls="--", alpha=0.4, lw=0.7)
        ax.set_xlabel(f"SHAP({display_name(feature_name)})")
        ax.xaxis.set_major_formatter(_fmt)
    else:
        ax.set_xticks(positions)
        ax.set_xticklabels(labels_display, fontsize=7, rotation=45, ha="right")
        ax.axhline(y=0, color="gray", ls="--", alpha=0.4, lw=0.7)
        ax.set_xlabel(display_name(feature_name))
        ax.set_ylabel(f"SHAP({display_name(feature_name)})")
        ax.yaxis.set_major_formatter(_fmt)


def plot_dependence_panel_by_sex(
    X: pd.DataFrame,
    robust_shap: np.ndarray,
    feature_names: list[str],
    sex_col: str = "female",
    top_k: int = 12,
    ncols: int = 4,
    outcome_label: str = "",
    figsize: tuple | None = None,
    alpha_scatter: float = 0.18,
    scatter_size: float = 5,
) -> plt.Figure:
    """Multi-panel SHAP dependence, stratified by sex (overlay on same axes).

    Each subplot shows scatter + smoothing spline + 95 % CI for Male and Female
    separately, allowing visual comparison of marginal effects.
    """
    setup_rcparams()

    sex_arr = X[sex_col].values.astype(float)
    groups = {
        "Male": sex_arr == 0,
        "Female": sex_arr == 1,
    }

    # Rank by overall mean |SHAP| (same ranking as unstratified version)
    mean_abs = np.nanmean(np.abs(robust_shap), axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:top_k]

    nrows = math.ceil(top_k / ncols)
    if figsize is None:
        figsize = (PNAS_FULL * 2, nrows * 2.8)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = axes.flatten() if top_k > 1 else [axes]

    for i, feat_i in enumerate(top_idx):
        ax = axes_flat[i]
        fname = feature_names[feat_i]
        feat_idx = feat_i

        col = X[fname]
        is_cat = (col.dtype.name == "category"
                  or fname in CATEGORICALS
                  or fname == sex_col)

        if is_cat:
            # Side-by-side violins colored by sex
            col_raw = col.values
            if col.dtype.name == "category":
                codes = col.cat.codes.values.astype(float)
                shap_valid = ~np.isnan(robust_shap[:, feat_idx])
                cat_valid = codes >= 0
                mask_all = shap_valid & cat_valid
                x_cat = col_raw[mask_all]
                s_cat = robust_shap[mask_all, feat_idx]
                sex_sub = {lbl: m[mask_all] for lbl, m in groups.items()}
            else:
                try:
                    x_float = col_raw.astype(float)
                    shap_valid = ~np.isnan(robust_shap[:, feat_idx])
                    mask_all = ~np.isnan(x_float) & shap_valid
                except (ValueError, TypeError):
                    mask_all = ~np.isnan(robust_shap[:, feat_idx])
                x_cat = col_raw[mask_all]
                s_cat = robust_shap[mask_all, feat_idx]
                sex_sub = {lbl: m[mask_all] for lbl, m in groups.items()}
            _plot_categorical_by_sex(x_cat, s_cat, sex_sub, fname, ax)
        else:
            # Overlay Male / Female continuous scatter + spline smooth
            for label, mask in groups.items():
                color = _SEX_COLORS[label]
                valid = mask & ~np.isnan(robust_shap[:, feat_idx])

                x_vals = col.values.astype(float)
                valid &= ~np.isnan(x_vals)
                x_v = x_vals[valid]
                s_v = robust_shap[valid, feat_idx]

                if len(x_v) < 10:
                    continue

                ax.scatter(x_v, s_v, s=scatter_size, alpha=alpha_scatter,
                           color=color, edgecolors="none", rasterized=True,
                           label=label)

                # Smoothing spline + CI
                if len(x_v) >= _MIN_OBS_SMOOTH:
                    x_grid = np.linspace(x_v.min(), x_v.max(), 200)
                    result = _fit_smooth_ci(x_v, s_v, x_grid)
                    if result is not None:
                        main_curve, ci_lo, ci_hi = result
                        ax.plot(x_grid, main_curve, color=color, lw=1.5)
                        ax.fill_between(x_grid, ci_lo, ci_hi,
                                        color=color, alpha=0.12)

            ax.axhline(y=0, color="gray", ls="--", alpha=0.4, lw=0.7)
            ax.set_xlabel(display_name(fname))
            ax.set_ylabel(f"SHAP({display_name(fname)})")
            ax.yaxis.set_major_formatter(FuncFormatter(_shap_tick_formatter))
            _apply_ordinal_ticks(ax, fname)

        ax.set_title(display_name(fname), fontsize=9)

    # Legend on first axis
    axes_flat[0].legend(fontsize=7, loc="best", framealpha=0.7)

    for j in range(len(top_idx), len(axes_flat)):
        axes_flat[j].set_visible(False)

    title = "SHAP Dependence by Sex (OOB-robust)"
    if outcome_label:
        title = f"{outcome_label} — {title}"
    fig.suptitle(title, fontsize=11, y=1.01)
    fig.tight_layout()
    return fig
