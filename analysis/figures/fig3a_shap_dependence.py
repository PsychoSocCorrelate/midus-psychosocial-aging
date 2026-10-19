#!/usr/bin/env python3
"""
Figure 3A: SHAP dependence plots for selected cs1 features.

3x4 grid: one row per outcome, four selected variables per row; scatter,
smoothing spline and 95% CI, in the outcome's colour.

Frailty uses the unadjusted cs1 model, because the A2 overlap-adjusted model
drops bio/lifestyle overlap variables (including BMI, sleep*, whr). The panel is
descriptive: it shows how these features co-vary with frailty and does not
claim they are non-overlapping predictors.

Usage:
    uv run python analysis/figures/fig3a_shap_dependence.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.core.data_loader import load_master
from analysis.core import artifact_paths as ap
from analysis.core.shap_dependence import (
    plot_dependence, _shap_tick_formatter, CATEGORY_LABELS, _resolve_label,
)

OUT = ROOT / "manuscript" / "figures" / "fig3"
OUT.mkdir(parents=True, exist_ok=True)

# Config

COLORS = {
    "Frailty":   "#E07A5F",
    "Cognition": "#3D7C98",
    "AL":        "#E5A84B",
}

outcome_display = {
    "Frailty":   "Frailty",
    "Cognition": "Cognition",
    "AL":        "Allostatic Load",
}

# Variables shown per outcome
SELECTIONS = {
    "Frailty":   ["controlOverHealth", "bmi", "sleepAvg", "lifeDiscriminate"],
    "Cognition": ["education", "wordGames", "educationACE_3item", "householdIncome"],
    "AL":        ["bmi", "whr", "drinkStatus", "vmLeisureAct_vig"],
}

OUTCOME_TO_LOAD = {
    "Frailty":   "FInoPsy_W1",
    "Cognition": "btactComp_W1",
    "AL":        f"{ap.AL_STEM}_W1",
}


VAR_DISPLAY = {
    "controlOverHealth": "Health control",
    "bmi":               "BMI",
    "sleepAvg":          "Sleep (avg hrs)",
    "lifeDiscriminate":  "Life discrimination",
    "education":         "Education",
    "wordGames":         "Word-game freq.",
    "educationACE_3item":"Childhood educ. adversity",
    "householdIncome":   "Household income",
    "whr":               "Waist-hip ratio",
    "drinkStatus":       "Drinking status",
    "vmLeisureAct_vig":  "Vigorous leisure activity",
}

plt.rcParams.update({
    'font.family':   'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':     8,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
})

# Variables rendered with a custom horizontal violin (forced orientation).
FORCE_HORIZONTAL_VIOLIN = {"educationACE_3item", "drinkStatus"}

# Short labels for horizontal violin ticks; full labels are in
# config/variable_config.yaml.
SHORT_CATEGORY_LABELS = {
    "drinkStatus": {
        "Past-month abstainer": "Abstainer",
        "Infrequent-non-binge": "Infrequent",
        "Frequent-non-binge":   "Frequent",
        "Binge":                "Binge",
    },
}

# Variables with a custom x-tick formatter applied after plot_dependence.
def _income_k_formatter(x, _pos=None):
    if x == 0:
        return "0"
    if abs(x) >= 1_000_000:
        return f"{x / 1_000_000:g}M"
    if abs(x) >= 1_000:
        return f"{x / 1_000:g}k"
    return f"{x:g}"

X_TICK_FORMATTERS = {
    "householdIncome": _income_k_formatter,
}


def _horizontal_categorical(ax, x_raw, shap_vals, feature_name, color,
                            label_fontsize=6.5):
    """Draw a horizontal violin plot for a categorical feature.

    Uses CATEGORY_LABELS[feature_name] verbatim for y-tick labels.
    Preserves the ordering of that mapping (YAML dict insertion order).
    """
    # Short-label override takes precedence over the verbose YAML map.
    label_map = SHORT_CATEGORY_LABELS.get(feature_name) \
                or CATEGORY_LABELS.get(feature_name, {})
    # Ordered list of (raw_key, display_label) from the YAML config.
    # Fall back to sorted unique values in the data if no map exists.
    if label_map:
        level_order = list(label_map.keys())
        level_display = [label_map[k] for k in level_order]
    else:
        try:
            level_order = sorted(set(x_raw), key=float)
        except (ValueError, TypeError):
            level_order = sorted(set(x_raw), key=str)
        level_display = [_resolve_label(feature_name, v) for v in level_order]

    def _norm_key(v):
        # Normalise '0.0' and 0 and '0' to a common string form so that
        # YAML int keys match float data values.
        try:
            return str(int(float(v)))
        except (ValueError, TypeError):
            return str(v)

    raw_norm = np.array([_norm_key(v) for v in x_raw])
    groups = []
    kept_labels = []
    for key, disp in zip(level_order, level_display):
        grp = shap_vals[raw_norm == _norm_key(key)]
        if len(grp) >= 2:
            groups.append(grp)
            kept_labels.append(disp)
    if not groups:
        return

    positions = list(range(len(groups)))
    vp = ax.violinplot(groups, positions=positions, vert=False,
                       showmedians=True, showextrema=False)
    for body in vp["bodies"]:
        body.set_facecolor(color)
        body.set_alpha(0.35)
        body.set_edgecolor(color)
    vp["cmedians"].set_color("black")
    vp["cmedians"].set_linewidth(1.1)

    ax.set_yticks(positions)
    ax.set_yticklabels(kept_labels, fontsize=label_fontsize)
    ax.axvline(x=0, color="gray", ls="--", alpha=0.4, lw=0.7)
    ax.xaxis.set_major_formatter(FuncFormatter(_shap_tick_formatter))
    ax.set_xlabel("SHAP", fontsize=7.5)
    ax.set_ylabel("")

# Load data

def load_outcome(outcome_label: str):
    """Return (X, robust_shap, feature_names, stability_map)."""
    outcome_name = OUTCOME_TO_LOAD[outcome_label]
    result = load_master(outcome_name, [1, 2])
    X, _y, features = result[0], result[1], result[2]

    rs_df = pd.read_csv(ap.CS1_DIR / "shap_raw" / f"{outcome_name}_T1T2_robust_shap.csv")
    # Align columns to feature_names order (they should already match)
    rs_cols = list(rs_df.columns)
    if rs_cols != features:
        rs_df = rs_df[features]
    robust_shap = rs_df.to_numpy()

    summary = pd.read_csv(
        ap.CS1_DIR / "shap" / f"{outcome_name}_T1T2_bootstrap_shap.csv", index_col=0
    )
    stab_map = summary["top20_stability"].to_dict() if "top20_stability" in summary else {}

    return X, robust_shap, features, stab_map


outcomes = ["Frailty", "Cognition", "AL"]
loaded = {}
for o in outcomes:
    print(f"Loading {o}...")
    X, rs, feats, stab = load_outcome(o)
    loaded[o] = dict(X=X, robust_shap=rs, features=feats, stab=stab)
    print(f"  X={X.shape}, robust_shap={rs.shape}")

# Figure

nrows = len(outcomes)
ncols = 4
fig_w = 19 / 2.54   # 7.48 in
fig_h = nrows * 5.0 / 2.54  # ~5.9 in

fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h),
                          gridspec_kw=dict(hspace=0.70, wspace=0.52))

for i, outcome in enumerate(outcomes):
    data = loaded[outcome]
    X = data["X"]
    robust_shap = data["robust_shap"]
    features = data["features"]
    stab_map = data["stab"]
    color = COLORS[outcome]

    for j, fname in enumerate(SELECTIONS[outcome]):
        ax = axes[i, j]
        if fname not in features:
            ax.text(0.5, 0.5, f"{fname}\nnot in feature set",
                    ha="center", va="center", fontsize=7,
                    transform=ax.transAxes, color="#AA3333")
            ax.set_xticks([]); ax.set_yticks([])
            continue

        if fname in FORCE_HORIZONTAL_VIOLIN:
            # Custom horizontal violin using CATEGORY_LABELS ordering/text
            feat_idx = features.index(fname)
            col_series = X[fname]
            shap_col = robust_shap[:, feat_idx]
            valid = ~np.isnan(shap_col)
            if col_series.dtype.name == "category":
                codes = col_series.cat.codes.values
                valid = valid & (codes >= 0)
                x_raw = col_series.values[valid]
            else:
                try:
                    x_float = col_series.values.astype(float)
                    valid = valid & ~np.isnan(x_float)
                    x_raw = col_series.values[valid]
                except (ValueError, TypeError):
                    x_raw = col_series.values[valid]
            s_v = shap_col[valid]
            _horizontal_categorical(ax, x_raw, s_v, fname, color)
            ax.tick_params(labelsize=7)
        else:
            plot_dependence(
                X, robust_shap, fname, features,
                ax=ax,
                alpha_scatter=0.22,
                scatter_size=5,
                color=color,
                ci_color=color,
                ci_alpha=0.20,
            )

            ax.yaxis.set_major_formatter(FuncFormatter(_shap_tick_formatter))

            # Compact axis labels (override the module's verbose SHAP(varname))
            ax.set_ylabel("SHAP", fontsize=7.5)
            ax.tick_params(labelsize=7)
            ax.set_xlabel(ax.get_xlabel(), fontsize=7.5)

            # Custom x-axis formatter for selected features
            if fname in X_TICK_FORMATTERS:
                ax.xaxis.set_major_formatter(
                    FuncFormatter(X_TICK_FORMATTERS[fname])
                )

        # Per-panel title: variable display + stability %
        disp = VAR_DISPLAY.get(fname, fname)
        stab = stab_map.get(fname)
        title = disp if stab is None else f"{disp}  ({stab:.0%})"
        ax.set_title(title, fontsize=8.5, pad=3)

    # Row label on the far left (outside the first panel)
    axes[i, 0].text(
        -0.48, 0.5, outcome_display[outcome],
        transform=axes[i, 0].transAxes,
        rotation=90, ha='center', va='center',
        fontsize=11, fontweight='bold', color=color,
    )

fig.subplots_adjust(left=0.10, right=0.98, top=0.94, bottom=0.07)

outpath_png = OUT / "fig3A.png"
fig.savefig(outpath_png, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved: {outpath_png}")

plt.close(fig)
