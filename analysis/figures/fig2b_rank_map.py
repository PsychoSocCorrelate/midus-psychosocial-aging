#!/usr/bin/env python3
"""Figure 2B - SHAP rank map: cross-sectional (cs-1) x longitudinal (a4).

Three panels (Frailty / Cognition / Allostatic Load), one per aging outcome,
map the percentile rank of each predictor's mean |SHAP| in the cross-sectional
model (cs-1) against its rank in the baseline-adjusted longitudinal model (a4).

Outcome order Frailty, Cognition, Allostatic Load. Text and markers are sized
for print.

Run:  uv run python analysis/figures/fig2b_rank_map.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.core.pub_style import (
    setup_rcparams, display_name, get_subdomain, get_tier,
    SUBDOMAIN_COLORS, PNAS_FULL,
    FONT_TITLE, FONT_LABEL, FONT_ANNOT, FONT_TICK,
)

from analysis.core import artifact_paths as ap
FIG_DIR = ROOT / "manuscript/figures/fig2"

SCATTER_EXCLUDE = {"age_W1"}
STABILITY_ANNOTATE = 0.80
TOP_N_PER_AXIS = 10

# Marker and font sizes
MS_UNSTABLE = 22
MS_STABLE = 55
FS_LABEL_ANNOT = 9   # scatter point labels
MUL_AXIS = 1.5       # axis label multiplier
MUL_TICK = 1.5       # tick multiplier
MUL_TITLE = 1.6      # panel title multiplier
MUL_RHO = 1.6        # rho annotation box
FS_LEGEND = FONT_ANNOT * 1.5

PANEL_TITLE_COLORS = {
    "Frailty":         "#E07A5F",
    "Cognition":       "#3D7C98",
    "Allostatic Load": "#E5A84B",
}

PANELS = [
    {"key": "FInoPsy", "label": "Frailty", "domain": "Frailty",
     "cs1_file": "FInoPsy_W1_T1T2_bootstrap_shap.csv",
     "a4_file": "FInoPsy_W2_long_bootstrap_shap.csv"},
    {"key": "btactComp", "label": "Cognition", "domain": "Cognition",
     "cs1_file": "btactComp_W1_T1T2_bootstrap_shap.csv",
     "a4_file": "btactComp_W2_long_bootstrap_shap.csv"},
    # AL composite. The key selects the artifact directory, so it and
    # the filenames change together.
    {"key": ap.AL_STEM, "label": "Allostatic Load", "domain": "Allostatic Load",
     "cs1_file": f"{ap.AL_STEM}_W1_T1T2_bootstrap_shap.csv",
     "a4_file": f"{ap.AL_STEM}_W2_long_bootstrap_shap.csv"},
]


def load_and_align(cs1_path: Path, a4_path: Path) -> pd.DataFrame:
    cs1 = pd.read_csv(cs1_path, index_col=0)
    a4 = pd.read_csv(a4_path, index_col=0)
    shared = sorted((set(cs1.index) & set(a4.index)) - SCATTER_EXCLUDE)
    print(f"    shared variables: {len(shared)} (cs1={len(cs1)}, a4={len(a4)})")
    df = pd.DataFrame({
        "cs1_shap": cs1.loc[shared, "mean_abs_shap"],
        "a4_shap": a4.loc[shared, "mean_abs_shap"],
        "cs1_stable": cs1.loc[shared, "stable"],
        "a4_stable": a4.loc[shared, "stable"],
        "cs1_stability": cs1.loc[shared, "top20_stability"],
        "a4_stability": a4.loc[shared, "top20_stability"],
    }, index=shared)
    n = len(df)
    df["cs1_rank"] = stats.rankdata(df["cs1_shap"]) / n
    df["a4_rank"] = stats.rankdata(df["a4_shap"]) / n
    df["subdomain"] = [get_subdomain(v) or "demographics" for v in df.index]
    df["tier"] = [get_tier(v) for v in df.index]
    df["display"] = [display_name(v) for v in df.index]
    return df


def select_annotations(df: pd.DataFrame) -> pd.DataFrame:
    top_cs1 = df.nlargest(TOP_N_PER_AXIS, "cs1_shap")
    top_a4 = df.nlargest(TOP_N_PER_AXIS, "a4_shap")
    candidates = pd.concat([top_cs1, top_a4]).drop_duplicates()
    in_top_cs1 = candidates.index.isin(top_cs1.index)
    in_top_a4 = candidates.index.isin(top_a4.index)
    cs1_ok = in_top_cs1 & (candidates["cs1_stability"].values >= STABILITY_ANNOTATE)
    a4_ok = in_top_a4 & (candidates["a4_stability"].values >= STABILITY_ANNOTATE)
    kept = candidates[cs1_ok | a4_ok]
    print(f"    top-10 union: {len(candidates)} vars -> {len(kept)} labeled")
    return kept


def plot_panel(df: pd.DataFrame, to_annotate: pd.DataFrame, ax):
    subdomains = sorted(df["subdomain"].unique())
    any_stable = df["cs1_stable"] | df["a4_stable"]
    for sub in subdomains:
        mask = df["subdomain"] == sub
        if not mask.any():
            continue
        color = SUBDOMAIN_COLORS.get(sub, "#888888")
        sub_df = df[mask]
        unstable = ~any_stable[mask]
        if unstable.any():
            ax.scatter(sub_df.loc[unstable, "cs1_rank"], sub_df.loc[unstable, "a4_rank"],
                       c=color, s=MS_UNSTABLE, alpha=0.30, edgecolors="none", zorder=1)
        stable = any_stable[mask]
        if stable.any():
            ax.scatter(sub_df.loc[stable, "cs1_rank"], sub_df.loc[stable, "a4_rank"],
                       c=color, s=MS_STABLE, alpha=0.85, edgecolors="white",
                       linewidths=0.5, zorder=2, label=sub.replace("_", " ").title())

    ax.plot([0, 1], [0, 1], ls="--", c="#BBBBBB", lw=0.6, zorder=0)
    ax.axhline(0.5, ls=":", c="#DDDDDD", lw=0.5, zorder=0)
    ax.axvline(0.5, ls=":", c="#DDDDDD", lw=0.5, zorder=0)

    from adjustText import adjust_text
    texts, xs, ys = [], [], []
    for var, row in to_annotate.iterrows():
        xs.append(row["cs1_rank"]); ys.append(row["a4_rank"])
        texts.append(ax.text(row["cs1_rank"], row["a4_rank"], row["display"],
                             fontsize=FS_LABEL_ANNOT, color="#222222",
                             bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.6)))
    if texts:
        adjust_text(texts, x=xs, y=ys, ax=ax,
                    force_text=(2.2, 2.4), force_points=(3.0, 3.0),
                    force_explode=(2.0, 2.0), expand=(1.6, 2.0),
                    expand_points=(1.8, 1.8), expand_text=(1.5, 1.6),
                    arrowprops=dict(arrowstyle="-", color="#888888", lw=0.4, shrinkA=2, shrinkB=3),
                    ensure_inside_axes=True, max_move=40)

    rho, pval = stats.spearmanr(df["cs1_shap"], df["a4_shap"])
    ax.text(0.97, 0.04, f"$\\rho_s$ = {rho:.2f}\np = {pval:.1e}",
            transform=ax.transAxes, fontsize=FONT_ANNOT * MUL_RHO,
            va="bottom", ha="right", color="#555555",
            bbox=dict(facecolor="white", edgecolor="#DDDDDD", alpha=0.9, boxstyle="round,pad=0.3"),
            zorder=5)
    ax.set_xlabel("Cross-sectional importance rank", fontsize=FONT_LABEL * MUL_AXIS)
    ax.set_ylabel("Longitudinal change importance rank", fontsize=FONT_LABEL * MUL_AXIS)
    ax.set_xlim(-0.03, 1.06); ax.set_ylim(-0.03, 1.06)
    ax.tick_params(labelsize=FONT_TICK * MUL_TICK)
    return rho, pval


def main():
    setup_rcparams()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_w = PNAS_FULL * 1.25 * 1.68
    fig_h = (PNAS_FULL * 0.52 + 0.6) * 1.25
    fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h))

    all_handles, all_labels = [], []
    for i, (panel, ax) in enumerate(zip(PANELS, axes)):
        print(f"\n[{panel['label']}]")
        df = load_and_align(ap.CS1_DIR / "shap" / panel["cs1_file"],
                            ap.A4_DIR / "shap" / panel["a4_file"])
        to_annotate = select_annotations(df)
        plot_panel(df, to_annotate, ax)
        ax.set_title(panel["label"], fontsize=FONT_TITLE * MUL_TITLE, loc="left", pad=6,
                     color=PANEL_TITLE_COLORS.get(panel["domain"], "#555555"), fontweight="bold")
        if i != 0:
            ax.set_ylabel("")
        h, l = ax.get_legend_handles_labels()
        all_handles.extend(h); all_labels.extend(l)

    seen, handles, labels = set(), [], []
    for h, l in zip(all_handles, all_labels):
        if l not in seen:
            seen.add(l); handles.append(h); labels.append(l)
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(5, len(handles)),
                   fontsize=FS_LEGEND, frameon=False, bbox_to_anchor=(0.5, -0.03),
                   handletextpad=0.3, columnspacing=1.0)
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.22, top=0.90, wspace=0.32)

    for ext in ("pdf", "png"):
        out = FIG_DIR / f"fig2B.{ext}"
        fig.savefig(out, format=ext, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"\nSaved: {out.relative_to(ROOT)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
