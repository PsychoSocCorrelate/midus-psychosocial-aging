#!/usr/bin/env python3
"""Figure 3B: cross-domain predictor landscape (top-20 rank heatmap).

Draws the 3-row heatmap of the union of top-20 features (by cross-sectional
mean |SHAP|) across three aging outcomes: frailty, cognition, allostatic load.

Encoding
--------
    Rows        : aging outcome.
    Columns     : individual predictors, grouped by a priori subdomain
                  (colored band above).
    Cell fill   : mean |SHAP| rescaled by within-outcome maximum
                  (per-row colormap; white = 0, saturated = row max).
    Cell text   : within-outcome rank (1-20). Blank if not top-20.
    Grey italic : bootstrap top-20 stability < 0.80 in that outcome.
    Bold label  : feature appears in the top 20 of >= 2 of the 3
                  displayed outcomes.

Only features entering at least one outcome's top 20 are shown as
columns. Subdomains that contribute no such feature are omitted from
the band; their absence is logged.

Usage:
    uv run python analysis/figures/fig3b_rank_heatmap.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.core.pub_style import setup_rcparams, display_name as _display_name_base, get_subdomain

# Config
from analysis.core import artifact_paths as ap
FIG_DIR = ROOT / "manuscript/figures/fig3"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TOP_N = 20
STABILITY_CUT = 0.80

# Gamma < 1 lifts low-end cell colour so non-top-20 cells are visible.
# The colorbars use the same PowerNorm so the mapping stays consistent.
COLOR_GAMMA = 0.55

# Figure-local display overrides (sentence-case, compact) on top of the
# global pub_style._DISPLAY_OVERRIDES. Covers features where the camel-
# case splitter produces awkward output (e.g. "Rec Support").
_LOCAL_LABELS = {
    "female":              "Sex",
    "race":                "Race",
    "education":           "Education",
    "educationACE_3item":  "ACE (3-item)",
    "householdIncome":     "Household income",
    "controlOverHealth":   "Health control",
    "healthLocOther":      "Health LoC (others)",
    "healthLocControl":    "Health LoC (self)",
    "posAffect":           "Positive affect",
    "negAffect":           "Negative affect",
    "mentalHealth":        "Mental health",
    "psychWellbeing":      "Psych. well-being",
    "pessimism":           "Pessimism",
    "interdependence":     "Interdependence",
    "mpqPotency":          "MPQ Potency",
    "mpqCloseness":        "MPQ Closeness",
    "live4Today":          "Live-for-today",
    "ventEmotion":         "Venting emotion",
    "lifeSatisfaction":    "Life satisf.",
    "endMiddleAge":        "Subjective aging",
    "lifeDiscriminate":    "Discrimination",
    "dailyDiscriminate":   "Daily discrimination",
    "recSupport":          "Received support",
    "giveSupport":         "Given support",
    "socContribution":     "Social contrib.",
    "friendQuality":       "Friend quality",
    "partnerAgeGap":       "Partner age gap",
    "relatCohabLength":    "Cohabitation yrs",
    "bmi":                 "BMI",
    "whr":                 "Waist-hip ratio",
    "physicianDiet":       "Physician diet",
    "physicalPTFreq":      "Physical therapy",
    "vmLeisureAct_vig":    "Vigorous leisure",
    "napFreq":             "Nap freq.",
    "sleepAvg":            "sleepAvg",
    "chiropractorFreq":    "chiropractorFreq",
    "wordGames":           "Word games",
    "useComputer":         "Computer use",
    "intellAging":         "Intellectual aging",
    "playCard":            "Card games",
    "readBook":            "Reading",
    "drinkStatus":         "Drinking status",
    "smokeDuration":       "Smoke duration",
    "occGroup":            "Occupation",
    "workIndustry":        "Work industry",
    "workStatus":          "Work status",
    "manageRole":          "Role management",
}


def display_name(var: str) -> str:
    return _LOCAL_LABELS.get(var, _display_name_base(var))

OUTCOMES = [
    {"key": "FInoPsy",   "label": "Frailty",
     "file": "FInoPsy_W1_T1T2_bootstrap_shap.csv",   "color": "#E07A5F"},
    {"key": "btactComp", "label": "Cognition",
     "file": "btactComp_W1_T1T2_bootstrap_shap.csv", "color": "#3D7C98"},
    # AL composite
    {"key": ap.AL_STEM, "label": "Allostatic Load",
     "file": f"{ap.AL_STEM}_W1_T1T2_bootstrap_shap.csv", "color": "#E5A84B"},
]

# Fixed display order for the subdomain band (short labels used above).
SUBDOMAIN_ORDER = [
    ("tier_1",                   "Demog."),
    ("personality",              "Personality"),
    ("psychological_wellbeing",  "Well-being"),
    ("mental_health_symptoms",   "Mental H."),
    ("coping_resilience",        "Coping"),
    ("life_evaluation",          "Life Eval."),
    ("social_relationships",     "Social Rel."),
    ("intimate_partner",         "Partner"),
    ("household_caregiving",     "Household"),
    ("health_behaviors",         "Health Beh."),
    ("cognitive_activities",     "Cog. Act."),
    ("substance_use",            "Substance"),
    ("work_occupation",          "Work"),
    ("socioeconomic_subjective", "SES (subj.)"),
    ("healthcare_access",        "Healthcare"),
]

BAND_GREYS = ["#B8B8B8", "#DEDEDE"]  # alternating


# Data assembly
def _load_outcome(info: dict) -> pd.DataFrame:
    df = pd.read_csv(ap.CS1_DIR / "shap" / info["file"], index_col=0)
    df = df.sort_values("mean_abs_shap", ascending=False)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def build_dataset():
    frames = {o["key"]: _load_outcome(o) for o in OUTCOMES}
    top20 = {k: set(df.head(TOP_N).index) for k, df in frames.items()}
    union = set().union(*top20.values())

    shared = {f: sum(f in top20[k] for k in top20) for f in union}
    sub = {f: (get_subdomain(f) or "unknown") for f in union}
    maxes = {k: float(df["mean_abs_shap"].max()) for k, df in frames.items()}
    return frames, top20, union, shared, sub, maxes


def order_columns(union, sub, frames):
    """Group by SUBDOMAIN_ORDER; within subdomain sort by best (lowest)
    rank across outcomes so the column-order is information-ordered."""
    def best_rank(f):
        ranks = []
        for o in OUTCOMES:
            df = frames[o["key"]]
            if f in df.index:
                ranks.append(int(df.loc[f, "rank"]))
        return min(ranks) if ranks else 10**6

    ordered = []
    for sub_id, _ in SUBDOMAIN_ORDER:
        members = sorted(
            [f for f in union if sub.get(f) == sub_id],
            key=best_rank,
        )
        ordered.extend((sub_id, f) for f in members)

    # Features with no subdomain go last.
    known = {s for s, _ in SUBDOMAIN_ORDER}
    leftover = sorted(
        [f for f in union if sub.get(f) not in known],
        key=best_rank,
    )
    ordered.extend(("unknown", f) for f in leftover)
    return ordered


# Plotting
def _txt_color(intensity: float) -> str:
    return "#FFFFFF" if intensity > 0.55 else "#1A1A1A"


def plot(frames, top20, ordered, shared, maxes, outpath_png, outpath_pdf):
    n_cols = len(ordered)

    present_subs = []
    seen = set()
    for sub_id, _ in ordered:
        if sub_id in seen:
            continue
        seen.add(sub_id)
        short = dict(SUBDOMAIN_ORDER).get(sub_id, sub_id)
        present_subs.append((sub_id, short))

    cmaps = {
        o["key"]: mcolors.LinearSegmentedColormap.from_list(
            f"cm_{o['key']}", ["#FFFFFF", o["color"]]
        )
        for o in OUTCOMES
    }

    # Figure geometry
    # Width scales with # columns; height gives enough headroom for
    # rotated feature labels without overlapping the subdomain band.
    fig_w = max(11.0, 0.285 * n_cols + 3.4)
    fig_h = 4.3
    fig = plt.figure(figsize=(fig_w, fig_h))

    # Main axes covers cells + band + rotated labels (colorbars below it).
    ax = fig.add_axes([0.10, 0.22, 0.88, 0.66])
    ax.set_aspect("auto")

    # y layout (data coords), bottom -> top:
    #   rows    : AL  1, Cognition 2, Frailty 3 (cell y-centers)
    #   band    : y = 3.7 .. 4.3   (subdomain title centered inside)
    #   labels  : rotated upward from y = 4.5
    y_rows = {0: 3.0, 1: 2.0, 2: 1.0}  # outcome index -> y-center
    y_band_bot = 3.70
    y_band_top = 4.30
    y_band_mid = 0.5 * (y_band_bot + y_band_top)
    y_label = 4.50

    # Cells
    for col_idx, (_, f) in enumerate(ordered):
        x = col_idx
        for oi, o in enumerate(OUTCOMES):
            key = o["key"]
            y = y_rows[oi]
            in_top = f in top20[key]
            df = frames[key]

            # Every cell is coloured by its actual mean|SHAP| scaled to
            # the within-outcome maximum, then gamma-lifted so non-top-20
            # cells remain visible rather than collapsing to near-white.
            if f in df.index:
                shap_val = float(df.loc[f, "mean_abs_shap"])
                raw = shap_val / maxes[key] if maxes[key] > 0 else 0.0
            else:
                shap_val = 0.0
                raw = 0.0
            intensity = raw ** COLOR_GAMMA
            color = cmaps[key](intensity)

            ax.add_patch(Rectangle(
                (x - 0.5, y - 0.5), 1, 1,
                facecolor=color, edgecolor="white", linewidth=0.6,
                zorder=1,
            ))

            if in_top:
                rank = int(df.loc[f, "rank"])
                stab = float(df.loc[f, "top20_stability"])
                is_shared = shared.get(f, 0) >= 2

                # Two independent style channels:
                #   weight  = shared (matches column-label bolding rule)
                #   italic  = unstable (stability < 0.80)
                # Unstable cells also shift to grey so they fade in.
                weight = "bold" if is_shared else "normal"
                style = "italic" if stab < STABILITY_CUT else "normal"
                color = ("#8C8C8C"
                         if stab < STABILITY_CUT
                         else _txt_color(intensity))

                ax.text(x, y, str(rank),
                        ha="center", va="center",
                        fontsize=12, color=color,
                        fontweight=weight, fontstyle=style,
                        zorder=3)

    # Subdomain band + titles
    for si, (sub_id, short) in enumerate(present_subs):
        members = [(i, f) for i, (sid, f) in enumerate(ordered)
                   if sid == sub_id]
        if not members:
            continue
        start = members[0][0]
        end = members[-1][0]
        color = BAND_GREYS[si % 2]
        ax.add_patch(Rectangle(
            (start - 0.5, y_band_bot),
            end - start + 1, y_band_top - y_band_bot,
            facecolor=color, edgecolor="white", linewidth=0.8,
            zorder=1,
        ))
        ax.text((start + end) / 2, y_band_mid, short,
                ha="center", va="center", fontsize=11,
                fontweight="bold", color="#222222", zorder=3)

    # Rotated column labels
    # Bold-vs-normal at 7pt rotated 55° is too subtle after anti-aliased
    # downsampling, so we reinforce the weight channel with a colour
    # channel: shared-feature labels render in full black + bold, single-
    # outcome labels in medium grey + normal.
    for col_idx, (_, f) in enumerate(ordered):
        is_bold = shared.get(f, 0) >= 2
        ax.text(col_idx, y_label, display_name(f),
                ha="left", va="bottom",
                rotation=55, rotation_mode="anchor",
                fontsize=(10.5 if is_bold else 10),
                fontweight=("bold" if is_bold else "normal"),
                color=("#000000" if is_bold else "#7A7A7A"),
                zorder=2)

    # Row labels (outcome name + row max)
    for oi, o in enumerate(OUTCOMES):
        y = y_rows[oi]
        ax.text(-0.9, y + 0.15, o["label"],
                ha="right", va="center",
                fontsize=15, fontweight="bold", color=o["color"])
        ax.text(-0.9, y - 0.22,
                f"(max {maxes[o['key']]:.3f})",
                ha="right", va="center",
                fontsize=11, fontstyle="italic", color="#666666")

    # Axis extents / cleanup
    ax.set_xlim(-4.8, n_cols + 0.5)
    ax.set_ylim(0.4, y_label + 3.6)  # headroom for rotated labels
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    # Horizontal colorbars (one per outcome)
    cb_h = 0.028
    cb_w = 0.14
    gap = 0.035
    total = 3 * cb_w + 2 * gap
    x0 = 0.5 - total / 2
    cb_y = 0.08

    for oi, o in enumerate(OUTCOMES):
        xoff = x0 + oi * (cb_w + gap)
        cax = fig.add_axes([xoff, cb_y, cb_w, cb_h])
        norm = mcolors.PowerNorm(
            gamma=COLOR_GAMMA, vmin=0, vmax=maxes[o["key"]],
        )
        cb = fig.colorbar(
            cm.ScalarMappable(norm=norm, cmap=cmaps[o["key"]]),
            cax=cax, orientation="horizontal",
        )
        cb.set_ticks([0, maxes[o["key"]]])
        cb.set_ticklabels(["0", f"{maxes[o['key']]:.3f}"])
        cb.ax.tick_params(labelsize=11, length=2, pad=1)
        cb.outline.set_linewidth(0.3)
        fig.text(xoff + cb_w / 2, cb_y + cb_h + 0.018, o["label"],
                 ha="center", va="bottom",
                 fontsize=12.5, fontweight="bold", color=o["color"])

    # Save
    fig.savefig(outpath_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(outpath_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {outpath_png.relative_to(ROOT)}")
    print(f"Saved: {outpath_pdf.relative_to(ROOT)}")


# Main
def _report(ordered, shared, present_subs):
    all_sub_ids = [s for s, _ in SUBDOMAIN_ORDER]
    present_ids = [s for s, _ in present_subs]
    absent = [s for s in all_sub_ids if s not in present_ids]

    shared2 = sorted(
        [f for f, c in shared.items() if c >= 2],
        key=lambda f: -shared[f],
    )
    shared3 = [f for f, c in shared.items() if c >= 3]

    print()
    print("Figure 3B summary")
    print(f"  Columns (union of top-20 across 3 outcomes): {len(ordered)}")
    print(f"  Subdomains represented: {len(present_subs)} of "
          f"{len(SUBDOMAIN_ORDER)}")
    print(f"  Absent subdomains      : {absent}")
    print(f"  Bolded features (top-20 in >=2 outcomes): {len(shared2)}")
    for f in shared2:
        print(f"      {display_name(f):<30s}  (in {shared[f]} top-20s)")
    print(f"  In all 3 top-20s      : "
          f"{[display_name(f) for f in shared3]}")


def main():
    setup_rcparams()
    frames, top20, union, shared, sub, maxes = build_dataset()
    ordered = order_columns(union, sub, frames)

    present_subs = []
    seen = set()
    for sub_id, _ in ordered:
        if sub_id in seen:
            continue
        seen.add(sub_id)
        short = dict(SUBDOMAIN_ORDER).get(sub_id, sub_id)
        present_subs.append((sub_id, short))

    _report(ordered, shared, present_subs)

    out_png = FIG_DIR / "fig3B.png"
    out_pdf = FIG_DIR / "fig3B.pdf"
    plot(frames, top20, ordered, shared, maxes, out_png, out_pdf)

    # Persist the underlying cell data for auditability.
    records = []
    for col_idx, (sub_id, f) in enumerate(ordered):
        for o in OUTCOMES:
            df = frames[o["key"]]
            in_top = f in top20[o["key"]]
            records.append(dict(
                column_index=col_idx,
                subdomain=sub_id,
                feature=f,
                display=display_name(f),
                outcome=o["key"],
                outcome_label=o["label"],
                in_top20=in_top,
                rank=(int(df.loc[f, "rank"]) if f in df.index else None),
                mean_abs_shap=(float(df.loc[f, "mean_abs_shap"])
                               if f in df.index else 0.0),
                top20_stability=(float(df.loc[f, "top20_stability"])
                                 if f in df.index else None),
                shared_across_displayed=shared.get(f, 0),
                bold_label=shared.get(f, 0) >= 2,
            ))
    pd.DataFrame(records).to_csv(FIG_DIR / "fig3B_cells.csv", index=False)
    print(f"Saved: {(FIG_DIR / 'fig3B_cells.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
