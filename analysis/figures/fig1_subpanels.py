#!/usr/bin/env python3
"""Figure 1 panels as standalone PNGs, each drawn at its print size.

Every panel is drawn at the size it takes in a PNAS full-width figure
(7.0 in, 17.8 cm), with type set in points at that size: 8 pt titles, 6.5 to
7 pt labels, nothing under 6 pt. Placed in PowerPoint at 100%, the type
prints at the size written here; rescaling a panel changes it.

Panels, in reading order (letters are added at placement, in the 0.17 in
left indent every panel reserves for one):
  design   A fig1_cohorts      two MIDUS cohorts on one calendar axis
           B fig1_predictors   177 predictors: Tier 1 and the 14 Tier 2 subdomains
           C fig1_outcomes     the three outcomes and the benchmark: the colour key
           - fig1_question     the question D-G answer, and the path through them
  results  D fig1_concurrent   cross-sectional layer: Tier 2 increment at Wave 1
           E fig1_change       longitudinal layer: Tier 2 increment for 9-year change
           F fig1_crossdomain  cross-domain layer: agreement of SHAP rankings
           G fig1_cotwin       within-family layer: MZ co-twin resemblance

Reading. The figure is meant to be read without its caption. One question
line names what D-G answer. Each result panel has a header: the finding, in
bold, on one or two lines, and below it what goes in, what comes out, and the
model. Every header in D-G reserves two finding lines, so panels in a row keep
one height and their axes stay level. Numbers stay on the marks and are not
repeated in the headers. Each qualitative word in a finding is guarded by an
assert on the numbers it summarizes, so a rerun that changes the picture fails
here rather than printing a stale finding. D's increment is the one over the
Tier 1 segment named on its top bar, and E's names its baseline in the finding.

Grid. D, E, F and G put x = 0 at the same inset (ROW_LABEL_W), so the
axes line up down each column; F's pair labels sit above their lines, not
beside them, to make that possible.

Visual hierarchy. A and B are neutral grey; outcome colour enters at C and
means the same outcome in every later panel; no text in the headers is
coloured. D and E are the largest panels and the only ones drawn with bars;
F and G are smaller dot plots. Text set in an outcome colour uses a darkened
shade of it (TEXT_COLORS), so it reads at 4.5:1 on white; the fills are the
Figure 2-3 hues unchanged.

Encodings. In D and E, light = what is held constant, saturated = the Tier 2
increment, on one R2 axis and one row grid. E's increments are a few pixels
on that axis, so E adds a magnified ΔR² scale. Both print the repeated-CV
interval under each increment; a12 has none for epigenetic aging. Bar
segments are labelled on the top bar, not in a legend. In G, Wave 1
status and 9-year change are filled and open markers, never light and dark,
and are not joined: they come from different pair sets, so no line invites a
reading of decline.

Numbers are read from output/; the only typed values are the cohort dates and
the 4,093 followed up, which are in Methods and in no table.

fig1_deck.py places the panels and writes the preview.

Run:  uv run python analysis/figures/fig1_subpanels.py
"""
from __future__ import annotations

import colorsys
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.textpath import TextToPath

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.core import artifact_paths as ap  # noqa: E402
from analysis.core.data_loader import get_tier_variables  # noqa: E402
from analysis.core.pub_style import get_subdomain  # noqa: E402

OUT = ROOT / "manuscript" / "figures" / "fig1" / "subpanels"

# Print geometry, inches. Two columns of COL, GUTTER apart, across 7.0 in.
FULL_W = 7.0
COL, GUTTER = 3.45, 0.10
A_W = 2.85
B_W = FULL_W - A_W - GUTTER    # row 1 is uneven: the predictor bars need the room
ROW1_H = 1.72
C_H = 0.50
Q_H = 0.19                     # the question line between design and results
FIND_LINES, FIND_PITCH = 2, 0.125  # D-G: finding lines reserved, and their pitch
RES_H = 2.125                  # D and E, the primary results
SEC_H = 1.655                  # F and G
HEAD_H = 0.395                 # two finding lines and the method line, D-G
ROW_LABEL_W = 0.82             # D-G: x = 0 sits at INDENT + ROW_LABEL_W
INDENT = 0.17                  # left indent reserved for the panel letter
DPI = 600

# Type, points at print size
FS_TITLE, FS_SUB, FS_LABEL, FS_TICK, FS_NOTE = 8, 6.5, 7, 6.5, 6
INK, MUTED = "#222222", "#5F6368"
GRID, GUIDE = "#ECEEF0", "#D5D9DD"

# Arial where installed, else Liberation Sans, which has Arial's metrics
_INSTALLED = {f.name for f in font_manager.fontManager.ttflist}
SANS = [f for f in ("Arial", "Liberation Sans", "Helvetica") if f in _INSTALLED] or ["DejaVu Sans"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": SANS,
    "font.size": FS_LABEL, "text.color": INK,
    "axes.labelcolor": INK, "axes.edgecolor": INK, "axes.linewidth": 0.5,
    "axes.labelsize": FS_LABEL,
    "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_LABEL,
    "xtick.color": INK, "ytick.color": INK,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.5, "ytick.major.size": 0,
    "xtick.major.pad": 1.5, "ytick.major.pad": 3,
    "hatch.linewidth": 0.4, "mathtext.default": "regular",
})

# Outcome colours and names as in Figures 2-3
COLORS = {"Frailty": "#E07A5F", "Cognition": "#3D7C98",
          "Allostatic Load": "#E5A84B", "Epigenetic Aging": "#8C8C8C"}
ORDER3 = ["Frailty", "Cognition", "Allostatic Load"]
ORDER4 = ORDER3 + ["Epigenetic Aging"]
AL = ap.AL_STEM
STEM = {"Frailty": "FInoPsy", "Cognition": "btactComp", "Allostatic Load": AL}
# a3's domain labels
A3_NAME = {"Frailty": "Frailty", "Cognition": "Cognition",
           "Allostatic Load": "Allostatic Load"}
DR2 = r"ΔR$^2_{\rm T2}$"  # the text's ΔR²T2


def lighten(hexc, f):
    r, g, b = mcolors.to_rgb(hexc)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


def contrast(c1, c2="#FFFFFF"):
    """WCAG contrast ratio of two colours."""
    def lum(c):
        ch = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4
              for x in mcolors.to_rgb(c)]
        return 0.2126 * ch[0] + 0.7152 * ch[1] + 0.0722 * ch[2]
    a, b = sorted((lum(c1), lum(c2)), reverse=True)
    return (a + 0.05) / (b + 0.05)


def text_shade(hexc, target=4.5):
    """The same hue darkened just enough to read as text on white."""
    h, l, s = colorsys.rgb_to_hls(*mcolors.to_rgb(hexc))
    while contrast(colorsys.hls_to_rgb(h, l, s)) < target and l > 0:
        l -= 0.01
    return mcolors.to_hex(colorsys.hls_to_rgb(h, l, s))


TEXT_COLORS = {k: text_shade(v) for k, v in COLORS.items()}


def pct(x, nd=1):
    """Percent with a true minus sign, as the text prints it."""
    return f"{100 * x:.{nd}f}".replace("-", "−")


def nolead(x, nd=2):
    """A bounded statistic without its leading zero, as the text prints it."""
    return f"{x:.{nd}f}".replace("0.", ".", 1).replace("-", "−")


def figure(w, h):
    return plt.figure(figsize=(w, h))


def axes_in(fig, left, bottom, width, height):
    """Axes placed in inches from the panel's bottom-left corner."""
    W, H = fig.get_size_inches()
    return fig.add_axes([left / W, bottom / H, width / W, height / H])


def heading(fig, title, sub=None):
    """A design panel's title and an optional subtitle, top left after the
    indent."""
    W, H = fig.get_size_inches()
    fig.text(INDENT / W, 1 - 0.02 / H, title, ha="left", va="top",
             fontsize=FS_TITLE, fontweight="bold")
    if sub:
        fig.text(INDENT / W, 1 - 0.17 / H, sub, ha="left", va="top",
                 fontsize=FS_SUB, color=MUTED)


def _wrap(s, width, fs, bold=False):
    """s on one line if it fits, else on two, broken where the longer line is
    shortest."""
    if _width_in(s, fs, bold) <= width:
        return [s]
    words = s.split()
    splits = [(" ".join(words[:i]), " ".join(words[i:])) for i in range(1, len(words))]
    lines = min(splits, key=lambda p: max(_width_in(x, fs, bold) for x in p))
    assert max(_width_in(x, fs, bold) for x in lines) <= width, s
    return list(lines)


def result_heading(fig, finding, method):
    """A result panel's header, HEAD_H deep: the finding in bold on one or two
    lines, and what goes in, what comes out, and the model on the next."""
    W, H = fig.get_size_inches()
    lines = _wrap(finding, W - INDENT - 0.04, FS_TITLE, bold=True)
    assert len(lines) <= FIND_LINES, finding
    assert _width_in(method, FS_SUB) <= W - INDENT - 0.04, method
    for i, line in enumerate(lines):
        fig.text(INDENT / W, 1 - (0.115 + FIND_PITCH * i) / H, line, ha="left",
                 va="baseline", fontsize=FS_TITLE, fontweight="bold")
    fig.text(INDENT / W, 1 - (0.225 + FIND_PITCH * (len(lines) - 1)) / H, method,
             ha="left", va="baseline", fontsize=FS_SUB, color=MUTED)


def strip(ax, keep=("bottom",)):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    # Fixed canvas, no tight bbox: the PNG is exactly the print size.
    fig.savefig(OUT / f"{name}.png", dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"  saved {name}.png")


_T2P = TextToPath()


def _width_in(s, fs, bold=False):
    """Rendered width of one line, inches, from the font's own metrics."""
    fp = FontProperties(family=plt.rcParams["font.sans-serif"], size=fs,
                        weight="bold" if bold else "normal")
    return _T2P.get_text_width_height_descent(s, fp, ismath=False)[0] / 72


# Data
def _r2(path, **match):
    df = pd.read_csv(ROOT / path)
    for k, v in match.items():
        df = df[df[k] == v]
    return df.set_index("tier_config")["r2_mean"]


def _a12(arm):
    ci = pd.read_csv(ROOT / "output/a12_delta_r2_ci/a12_delta_r2_ci.csv")
    return ci[ci["arm"] == arm].set_index("outcome")


def concurrent():
    """(Tier 1 R2, full R2, CV-interval low, high) per outcome at Wave 1. a12
    did not resample epigenetic aging, so its interval is (None, None)."""
    a12 = _a12("concurrent")
    out = {}
    for dom in ORDER4:
        code = "phenoAge_res" if dom == "Epigenetic Aging" else f"{STEM[dom]}_W1"
        r = _r2("output/cs1_cross_sectional/cs1_r2_table.csv", outcome=code)
        lo = hi = None
        if code in a12.index:  # the increment drawn is the one a12 resampled
            assert abs((r["T1T2"] - r["T1"]) - a12.loc[code, "point_seed42"]) < 1e-9, code
            lo, hi = a12.loc[code, "p2.5"], a12.loc[code, "p97.5"]
        out[dom] = (r["T1"], r["T1T2"], lo, hi)
    return out


def change():
    """(Wave 1 status + Tier 1 R2, full R2, CV-interval low, high) per outcome."""
    a12 = _a12("conditional")
    out = {}
    for dom in ORDER3:
        code = f"{STEM[dom]}_W2"
        r = _r2("output/a4_longitudinal/a4_r2_table.csv", outcome=code, cohort="pooled")
        assert abs((r["Yw1T1T2"] - r["Yw1T1"]) - a12.loc[code, "point_seed42"]) < 1e-9, code
        out[dom] = (r["Yw1T1"], r["Yw1T1T2"], a12.loc[code, "p2.5"], a12.loc[code, "p97.5"])
    return out


def icc():
    """MZ outcome ICC and pair count, at Wave 1 ("cs") and in change."""
    df = pd.read_csv(ROOT / "output/a6_sibling/a6_icc_denominators.csv")
    out = {}
    for a in ("cs", "change"):
        for dom in ORDER3:
            row = df[(df.analysis == a) & (df.group == "MZ") & (df.outcome == STEM[dom])]
            assert len(row) == 1, (a, dom)
            out[a, dom] = (float(row.icc.iloc[0]), int(row.n_pairs.iloc[0]))
    return out


def twin_pairs():
    """Range of all-twin pair counts over outcomes, at Wave 1 and in change."""
    df = pd.read_csv(ROOT / "output/a6_sibling/a6_icc_denominators.csv")
    df = df[(df.group == "All_paired") & df.outcome.isin(STEM.values())]
    return {a: (int(g.n_pairs.min()), int(g.n_pairs.max())) for a, g in df.groupby("analysis")}


def rho():
    df = pd.read_csv(ROOT / "output/a3_domain_divergence/a3_pairwise_rho.csv")
    out = []
    for a, b in (("Frailty", "Cognition"), ("Frailty", "Allostatic Load"),
                 ("Cognition", "Allostatic Load")):
        x, y = A3_NAME[a], A3_NAME[b]
        row = df[((df.domain_1 == x) & (df.domain_2 == y)) |
                 ((df.domain_1 == y) & (df.domain_2 == x))]
        assert len(row) == 1, (a, b)
        out.append((a, b, float(row.observed_rho.iloc[0]), float(row.null_mean.iloc[0])))
    return out


def cohort_ns():
    """Total and per-cohort N from output/d1_descriptive (SI Table S11)."""
    ns = {}
    with open(ROOT / "output/d1_descriptive/sample_descriptives.csv") as fh:
        for row in csv.reader(fh):
            key = row[0].strip()
            if key in ("N (total)", "MIDUS2", "MIDUSR"):
                ns[key] = int(row[1].split(" ")[0].replace(",", ""))
    return ns


# A: cohorts, neutral
FOLLOWED_UP = 4093  # Methods: baseline participants with Wave 2 data; no table carries it
COHORT_TONE = {"MIDUS 2": "#4A5662", "Refresher": "#6A7682"}  # white text 7.6:1 and 4.6:1


def panel_cohorts():
    ns, pairs = cohort_ns(), twin_pairs()
    assert ns["MIDUS2"] + ns["MIDUSR"] == ns["N (total)"]
    fig = figure(A_W, ROW1_H)
    heading(fig, "Two MIDUS cohorts, ~9-year follow-up")
    ax = axes_in(fig, INDENT + 0.62, 0.60, A_W - INDENT - 0.62 - 0.08, ROW1_H - 0.60 - 0.30)
    x0, x1 = 2003.5, 2023.0
    ax.set_xlim(x0, x1); ax.set_ylim(-0.55, 1.45)
    cohorts = [  # name, N, (W1 start, end), (W2 start, end), y
        ("MIDUS 2", ns["MIDUS2"], (2004, 2009), (2013, 2014), 1.0),
        ("Refresher", ns["MIDUSR"], (2011, 2014), (2020, 2022), 0.0),
    ]
    for name, n, w1, w2, y in cohorts:
        c = COHORT_TONE[name]
        ax.plot([w1[1] + 1, w2[0]], [y, y], color=c, lw=0.8, ls=(0, (2, 1.5)),
                zorder=1, solid_capstyle="butt")
        for (a, b), lab in ((w1, "W1"), (w2, "W2")):
            ax.add_patch(FancyBboxPatch((a, y - 0.22), b + 1 - a, 0.44,
                                        boxstyle="round,pad=0,rounding_size=0.12",
                                        mutation_aspect=0.2, fc=c, ec="none", zorder=2))
            ax.text((a + b + 1) / 2, y, lab, ha="center", va="center", fontsize=FS_NOTE,
                    color="white", fontweight="bold", zorder=3)
        ax.text(x0 - 0.3, y + 0.12, name, ha="right", va="center", fontsize=FS_LABEL,
                fontweight="bold", color=c, clip_on=False)
        ax.text(x0 - 0.3, y - 0.30, f"N = {n:,}", ha="right", va="center",
                fontsize=FS_NOTE, color=MUTED, clip_on=False)
    strip(ax)
    ax.set_yticks([])
    ax.set_xticks([2004, 2008, 2012, 2016, 2020])
    ax.spines["bottom"].set_bounds(2004, 2023)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(axis="x", colors=MUTED, labelcolor=INK)
    W, H = fig.get_size_inches()
    lines = [
        (f"Pooled baseline N = {ns['N (total)']:,}; followed up {FOLLOWED_UP:,} "
         f"({FOLLOWED_UP / ns['N (total)']:.1%})", INK),
        (f"Twin pairs: {pairs['cs'][0]}–{pairs['cs'][1]} at Wave 1, "
         f"{pairs['change'][0]}–{pairs['change'][1]} for change", INK),
        ("held out from frailty and cognition training", MUTED),
    ]
    for i, (t, c) in enumerate(lines):
        fig.text(INDENT / W, (0.28 - 0.11 * i) / H, t, ha="left", va="center",
                 fontsize=FS_NOTE, color=c)
    save(fig, "fig1_cohorts")


# B: predictors, neutral
SUB = {  # display label for each Tier 2 subdomain
    "social_relationships": "Social relations", "personality": "Personality",
    "health_behaviors": "Health behaviors", "psychological_wellbeing": "Well-being",
    "work_occupation": "Work", "coping_resilience": "Coping",
    "intimate_partner": "Partner", "life_evaluation": "Life evaluation",
    "cognitive_activities": "Cognitive activities", "household_caregiving": "Household",
    "substance_use": "Substance use", "socioeconomic_subjective": "Financial",
    "mental_health_symptoms": "Mental health", "healthcare_access": "Healthcare",
}
TIER1_FILL, TIER2_BAR = "#ECEEF0", "#A3AAB1"


def tier_counts():
    """Tier 1 size and Tier 2 size per subdomain, as the models use them."""
    counts = defaultdict(int)
    for v in get_tier_variables([1, 2], exclude_high_missing=True):
        counts[get_subdomain(v) or "tier_1"] += 1
    n_t1 = counts.pop("tier_1")
    assert set(counts) == set(SUB), set(counts) ^ set(SUB)
    return n_t1, counts


def panel_predictors():
    """Tier 1 as one block; Tier 2 as its 14 subdomains, ranked, in two columns
    of bars. A treemap of 14 cells does not hold legible labels at this size."""
    n_t1, counts = tier_counts()
    n_t2 = sum(counts.values())
    keys = sorted(SUB, key=lambda k: (-counts[k], SUB[k]))

    fig = figure(B_W, ROW1_H)
    heading(fig, f"{n_t1 + n_t2} predictors in two tiers")
    W, H = fig.get_size_inches()
    ax = axes_in(fig, 0, 0, W, H)
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    top, bottom = H - 0.24, 0.04

    # Tier 1: held constant. Tier 2: tested jointly. These are the roles D-F give them.
    x_t1, w_t1 = INDENT, 0.82
    ax.add_patch(Rectangle((x_t1, bottom), w_t1, top - bottom, fc=TIER1_FILL, ec="none"))
    cx = x_t1 + w_t1 / 2
    ax.text(cx, top - 0.08, "Tier 1", ha="center", va="top", fontsize=FS_LABEL,
            fontweight="bold")
    ax.text(cx, top - 0.22, f"background\n{n_t1} variables", ha="center", va="top",
            fontsize=FS_NOTE, color=MUTED, linespacing=1.2)
    ax.text(cx, bottom + 0.07, "held constant", ha="center", va="bottom",
            fontsize=FS_NOTE, fontweight="bold", color=MUTED)
    ax.text(cx, (top - 0.50 + bottom + 0.12) / 2,
            "e.g. sex, race,\neducation,\nhousehold income,\nchildhood\nadversity",
            ha="center", va="center", fontsize=FS_NOTE, color=MUTED,
            fontstyle="italic", linespacing=1.2)

    # Tier 2
    x2 = x_t1 + w_t1 + 0.12
    ax.text(x2, top - 0.08, "Tier 2", ha="left", va="top", fontsize=FS_LABEL,
            fontweight="bold")
    sub_x = x2 + _width_in("Tier 2", FS_LABEL, bold=True) + 0.06
    sub = f"{n_t2} psychosocial and behavioral variables, "
    ax.text(sub_x, top - 0.08, sub, ha="left", va="top", fontsize=FS_NOTE, color=MUTED)
    ax.text(sub_x + _width_in(sub, FS_NOTE), top - 0.08, "tested jointly", ha="left",
            va="top", fontsize=FS_NOTE, fontweight="bold", color=MUTED)
    rows = (len(keys) + 1) // 2
    y_first, pitch = top - 0.36, (top - 0.36 - (bottom + 0.06)) / (rows - 1)
    col_w = (W - x2 - 0.02) / 2
    lab_w = max(_width_in(SUB[k], FS_SUB) for k in keys) + 0.06
    num_w = _width_in("20", FS_SUB) + 0.04
    bar_max = col_w - lab_w - num_w - 0.12
    top_n = max(counts.values())
    for j, k in enumerate(keys):
        col, row = divmod(j, rows)
        x = x2 + col * col_w
        y = y_first - row * pitch
        ax.text(x, y, SUB[k], ha="left", va="center", fontsize=FS_SUB)
        L = bar_max * counts[k] / top_n
        ax.add_patch(Rectangle((x + lab_w, y - 0.04), L, 0.08, fc=TIER2_BAR, ec="none"))
        ax.text(x + lab_w + L + 0.04, y, str(counts[k]), ha="left", va="center",
                fontsize=FS_SUB, color=MUTED)
    save(fig, "fig1_predictors")


# C: outcomes, where colour enters
OUTCOMES = [
    ("Frailty", "Deficit-accumulation index", "physical limitations, chronic conditions"),
    ("Cognition", "BTACT composite", "telephone cognitive battery"),
    ("Allostatic Load", "16 biomarkers, 5 systems", "multisystem dysregulation"),
    ("Epigenetic Aging", "PhenoAge, residualized", "benchmark, Wave 1 only"),
]


def panel_outcomes():
    fig = figure(FULL_W, C_H)
    W, H = fig.get_size_inches()
    heading(fig, "Aging outcomes")
    ax = axes_in(fig, 0, 0, W, H)
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    x, cell = INDENT, (W - INDENT) / 4
    for i, (dom, what, gloss) in enumerate(OUTCOMES):
        x0 = x + i * cell
        bench = dom == "Epigenetic Aging"
        ax.add_patch(Rectangle((x0, 0.04), 0.05, 0.28, fc=COLORS[dom], ec="none"))
        ax.text(x0 + 0.10, 0.27, dom, ha="left", va="center", fontsize=FS_LABEL,
                fontweight="bold", color=MUTED if bench else TEXT_COLORS[dom])
        ax.text(x0 + 0.10, 0.155, what, ha="left", va="center", fontsize=FS_NOTE, color=INK)
        ax.text(x0 + 0.10, 0.05, gloss, ha="left", va="center", fontsize=FS_NOTE,
                color=MUTED, fontstyle="italic")
    save(fig, "fig1_outcomes")


# The question D-G answer, between design and results
def panel_question():
    n_t2 = sum(tier_counts()[1].values())
    q = (f"How well do {n_t2} psychosocial and behavioral characteristics jointly predict "
         "concurrent aging status and longitudinal aging?")
    fig = figure(FULL_W, Q_H)
    W, H = fig.get_size_inches()
    assert _width_in(q, FS_TITLE, bold=True) <= W - INDENT - 0.04, q  # one line
    fig.text(INDENT / W, 1 - 0.115 / H, q, ha="left", va="baseline",
             fontsize=FS_TITLE, fontweight="bold")
    save(fig, "fig1_question")


# D and E: Tier 2 increments, one R2 axis and one row grid
BAR_H, X_MAX = 0.64, 0.60
# Axes top sits below the header and the two-line ΔR² column header.
GEOM = dict(left=INDENT + ROW_LABEL_W, bottom=0.36, ax_w=1.38, ax_h=1.12)
GEOM["col"] = GEOM["left"] + GEOM["ax_w"] + 0.20   # the ΔR² column, x in inches
assert GEOM["bottom"] + GEOM["ax_h"] + 0.25 <= RES_H - HEAD_H


def _increment_axes(fig, wave):
    g = GEOM
    ax = axes_in(fig, g["left"], g["bottom"], g["ax_w"], g["ax_h"])
    ax.set_xlim(0, X_MAX); ax.set_ylim(3.5, -0.5)  # four rows, top down
    ticks = np.arange(0, X_MAX + 1e-9, 0.2)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(round(100 * t))}" for t in ticks])
    ax.set_xlabel(f"Cross-validated R$^2$, Wave {wave} outcome (%)", labelpad=2)
    for t in (0.2, 0.4):
        ax.axvline(t, color=GRID, lw=0.5, zorder=0)
    strip(ax)
    return ax


def _row_y(ax, i):
    """Figure-inch y of row i's centre."""
    g = GEOM
    y0, y1 = ax.get_ylim()
    return g["bottom"] + g["ax_h"] * (y0 - i) / (y0 - y1)


def _column_header(fig, text):
    W, H = fig.get_size_inches()
    g = GEOM
    fig.text(g["col"] / W, (g["bottom"] + g["ax_h"] + 0.06) / H, text, ha="left",
             va="bottom", fontsize=FS_NOTE, color=MUTED, linespacing=1.15)


def _segment_labels(ax, spans):
    """Name the bar segments once, above the top bar: (label, x0, x1) in data
    units. Replaces a legend, which made the reader match swatches."""
    for label, x0, x1 in spans:
        ax.text((x0 + x1) / 2, -0.38, label, ha="center", va="bottom",
                fontsize=FS_NOTE, color=MUTED, clip_on=False)


def _legend(fig, items, x=INDENT):
    W, H = fig.get_size_inches()
    fig.legend(handles=items, loc="lower left", ncol=len(items),
               bbox_to_anchor=(x / W, 0.0), frameon=False, fontsize=FS_NOTE,
               handlelength=1.2, handleheight=0.8, handletextpad=0.4, columnspacing=1.0,
               borderaxespad=0.1, borderpad=0.1)


def _row_labels(ax, labels, muted=()):
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    for i in muted:
        ax.get_yticklabels()[i].set_color(MUTED)


def panel_concurrent():
    data = concurrent()
    inc = {d: v[1] - v[0] for d, v in data.items()}
    # the finding's words, held to the numbers they summarize: "substantial"
    # is at least 10 points of R² for each of the three outcomes; epigenetic
    # aging is the benchmark and is not covered by it
    assert min(inc[d] for d in ORDER3) > 0.10, "substantial"
    fig = figure(COL, RES_H)
    result_heading(fig, "Psychosocial predictors provide substantial incremental "
                        "prediction of concurrent aging status",
                   "Wave 1 predictors → Wave 1 outcome · XGBoost, 5-fold CV")
    ax = _increment_axes(fig, 1)
    W, H = fig.get_size_inches()
    _column_header(fig, f"Tier 2 adds, {DR2} (%)\nrepeated-CV interval")
    for i, dom in enumerate(ORDER4):
        t1, full, lo, hi = data[dom]
        c = COLORS[dom]
        base = max(t1, 0.0)  # Tier 1 alone is below zero for epigenetic aging
        ax.barh(i, base, color=lighten(c, 0.62), height=BAR_H, lw=0, zorder=2)
        ax.barh(i, full - base, left=base, color=c, height=BAR_H, lw=0, zorder=2)
        ax.text(max(full, 0) + 0.012, i, pct(full), va="center", ha="left",
                fontsize=FS_NOTE, color=MUTED)
        # The interval is on ΔR², not on the full R² the bar ends at, so it is
        # printed under the increment rather than drawn on the bar.
        y = _row_y(ax, i)
        lift = 0.045 if lo is not None else 0.0
        fig.text(GEOM["col"] / W, (y + lift) / H, f"+{pct(full - t1)}%", va="center",
                 ha="left", fontsize=FS_LABEL + 0.5, fontweight="bold",
                 color=TEXT_COLORS[dom] if dom != "Epigenetic Aging" else MUTED)
        if lo is not None:
            fig.text(GEOM["col"] / W, (y - 0.075) / H, f"[{pct(lo)}, {pct(hi)}]",
                     va="center", ha="left", fontsize=FS_NOTE, color=MUTED)
    t1, full = data["Frailty"][:2]
    _segment_labels(ax, [("Tier 1", 0, t1), ("+ Tier 2", t1, full)])
    _row_labels(ax, ORDER4, muted=(3,))
    save(fig, "fig1_concurrent")


DR2_LIM = (-0.01, 0.04)  # E's magnified ΔR² scale


def panel_change():
    data = change()
    inc = {d: v[1] - v[0] for d, v in data.items()}
    # the finding's words, held to the numbers they summarize
    assert max(inc.values()) < 0.03, "adds little"
    fig = figure(COL, RES_H)
    result_heading(fig, "Beyond Wave 1 status, Tier 2 adds little for 9-year change",
                   "Wave 1 predictors and outcome → Wave 2 outcome · XGBoost, 5-fold CV")
    ax = _increment_axes(fig, 2)
    W, H = fig.get_size_inches()
    _column_header(fig, f"Tier 2 adds, {DR2} (%)\nrepeated-CV interval")

    g = GEOM
    zx = axes_in(fig, g["col"] + 0.02, g["bottom"], W - g["col"] - 0.12, g["ax_h"])
    zx.set_xlim(*DR2_LIM); zx.set_ylim(3.5, -0.5)
    zx.axvline(0, color=MUTED, lw=0.5, ls=(0, (2, 1.5)), zorder=1)
    zx.set_xticks([0, 0.02, 0.04]); zx.set_xticklabels(["0", "2", "4"])
    zx.set_yticks([])
    zx.set_xlabel("magnified", labelpad=2, fontsize=FS_NOTE, color=MUTED)
    strip(zx)

    for i, dom in enumerate(ORDER3):
        base, full, lo, hi = data[dom]
        c, tc = COLORS[dom], TEXT_COLORS[dom]
        ax.barh(i, base, color=lighten(c, 0.70), edgecolor=lighten(c, 0.30),
                hatch="//////", height=BAR_H, lw=0, zorder=2)
        ax.barh(i, full - base, left=base, color=c, height=BAR_H, lw=0, zorder=3)
        ax.text(full + 0.012, i, pct(full), va="center", ha="left",
                fontsize=FS_NOTE, color=MUTED)
        # magnified increment: point and repeated-CV interval
        # labels sit just right of zero, above and below the interval, so
        # neither the zero line nor a neighbouring row runs through them
        d = full - base
        zx.plot([lo, hi], [i, i], color=c, lw=1.4, solid_capstyle="butt", zorder=2)
        zx.scatter([d], [i], s=16, color=tc, lw=0, zorder=3)
        zx.text(0.0015, i - 0.12, f"+{pct(d)}%", ha="left", va="bottom",
                fontsize=FS_LABEL + 0.5, fontweight="bold", color=tc)
        zx.text(0.0015, i + 0.13, f"[{pct(lo)}, {pct(hi)}]", ha="left", va="top",
                fontsize=FS_NOTE, color=MUTED)
    ax.text(0.0, 3, "Epigenetic Aging: Wave 1 only", va="center", ha="left",
            fontsize=FS_NOTE, color=MUTED, fontstyle="italic")
    base, full = data["Frailty"][:2]
    _segment_labels(ax, [("Wave 1 status + Tier 1", 0, base)])
    ax.text(full + 0.012, -0.38, "+ Tier 2", ha="left", va="bottom", fontsize=FS_NOTE,
            color=MUTED, clip_on=False)
    ax.plot([full - (full - base) / 2, full + 0.010], [-0.33, -0.36], color=MUTED,
            lw=0.4, clip_on=False, solid_capstyle="butt")
    _row_labels(ax, ORDER3)
    save(fig, "fig1_change")


# F: cross-domain SHAP-rank agreement
def panel_crossdomain():
    rows = rho()
    shared = float(np.mean([r[3] for r in rows]))
    assert 0 < min(r[2] for r in rows) and max(r[2] for r in rows) < shared - 0.3, "divergent"
    fig = figure(COL, SEC_H)
    result_heading(fig, "Aging phenotypes exhibit divergent psychosocial feature "
                        "importance rankings",
                   "Mean |SHAP| rankings of all 177 predictors in D's XGBoost models")
    W, H = fig.get_size_inches()
    left, bottom, ax_h = INDENT + ROW_LABEL_W, 0.36, 0.82
    ax = axes_in(fig, left, bottom, W - left - 0.12, ax_h)
    ax.set_xlim(0, 1.0); ax.set_ylim(2.5, -0.5)
    ax.axvline(shared, color=MUTED, lw=0.6, ls=(0, (2, 1.5)), zorder=1)
    ax.text(shared, -0.52, nolead(shared), ha="center", va="bottom", fontsize=FS_NOTE,
            fontweight="bold", color=MUTED, clip_on=False)
    ax.text(shared - 0.025, -0.30, "expected if one\nranking were shared",
            ha="right", va="top", fontsize=FS_NOTE, color=MUTED, linespacing=1.1)
    for i, (a, b, r, _) in enumerate(rows):
        ax.plot([0, r], [i, i], color=GUIDE, lw=0.8, zorder=2, solid_capstyle="butt")
        ax.scatter([r], [i], s=16, color=INK, zorder=3, lw=0)
        ax.text(r + 0.025, i, nolead(r), va="center", ha="left", fontsize=FS_LABEL,
                fontweight="bold")
        # the pair, above its line from x = 0: two outcome swatches, then names
        y = (bottom + ax_h * (2.5 - i) / 3.0 + 0.075) / H
        for j, dom in enumerate((a, b)):
            fig.patches.append(Rectangle(((left + j * 0.07) / W, y), 0.055 / W, 0.07 / H,
                                         transform=fig.transFigure, fc=COLORS[dom],
                                         ec="none"))
        fig.text((left + 0.16) / W, y + 0.035 / H, f"{a} · {b}", va="center", ha="left",
                 fontsize=FS_SUB)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", ".25", ".50", ".75", "1"])
    ax.set_xlabel("Spearman ρ between outcomes' rankings", labelpad=2)
    strip(ax)
    save(fig, "fig1_crossdomain")


# G: within-family, MZ co-twin resemblance
def panel_cotwin():
    d = icc()
    # the finding's words, held to the numbers they summarize
    assert min(v for v, _ in d.values()) > 0.1, "familial resemblance"
    fig = figure(COL, SEC_H)
    result_heading(fig, "Aging status and 9-year change exhibit familial resemblance",
                   "Held-out MZ twin pairs, outcome only, no predictors · null mixed model")
    W, H = fig.get_size_inches()
    ax = axes_in(fig, INDENT + ROW_LABEL_W, 0.44, 1.95, 0.74)
    ax.set_xlim(0, 0.8); ax.set_ylim(2.5, -0.5)
    off = 0.19
    for i, dom in enumerate(ORDER3):
        c = COLORS[dom]
        for k, a in enumerate(("cs", "change")):
            v, n = d[a, dom]
            y = i - off + 2 * off * k
            ax.plot([0, v], [y, y], color=GUIDE, lw=0.6, zorder=1, solid_capstyle="butt")
            ax.scatter([v], [y], s=20, zorder=3, lw=0.9,
                       facecolor=c if k == 0 else "white", edgecolor=c)
            ax.text(v + 0.022, y, f"{nolead(v)}  ({n})", va="center", ha="left",
                    fontsize=FS_NOTE, color=INK if k == 0 else MUTED,
                    fontweight="bold" if k == 0 else "normal")
    for t in (0.2, 0.4, 0.6):
        ax.axvline(t, color=GRID, lw=0.5, zorder=0)
    _row_labels(ax, ORDER3)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_xticklabels(["0", ".2", ".4", ".6", ".8"])
    ax.set_xlabel("MZ intraclass correlation (n pairs)", labelpad=2)
    strip(ax)
    mk = dict(marker="o", ls="none", markersize=4, markeredgewidth=0.9)
    _legend(fig, [Line2D([], [], color=MUTED, markerfacecolor=MUTED, label="Wave 1 status", **mk),
                  Line2D([], [], color=MUTED, markerfacecolor="white",
                         label="9-year change (Wave 2 given Wave 1)", **mk)])
    save(fig, "fig1_cotwin")


PANELS = [panel_cohorts, panel_predictors, panel_outcomes, panel_question,
          panel_concurrent, panel_change, panel_crossdomain, panel_cotwin]

if __name__ == "__main__":
    print("Figure 1 panels, at print size:")
    for p in PANELS:
        p()
    print(f"\nAll panels in: {OUT.relative_to(ROOT)}")
