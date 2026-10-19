#!/usr/bin/env python3
"""
Figure 2A: variance decomposition, stacked R2 bars (cross-sectional T1+T2;
longitudinal Yw1+T1+T2), labelled in percent as the text reports them.

Inputs: output/cs1_cross_sectional/cs1_r2_table.csv and
output/a4_longitudinal/a4_r2_table.csv (pooled); AL rows use the 16-marker
composite.

Usage:
    uv run python analysis/figures/fig2a_increments.py
"""
import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.core import artifact_paths as ap  # noqa: E402
OUT = ROOT / "manuscript" / "figures" / "fig2"
OUT.mkdir(parents=True, exist_ok=True)

CS_CSV   = ROOT / "output/cs1_cross_sectional/cs1_r2_table.csv"
LONG_CSV = ROOT / "output/a4_longitudinal/a4_r2_table.csv"

# Style
COLORS = {
    "Frailty":   "#E07A5F",
    "Cognition": "#3D7C98",
    "AL":        "#E5A84B",
}

def lighten(hex_color, factor):
    rgb = mcolors.to_rgb(hex_color)
    return tuple(c + (1.0 - c) * factor for c in rgb)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11,
    'axes.linewidth': 0.7,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'ytick.labelsize': 10.5,
})

FONT_SMALL = 9

# Data

domains = ["Frailty", "Cognition", "AL"]
domain_display = {"Frailty": "Frailty", "Cognition": "Cognition",
                  "AL": "Allostatic\nLoad"}

# AL composite
CS_MAP = {"Frailty": "FInoPsy_W1", "Cognition": "btactComp_W1",
          "AL": f"{ap.AL_STEM}_W1"}
LONG_MAP = {"Frailty": "FInoPsy_W2", "Cognition": "btactComp_W2",
            "AL": f"{ap.AL_STEM}_W2"}

def load_cs_data():
    df = pd.read_csv(CS_CSV)
    out = {}
    for dom, outcome in CS_MAP.items():
        sub = df[df["outcome"] == outcome].set_index("tier_config")["r2_mean"]
        out[dom] = {"t1": float(sub["T1"]), "t1t2": float(sub["T1T2"])}
    return out

def load_long_data():
    df_a4 = pd.read_csv(LONG_CSV)
    df_a4 = df_a4[df_a4["cohort"] == "pooled"]
    out = {}
    for dom, outcome in LONG_MAP.items():
        sub = df_a4[df_a4["outcome"] == outcome].set_index("tier_config")["r2_mean"]
        out[dom] = {"yw1": float(sub["Yw1"]),
                    "yw1t1": float(sub["Yw1T1"]),
                    "yw1t1t2": float(sub["Yw1T1T2"])}
    return out

CS_DATA = load_cs_data()
LONG_DATA = load_long_data()

print("CS_DATA:")
for d in domains:
    print(f"  {d}: T1={CS_DATA[d]['t1']:.4f}  T1T2={CS_DATA[d]['t1t2']:.4f}")
print("LONG_DATA:")
for d in domains:
    v = LONG_DATA[d]
    print(f"  {d}: Yw1={v['yw1']:.4f}  Yw1T1={v['yw1t1']:.4f}  Yw1T1T2={v['yw1t1t2']:.4f}")

def decompose_cs(d):
    return d["t1"], d["t1t2"] - d["t1"]

def decompose_long(d):
    t1_incr = d["yw1t1"] - d["yw1"]
    t2_incr = d["yw1t1t2"] - d["yw1t1"]
    if t1_incr < 0:
        return d["yw1t1"], 0, t2_incr
    return d["yw1"], t1_incr, t2_incr

ratios = {}
for d in domains:
    cs_t2 = CS_DATA[d]["t1t2"] - CS_DATA[d]["t1"]
    long_t2 = max(0.001, LONG_DATA[d]["yw1t1t2"] - LONG_DATA[d]["yw1t1"])
    ratios[d] = cs_t2 / long_t2

# Figure

fig, ax_a = plt.subplots(figsize=(5.5, 5.0))

bar_width = 0.30
intra_gap = 0.10
inter_gap = 0.70

HATCH_YW1 = '\\\\\\\\'
HATCH_T1 = '..'

cs_pos = []; long_pos = []; grp_centers = []
xc = 0
for i in range(3):
    cx = xc; lx = xc + bar_width + intra_gap
    cs_pos.append(cx); long_pos.append(lx)
    grp_centers.append((cx + lx) / 2)
    xc = lx + bar_width + inter_gap

for i, d in enumerate(domains):
    color = COLORS[d]
    color_t1 = lighten(color, 0.55)
    color_yw1 = lighten(color, 0.70)
    color_t2 = color

    # CS bar: T1 (bottom) + T2 (top)
    cs_t1, cs_t2 = decompose_cs(CS_DATA[d])
    ax_a.bar(cs_pos[i], cs_t1, bar_width, bottom=0,
             color=color_t1, edgecolor='#AAAAAA', linewidth=0,
             hatch=HATCH_T1, zorder=3)
    ax_a.bar(cs_pos[i], cs_t2, bar_width, bottom=cs_t1,
             color=color_t2, edgecolor='white', linewidth=0, zorder=3)

    if cs_t1 > 0.06:
        ax_a.text(cs_pos[i], cs_t1 / 2, f"{cs_t1:.0%}",
                  ha='center', va='center', fontsize=FONT_SMALL,
                  color='#444444', zorder=5)

    cs_total = cs_t1 + cs_t2
    ax_a.text(cs_pos[i], cs_total + 0.010,
              f'+{cs_t2:.1%}', ha='center', va='bottom',
              fontsize=9.5, fontweight='bold', color=color)

    # Long bar: Yw1 + T1 + T2
    yw1, long_t1, long_t2 = decompose_long(LONG_DATA[d])

    ax_a.bar(long_pos[i], yw1, bar_width, bottom=0,
             color=color_yw1, edgecolor='#AAAAAA', linewidth=0,
             hatch=HATCH_YW1, zorder=3)
    if long_t1 > 0.001:
        ax_a.bar(long_pos[i], long_t1, bar_width, bottom=yw1,
                 color=color_t1, edgecolor='#AAAAAA', linewidth=0,
                 hatch=HATCH_T1, zorder=3)
    ax_a.bar(long_pos[i], long_t2, bar_width, bottom=yw1 + long_t1,
             color=color_t2, edgecolor='white', linewidth=0, zorder=3)

    ax_a.text(long_pos[i], yw1 / 2, f"{yw1:.0%}",
              ha='center', va='center', fontsize=FONT_SMALL,
              color='#444444', zorder=5)

    total_long = yw1 + long_t1 + long_t2
    ax_a.text(long_pos[i], total_long + 0.010,
              f'+{long_t2:.1%}', ha='center', va='bottom',
              fontsize=9.5, fontweight='bold', color=color)

    group_max = max(cs_total, total_long)
    mid_x = (cs_pos[i] + long_pos[i]) / 2
    ax_a.annotate(f'{ratios[d]:.0f}:1',
                  xy=(mid_x, group_max + 0.055),
                  ha='center', va='bottom', fontsize=10.5,
                  fontweight='bold', color='#4A4A4A',
                  bbox=dict(boxstyle='round,pad=0.2', facecolor='#F0F0F0',
                            edgecolor='#CCCCCC', linewidth=0.5))

for i, d in enumerate(domains):
    ax_a.text(cs_pos[i], -0.020, "CS", ha='center', va='top',
              fontsize=9.5, color='#555555', clip_on=False)
    ax_a.text(long_pos[i], -0.020, "Long", ha='center', va='top',
              fontsize=9.5, color='#555555', clip_on=False)
    ax_a.text(grp_centers[i], -0.055, domain_display[d],
              ha='center', va='top', fontsize=11.5,
              fontweight='bold', color=COLORS[d], clip_on=False)

ax_a.set_ylabel('R²', fontsize=12.5)
ax_a.set_ylim(0, 0.88)
ax_a.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
ax_a.set_xlim(-0.22, xc - inter_gap + bar_width + 0.25)
ax_a.set_xticks([])
ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)
ax_a.set_title('A  Variance decomposition', fontsize=13, fontweight='bold',
               loc='left', pad=8)

legend_handles = [
    mpatches.Patch(facecolor=lighten('#888888', 0.70), edgecolor='#AAAAAA',
                   hatch=HATCH_YW1, label='$Y_{W1}$ (baseline)'),
    mpatches.Patch(facecolor=lighten('#888888', 0.55), edgecolor='#AAAAAA',
                   hatch=HATCH_T1, label='T1: background'),
    mpatches.Patch(facecolor='#888888', label='T2: psychosocial and behavioral'),
]
ax_a.legend(handles=legend_handles, fontsize=9, loc='upper right',
            framealpha=0.9, handlelength=1.5,
            bbox_to_anchor=(0.98, 0.98))

fig.subplots_adjust(left=0.12, right=0.97, top=0.90, bottom=0.14)
for fmt in ('png', 'pdf'):
    fig.savefig(OUT / f'fig2A.{fmt}', dpi=300,
                bbox_inches='tight', facecolor='white')
    print(f"Saved: {OUT / f'fig2A.{fmt}'}")
plt.close()
