"""Shared publication style for figure and table builders."""

import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# Paths
ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "output" / "table_figure"
CSV_DIR = OUTPUT_DIR / "csv"
MAIN_DIR = OUTPUT_DIR / "main"
SUPP_DIR = OUTPUT_DIR / "supplement"

# Input directories
A3_DIR = ROOT / "output" / "a3_domain_divergence"
A4_DIR = ROOT / "output" / "a4_longitudinal"
A6_DIR = ROOT / "output" / "a6_sibling"
A7_DIR = ROOT / "output" / "a7_mortality"
CS1_DIR = ROOT / "output" / "cs1_cross_sectional"
A9_DIR = ROOT / "output" / "a9_concurrent"

# PNAS dimensions (cm to inches)
CM2IN = 1 / 2.54
PNAS_SINGLE = 8.7 * CM2IN      # ~3.43 in
PNAS_1_5 = 11.4 * CM2IN        # ~4.49 in
PNAS_FULL = 17.8 * CM2IN       # ~7.01 in

# Font sizes
FONT_TITLE = 11
FONT_LABEL = 10
FONT_ANNOT = 8
FONT_TICK = 8

# Color palettes
DOMAIN_COLORS = {
    "Frailty": "#2CA6A4",
    "Cognition": "#7B68AE",
    "Allostatic Load": "#E8913A",
    "Epigenetic Aging": "#8C8C8C",
}

TIER_COLORS = {
    1: "#4682B4",   # steel blue
    2: "#CD5C5C",   # coral
    3: "#8FBC8F",   # sage
}

SUBDOMAIN_COLORS = {
    "demographics":             "#555555",
    "personality":              "#E24A33",
    "psychological_wellbeing":  "#348ABD",
    "mental_health_symptoms":   "#988ED5",
    "coping_resilience":        "#FBC15E",
    "life_evaluation":          "#8EBA42",
    "social_relationships":     "#FFB5B8",
    "intimate_partner":         "#D4281A",
    "household_caregiving":     "#6D904F",
    "health_behaviors":         "#E5AE38",
    "cognitive_activities":     "#30A2DA",
    "substance_use":            "#FC4F30",
    "work_occupation":          "#A0A0A0",
    "socioeconomic":            "#5B8C5A",
    "healthcare_access":        "#C4AD66",
}

# Domain short labels for mapping
DOMAIN_SHORT = {
    "FInoPsy": "Frailty",
    "btactComp": "Cognition",
    "AL": "Allostatic Load",
    "phenoAge": "Epigenetic Aging",
    "phenoAge_res": "Epigenetic Aging",
    "grimAgeN_res": "Epigenetic Aging",
    "dunedinPace_res": "Epigenetic Aging",
    "episoMem": "Cognition",
    "efFactor": "Cognition",
}


# Variable tier / subdomain lookup
def load_variable_tiers():
    """Parse variable_config.yaml into {variable_name: tier_int}."""
    cfg_path = ROOT / "config" / "variable_config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    tiers = {}
    for var in cfg.get("tier1", []):
        tiers[var] = 1
    for var in cfg.get("tier2", []):
        tiers[var] = 2
    for var in cfg.get("tier3", []):
        tiers[var] = 3
    return tiers


def load_variable_subdomains():
    """Parse variable_config.yaml into {variable_name: subdomain_str}."""
    cfg_path = ROOT / "config" / "variable_config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    subs = {}
    for sub_name, var_list in cfg.get("subdomain", {}).items():
        for var in var_list:
            subs[var] = sub_name
    return subs


# Lazy-loaded caches
_TIER_CACHE = None
_SUB_CACHE = None


def get_tier(var):
    """Return tier int (1/2/3) for a variable, or 0 if unknown."""
    global _TIER_CACHE
    if _TIER_CACHE is None:
        _TIER_CACHE = load_variable_tiers()
    return _TIER_CACHE.get(var, 0)


def get_subdomain(var):
    """Return subdomain string for a variable, or '' if unknown."""
    global _SUB_CACHE
    if _SUB_CACHE is None:
        _SUB_CACHE = load_variable_subdomains()
    return _SUB_CACHE.get(var, "")


# Display name conversion
_DISPLAY_OVERRIDES = {
    "bmi": "BMI",
    "whr": "Waist-hip ratio",
    "FInoPsy": "Frailty index",
    "btactComp": "BTACT composite",
    "AL": "Allostatic load",
    "HMOClinic": "HMO clinic visits",
    "mpqWellbeing": "MPQ Well-being",
    "mpqPotency": "MPQ Potency",
    "mpqAchievement": "MPQ Achievement",
    "mpqCloseness": "MPQ Closeness",
    "mpqStress": "MPQ Stress reaction",
    "mpqAggression": "MPQ Aggression",
    "mpqAlienation": "MPQ Alienation",
    "mpqControl": "MPQ Control",
    "mpqTradition": "MPQ Traditionalism",
    "mpqHarmAvoid": "MPQ Harm avoidance",
    "vmJobAct_vig": "Vigorous job activity",
    "vmLeisureAct_vig": "Vigorous leisure activity",
    "vmChoresAct_vig": "Vigorous chores activity",
    "educationACE_3item": "Childhood educ. adversity",
    "famDysfuncACE_3item": "Family dysfunction ACE",
    "ecoACE_1item": "Economic adversity ACE",
    "smokeExposure_3item": "Smoke exposure (3-item)",
    "drinkExposure_PA59": "Drink exposure (PA59)",
    "hhPersonalEarn": "HH personal earnings",
    "hhPension": "HH pension income",
    "hhSocialSecurity": "HH Social Security",
    "FInoPsy_W1": "Frailty (baseline)",
    "btactComp_W1": "BTACT (baseline)",
    "phenoAge_res": "PhenoAge acceleration",
    "age_W1": "Age (baseline)",
}


def display_name(var):
    """Convert camelCase variable name to readable display label."""
    if var in _DISPLAY_OVERRIDES:
        return _DISPLAY_OVERRIDES[var]
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", var)
    # Split on underscores
    name = name.replace("_", " ")
    # Capitalize first letter only
    if name:
        name = name[0].upper() + name[1:]
    return name


# Matplotlib setup
def setup_rcparams():
    """Apply PNAS-compatible matplotlib defaults."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": FONT_LABEL,
        "axes.titlesize": FONT_TITLE,
        "axes.labelsize": FONT_LABEL,
        "xtick.labelsize": FONT_TICK,
        "ytick.labelsize": FONT_TICK,
        "legend.fontsize": FONT_ANNOT,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42,       # TrueType for journal compatibility
        "ps.fonttype": 42,
    })


# Save helpers
def ensure_dirs():
    """Create output directories if needed."""
    for d in [CSV_DIR, MAIN_DIR, SUPP_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def save_fig(fig, name, subdir="main"):
    """Save figure as PDF + PNG at 300 DPI."""
    ensure_dirs()
    out_dir = MAIN_DIR if subdir == "main" else SUPP_DIR
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}", format=ext)
    plt.close(fig)
    print(f"  Saved {name}.pdf/.png to {out_dir.relative_to(ROOT)}/")


def save_table(df, name, subdir="main", latex=True):
    """Save table as CSV + optionally LaTeX (booktabs)."""
    ensure_dirs()
    out_dir = MAIN_DIR if subdir == "main" else SUPP_DIR

    csv_path = CSV_DIR / f"{name}.csv"
    df.to_csv(csv_path, index=False)

    # Also save formatted CSV to output dir
    df.to_csv(out_dir / f"{name}.csv", index=False)

    # LaTeX
    if latex:
        tex_path = out_dir / f"{name}.tex"
        tex_str = df_to_latex(df, name)
        tex_path.write_text(tex_str)

    print(f"  Saved {name}.csv/.tex to {out_dir.relative_to(ROOT)}/")


def df_to_latex(df, label="table"):
    """Convert DataFrame to LaTeX booktabs table string."""
    n_cols = len(df.columns)
    col_spec = "l" + "r" * (n_cols - 1)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
    ]

    header = " & ".join(latex_escape(str(c)) for c in df.columns) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for _, row in df.iterrows():
        row_str = " & ".join(latex_escape(str(v)) for v in row.values) + r" \\"
        lines.append(row_str)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(rf"\label{{tab:{label}}}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# Formatting helpers
def fmt_ci(lo, hi, decimals=3):
    """Format confidence interval as [lo, hi]."""
    return f"[{lo:.{decimals}f}, {hi:.{decimals}f}]"


def fmt_p(p):
    """Format p-value with scientific notation for small values."""
    if p < 0.001:
        return f"{p:.1e}"
    elif p < 0.01:
        return f"{p:.3f}"
    else:
        return f"{p:.2f}"


def fmt_r2(val, decimals=2):
    """Format R-squared value."""
    if pd.isna(val):
        return "—"
    return f"{val:.{decimals}f}"


def fmt_hr(hr, lo, hi):
    """Format hazard ratio with CI."""
    return f"{hr:.3f} [{lo:.3f}, {hi:.3f}]"


def latex_escape(s):
    """Escape special LaTeX characters."""
    s = str(s)
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "^": r"\^{}",
        "~": r"\textasciitilde{}",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s
