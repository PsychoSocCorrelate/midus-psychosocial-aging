"""Figure S7: exploratory profile comparison.

Three panels (frailty / cognition / allostatic load). Each panel shows the focal
predictors' SHAP contributions under the favorable / neutral / unfavorable profiles
(grouped horizontal bars), with the model-predicted (age-residualized) outcome and its
95% CI annotated per profile. Reads the outputs of analysis/profile_comparison/profile_comparison.py.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.core import pub_style as ps
from analysis.core import artifact_paths as ap  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "output" / "profile_comparison"
SUPPLEMENT = ROOT / "manuscript" / "supplement"

ORDER = ["favorable", "neutral", "unfavorable"]
PROFILE_COLOR = {"favorable": "#1f6b5a", "neutral": "#9a6726", "unfavorable": "#a13d37"}
PROFILE_LABEL = {"favorable": "Favorable", "neutral": "Neutral", "unfavorable": "Unfavorable"}
OUTCOME_KEYS = ["FInoPsy_W1", "btactComp_W1", f"{ap.AL_STEM}_W1"]
OUTCOME_TITLE = {"FInoPsy_W1": "Frailty", "btactComp_W1": "Cognition",
                 f"{ap.AL_STEM}_W1": "Allostatic load"}
DOMAIN_KEY = {"FInoPsy_W1": "Frailty", "btactComp_W1": "Cognition",
              f"{ap.AL_STEM}_W1": "Allostatic Load"}
# Focal predictors whose model direction is confounded and must not read as a recommendation.
# The "†" (categorical) and "*" (confounded) glyphs are decoded in the figure caption and Table S16,
# not on the figure, to avoid duplicating the caption's explanatory prose.
CONFOUNDED = {"drinkStatus"}


def _fmt_signed(x, dec=2):
    """Signed fixed-decimal string, collapsing negative/positive zero to a clean '0.00'."""
    s = f"{x:+.{dec}f}"
    return "0." + "0" * dec if s.lstrip("+-") == "0." + "0" * dec else s


def _outcome_sd(pred, okey):
    """SD of the (age-residualized) outcome, from profile_predictions.csv."""
    vals = pred.loc[pred["outcome_key"] == okey, "outcome_sd"].dropna()
    return float(vals.iloc[0]) if len(vals) else None


def main():
    ps.setup_rcparams()
    pred = pd.read_csv(DATA / "profile_predictions.csv")
    shap = pd.read_csv(DATA / "profile_shap.csv")

    fig, axes = plt.subplots(1, 3, figsize=(ps.PNAS_FULL, 4.5))
    for ax, okey in zip(axes, OUTCOME_KEYS):
        sub = shap[shap["outcome_key"] == okey]
        feats = list(dict.fromkeys(sub["feature"]))  # preserve key_fields order
        # Order focal predictors by |favorable - unfavorable| SHAP spread (top drivers first).
        def spread(f):
            fav = sub[(sub.feature == f) & (sub.profile == "favorable")]["shap"].values
            unf = sub[(sub.feature == f) & (sub.profile == "unfavorable")]["shap"].values
            return abs((fav[0] if len(fav) else 0) - (unf[0] if len(unf) else 0))
        feats = sorted(feats, key=spread)  # ascending -> biggest at top (barh y grows up)
        labels = [sub[sub.feature == f]["label"].iloc[0] for f in feats]

        y = np.arange(len(feats))
        h = 0.26
        for k, prof in enumerate(ORDER):
            ps_sub = {f: sub[(sub.feature == f) & (sub.profile == prof)] for f in feats}
            vals = np.array([ps_sub[f]["shap"].iloc[0] for f in feats])
            # Asymmetric bootstrap SHAP CIs -> error-bar half-widths (clip tiny negatives from rounding).
            err_lo = np.clip(vals - np.array([ps_sub[f]["shap_ci_lo"].iloc[0] for f in feats]), 0, None)
            err_hi = np.clip(np.array([ps_sub[f]["shap_ci_hi"].iloc[0] for f in feats]) - vals, 0, None)
            ax.barh(y + (1 - k) * h, vals, height=h, color=PROFILE_COLOR[prof],
                    label=PROFILE_LABEL[prof], edgecolor="white", linewidth=0.4,
                    xerr=np.vstack([err_lo, err_hi]),
                    error_kw=dict(ecolor="#3a3a3a", elinewidth=0.5, capsize=1.3, capthick=0.5))
        ax.axvline(0, color="#888", lw=0.8, ls=(0, (2, 2)))
        ax.set_yticks(y)
        # Mark categorical predictors (†) and confounded ones (*), keyed to the figure notes.
        vtype = {f: sub[sub.feature == f]["var_type"].iloc[0] for f in feats}

        def _mark(feat, lab):
            return lab + (" †" if vtype.get(feat) == "categorical" else "") + \
                (" *" if feat in CONFOUNDED else "")
        ylabels = [_mark(feat, lab) for feat, lab in zip(feats, labels)]
        ax.set_yticklabels(ylabels, fontsize=ps.FONT_TICK)
        ax.set_xlabel("SHAP (age-resid.)", fontsize=ps.FONT_LABEL)
        ax.tick_params(labelsize=ps.FONT_TICK)

        # Title, then a format-key line (with the outcome SD for scale), then per-profile
        # predicted outcome [95% CI] (percentile), all stacked above the axes.
        ax.set_title(OUTCOME_TITLE[okey], fontsize=ps.FONT_TITLE, pad=56,
                     color=ps.DOMAIN_COLORS[DOMAIN_KEY[okey]])
        sd = _outcome_sd(pred, okey)
        sd_txt = f"SD = {sd:.2f}" if sd is not None else "SD n/a"
        ax.text(0.0, 1.255, f"{sd_txt}   ·   pred [95% CI] pctl", transform=ax.transAxes,
                fontsize=ps.FONT_ANNOT - 2, color="#555", va="bottom", ha="left", style="italic")
        p = pred[pred["outcome_key"] == okey].set_index("profile")
        y0 = 1.175
        for prof in ORDER:
            r = p.loc[prof]
            txt = (f"{PROFILE_LABEL[prof][:3]} {_fmt_signed(r['prediction'])} "
                   f"[{_fmt_signed(r['ci_lo'])}, {_fmt_signed(r['ci_hi'])}] p{r['percentile']:.0f}")
            ax.text(0.0, y0, txt, transform=ax.transAxes, fontsize=ps.FONT_ANNOT - 1.5,
                    color=PROFILE_COLOR[prof], va="bottom", ha="left")
            y0 -= 0.075

    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="lower center", ncol=3, fontsize=ps.FONT_ANNOT,
               frameon=False, bbox_to_anchor=(0.5, 0.035))
    fig.subplots_adjust(left=0.13, right=0.985, top=0.74, bottom=0.205, wspace=0.95)
    ps.save_fig(fig, "fig_s7_profile_comparison", subdir="supplement")

    # Sync into the manuscript supplement tree (canonical home for SI assets).
    SUPPLEMENT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        src = ps.SUPP_DIR / f"fig_s7_profile_comparison.{ext}"
        shutil.copyfile(src, SUPPLEMENT / f"fig_s7_profile_comparison.{ext}")
    print(f"  Synced fig_s7_profile_comparison.png/.pdf to {SUPPLEMENT.relative_to(ROOT)}/")


if __name__ == "__main__":
    raise SystemExit(main())
