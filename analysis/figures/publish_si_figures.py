#!/usr/bin/env python3
"""Copy pipeline-produced figures to the paths the supplement uses.

Every destination names its source; a missing source is an error.

    uv run python analysis/figures/publish_si_figures.py
    uv run python analysis/figures/publish_si_figures.py --check
"""

from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.core import artifact_paths as ap  # noqa: E402

AL = ap.AL_STEM
CS1_FIG = ap.CS1_DIR / "figures"
A11_FIG = ap.OUTPUT / "a11_mortality_shap" / "figures"
D1 = ap.OUTPUT / "d1_descriptive"
SUP = ROOT / "manuscript" / "supplement"

# destination -> source. Both are repository-relative; every pair is stated, so a
# figure that the paper carries and the pipeline does not produce is absent from
# this table rather than quietly approximated by a neighbouring one.
PUBLISH: dict[Path, Path] = {
    SUP / "fig_s1_change_score_distributions.png": D1 / "fig_change_score_distributions.png",
    SUP / "fig_s2_shap_dependence_frailty.png": CS1_FIG / "FInoPsy_W1_T1T2_dependence.png",
    SUP / "fig_s3_shap_dependence_cognition.png": CS1_FIG / "btactComp_W1_T1T2_dependence.png",
    SUP / "fig_s4_shap_dependence_allostatic_load.png": CS1_FIG / f"{AL}_W1_T1T2_dependence.png",
    SUP / "fig_s5_shap_dependence_phenoage.png": CS1_FIG / "phenoAge_res_T1T2_dependence.png",
    SUP / "fig_s6_mortality_validation.png": A11_FIG / "mortality_dual_scatter.png",
    # Same panels again under the outcome-keyed names the SHAP_dependence folder uses.
    SUP / "SHAP_dependence" / "FInoPsy_W1_T1T2_dependence.png":
        CS1_FIG / "FInoPsy_W1_T1T2_dependence.png",
    SUP / "SHAP_dependence" / "btactComp_W1_T1T2_dependence.png":
        CS1_FIG / "btactComp_W1_T1T2_dependence.png",
    SUP / "SHAP_dependence" / f"{AL}_W1_T1T2_dependence.png":
        CS1_FIG / f"{AL}_W1_T1T2_dependence.png",
    SUP / "SHAP_dependence" / "phenoAge_res_T1T2_dependence.png":
        CS1_FIG / "phenoAge_res_T1T2_dependence.png",
}

# Carried by the paper with nothing in the pipeline that produces them. Listed so
# that their absence from PUBLISH is a recorded fact rather than an oversight.
NO_SOURCE = {
    "manuscript/figures/fig1/**": "assembled by fig1_deck.py (step fig1) from the generated subpanels",
}


def main() -> int:
    ap_ = argparse.ArgumentParser(description=__doc__)
    ap_.add_argument("--check", action="store_true",
                     help="report what is stale or missing and change nothing")
    args = ap_.parse_args()

    missing, stale, copied = [], [], []
    for dest, src in PUBLISH.items():
        if not src.exists():
            missing.append((dest, src))
            continue
        if dest.exists() and filecmp.cmp(src, dest, shallow=False):
            continue
        stale.append(dest)
        if not args.check:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            copied.append(dest)

    for dest, src in missing:
        print(f"  MISSING SOURCE  {dest.relative_to(ROOT)}\n"
              f"                  needs {src.relative_to(ROOT)}")
    for d in copied:
        print(f"  published  {d.relative_to(ROOT)}")
    if args.check:
        for d in stale:
            print(f"  STALE      {d.relative_to(ROOT)}")

    print(f"\n{len(PUBLISH)} destination(s): {len(stale)} "
          f"{'stale' if args.check else 'republished'}, {len(missing)} missing a source, "
          f"{len(PUBLISH) - len(stale) - len(missing)} already current")
    for path, why in NO_SOURCE.items():
        print(f"  not published: {path} -- {why}")
    return 1 if missing or (args.check and stale) else 0


if __name__ == "__main__":
    sys.exit(main())
