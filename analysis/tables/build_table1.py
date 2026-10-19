#!/usr/bin/env python3
"""
Build Table 1 and the full within/between-family decomposition tables
(cross-sectional and change) with FDR-adjusted within-family predictor lists.

Sources:
  - output/a6_sibling/a6_cs_{All_paired,MZ_sensitivity}_{outcome}.csv
  - output/a6_sibling/a6_{All_paired,MZ_sensitivity}_{outcome}.csv
  - output/a6_sibling/a6_icc_denominators.csv (ICC by zygosity, computed)

Output: manuscript/tables/family_decomposition/family_decomposition.xlsx + *.csv (mirror)
        manuscript/tables/table1/table1.csv + table1_provenance.csv + table1.md
Table 1 also reads output/within_family_mde/within_family_mde_summary.csv for its
change-layer MDE column.

The Overview sheet is the manuscript's Table 1. The step also writes that sheet to
manuscript/tables/table1/ in the paper's column layout, so both are rendered from
one computation. The SI tables S4-S10 are built from the full tables.
"""

from pathlib import Path
import pandas as pd
import openpyxl as xl
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side

ROOT = Path(__file__).resolve().parents[2]
A6 = ROOT / "output/a6_sibling"
OUT = ROOT / "manuscript/tables/family_decomposition"
T1_OUT = ROOT / "manuscript/tables/table1"

# Stem of the allostatic-load outcome columns.
AL_STEM = "ALcomp"

DOMAINS = [("FInoPsy", "Frailty"),
           ("btactComp", "Cognition"),
           (AL_STEM, "Allostatic load")]

# Change-layer median minimum detectable effects, for Table 1
MDE_SUMMARY = ROOT / "output/within_family_mde/within_family_mde_summary.csv"

# ICC by zygosity, read from a6_icc_denominators.csv
ICC_SOURCE = A6 / "a6_icc_denominators.csv"


def _load_icc(analysis: str) -> dict:
    """{outcome: {group: (n_pairs, icc)}} for one analysis layer ("cs" or "change")."""
    df = pd.read_csv(ICC_SOURCE)
    df = df[df["analysis"] == analysis]
    out: dict[str, dict[str, tuple[int, float]]] = {}
    for _, r in df.iterrows():
        out.setdefault(r["outcome"], {})[r["group"]] = (int(r["n_pairs"]), float(r["icc"]))
    missing = [o for o, _ in DOMAINS if o not in out]
    if missing:
        raise KeyError(
            f"{ICC_SOURCE.name} has no {analysis} rows for {missing}. "
            f"Run a6_family_decomposition before this step."
        )
    return out


ICC_W1 = _load_icc("cs")
ICC_CHANGE = _load_icc("change")


def fmt_ci(lo, hi):
    if pd.isna(lo) or pd.isna(hi):
        return "—"
    return f"[{lo:+.4f}, {hi:+.4f}]"


def fmt_sci(p):
    if pd.isna(p):
        return "—"
    if p < 1e-3:
        return f"{p:.1e}"
    return f"{p:.3f}"


def load_results(analysis_kind: str, group: str, outcome: str) -> pd.DataFrame:
    """analysis_kind ∈ {'cs', 'change'}, group ∈ {'All_paired','MZ_sensitivity'}."""
    stem = f"a6_cs_{group}_{outcome}" if analysis_kind == "cs" else f"a6_{group}_{outcome}"
    path = A6 / f"{stem}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df.assign(
        outcome=outcome,
        domain=dict(DOMAINS)[outcome],
        analysis=analysis_kind,
        group=group,
    )
    return df


def build_full_results() -> pd.DataFrame:
    rows = []
    for kind in ["cs", "change"]:
        for group in ["All_paired", "MZ_sensitivity"]:
            for outcome, _ in DOMAINS:
                df = load_results(kind, group, outcome)
                if df.empty:
                    continue
                rows.append(df)
    full = pd.concat(rows, ignore_index=True)
    full = attach_mz_counts(full)

    keep_cols = [
        "domain", "outcome", "analysis", "group",
        "focal_var", "n_pairs", "n_MZ", "n_MZ_status",
        "beta_W", "beta_W_ci_lo", "beta_W_ci_hi",
        "beta_W_std", "beta_W_p", "beta_W_q",
        "beta_B", "beta_B_std", "beta_B_p",
        "attenuation_ratio",
    ]
    full = full[keep_cols].copy()
    full = full.sort_values(
        ["domain", "analysis", "group", "beta_W_q", "beta_W_p"],
        ascending=[True, True, True, True, True],
        na_position="last",
    ).reset_index(drop=True)
    return full


def attach_mz_counts(full: pd.DataFrame) -> pd.DataFrame:
    """Attach predictor-complete MZ denominators to every coefficient row."""
    keys = ["analysis", "outcome", "focal_var"]
    mz = (
        full[full["group"] == "MZ_sensitivity"][keys + ["n_pairs"]]
        .rename(columns={"n_pairs": "n_MZ"})
        .drop_duplicates(keys)
    )
    out = full.merge(mz, on=keys, how="left")
    mask = out["group"] == "MZ_sensitivity"
    out.loc[mask, "n_MZ"] = out.loc[mask, "n_pairs"]
    out["n_MZ_status"] = "matched MZ model-complete count"
    out.loc[mask, "n_MZ_status"] = "row is MZ_sensitivity"
    out.loc[
        out["n_MZ"].isna(), "n_MZ_status"
    ] = "not modeled in MZ_sensitivity after discordance/power filters"
    return out


def build_fdr_summary(full: pd.DataFrame) -> pd.DataFrame:
    """One row per FDR-surviving predictor (q ≤ 0.05)."""
    sig = full[full["beta_W_q"] <= 0.05].copy()
    sig["β_W raw [95% CI]"] = [
        f"{b:+.4f} {fmt_ci(lo, hi)}"
        for b, lo, hi in zip(
            sig["beta_W"], sig["beta_W_ci_lo"], sig["beta_W_ci_hi"]
        )
    ]
    sig["β_W std"] = sig["beta_W_std"].map(lambda x: f"{x:+.3f}" if pd.notna(x) else "—")
    sig["β_W q"] = sig["beta_W_q"].map(fmt_sci)
    sig["β_W p"] = sig["beta_W_p"].map(fmt_sci)

    out = sig[[
        "domain", "analysis", "group", "focal_var", "n_pairs", "n_MZ", "n_MZ_status",
        "β_W raw [95% CI]", "β_W std", "β_W p", "β_W q",
    ]].rename(columns={
        "domain": "Domain",
        "analysis": "Analysis",
        "group": "Group",
        "focal_var": "Variable",
        "n_pairs": "n_pairs",
        "n_MZ": "n_MZ",
        "n_MZ_status": "n_MZ_status",
    })
    return out


def build_overview(full: pd.DataFrame) -> pd.DataFrame:
    """Compact domain × analysis summary row."""
    rows = []
    for outcome, domain in DOMAINS:
        # Counts of FDR survivors per (analysis, group)
        def _survivors(kind, grp):
            d = full[
                (full["outcome"] == outcome)
                & (full["analysis"] == kind)
                & (full["group"] == grp)
            ]
            total = len(d)
            sig = int((d["beta_W_q"] <= 0.05).sum())
            return total, sig

        cs_ap_t, cs_ap_s = _survivors("cs", "All_paired")
        cs_mz_t, cs_mz_s = _survivors("cs", "MZ_sensitivity")
        ch_ap_t, ch_ap_s = _survivors("change", "All_paired")
        ch_mz_t, ch_mz_s = _survivors("change", "MZ_sensitivity")

        icc_w1_mz = ICC_W1[outcome]["MZ"]
        icc_ch_mz = ICC_CHANGE[outcome]["MZ"]

        rows.append({
            "Domain": domain,
            "Outcome (var)": outcome,
            "CS n_pairs (All / MZ)": f"{ICC_W1[outcome]['All_paired'][0]} / {icc_w1_mz[0]}",
            "Change n_pairs (All / MZ)": f"{ICC_CHANGE[outcome]['All_paired'][0]} / {icc_ch_mz[0]}",
            "ICC_MZ at W1 (CS)": f"{icc_w1_mz[1]:.3f}",
            "ICC_MZ of change": f"{icc_ch_mz[1]:.3f}",
            "CS FDR-surviving β_W (All_paired)": f"{cs_ap_s}/{cs_ap_t}",
            "CS FDR-surviving β_W (MZ_sens)": f"{cs_mz_s}/{cs_mz_t}",
            "Change FDR-surviving β_W (All_paired)": f"{ch_ap_s}/{ch_ap_t}",
            "Change FDR-surviving β_W (MZ_sens)": f"{ch_mz_s}/{ch_mz_t}",
        })
    return pd.DataFrame(rows)


def build_icc_table() -> pd.DataFrame:
    rows = []
    for outcome, domain in DOMAINS:
        for group in ["MZ", "DZ_SS", "All_paired"]:
            n1, icc1 = ICC_W1[outcome][group]
            n2, icc2 = ICC_CHANGE[outcome][group]
            rows.append({
                "Domain": domain,
                "Outcome": outcome,
                "Sibling stratum": group,
                "n_pairs (CS / change)": f"{n1} / {n2}",
                "ICC at W1 (CS)": f"{icc1:.3f}",
                "ICC of change (W2|W1)": f"{icc2:.3f}",
            })
    return pd.DataFrame(rows)


def build_denominator_table(full: pd.DataFrame) -> pd.DataFrame:
    """CSV bridge between displayed and model-complete counts."""
    rows = []
    for outcome, domain in DOMAINS:
        for analysis, icc_source in [("cs", ICC_W1), ("change", ICC_CHANGE)]:
            for group in ["All_paired", "MZ_sensitivity"]:
                model = full[
                    (full["outcome"] == outcome)
                    & (full["analysis"] == analysis)
                    & (full["group"] == group)
                ]
                if model.empty:
                    continue
                display_group = "MZ" if group == "MZ_sensitivity" else "All_paired"
                display_n = icc_source[outcome][display_group][0]
                model_counts = model["n_pairs"].dropna().astype(int)
                rows.append({
                    "Domain": domain,
                    "Outcome": outcome,
                    "Analysis": analysis,
                    "Group": group,
                    "display_denominator_type": "overview/ICC",
                    "display_n_pairs": int(display_n),
                    "model_denominator_type": "model-complete coefficient row",
                    "model_n_pairs_min": int(model_counts.min()),
                    "model_n_pairs_max": int(model_counts.max()),
                    "model_n_MZ_min": (
                        int(model["n_MZ"].dropna().astype(int).min())
                        if model["n_MZ"].notna().any() else None
                    ),
                    "model_n_MZ_max": (
                        int(model["n_MZ"].dropna().astype(int).max())
                        if model["n_MZ"].notna().any() else None
                    ),
                    "note": (
                        "AL rows use AL-specific sibling denominators"
                        if outcome == AL_STEM else ""
                    ),
                })
    return pd.DataFrame(rows)


# Per-sheet table notes (footer below each table).
# Each entry: list of paragraph strings; blank strings = paragraph break.
SHEET_NOTES = {
    "Overview": [
        "Note. Two parallel Mundlak mixed-effects models on the MIDUS 2 and MIDUS Refresher twin subsample: "
        "(i) cross-sectional at W1 (Y_w1 on B/W-partitioned focal X + age_W1 B/W + sex + pair random intercept) and "
        "(ii) 9-year change (Y_w2 on the same partition plus the autoregressive B/W-partitioned Y_w1 and age_W2). "
        "Focal predictors differ by analysis: CS uses the top-20-∩-stable variables (top20_stability ≥ 0.60) "
        "from cs1_cross_sectional bootstrap SHAP; change uses the stable A4 longitudinal SHAP set, excluding the autoregressive W1 outcome. "
        "All_paired = MZ + DZ_SS + DZ_OS twin pairs (primary); MZ_sens = MZ pairs only (sensitivity). "
        "ICC_MZ is the outcome null-model intraclass correlation (σ²_u / (σ²_u + σ²_e)); "
        "overview denominators are domain-level ICC counts, while coefficient rows carry predictor-complete n_pairs and n_MZ. "
        "Allostatic-load rows use the allostatic-load biomarker sample and should not be assumed to share frailty/cognition denominators. "
        "CS stratum is consistently higher than change across domains, motivating the CS analysis as the primary within-family test. "
        "\"FDR-surviving β_W\" = predictors with Benjamini–Hochberg q ≤ 0.05 within each (analysis × group × outcome) predictor set. "
        "Following Turkheimer (2000) and Harden (2021), β_W indexes familial resemblance of psychosocial–aging coupling net of shared background, not heritability of the trait. "
        "Full predictor-level results appear in the \"CS full\", \"Change full\", and \"FDR-surviving β_W\" sheets; methods detail in \"Notes\"."
    ],
    "ICC by zygosity": [
        "Note. Null-model intraclass correlations (ICC = σ²_u / (σ²_u + σ²_e)) from mixed-effects regressions "
        "with pair random intercept and age (decomposed B/W) + sex fixed covariates. "
        "CS column: ICC of the Wave-1 outcome only (no lagged Y). "
        "Change column: ICC of the Wave-2 outcome after conditioning on Wave-1 outcome (B/W-partitioned); "
        "this is the familial resemblance of the residual change, not of raw W2 state. "
        "Sibling strata: MZ (monozygotic), DZ_SS (dizygotic same-sex), All_paired (MZ + DZ_SS + DZ_OS). "
        "CS n_pairs exceeds change n_pairs because the change model additionally requires W2 outcome presence."
    ],
    "FDR-surviving β_W": [
        "Note. All within-family (β_W) predictors with Benjamini–Hochberg q ≤ 0.05, pooled across analyses. "
        "Analysis: cs = cross-sectional W1; change = W1→W2 with autoregressive W1 control. "
        "Group: All_paired = primary; MZ_sensitivity = MZ-only robustness check. "
        "β_W raw = unstandardized coefficient; β_W std = Snijders–Bosker level-specific standardized; 95% CI from 1000× pairs-level bootstrap. "
        "n_pairs is predictor-complete for the displayed group; n_MZ is the matched predictor-complete MZ denominator for the same coefficient where available, with n_MZ_status marking skipped MZ models. "
        "q-values are BH-FDR across all predictors within each (analysis × group × outcome) table. "
        "Between-family (β_B) coefficients for the same variable–outcome pairs are in the \"CS full\" / \"Change full\" sheets."
    ],
    "CS full": [
        "Note. Full cross-sectional (W1) within/between-family decomposition. "
        "Each row: one focal predictor × group × outcome. Model: Y_w1 = β_B · X̄_j + β_W · (X_ij − X̄_j) + age_W1 (B/W) + sex + (1|pair_id). "
        "β_W = within-family effect; β_B = between-family effect; β_W/β_B = attenuation ratio (reported when |β_B| > 0.5·SE(β_B)). "
        "Standardized β use Snijders–Bosker (2012) level-specific SDs. CI from 1000× pairs-level bootstrap. "
        "q-values are BH-FDR within each (group × outcome) table; ✓ in the final column flags q ≤ 0.05. "
        "n_pairs and n_MZ are predictor-complete coefficient-row denominators; n_MZ_status marks variables not fitted in the MZ sensitivity after discordance/power filters. Table S4/Overview uses domain-level ICC denominators. "
        "Focal set: top-20 mean|SHAP| ∩ stable (top20_stability ≥ 0.60) from cs1_cross_sectional bootstrap SHAP per outcome, after pair-level filters. "
        "Categorical dummies (workStatus, occGroup, workIndustry) are estimated when at least two pairs are discordant within-group."
    ],
    "Change full": [
        "Note. Full 9-year change (W1 → W2) within/between-family decomposition. "
        "Model: Y_w2 = β_B · X̄_j + β_W · (X_ij − X̄_j) + α_B · Ȳ_w1,j + α_W · (Y_w1,ij − Ȳ_w1,j) + age_W2 (B/W) + sex + (1|pair_id). "
        "The autoregressive Y_w1 terms absorb baseline state, so β_W reports within-pair predictor effect on residual 9-year change. "
        "Standardized β: Snijders–Bosker level-specific. CI from 1000× pairs-level bootstrap. "
        "q-values are BH-FDR within each (group × outcome) table; ✓ flags q ≤ 0.05. "
        "Focal set: stable A4 longitudinal bootstrap SHAP per outcome, excluding the autoregressive W1 outcome. "
        "Sample sizes are roughly half the CS sizes because complete W1+W2 outcome data are required."
        " n_pairs and n_MZ are predictor-complete coefficient-row denominators; n_MZ_status marks variables not fitted in the MZ sensitivity after discordance/power filters. Table S4/Overview uses domain-level ICC denominators."
    ],
}


NOTES_LINES = [
    "Within/between-family decomposition of psychosocial–aging coupling.",
    "",
    "Two parallel Mundlak models fit on twin pairs from MIDUS 2 and the MIDUS Refresher (Wave 1), followed at MIDUS 3 and Refresher 2 (Wave 2):",
    "",
    "  Cross-sectional (W1):   Y_w1 ~ β_B · X̄_j + β_W · (X_ij − X̄_j) + age_W1 (B/W) + sex + (1|pair_id)",
    "  Change (W1 → W2):       Y_w2 ~ β_B · X̄_j + β_W · (X_ij − X̄_j) + α_B·Ȳ_w1 + α_W·(Y_ij−Ȳ_w1) + age_W2 (B/W) + sex + (1|pair_id)",
    "",
    "β_B = between-family coefficient (shared-background signal).",
    "β_W = within-family coefficient (differentiation net of shared background).",
    "Both β are reported raw and Snijders–Bosker (2012) level-specific standardized; 95% CI from 1000× pairs-level bootstrap.",
    "q-values: Benjamini–Hochberg FDR across all predictors within each (analysis × group × outcome) table.",
    "",
    "Predictor sets (separate for each analysis):",
    "  CS:     top-20 mean|SHAP| ∩ stable (top20_stability ≥ 0.60) from cs1_cross_sectional bootstrap SHAP (W1 outcome).",
    "  Change: stable A4 longitudinal bootstrap SHAP vars (excluding the autoregressive W1 outcome).",
    "",
    "Primary model: All_paired (MZ + DZ_SS + DZ_OS twin pairs).",
    "Sensitivity:  MZ_sensitivity (MZ pairs only).",
    "Denominators: Overview and ICC sheets report domain-level outcome/ICC counts; CS full, Change full, and FDR-surviving sheets report predictor-complete coefficient-row n_pairs plus matched n_MZ and n_MZ_status.",
    "Allostatic-load sibling rows use allostatic-load biomarker-specific sample counts because the AL biomarker longitudinal sample keeps the sibling-exclusion exception for power.",
    "Categorical variables (workStatus, occGroup, workIndustry) are expanded to k-1 dummies with substantive groupings; dummies with fewer than two discordant pairs are excluded.",
    "",
    "Sample sizes differ across analyses because the change model additionally requires W2 outcome presence; CS retains roughly 2× more pairs per domain.",
    "Outcome null-model ICC_MZ by zygosity: higher cross-sectionally than for 9-year change in all three outcomes.",
    "",
    "Within-pair β_W effects are interpreted following Turkheimer (2000) and Harden (2021) as the familial resemblance of psychosocial–aging coupling, not heritability of the trait. The design controls shared family background but not non-shared confounders.",
    "",
    "Sheets:",
    "  1. Overview           — Domain × analysis summary (ICC and FDR-surviving counts).",
    "  2. ICC by zygosity    — Null-model outcome ICC at W1 (CS) and of change (W1→W2) for MZ, DZ_SS, All_paired strata.",
    "  3. FDR-surviving β_W  — All within-family predictors with q ≤ 0.05, flat across analyses.",
    "  4. CS full            — All CS (W1) decomposition rows (both groups, all domains).",
    "  5. Change full        — All change decomposition rows (both groups, all domains).",
    "  6. Notes              — This page.",
]


def _autosize(ws):
    for col in ws.columns:
        length = 0
        letter = col[0].column_letter
        for cell in col:
            v = "" if cell.value is None else str(cell.value)
            length = max(length, len(v))
        ws.column_dimensions[letter].width = min(max(length + 2, 10), 60)


def _write_sheet(ws, df, title=None, footer_paragraphs=None):
    row0 = 1
    if title:
        ws.cell(row=1, column=1, value=title).font = Font(bold=True, size=12)
        row0 = 3
    for j, col in enumerate(df.columns, start=1):
        c = ws.cell(row=row0, column=j, value=str(col))
        c.font = Font(bold=True)
        c.fill = PatternFill("solid", fgColor="E6E6E6")
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    thin = Side(style="thin", color="BBBBBB")
    last_data_row = row0
    for i, record in enumerate(df.itertuples(index=False), start=row0 + 1):
        for j, val in enumerate(record, start=1):
            c = ws.cell(row=i, column=j, value=val)
            c.alignment = Alignment(vertical="center", wrap_text=True)
            c.border = Border(left=thin, right=thin, top=thin, bottom=thin)
        last_data_row = i
    _autosize(ws)

    if footer_paragraphs:
        n_cols = len(df.columns)
        foot_row = last_data_row + 2
        for para in footer_paragraphs:
            ws.cell(row=foot_row, column=1, value=para)
            if n_cols > 1:
                ws.merge_cells(
                    start_row=foot_row, start_column=1,
                    end_row=foot_row, end_column=n_cols,
                )
            cell = ws.cell(row=foot_row, column=1)
            cell.alignment = Alignment(
                horizontal="left", vertical="top", wrap_text=True,
            )
            cell.font = Font(italic=True, size=10)
            # Rough row height to fit wrapped text.
            approx_chars_per_line = max(n_cols * 14, 60)
            lines = max(1, len(para) // approx_chars_per_line + 1)
            ws.row_dimensions[foot_row].height = 15 * lines + 6
            foot_row += 1


def _write_notes(ws, lines):
    for i, line in enumerate(lines, start=1):
        c = ws.cell(row=i, column=1, value=line)
        c.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)
        if i == 1:
            c.font = Font(bold=True, size=12)
    ws.column_dimensions["A"].width = 120


def format_full(df: pd.DataFrame) -> pd.DataFrame:
    """Formatted version of the full decomposition table."""
    out = df.copy()
    out["β_W raw [95% CI]"] = [
        f"{b:+.4f} {fmt_ci(lo, hi)}"
        for b, lo, hi in zip(
            out["beta_W"], out["beta_W_ci_lo"], out["beta_W_ci_hi"]
        )
    ]
    out["β_W std"] = out["beta_W_std"].map(lambda x: f"{x:+.3f}" if pd.notna(x) else "—")
    out["β_W p"] = out["beta_W_p"].map(fmt_sci)
    out["β_W q"] = out["beta_W_q"].map(fmt_sci)
    out["β_B raw"] = out["beta_B"].map(lambda x: f"{x:+.4f}" if pd.notna(x) else "—")
    out["β_B std"] = out["beta_B_std"].map(lambda x: f"{x:+.3f}" if pd.notna(x) else "—")
    out["β_B p"] = out["beta_B_p"].map(fmt_sci)
    out["β_W/β_B"] = out["attenuation_ratio"].map(lambda x: f"{x:+.2f}" if pd.notna(x) else "—")
    out["FDR-sig (q≤0.05)"] = out["beta_W_q"].map(lambda x: "✓" if pd.notna(x) and x <= 0.05 else "")

    return out[[
        "domain", "outcome", "group", "focal_var", "n_pairs", "n_MZ", "n_MZ_status",
        "β_W raw [95% CI]", "β_W std", "β_W p", "β_W q",
        "β_B raw", "β_B std", "β_B p", "β_W/β_B",
        "FDR-sig (q≤0.05)",
    ]].rename(columns={
        "domain": "Domain", "outcome": "Outcome", "group": "Group",
        "focal_var": "Variable", "n_pairs": "n_pairs", "n_MZ": "n_MZ",
        "n_MZ_status": "n_MZ_status",
    })


# Table 1 in the paper's column layout. The outcome variable is dropped from the
# table and kept in table1_provenance.csv.
T1_COLUMNS = [
    ("Domain", "Domain"),
    ("CS n_pairs (All / MZ)", "CS N (All / MZ)"),
    ("Change n_pairs (All / MZ)", "Change N (All / MZ)"),
    ("ICC_MZ at W1 (CS)", "ICC at W1 (MZ)"),
    ("ICC_MZ of change", "ICC of change (MZ)"),
    ("CS FDR-surviving β_W (All_paired)", "CS βW (All)"),
    ("Change FDR-surviving β_W (All_paired)", "Change βW (All)"),
    ("Change median MDE (All / MZ)", "Change MDE (All / MZ)"),
]

T1_CAPTION = "Table 1. Within- and between-family decomposition"

# Table 1 note, verbatim from the manuscript.
T1_NOTE = (
    "Note. Between- and within-family decomposition by mixed-effects models in the twin "
    "subsample, one row per aging outcome. Counts are pair counts. Cross-sectional (CS) "
    "models at Wave 1 (W1) control for age, sex, and pair structure, with demographic "
    "variables partitioned into between-family and within-family components. Change "
    "models predict Wave 2 outcomes with the same partition plus the W1 outcome entered "
    "as an autoregressive term. Focal predictors are the stable top-ranked variables "
    "from the corresponding predictive model (top-20 stability ≥ .60), so the number "
    "tested differs by outcome and analysis. All = all pairs (MZ + same-sex DZ + "
    "opposite-sex DZ); MZ = monozygotic pairs. ICC is the outcome intraclass "
    "correlation (σ²_u / (σ²_u + σ²_e)) among MZ pairs. βW columns give the number of "
    "within-family effects surviving FDR correction (Benjamini–Hochberg q ≤ .05) over "
    "the number tested in all pairs, within each analysis × outcome predictor set; "
    "MZ-only results are in SI Tables S4-S10. Change MDE is the median minimum "
    "detectable effect over the predictors tested, in βWstd units (80% power, two-sided "
    "α = .05). N = number of pairs."
)


def _change_mde_cells(overview: pd.DataFrame) -> list[str]:
    """Median change-layer MDE per outcome, all pairs / MZ pairs, as Table 1 prints it."""
    mde = pd.read_csv(MDE_SUMMARY)
    cells = []
    for outcome in overview["Outcome (var)"]:
        val = {s: mde.loc[(mde["sample"] == s) & (mde["outcome_code"] == outcome),
                          "median_MDE_std"].iloc[0] for s in ("All", "MZ")}
        cells.append(f"{val['All']:.2f} / {val['MZ']:.2f}")
    return cells


def build_table1(overview: pd.DataFrame) -> pd.DataFrame:
    """The paper's Table 1: the overview, minus the variable column, renamed."""
    overview = overview.assign(**{"Change median MDE (All / MZ)": _change_mde_cells(overview)})
    missing = [src for src, _ in T1_COLUMNS if src not in overview.columns]
    if missing:
        raise KeyError(f"build_overview no longer emits {missing}; Table 1 cannot be shaped")
    return overview[[src for src, _ in T1_COLUMNS]].rename(
        columns=dict(T1_COLUMNS))


def build_table1_provenance(overview: pd.DataFrame) -> pd.DataFrame:
    """Every Table 1 cell beside the variable it was computed on and its source.

    The displayed ICC is rounded to three decimals; the unrounded value is
    included for checking it.
    """
    icc = pd.read_csv(ICC_SOURCE)
    rows = []
    for _, r in overview.iterrows():
        outcome = r["Outcome (var)"]
        def raw(analysis):
            m = icc[(icc["analysis"] == analysis) & (icc["outcome"] == outcome)
                    & (icc["group"] == "MZ")]
            return float(m["icc"].iloc[0])
        rows.append({
            "Domain": r["Domain"],
            "outcome_variable": outcome,
            "icc_w1_mz_unrounded": raw("cs"),
            "icc_change_mz_unrounded": raw("change"),
            "icc_source": f"output/a6_sibling/{ICC_SOURCE.name}",
            "beta_w_source": "output/a6_sibling/a6_[cs_]{All_paired,MZ_sensitivity}_"
                             f"{outcome}.csv, column beta_W_q",
            "fdr_rule": "Benjamini-Hochberg q <= 0.05 within each "
                        "(analysis x group x outcome) predictor set",
            "mde_source": "output/within_family_mde/within_family_mde_summary.csv, "
                          "column median_MDE_std",
        })
    return pd.DataFrame(rows)


def write_table1(overview: pd.DataFrame) -> None:
    T1_OUT.mkdir(parents=True, exist_ok=True)
    t1 = build_table1(overview)
    t1.to_csv(T1_OUT / "table1.csv", index=False)
    build_table1_provenance(overview).to_csv(
        T1_OUT / "table1_provenance.csv", index=False)

    md = [f"# {T1_CAPTION}", ""]
    md.append("| " + " | ".join(t1.columns) + " |")
    md.append("| " + " | ".join("---" for _ in t1.columns) + " |")
    for _, r in t1.iterrows():
        md.append("| " + " | ".join(str(v) for v in r) + " |")
    md += ["", T1_NOTE, "",
           "## Provenance", "",
           "Produced by `analysis/tables/build_table1.py`, whose overview sheet is this "
           "table. Numbers come from `output/a6_sibling/`; see `table1_provenance.csv` "
           "for the per-row sources and the unrounded ICCs. The MDE column comes from "
           "`output/within_family_mde/within_family_mde_summary.csv`.", ""]
    (T1_OUT / "table1.md").write_text("\n".join(md) + "\n")
    print(f"Wrote {T1_OUT / 'table1.csv'}")
    for n in ("table1_provenance.csv", "table1.md"):
        print(f"  + {n}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    full = build_full_results()
    overview = build_overview(full)
    icc_tbl = build_icc_table()
    fdr_tbl = build_fdr_summary(full)
    denom_tbl = build_denominator_table(full)

    cs_full = format_full(full[full["analysis"] == "cs"])
    ch_full = format_full(full[full["analysis"] == "change"])

    # CSV mirrors
    overview.to_csv(OUT / "overview.csv", index=False)
    icc_tbl.to_csv(OUT / "icc.csv", index=False)
    fdr_tbl.to_csv(OUT / "fdr_surviving.csv", index=False)
    denom_tbl.to_csv(OUT / "denominators.csv", index=False)
    cs_full.to_csv(OUT / "cs_full.csv", index=False)
    ch_full.to_csv(OUT / "change_full.csv", index=False)

    # XLSX
    wb = xl.Workbook()
    wb.remove(wb.active)

    ws = wb.create_sheet("Overview")
    _write_sheet(
        ws, overview,
        title="Within/between-family decomposition — domain × analysis overview",
        footer_paragraphs=SHEET_NOTES["Overview"],
    )
    ws = wb.create_sheet("ICC by zygosity")
    _write_sheet(
        ws, icc_tbl,
        title="Outcome null-model ICC at W1 vs. of change (W1→W2)",
        footer_paragraphs=SHEET_NOTES["ICC by zygosity"],
    )
    ws = wb.create_sheet("FDR-surviving β_W")
    _write_sheet(
        ws, fdr_tbl,
        title="Within-family predictors with β_W q ≤ 0.05 (BH-FDR)",
        footer_paragraphs=SHEET_NOTES["FDR-surviving β_W"],
    )
    ws = wb.create_sheet("CS full")
    _write_sheet(
        ws, cs_full,
        title="Cross-sectional (W1) decomposition — all predictors",
        footer_paragraphs=SHEET_NOTES["CS full"],
    )
    ws = wb.create_sheet("Change full")
    _write_sheet(
        ws, ch_full,
        title="Change (W1 → W2) decomposition — all predictors",
        footer_paragraphs=SHEET_NOTES["Change full"],
    )
    ws = wb.create_sheet("Notes")
    _write_notes(ws, NOTES_LINES)

    path = OUT / "family_decomposition.xlsx"
    wb.save(path)
    print(f"Wrote {path} ({path.stat().st_size/1024:.1f} KB)")
    for p in OUT.glob("*.csv"):
        print(f"  + {p.name}")

    write_table1(overview)


if __name__ == "__main__":
    main()
