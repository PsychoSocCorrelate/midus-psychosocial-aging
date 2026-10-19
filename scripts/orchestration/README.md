# Pipeline steps

`registry.json` declares every step: its script, inputs, outputs and group. The five wrappers
here each run one group of it, and `runner.py` holds what they share. Commands run from the
repository root; the order to run them in is in the top-level `README.md`.

| Wrapper | Group | Writes |
| --- | --- | --- |
| `rebuild_inputs.py` | survey QC, CFA composites, outcomes, master datasets | `data/processed/` |
| `run_analyses.py` | analyses | `output/` |
| `render_figures.py` | figures | `manuscript/figures/`, `manuscript/supplement/` |
| `build_tables.py` | tables | `manuscript/tables/`, `manuscript/supplement/` |
| `validate_all.py` | lint and compile checks | nothing |

Without `--execute` a wrapper only lists its steps. Heavy steps, which fit models, also need
`--include-heavy`, or they are skipped. `--step ID` runs one step, including the two that are
off by default (`profile_models`, `a12_delta_r2_ci`).

## Analyses

| Step | Result | Script | Outputs |
| --- | --- | --- | --- |
| `d1_descriptive` | Descriptives, attrition, reliability, change scores | `analysis/pipelines/d1_descriptive.py` | `output/d1_descriptive/` |
| `outcome_correlations_by_wave` | Outcome correlations at each wave | `analysis/pipelines/outcome_correlations_by_wave.py` | `output/outcome_correlations_by_wave/` |
| `cs1_cross_sectional` (heavy) | Wave 1 cross-sectional models, SHAP, TabPFN benchmark | `analysis/pipelines/cs1_cross_sectional.py` | `output/cs1_cross_sectional/` |
| `a4_longitudinal` (heavy) | Baseline-conditioned 9-year change models | `analysis/pipelines/a4_longitudinal.py` | `output/a4_longitudinal/` |
| `a4b_concurrent_sensitivity` (heavy) | Change models with Wave 2 predictors | `analysis/pipelines/a4b_longitudinal.py` | `output/a4b_longitudinal/` |
| `a9_concurrent` (heavy) | Wave 2 cross-sectional models | `analysis/pipelines/a9_concurrent.py` | `output/a9_concurrent/` |
| `a12_delta_r2_ci` (heavy, off by default) | Repeated cross-validation intervals for the Tier 2 increments | `analysis/pipelines/a12_delta_r2_ci.py` | `output/a12_delta_r2_ci/` |
| `a3_domain_divergence` (heavy) | Cross-domain agreement and transfer of predictors | `analysis/pipelines/a3_domain_divergence.py` | `output/a3_domain_divergence/` |
| `recurring_predictors` | Predictors in the top 20 of at least four of six models | `analysis/pipelines/recurring_predictors.py` | `output/recurring_predictors/` |
| `a6_family_decomposition` (heavy) | Between- and within-family decomposition in twin pairs | `analysis/pipelines/a6_sibling_decomposition.py` | `output/a6_sibling/` |
| `within_family_mde` | Within-family minimum detectable effects | `analysis/pipelines/within_family_mde.py` | `output/within_family_mde/` |
| `a7_mortality_primary`, `a7_mortality_fixed9` | Mortality validation, through 2025 and fixed 9 years | `analysis/pipelines/a7_mortality.py` | `output/a7_mortality/` |
| `a11_mortality_shap` (heavy) | Mortality model SHAP | `analysis/pipelines/a11_mortality_shap.py` | `output/a11_mortality_shap/` |
| `a2_frailty_overlap` (heavy) | Frailty models without content-overlapping predictors | `analysis/sensitivity/a2_frailty_overlap.py` | `output/a2_frailty_overlap/` |
| `frailty_lonely_sensitivity` | Frailty index without the loneliness item | `analysis/sensitivity/frailty_lonely_sensitivity.py` | `output/frailty_lonely_sensitivity/` |
| `al_twins_heldout` (heavy) | Allostatic-load models with twin pairs held out | `analysis/sensitivity/al_twins_heldout.py` | `output/al_twins_heldout/` |
| `a2c_al_drink_quick` | Allostatic load and drinking status | `analysis/sensitivity/a2c_AL_drinking_sensitivity.py` | `output/a2c_AL_drinking_sensitivity/` |
| `profile_models` (heavy, off by default) | Models for the profile comparison | `analysis/profile_comparison/fit_models.py` | `output/profile_comparison/models/` |
| `profile_specification` | Profile presets | `analysis/profile_comparison/build_profiles.py` | `config/profile_specification.yaml` |
| `profile_comparison` (heavy) | Exploratory profile comparison | `analysis/profile_comparison/profile_comparison.py` | `output/profile_comparison/` |

## Figures and tables of the paper

| Paper item | Step | Script | File |
| --- | --- | --- | --- |
| Fig. 1 | `fig1_subpanels`, then `fig1` | `analysis/figures/fig1_subpanels.py`, `fig1_deck.py` | `manuscript/figures/fig1/` |
| Fig. 2A | `fig2a` | `analysis/figures/fig2a_increments.py` | `manuscript/figures/fig2/fig2A.png` |
| Fig. 2B | `fig2b` | `analysis/figures/fig2b_rank_map.py` | `manuscript/figures/fig2/fig2B.png` |
| Fig. 3A | `fig3a` | `analysis/figures/fig3a_shap_dependence.py` | `manuscript/figures/fig3/fig3A.png` |
| Fig. 3B | `fig3b` | `analysis/figures/fig3b_rank_heatmap.py` | `manuscript/figures/fig3/fig3B.png` |
| Table 1 | `table1` | `analysis/tables/build_table1.py` | `manuscript/tables/table1/` |
| SI Tables S1-S18 | `supplement_tables` | `analysis/tables/build_supplemental_tables.py` | `manuscript/supplement/table_s*.xlsx` |
| SI Figs. S1-S6 | `si_dependence_figures` (heavy), then `publish_si_figures` | `analysis/pipelines/cs1_cross_sectional.py`, `analysis/figures/publish_si_figures.py` | `manuscript/supplement/fig_s*.png` |
| SI Fig. S7 | `fig_s7_profile_comparison` | `analysis/figures/figs7_profile_comparison.py` | `manuscript/supplement/fig_s7_profile_comparison.png` |

`table1` also writes the full within- and between-family tables to
`manuscript/tables/family_decomposition/`, which SI Tables S4-S10 are built from.
`supplement_tables_docx` sets SI Tables S1-S18 as one Word document.
