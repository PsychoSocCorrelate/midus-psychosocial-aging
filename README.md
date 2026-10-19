# Psychosocial and behavioral correlates of frailty, cognition, and allostatic load

Analysis code for "Psychosocial and behavioral correlates of frailty, cognition, and allostatic
load across 9 years and within twin pairs" (MIDUS 2 and MIDUS Refresher, two waves). It runs
the MIDUS source files to the result tables, figures and supplementary tables of the paper.

The repository contains the code, configuration, the shareable MIDUS key maps, and the aggregate
result tables (`output/**/*.csv`). It contains no individual-level data. The survey, cognition,
biomarker and mortality files are public MIDUS downloads; the twin-pair identifiers and zygosity
are not publicly distributed and are requested through the IGEMS consortium. Both routes are in
`data/raw/data_access.md`; place the files under `data/raw/` as listed in `data/raw/README.md`.
The aggregate results can also be explored at https://psychosoccorrelate.github.io/.

## Analytical workflow

1. **Harmonize the inputs** (`preprocess/`). Survey items from the four cohort-waves are mapped
   to common variables, scale composites are scored after testing measurement invariance across
   cohorts, and the outcomes are built: a frailty index excluding mental-health items, the BTACT
   cognitive composite, a 16-marker allostatic-load composite and DNA methylation clocks. Twin
   pairs come from the IGEMS file.
2. **Predict** (`analysis/pipelines/`). XGBoost models test how much a block of 161 psychosocial
   and behavioral predictors (Tier 2) adds beyond 16 background variables (Tier 1), as the gain in
   cross-validated R², at Wave 1 and for 9-year change conditional on the Wave 1 outcome.
   Bootstrap SHAP ranks the predictors; TabPFN is a cross-check of the cross-sectional models.
3. **Compare domains**. The predictor rankings are compared across outcomes, and each outcome's
   top predictors are tested on the others.
4. **Decompose within families**. Top predictors are refit in twin pairs with between- and
   within-pair terms, with intraclass correlations by zygosity and minimum detectable effects.
5. **Check robustness** (`analysis/sensitivity/`). Repeated cross-validation intervals, frailty
   models without content-overlapping predictors, allostatic-load models with twin pairs held
   out, and a mortality validation of the predictor blocks.

The tables and figures are also part of the replication steps; details are in
`scripts/orchestration/README.md`.

## Requirements

- Python 3.11 and [uv](https://docs.astral.sh/uv/). Dependencies are pinned in `requirements.txt`.
- An NVIDIA GPU with CUDA. XGBoost models, frailty-item imputation and TabPFN run on the GPU.
- R 4.5 for the CFA composites; package versions are in `renv.lock`.

```bash
uv venv --python 3.11
uv pip install -r requirements.txt
Rscript -e 'install.packages("renv"); renv::restore(prompt = FALSE)'
```

## Reproducing the paper

Every command runs from the repository root. Each wrapper runs one stage of the steps declared in
`scripts/orchestration/registry.json`; without `--execute` it only prints the steps it would run.

```bash
uv run python scripts/orchestration/rebuild_inputs.py --execute                  # stage 1, to data/processed/
uv run python scripts/orchestration/run_analyses.py --execute --include-heavy --step profile_models
uv run python scripts/orchestration/run_analyses.py --execute --include-heavy    # stages 2-5, to output/
uv run python scripts/orchestration/run_analyses.py --execute --include-heavy --step a12_delta_r2_ci
uv run python scripts/orchestration/render_figures.py --execute --include-heavy  # figures
uv run python scripts/orchestration/build_tables.py --execute                    # Table 1, SI Tables S1-S18
```

`--include-heavy` admits the model-fitting steps; without it they are skipped. Two steps run
only when named: `profile_models`, which fits the models the SI profile comparison reads, and
`a12_delta_r2_ci`, the repeated cross-validation intervals. On two RTX 5090 GPUs the heavy steps
take about 14 hours. With one GPU, run `a3_domain_divergence` with `--n-gpus 1`.

## Checks without MIDUS data

```bash
uv run python scripts/orchestration/validate_all.py --execute
uv run --with pytest pytest -q
```

## Layout

```text
preprocess/            raw MIDUS and IGEMS files to analysis-ready and master datasets
analysis/core/         data loader, XGBoost and SHAP pipeline, provenance, plotting style
analysis/pipelines/    main analyses, one script per step
analysis/sensitivity/  sensitivity analyses
analysis/profile_comparison/  exploratory profile comparison (SI)
analysis/figures/      figure builders
analysis/tables/       Table 1 and SI table builders
analysis/tests/        unit tests
config/                predictor tiers and subdomains, labels, biomarker definitions
scripts/orchestration/ step registry, wrappers and the step reference
scripts/validation/    lint checks
data/key_map/          MIDUS variable and frailty scoring keys
data/raw/              raw-data contract and access guide; source files go here
output/                aggregate result tables, by step
```

## License

The code is under the MIT License (`LICENSE`): everything in `analysis/`, `preprocess/`,
`config/` and `scripts/`, the key maps in `data/key_map/`, and `requirements.txt` and `renv.lock`.
The aggregate result tables in `output/`, the documentation (this file and the other `README.md`
files, and `data/raw/data_access.md`) and the content of the companion site are under the
Creative Commons Attribution 4.0 International License (`LICENSE-CC-BY-4.0`). To reuse them,
cite the paper and this repository.

Neither license extends to third-party material. The MIDUS and IGEMS data are not distributed
here and stay under their providers' terms; the results derive from MIDUS data, so work reusing
them should also acknowledge MIDUS (https://midus.wisc.edu/citation-acknowledgement/). The
MIDUS variable names and labels in the key maps come from the MIDUS documentation, and the
dependencies in `requirements.txt` and `renv.lock` carry their own licenses.
