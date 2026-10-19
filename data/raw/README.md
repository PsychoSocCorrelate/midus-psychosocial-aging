# data/raw/

Source files go here, under the names and subdirectories below. Access routes are in
`data_access.md`. The key maps the preprocessing also reads are tracked in `data/key_map/`.

## wave1/

| File | Description | Used by | Source |
|------|-------------|---------|--------|
| `M2_P1_SURVEY_N4963_20200720.sav` | MIDUS 2 main survey | `preprocess/survey/qc_pipeline.py` (cohort `midus2`) | [download](https://midus-study.github.io/public-documentation/M2P1/M2P1_Survey/Data/M2_P1_SURVEY_N4963_20200720.sav) |
| `MR1_P1_SURVEY_N3577_20250407.sav` | MIDUS Refresher 1 main survey | `preprocess/survey/qc_pipeline.py` (cohort `midusr`) | [download](https://midus-study.github.io/public-documentation/MR1P1/Data/MR1_P1_SURVEY_N3577_20250407.sav) |
| `M2_P3_BTACT_N4512_20211123.sav` | MIDUS 2 cognitive battery (BTACT) | `preprocess/cognition/build_cog_longitudinal.py` | [download](https://midus-study.github.io/public-documentation/M2P3/Data/M2_P3_BTACT_N4512_20211123.sav) |
| `MR1_P3_BTACT_N2763_20251119.SAV` | MIDUS Refresher 1 cognitive battery | `preprocess/cognition/build_cog_longitudinal.py` | [download](https://midus-study.github.io/public-documentation/MR1P3/Data/MR1_P3_BTACT_N2763_20251119.SAV) |

## wave2/

| File | Description | Used by | Source |
|------|-------------|---------|--------|
| `M3_P1_SURVEY_N3294_20251029.sav` | MIDUS 3 main survey | `preprocess/survey/qc_pipeline.py` (cohort `midus3`); `preprocess/master/build_master_dataset.py` age recovery | [download](https://midus-study.github.io/public-documentation/M3P1/Data/M3_P1_SURVEY_N3294_20251029.sav) |
| `MR2_P1_SURVEY_N2154_20251003.sav` | MIDUS Refresher 2 main survey | `preprocess/survey/qc_pipeline.py` (cohort `mr2`); `preprocess/master/build_master_dataset.py` age recovery | [download](https://midus-study.github.io/public-documentation/MR2P1/Data/MR2_P1_SURVEY_N2154_20251003.sav) |
| `M3_P3_BTACT_N3291_20210922.sav` | MIDUS 3 cognitive battery | `preprocess/cognition/build_cog_longitudinal.py` | [download](https://midus-study.github.io/public-documentation/M3P3/Data/M3_P3_BTACT_N3291_20210922.sav) |
| `MR2_P3_BTACT+MOCA_N1934_20260224.sav` | MIDUS Refresher 2 BTACT/MOCA battery | `preprocess/cognition/build_cog_longitudinal.py` | [download](https://midus-study.github.io/public-documentation/MR2P3/Data/MR2_P3_BTACT+MOCA_N1934_20260224.sav) |

## biomarker/

| File | Description | Used by | Source |
|------|-------------|---------|--------|
| `M2_BIO_AGGREGATE_N1255_20240405.sav` | MIDUS 2 biomarker panel | `preprocess/biomarkers/allostatic_load.py` | [download](https://midus-study.github.io/public-documentation/M2P4/Data/M2_BIO_AGGREGATE_N1255_20240405.sav) |
| `M2_BIO_MEDICATION_STACKED_N7174_20240404.SAV` | MIDUS 2 medication stack | `preprocess/biomarkers/allostatic_load.py` | [download](https://midus-study.github.io/public-documentation/M2P4/Stacked%20Medication/M2_BIO_MEDICATION_STACKED_N7174_20240404.SAV) |
| `M3_BIO_AGGREGATE_N747_20231016.sav` | MIDUS 3 biomarker panel | `preprocess/biomarkers/allostatic_load.py` | [download](https://midus-study.github.io/public-documentation/M3P4/Data/M3_BIO_AGGREGATE_N747_20231016.sav) |
| `M3_BIO_MEDICATION_STACKED_N5751_20231016.SAV` | MIDUS 3 medication stack | `preprocess/biomarkers/allostatic_load.py` | [download](https://midus-study.github.io/public-documentation/M3P4/Stacked%20Medication/M3_BIO_MEDICATION_STACKED_N5751_20231016.SAV) |
| `MR1_BIO_AGGREGATE_N863_20240405.sav` | MIDUS Refresher 1 biomarker panel | `preprocess/biomarkers/allostatic_load.py` | [download](https://midus-study.github.io/public-documentation/MR1P4/Data/MR1_BIO_AGGREGATE_N863_20240405.sav) |
| `MR1_BIO_MEDICATION_STACKED_N5118_20240404.sav` | MIDUS Refresher 1 medication stack | `preprocess/biomarkers/allostatic_load.py` | [download](https://midus-study.github.io/public-documentation/MR1P4/Stacked%20Medication/MR1_BIO_MEDICATION_STACKED_N5118_20240404.sav) |
| `M2MR1_GEN_DNAmAge_N2118_20230822.sav` | MIDUS 2 and Refresher 1 DNAm age clocks | `preprocess/biomarkers/dnam_age.py` | [download](https://midus-study.github.io/public-documentation/Genetics/DNA/M2MR1_Methylation/M2MR1_GEN_DNAmAge_N2118_20230822.sav) |

## mortality/

| File | Description | Used by | Source |
|------|-------------|---------|--------|
| `MIDUS_Core_MortalityCauseData_N2822_20260511.sav` | Mortality cause data for the MIDUS core sample | `analysis/core/mortality_data.py` | [download](https://midus-study.github.io/public-documentation/Mortality/Core/MIDUS_Core_MortalityCauseData_N2822_20260511.sav) |
| `MIDUS_Refresher_MortalityCauseData_N454_20260511.sav` | Mortality cause data for the MIDUS Refresher sample | `analysis/core/mortality_data.py` | [download](https://midus-study.github.io/public-documentation/Mortality/Refresher/MIDUS_Refresher_MortalityCauseData_N454_20260511.sav) |

## root of data/raw/

| File | Description | Used by | Source |
|------|-------------|---------|--------|
| `admin_midus_20nov2024.sav` | Twin-pair identifiers and zygosity for MIDUS twins | `preprocess/master/build_master_dataset.py`, `preprocess/master/build_master_concurrent.py` | IGEMS consortium, on request; not publicly distributed; see `data_access.md` |
