# Data access

This repository does not redistribute individual-level records. Reproducing the analyses requires
the survey, cognition, biomarker, mortality and twin-pair files listed in `README.md` in this
directory, each obtained from its provider and used under that provider's terms.

## MIDUS files

The survey, cognition, biomarker, DNA methylation age and mortality files are public MIDUS data.
The analyzed versions were downloaded from the MIDUS Portal; `README.md` gives each file's name,
release date and download link. The same studies are archived at ICPSR, though for some files
the Portal release is newer than any ICPSR version.

- MIDUS data access: https://midus.wisc.edu/data-access/
- ICPSR/NACDA MIDUS series: https://www.icpsr.umich.edu/sites/nacda/midus

The MIDUS terms of use do not permit redistribution of these files.

## Twin-pair identifiers

Twin-pair identifiers and zygosity (`admin_midus_20nov2024.sav`) are not publicly distributed.
They were obtained through the Interplay of Genes and Environment across Multiple Studies (IGEMS)
consortium and can be requested from IGEMS (IGEMS-Admin@usc.edu).
Without this file the master dataset builders still run, but family structure is empty: the
within-family analyses cannot be reproduced, and the frailty and cognition samples keep the
complete twin pairs that the paper holds out, so their estimates will differ from those reported.

## Where files go

Place the files under `data/raw/` with the names in `README.md`. Everything the pipeline derives
from them is written to `data/processed/` and to the individual-level SHAP matrices under
`output/**/shap_raw/`; both are excluded from version control and must not be shared.
