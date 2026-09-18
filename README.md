# The Biomarker Graveyard in Lung Adenocarcinoma

Analysis code and result files for the manuscript

> **The Biomarker Graveyard in Lung Adenocarcinoma: A Reproducible Directional Survival Signal Across Classical and Foundation-Model Machine Learning**
> Jesús Gil Ruiz, José Rivera Torres, Diego Rubén Rodríguez Regadera.

We screened 11,484 genes for association with overall survival in lung adenocarcinoma across three
independent public cohorts, calibrating the null by permutation instead of assuming it, and asked
whether a reproducible collective signal survives where per-gene false-discovery-rate correction
finds nothing.

**Headline results reproduced by this code**

- Of the 949 nominal hits of the discovery cohort, **0** survive the Benjamini–Hochberg correction.
- A fixed **121-gene panel** keeps the same direction and nominal significance in all three cohorts
  at once: 89–188× over a label-permutation null and 2.7–3.0× over a stricter gene-resampling null,
  and it survives stage and age adjustment.
- Twelve survival models covering eleven families, trained on the panel in one cohort and evaluated
  on the others, beat random size-matched gene panels in 9 of 12 configurations (C-index up to 0.77).
- Three pretrained foundation models (an in-context survival network, a bulk-RNA-seq embedding and a
  tabular model via pseudo-observations) do **not** improve on the classical ensembles; the embedding
  degrades performance below a random-noise control. This negative result is reported as it came out.
- Of 1,080 genes with a direction already specified in the Human Protein Atlas, 91 show the same
  cross-cohort concordance, a 1.5-fold excess over chance, but only 2 pass the per-gene FDR.

## Data

No data are redistributed here. All three cohorts and the external biomarker corpus are public and
are downloaded by the fetch scripts:

| Role | Source | n | Events |
|---|---|---|---|
| Discovery | CPTAC-LUAD (`cptac` Python package) | 106 | 24 |
| Replication 1 | TCGA-LUAD via the GDC REST API (open access, no token) | 508 | 183 |
| Replication 2 | GSE68465 (Shedden et al.) via NCBI GEO | 442 | 236 |
| External validation | Human Protein Atlas — Pathology Atlas | 1,080 genes | — |

`data/ACCESS.md` documents each access route as it was actually verified, and `data/DATASHEET.md`
describes the harmonised cohorts. Both are in Spanish, the working language of the project.

## Layout

```
src/            analysis pipeline, one script per phase, plus cohort loaders and figure generation
nullcal_pkg/    null-calibration engine used by the discovery screen
results/        every JSON/CSV this pipeline produces; the manuscript's numbers come from here
logs/           execution logs, including the end-to-end reproducibility audit
tests/          unit tests
env/            conda environment and pinned requirements
```

## Reproducing

```bash
conda env create -f env/environment.yml && conda activate biomarker
python src/fetch_cptac_luad.py && python src/fetch_tcga_gdc.py && python src/fetch_gse68465.py
python src/phase2_discovery.py          # naive screen vs permutation-calibrated null
python src/phase3_replication.py        # cross-cohort directional replication
python src/phase4_hpa_validation.py     # external validation against the HPA corpus
python src/phase8_ml_survival_panel.py  # classical machine-learning validation
python src/audit_reproducibility.py     # regenerates the central counts from the raw loaders
```

`src/audit_reproducibility.py` re-derives the manuscript's central counts (the 121-gene panel and the
91/1,080 HPA validation count) directly from the raw cohort pipelines; its output is in
`logs/audit_reproducibility_20260814_final.log`.

## Citing

Please cite the manuscript. This repository is archived on Zenodo; the DOI of the release matching
the submitted version is given in the manuscript's Data Availability statement.

## License

MIT — see `LICENSE`.
