# Cross-cohort reproducibility of prognostic gene-expression associations in lung adenocarcinoma

Analysis code and result files for the manuscript

> **Cross-cohort reproducibility of prognostic gene-expression associations in lung adenocarcinoma**
> Jesús Gil Ruiz, José Rivera Torres, Diego Rubén Rodríguez Regadera.

We screened 11,484 genes for association with overall survival in lung adenocarcinoma across three
public cohorts, calibrating the null by permutation instead of assuming it, and asked whether a
reproducible collective signal survives where per-gene false-discovery-rate correction finds
nothing. The three cohorts are separately assembled, but all three contribute to the selection of
the gene set below, so no analysis here constitutes independent external validation.

**Headline results reproduced by this code**

- Of the 949 nominal hits of the discovery cohort in the median-split log-rank screen, **0**
  survive the Benjamini–Hochberg correction. In the continuous-covariate sensitivity, 3 of 1,062
  survive BH and none survive BY (see the v1.0.2 note below).
- A **121-gene set**, selected jointly from all three cohorts, keeps the same direction and nominal
  significance in all three at once: 89–188× over a label-permutation null and 2.7–3.0× over a
  stricter gene-resampling null. After adjustment for stage and age, 120 of 121 retain their
  direction and 58 remain nominally significant in all three cohorts; the enrichment against the
  gene-resampling null was not recomputed on the adjusted results.
- Twelve survival models covering eleven families, trained on the gene set in one cohort and
  evaluated on the others, beat random size-matched gene panels in 9 of 12 configurations (C-index
  up to 0.77). These are exploratory: the evaluated cohorts contributed their outcomes to the gene
  selection, so this is not an unbiased estimate of external predictive performance, and the random
  panels did not undergo that selection.
- Three pretrained foundation models (an in-context survival network, a bulk-RNA-seq embedding and a
  tabular model via pseudo-observations) do **not** improve on the classical survival models; the embedding
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
| Published-candidate audit | Human Protein Atlas — Pathology Atlas | 1,080 genes | — |

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
pip install -e nullcal_pkg            # the null-calibration engine, installed from this repository
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

**Notes on the two foundation-model scripts.** `src/phase9_survivalpfn.py` needs a checkout of
SurvivalPFN and its own conda environment (torch 2.5.1); point `SURVIVALPFN_DIR` at the checkout.
`src/phase10_bulkrnabert_embeddings.py` needs two derived inputs that are not archived here: the
Ensembl gene list shipped with the BulkRNABert checkpoint (`BULKRNABERT_ENSG_ORDER`) and an
Ensembl-to-HGNC map for that list obtained from mygene.info (`ENSG_TO_SYMBOL_MAP`). The embeddings
and every number that script produced are archived in `results/phase10_bulkrnabert_embeddings.*`.

## Citing

Please cite the manuscript. This repository is archived on Zenodo; the DOI of the release matching
the submitted version is given in the manuscript's Data Availability statement.

## License

MIT — see `LICENSE`.

## v1.0.2 — corrected FDR in the continuous-covariate sensitivity

In versions up to v1.0.1, `results/discovery_cox_naive_vs_calibrated.csv` carried non-finite
q-values for all 11,484 genes. Twenty-two genes with constant expression produced NaN statistics
when standardised, and the cumulative-minimum step of the BH/BY procedures propagated those NaNs
to every gene. Comparing NaN with alpha returns `False`, so the saved survivor count was 0 as an
artefact rather than as a result.

`src/phase2_cox_sensitivity.py` now assigns constant genes a zero statistic and p=1, keeping the
gene universe unchanged, and refuses to run the FDR correction on a non-finite p-value vector.
Recomputed on the same permutation p-values, **three genes survive BH at alpha=0.05 (GOLT1B,
ARNTL2, GAPDH) and none survive BY**; the nominal-hit count (1,062) and the universe (11,484) are
unchanged. The median-split log-rank screen, which carries the manuscript's headline result, was
never affected.

The pre-repair file is kept as `results/discovery_cox_naive_vs_calibrated_PRE_NANFIX.csv.bak` for
reference, and `src/audit_astra_r2.py` audits against it.
