# Stratified reproducibility analysis of HPA prognostic calls (LUAD)

Scripts and outputs for the analyses in the manuscript "External reproducibility of Human Protein Atlas prognostic calls in lung adenocarcinoma is graded by coupling to a proliferation-associated expression axis". Files are in `analisis/` (file names are in Spanish; step numbers follow the manuscript order). `PROTOCOLO_CONGELADO.md` is the written protocol (in Spanish), produced before the stratified analysis and retrospective, not a preregistration.

| Step | Script | Content |
|---|---|---|
| 1 | `paso1_identidad.py`, `paso1b_cptac_v2.py` | Ensembl-level identity audit; corrected CPTAC-LUAD matrix (row with highest mean expression per duplicated symbol) |
| 3 | `paso3_estratificacion.py`, `paso3c_cruce_validated.py` | Reproduction rate by quintile of mean absolute Spearman correlation with the 13-gene metagene; label-permutation null; Cochran-Armitage test |
| 4 | `paso4b_magnitudes.py` | Cox models M0 (gene+age+stage) versus M1 (+metagene); patient bootstrap (B=200, shared indices) |
| 5 | `paso5_conjuncion.py` | Partial-conjunction test (BH and BY) |
| 6 | `paso6_controles.py` | Alveolar and lymphocyte modules; terciles; coupling computed in TCGA only |
| 7 | `paso7_ppi_228.py` | STRING edges among the 228-gene panel |
| 8 | `paso8_dependencia_centro.py`, `paso8b_submuestreo.py`, `paso4c_magnitudes.py` | Gene-dependence-aware permutation reference for the trend statistic; analyses stratified by recruiting centre in GSE68465 (log-rank, permutation, Cox); SFTPB sensitivity |
| - | `fig_central.py`, `fig0_diseno.py` | Figures 0 and 2 |

Main results (corrected CPTAC-LUAD matrix, evaluation in CPTAC-LUAD n=106 and GSE68465 n=442): 1,067 evaluable calls, 92 nominally significant with the HPA direction in both cohorts; reproduction by coupling quintile 1/214, 10/213, 10/213, 24/213, 47/214 (Cochran-Armitage z=8.35, p=6.8e-17, treating genes as independent); 0/1,067 pass the partial-conjunction test (best q=0.057); median attenuation of the log hazard ratio by the metagene 21% (95% bootstrap CI -13 to 46%). Machine-readable outputs: `paso3_quintiles.csv`, `paso4b_resultado.json`, `paso5_conjuncion.csv`, `paso6_controles.json`. `results_v2/` holds the regenerated main pipeline.


## Current outputs and caveats

- Current (v1.1.0): `paso3_quintiles.csv`, `paso3_corpus_genes.csv`, `paso4b_resultado.json` (unstratified attenuation, replicate values not stored), `paso4c_centro_resultado.json` (centre-stratified attenuation, replicates stored), `paso5_conjuncion.csv`, `paso6_controles.json`, `paso8_resultado.json`. Earlier magnitude outputs (`paso4_*`, 28% estimate) were removed because that estimator was superseded.
- `paso3_corpus_genes.csv` keeps historical columns from the v1 identity map (e.g. `ens_coincide`); the identity check actually applied in v2 is `ens_ok`.
- `paso5_conjuncion.py` writes its output relative to the working directory; run it from `analisis/`.
- Patient bootstrap with re-testing of every gene (tried in `paso8`) is not valid for intervals on the reproduction rates because duplicated patients inflate significance; only the permutation references are used.
- Centre-stratified results: with GSE68465 stratified by site, 82 genes reproduce (2/12/7/19/42 per quintile, z=7.32) and the lowest quintile is above the within-site permutation null (empirical p=0.013).
