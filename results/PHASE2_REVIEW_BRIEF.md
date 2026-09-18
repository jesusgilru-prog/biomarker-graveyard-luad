# The Biomarker Graveyard — Phase 2 result, for review

## Study
Auditing how many published-style cancer prognostic biomarkers survive a rigorous
null-signal control. Cancer type: lung adenocarcinoma (LUAD), picked because the
originally-planned breast cohort had only 2/105 death events (1.9%) — too few to
screen anything.

## Cohorts
- **Discovery**: CPTAC-LUAD (prospective CPTAC3, RNA-seq FPKM). n=106, 24 death events
  (22.6%). Patient IDs are CPTAC3-native (`C3L-`/`C3N-`), verified NOT to overlap TCGA.
- **Replication 1**: TCGA-LUAD via GDC REST API (open access, no token/DUA), random
  sample of 300 primary-tumor cases (seed=42) → 294 with usable survival, 101 events
  (34.4%). RNA-seq TPM (STAR-Counts), independent processing pipeline from CPTAC.
- **Replication 2**: GSE68465 (Shedden et al. 2008, GEO), Affymetrix HG-U133A
  microarray — different assay technology entirely. n=442, 236 events (53.4%).
- Candidate panel: 11,484 genes present in all three cohorts simultaneously (by HGNC
  symbol).

## Method
- Statistic: two-group log-rank chi-square (median-split dichotomization of
  expression), closed-form Mantel-Haenszel formula. **Validated exactly against
  `lifelines.statistics.logrank_test` on 8 real genes before use** (bit-identical
  chi2 values).
- Naive p-value: asymptotic chi2(1) survival function on the observed statistic —
  what an uncorrected single-gene biomarker paper would report.
- Calibration: permutation-based null (shuffle patient outcome labels, preserving
  time+event pairing), empirical p-value, BH-FDR and BY-FDR correction at alpha=0.05.
  Implemented via a small pre-built/pre-tested library (`nullcal`, MIT, ~150 lines,
  numpy/scipy only, 7/7 unit tests pass) — not reimplemented for this study.
- **Adaptive 3-stage permutation design** (needed because a flat 1000-permutation
  run over all 11,484 genes gave a p-value floor of 1/1001≈0.001, mathematically too
  coarse for BH to ever let anything survive at N=11,484 tests):
  - Stage A: analytic chi2(1) p-value for all 11,484 genes (no permutation needed).
    → 949 genes pass naive p<0.05 (8.3%). The other 10,535 have naive p>=0.05 and are
    mathematically guaranteed to never survive BH/BY at alpha=0.05 regardless of
    permutation precision (BH's threshold at the maximum possible rank N equals
    alpha itself, and only 8.3% of tests are naively significant here).
  - Stage B: 20,000 permutations on those 949 genes only.
  - Stage C: 1,000,000 permutations on the 20 genes closest to the BH decision
    boundary (floor ~1e-6), to resolve the boundary case definitively rather than
    leave it floor-limited.

## Result
| Rigor level | N genes (of 11,484 candidates) |
|---|---|
| Naive (p<0.05, uncorrected) | 949 (8.3%) |
| Survive BH-FDR (permutation-calibrated, alpha=0.05) | **0 (0.0%)** |
| Survive BY-FDR (more conservative, for correlated features) | **0 (0.0%)** |

Best gene after 1,000,000 permutations: CHCHD3, empirical p=1.2e-5, q_BH=0.080 (still
above 0.05).

**Supplementary, cheaper check** (no permutation, just naive p<0.05 in each cohort
separately): of the 949 naive discovery hits, 23.5% are also naively significant in
TCGA-LUAD, 26.0% in GSE68465, and **10.0% in BOTH replication cohorts simultaneously**
— vs. 0.25% expected by chance alone if independent 5%×5% (a 40x enrichment over pure
noise). So there appears to be real, reproducible signal in the aggregate, but no
single gene individually clears a genome-wide permutation-FDR bar at this discovery
sample size (n=106, 24 events).

## What we want reviewed
1. Is the 3-stage adaptive permutation design statistically sound, or does it
   introduce bias / p-hacking risk (e.g. by choosing which genes to escalate based on
   stage-A/B results)?
2. Is "0% survive BH-FDR, but 10% naive-to-naive replicate across 2 independent
   cohorts at 40x chance" an internally consistent, honest way to report this, or is
   it confusing / does it need a different framing?
3. Is the discovery cohort (n=106, 24 events) simply too underpowered to draw the
   "0% survives" conclusion at all, i.e. should the paper's headline be about power
   limitations rather than biomarker validity?
4. Any statistical red flags in using median-split log-rank as the sole screening
   statistic (vs. e.g. continuous Cox score test, which has more power)?
5. Given this result is much more extreme than what was originally pitched ("a
   fraction survives"), does the overall project remain publishable, and at what
   framing/venue?
