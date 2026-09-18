"""End-to-end reproducibility audit, per Codex's plan step 1
(revision_debate_veredicto_q1/ACTA.md): from the raw-data loaders only (no intermediate
results/*.csv trusted), regenerate the two numbers the manuscript's central claims rest
on, and assert they match the recorded values exactly. This is a correctness check, not a
new analysis -- if it fails, the manuscript is wrong and must not be submitted as-is.

Checks:
1. The 121-gene cross-cohort directional panel (Section 3.2 of the manuscript).
2. The 91/1,080 HPA external-validation count (Section 3.3), naive+direction-concordant
   in both CPTAC and GSE68465 (the number that also feeds the empirical-null calibration
   in phase4_hpa_validation.py -- this script checks only the deterministic part; the
   permutation-based empirical p-value/fold-enrichment already has its own dedicated,
   separately-run script and is not re-run here since it is stochastic by design, not a
   candidate for exact-match auditing the way the two counts above are).

Run: python src/audit_reproducibility.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import (hpa_lung_adenocarcinoma_claims, load_cptac_luad_discovery,  # noqa: E402
                   load_tcga_luad_replication, load_gse68465_replication)
from survival_stats import _logrank_components  # noqa: E402

EXPECTED_PANEL_SIZE = 121
EXPECTED_PANEL_WITHOUT_TCGA = 231
EXPECTED_HPA_DIRECTIONAL_IN_PANEL = 1080
EXPECTED_HPA_BOTH_CONCORDANT = 91


def chi2_dir(x, y):
    c, o = _logrank_components(x, y)
    return c, (1 if o > 0 else (-1 if o < 0 else 0))


def main():
    ok = True

    X1, y1 = load_cptac_luad_discovery()
    X2, y2 = load_tcga_luad_replication()
    X3, y3 = load_gse68465_replication()
    panel = sorted(set(X1.index) & set(X2.index) & set(X3.index))
    ya1 = y1[["os_time_days", "os_event"]].values
    ya2 = y2[["os_time_days", "os_event"]].values
    ya3 = y3[["os_time_months", "os_event"]].values

    print(f"Shared gene panel: {len(panel)} genes "
          f"({'OK' if len(panel) == 11484 else 'MISMATCH, expected 11484'})")

    genes_3cohorts = []
    genes_sin_tcga = []
    for g in panel:
        c1, d1 = chi2_dir(X1.loc[g].values, ya1)
        c2, d2 = chi2_dir(X2.loc[g].values, ya2)
        c3, d3 = chi2_dir(X3.loc[g].values, ya3)
        p1, p2, p3 = sps.chi2.sf(c1, 1), sps.chi2.sf(c2, 1), sps.chi2.sf(c3, 1)
        if p1 < 0.05 and p2 < 0.05 and p3 < 0.05 and d1 == d2 == d3 and d1 != 0:
            genes_3cohorts.append(g)
        if p1 < 0.05 and p3 < 0.05 and d1 == d3 and d1 != 0:
            genes_sin_tcga.append(g)

    n_panel = len(genes_3cohorts)
    n_no_tcga = len(genes_sin_tcga)
    subset_ok = set(genes_3cohorts) <= set(genes_sin_tcga)

    print(f"\n[Check 1] 3-cohort directional panel: {n_panel} genes "
          f"(expected {EXPECTED_PANEL_SIZE}) -> "
          f"{'PASS' if n_panel == EXPECTED_PANEL_SIZE else 'FAIL'}")
    ok &= (n_panel == EXPECTED_PANEL_SIZE)

    print(f"[Check 2] Panel without TCGA (CPTAC+GSE68465 only): {n_no_tcga} genes "
          f"(expected {EXPECTED_PANEL_WITHOUT_TCGA}) -> "
          f"{'PASS' if n_no_tcga == EXPECTED_PANEL_WITHOUT_TCGA else 'FAIL'}")
    ok &= (n_no_tcga == EXPECTED_PANEL_WITHOUT_TCGA)

    print(f"[Check 3] 3-cohort panel is an exact subset of the no-TCGA panel: "
          f"{'PASS' if subset_ok else 'FAIL'}")
    ok &= subset_ok

    # ---- HPA external validation (deterministic part) ----
    hpa = hpa_lung_adenocarcinoma_claims()
    directional = hpa[hpa["call"].str.contains("favorable", na=False)].copy()
    directional["hpa_direction"] = directional["call"].apply(
        lambda c: -1 if "favorable" in c and "unfavorable" not in c else 1)
    directional = directional[directional["gene"].isin(panel)].drop_duplicates(subset="gene")
    hpa_genes = directional["gene"].tolist()
    hpa_dir = directional.set_index("gene")["hpa_direction"].reindex(hpa_genes).values

    n_hpa_directional = len(hpa_genes)
    print(f"\n[Check 4] HPA directional genes in shared panel: {n_hpa_directional} "
          f"(expected {EXPECTED_HPA_DIRECTIONAL_IN_PANEL}) -> "
          f"{'PASS' if n_hpa_directional == EXPECTED_HPA_DIRECTIONAL_IN_PANEL else 'FAIL'}")
    ok &= (n_hpa_directional == EXPECTED_HPA_DIRECTIONAL_IN_PANEL)

    both_concordant = 0
    for g, hd in zip(hpa_genes, hpa_dir):
        c1, d1 = chi2_dir(X1.loc[g].values, ya1)
        c3, d3 = chi2_dir(X3.loc[g].values, ya3)
        p1, p3 = sps.chi2.sf(c1, 1), sps.chi2.sf(c3, 1)
        if p1 < 0.05 and d1 == hd and p3 < 0.05 and d3 == hd:
            both_concordant += 1

    print(f"[Check 5] HPA both-cohort naive+concordant: {both_concordant} "
          f"(expected {EXPECTED_HPA_BOTH_CONCORDANT}) -> "
          f"{'PASS' if both_concordant == EXPECTED_HPA_BOTH_CONCORDANT else 'FAIL'}")
    ok &= (both_concordant == EXPECTED_HPA_BOTH_CONCORDANT)

    print(f"\n{'ALL CHECKS PASSED' if ok else 'AUDIT FAILED -- DO NOT SUBMIT UNTIL RESOLVED'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
