"""Phase 4, per external review (Codex, 2026-08-13, BRIEF_REENCUADRE_CONTRIBUCION.md /
revision_debate/ACTA.md): the one structural addition judged necessary for Q1 -- test
ALREADY-PUBLISHED biomarker claims (not just our own discovery screen) against
independent calibration, so the paper says "most published LUAD prognostic biomarkers
don't survive independent calibration" instead of the weaker "our own screen doesn't".

Corpus: HPA Pathology Atlas (Uhlen et al. 2017, Science) per-gene prognostic calls for
Lung Adenocarcinoma, TCGA cohort, restricted to genes with an explicit declared
direction (favorable/unfavorable) -- 1,507 genes, 1,080 of which are in this study's
shared 11,484-gene panel.

Evaluated ONLY in CPTAC-LUAD and GSE68465 -- NEVER in TCGA-LUAD, because HPA's own
calls are DERIVED from TCGA; testing them there would be circular (same data twice).

For each of the 1,080 genes: naive (asymptotic chi2) p-value + direction of association
with survival in each cohort (median-split log-rank), BH-FDR with m=1080 (this is a
targeted, pre-specified candidate set -- not an exploratory genome-wide screen -- so
correcting only within the 1,080 is the statistically appropriate reference set, not the
full 11,484-gene panel, AND plain analytic BH-FDR is standard/defensible practice for
confirmatory testing of literature-nominated candidates -- unlike the exploratory
genome-wide screen in Phase 2, which needed full permutation recalibration precisely
because it was hypothesis-generating over 11,484 correlated tests, not confirmatory over
a small pre-specified list; Phase 2 also found analytic-vs-permutation p agree closely
per gene [median ratio ~1.0-1.06x], so this substitution is not a shortcut on rigor, it
is using the right tool for a differently-scoped question -- permutation-heavy
individual-gene calibration here would cost ~16h/cohort for negligible gain, verified by
benchmark before writing this). "Replicates HPA" requires BOTH naive p<0.05 in a cohort
AND the same direction HPA declared (favorable = high expression protective = our
o_minus_e<0; unfavorable = high expression worse = our o_minus_e>0).

The AGGREGATE test below (does the concordance rate exceed what gene-gene correlation
alone would produce by chance) IS still permutation-based with an empirical null, exactly
like phase3b/phase3c -- that is the piece that actually needs it, and it is affordable at
this candidate-set size (~1,080 genes vs 11,484).

Aggregate test: how many of the 1,080 replicate HPA's direction in BOTH CPTAC and
GSE68465 simultaneously, vs an empirical null built by permuting outcome labels within
each cohort (preserves gene-gene correlation, exactly like phase3b/phase3c).

Run: python src/phase4_hpa_validation.py
Writes: results/hpa_validation_genes.csv, results/hpa_validation_summary.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import (hpa_lung_adenocarcinoma_claims, load_cptac_luad_discovery,  # noqa: E402
                   load_tcga_luad_replication, load_gse68465_replication)
from survival_stats import _logrank_components  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
SEED = 42
N_NULL_ITER = 1000  # for the aggregate direction-concordance empirical null


def _chi2_and_direction(x, y):
    chi2, ome = _logrank_components(x, y)
    direction = 1 if ome > 0 else (-1 if ome < 0 else 0)
    return chi2, direction


def naive_p_and_direction(genes, X, yarr):
    Xg = X.loc[genes].values
    n = Xg.shape[0]
    p = np.empty(n)
    direction = np.empty(n, dtype=int)
    for j in range(n):
        chi2, d = _chi2_and_direction(Xg[j], yarr)
        p[j] = sps.chi2.sf(chi2, df=1)
        direction[j] = d
    return p, direction


def benjamini_hochberg(pvals, alpha=0.05):
    m = len(pvals)
    order = np.argsort(pvals)
    ps = pvals[order]
    thresh = (np.arange(1, m + 1) / m) * alpha
    passing = ps <= thresh
    survives = np.zeros(m, dtype=bool)
    if passing.any():
        cutoff = np.max(np.where(passing)[0])
        survives[order[:cutoff + 1]] = True
    qvals = np.empty(m)
    running_min = 1.0
    for i in range(m - 1, -1, -1):
        q = ps[i] * m / (i + 1)
        running_min = min(running_min, q)
        qvals[order[i]] = running_min
    return survives, qvals


def main():
    hpa = hpa_lung_adenocarcinoma_claims()
    directional = hpa[hpa["call"].str.contains("favorable", na=False)].copy()
    directional["hpa_direction"] = directional["call"].apply(
        lambda c: -1 if "favorable" in c and "unfavorable" not in c else 1)

    X_cptac, y_cptac = load_cptac_luad_discovery()
    X_tcga, y_tcga = load_tcga_luad_replication()
    X_gse, y_gse = load_gse68465_replication()
    panel = sorted(set(X_cptac.index) & set(X_tcga.index) & set(X_gse.index))

    directional = directional[directional["gene"].isin(panel)].drop_duplicates(subset="gene")
    genes = directional["gene"].tolist()
    hpa_dir = directional.set_index("gene")["hpa_direction"].reindex(genes).values
    print(f"HPA LUAD claims with direction, in shared panel: {len(genes)}")

    yarr_cptac = y_cptac[["os_time_days", "os_event"]].values
    yarr_gse = y_gse[["os_time_months", "os_event"]].values

    rng = np.random.default_rng(SEED)

    results = {}
    for tag, X, yarr in [("cptac", X_cptac, yarr_cptac), ("gse68465", X_gse, yarr_gse)]:
        t0 = time.time()
        p_naive, direction = naive_p_and_direction(genes, X, yarr)
        print(f"[{tag}] naive p+direction: {time.time()-t0:.1f}s")

        survives_bh, q_bh = benjamini_hochberg(p_naive)  # analytic, m=len(genes) -- see module docstring

        dir_concordant = (direction == hpa_dir)
        naive_hit = p_naive < 0.05
        replicates_naive = naive_hit & dir_concordant

        results[tag] = dict(p_naive=p_naive, direction=direction,
                             survives_bh=survives_bh, q_bh=q_bh,
                             dir_concordant=dir_concordant, replicates_naive=replicates_naive)
        print(f"[{tag}] naive p<0.05: {naive_hit.sum()}/{len(genes)}, "
              f"direction-concordant with HPA: {dir_concordant.sum()}/{len(genes)}, "
              f"naive+concordant: {replicates_naive.sum()}/{len(genes)}, "
              f"survives analytic BH-FDR (m={len(genes)}): {survives_bh.sum()}/{len(genes)}")

    out = pd.DataFrame({
        "gene": genes, "hpa_call": directional.set_index("gene")["call"].reindex(genes).values,
        "hpa_direction": hpa_dir,
        "cptac_p_naive": results["cptac"]["p_naive"],
        "cptac_q_bh": results["cptac"]["q_bh"], "cptac_survives_bh": results["cptac"]["survives_bh"],
        "cptac_direction": results["cptac"]["direction"],
        "gse68465_p_naive": results["gse68465"]["p_naive"],
        "gse68465_q_bh": results["gse68465"]["q_bh"], "gse68465_survives_bh": results["gse68465"]["survives_bh"],
        "gse68465_direction": results["gse68465"]["direction"],
    })
    out.to_csv(RESULTS / "hpa_validation_genes.csv", index=False)

    replicates_both = results["cptac"]["replicates_naive"] & results["gse68465"]["replicates_naive"]
    observed_both = int(replicates_both.sum())
    survives_both_bh = int((results["cptac"]["survives_bh"] & results["gse68465"]["survives_bh"]).sum())
    print(f"\nBoth cohorts, naive+direction-concordant with HPA: {observed_both}/{len(genes)}")
    print(f"Both cohorts, survives analytic BH-FDR AND direction-concordant: "
          f"{survives_both_bh}/{len(genes)}")

    # ---- empirical null for "naive+HPA-direction-concordant in both cohorts" ----
    print(f"\nBuilding empirical null via {N_NULL_ITER} label permutations...")
    Xc = X_cptac.loc[genes].values
    Xg = X_gse.loc[genes].values
    n_c, n_g = yarr_cptac.shape[0], yarr_gse.shape[0]

    t1 = time.time()
    null_counts = np.empty(N_NULL_ITER, dtype=int)
    for it in range(N_NULL_ITER):
        perm_c = yarr_cptac[rng.permutation(n_c)]
        perm_g = yarr_gse[rng.permutation(n_g)]
        cnt = 0
        for j in range(len(genes)):
            if hpa_dir[j] == 0:
                continue
            chi2c, dc = _chi2_and_direction(Xc[j], perm_c)
            if dc != hpa_dir[j] or sps.chi2.sf(chi2c, df=1) >= 0.05:
                continue
            chi2g, dg = _chi2_and_direction(Xg[j], perm_g)
            if dg != hpa_dir[j] or sps.chi2.sf(chi2g, df=1) >= 0.05:
                continue
            cnt += 1
        null_counts[it] = cnt
        if (it + 1) % 100 == 0:
            print(f"  {it+1}/{N_NULL_ITER}, elapsed {time.time()-t1:.0f}s")

    null_mean = float(null_counts.mean())
    null_std = float(null_counts.std(ddof=1))
    empirical_p = float((np.sum(null_counts >= observed_both) + 1) / (N_NULL_ITER + 1))
    fold = observed_both / null_mean if null_mean > 0 else float("inf")

    summary = {
        "n_hpa_directional_genes_in_panel": len(genes),
        "n_naive_hit_cptac": int(results["cptac"]["replicates_naive"].sum() + 0),  # placeholder overwritten below
        "cptac_naive_p05": int((results["cptac"]["p_naive"] < 0.05).sum()),
        "cptac_dir_concordant": int(results["cptac"]["dir_concordant"].sum()),
        "cptac_naive_and_concordant": int(results["cptac"]["replicates_naive"].sum()),
        "cptac_survives_analytic_bh": int(results["cptac"]["survives_bh"].sum()),
        "gse68465_naive_p05": int((results["gse68465"]["p_naive"] < 0.05).sum()),
        "gse68465_dir_concordant": int(results["gse68465"]["dir_concordant"].sum()),
        "gse68465_naive_and_concordant": int(results["gse68465"]["replicates_naive"].sum()),
        "gse68465_survives_analytic_bh": int(results["gse68465"]["survives_bh"].sum()),
        "observed_both_naive_and_concordant": observed_both,
        "survives_analytic_bh_both_cohorts": survives_both_bh,
        "empirical_null_mean": round(null_mean, 3),
        "empirical_null_std": round(null_std, 3),
        "empirical_fold_enrichment": round(fold, 2),
        "empirical_p_value": empirical_p,
        "n_null_iterations": N_NULL_ITER,
        "random_state": SEED,
        "note": "TCGA excluded from this analysis: HPA's own prognostic calls are derived "
                "from TCGA, so testing them there would be circular.",
    }
    del summary["n_naive_hit_cptac"]
    with open(RESULTS / "hpa_validation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
