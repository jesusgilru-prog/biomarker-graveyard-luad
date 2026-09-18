"""Phase 5, per external review round 1 (Codex + Fable, 2026-08-14,
paper/revision_r1/ACTA.md and the Fable agent report in that same round): four analyses
requested to preempt the sharpest anticipated referee objections, all analytic/combinatorial
(no new permutation runs needed -- reuses data already on disk).

1. Gene-resampling (hypergeometric) null for the 121-gene panel, at all three role-swap
   configurations. The permutation null used in Section 3.2 destroys ALL cross-cohort
   survival signal (it shuffles outcome labels), so enrichment over it (89-188x) answers
   "is there any cross-cohort directional signal at all" -- a null nobody disputes. The
   sharper, referee-proof question is "are the discovery cohort's naive hits specifically
   enriched among genes that already show cross-cohort concordance in the OTHER two
   cohorts, beyond what a same-size random gene subset would show". That is a
   combinatorial (hypergeometric) question, not a permutation one, and is computed here.

2. The same joint-vs-marginal decomposition for the HPA external validation (Section 3.3):
   91 observed vs. 59.1 expected under independence of the two cohorts' marginal hit
   rates (185 CPTAC, 345 GSE68465) -- a 1.54x joint excess, not the 136.6x fold reported
   against the fully-destructive permutation null.

3. Effective dimensionality (Galwey 2009 Meff, same method as the existing GSE68465-anomaly
   analysis in Section 3.4) computed WITHIN the 121-gene panel alone, to address whether it
   behaves like ~121 independent signals or a much lower-dimensional module (e.g. a single
   proliferation axis) -- directly relevant to the Venet, Dumont & Detours (2011) critique
   that proliferation-correlated gene sets broadly predict cancer outcome regardless of
   biological specificity.

4. A standard directional meta-analysis baseline (Stouffer's Z, direction-signed, combining
   all 3 cohorts) with BH-FDR, as the "why not just do this" comparison a statistical
   reviewer will ask for.

Run: python src/phase5_reviewer_robustness.py
Writes: results/phase5_gene_resampling_null.json, results/phase5_hpa_joint_excess.json,
        results/phase5_panel_meff.json, results/phase5_stouffer_meta_analysis.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import (hpa_lung_adenocarcinoma_claims, load_cptac_luad_discovery,  # noqa: E402
                   load_tcga_luad_replication, load_gse68465_replication)
from survival_stats import _logrank_components  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"


def chi2_p_dir(x, y):
    c, o = _logrank_components(x, y)
    d = 1 if o > 0 else (-1 if o < 0 else 0)
    return c, sps.chi2.sf(c, df=1), d


def load_all():
    X1, y1 = load_cptac_luad_discovery()
    X2, y2 = load_tcga_luad_replication()
    X3, y3 = load_gse68465_replication()
    panel = sorted(set(X1.index) & set(X2.index) & set(X3.index))
    ya1 = y1[["os_time_days", "os_event"]].values
    ya2 = y2[["os_time_days", "os_event"]].values
    ya3 = y3[["os_time_months", "os_event"]].values
    return {"cptac": (X1, ya1), "tcga": (X2, ya2), "gse68465": (X3, ya3)}, panel


def compute_full_stats(cohorts, panel):
    """chi2, p, direction for every gene in every cohort, computed once and reused."""
    out = {}
    for tag, (X, ya) in cohorts.items():
        chi2s, ps, ds = {}, {}, {}
        for g in panel:
            c, p, d = chi2_p_dir(X.loc[g].values, ya)
            chi2s[g], ps[g], ds[g] = c, p, d
        out[tag] = dict(chi2=chi2s, p=ps, d=ds)
    return out


def gene_resampling_null(stats, panel):
    """Analysis 1: for each reference cohort, hypergeometric enrichment of its naive hits
    within the OTHER two cohorts' mutual concordant set."""
    M = len(panel)
    tags = ["cptac", "tcga", "gse68465"]
    results = {}
    for ref in tags:
        others = [t for t in tags if t != ref]
        ref_naive = {g for g in panel if stats[ref]["p"][g] < 0.05}
        others_concordant = {
            g for g in panel
            if stats[others[0]]["p"][g] < 0.05 and stats[others[1]]["p"][g] < 0.05
            and stats[others[0]]["d"][g] == stats[others[1]]["d"][g]
            and stats[others[0]]["d"][g] != 0
        }
        overlap_naive = len(ref_naive & others_concordant)
        expected = len(ref_naive) * len(others_concordant) / M
        fold = overlap_naive / expected if expected > 0 else float("inf")
        pval = sps.hypergeom.sf(overlap_naive - 1, M, len(others_concordant), len(ref_naive))
        # of that naive overlap, how many also match the reference cohort's own direction
        # (this reproduces the reported 121-gene panel size as an internal consistency check)
        triple = sum(1 for g in (ref_naive & others_concordant)
                     if stats[ref]["d"][g] == stats[others[0]]["d"][g])
        results[ref] = dict(
            universe=M, n_reference_naive=len(ref_naive),
            n_others_concordant=len(others_concordant),
            observed_naive_overlap=overlap_naive,
            expected_under_independence=round(expected, 2),
            fold_over_gene_resampling_null=round(fold, 2),
            hypergeometric_p=pval,
            triple_concordant_check=triple,
        )
    return results


def hpa_joint_excess(stats, panel):
    """Analysis 2: HPA external validation, joint-vs-marginal decomposition."""
    hpa = hpa_lung_adenocarcinoma_claims()
    directional = hpa[hpa["call"].str.contains("favorable", na=False)].copy()
    directional["hpa_direction"] = directional["call"].apply(
        lambda c: -1 if "favorable" in c and "unfavorable" not in c else 1)
    directional = directional[directional["gene"].isin(panel)].drop_duplicates(subset="gene")
    genes = directional["gene"].tolist()
    hpa_dir = directional.set_index("gene")["hpa_direction"].reindex(genes).values
    M = len(genes)

    cptac_ok = gse_ok = both_ok = 0
    for g, hd in zip(genes, hpa_dir):
        ok1 = stats["cptac"]["p"][g] < 0.05 and stats["cptac"]["d"][g] == hd
        ok3 = stats["gse68465"]["p"][g] < 0.05 and stats["gse68465"]["d"][g] == hd
        cptac_ok += ok1
        gse_ok += ok3
        both_ok += ok1 and ok3

    expected = cptac_ok * gse_ok / M
    pval = sps.hypergeom.sf(both_ok - 1, M, gse_ok, cptac_ok)
    return dict(
        n_hpa_directional_genes=M, cptac_marginal=cptac_ok, gse68465_marginal=gse_ok,
        observed_joint=both_ok, expected_under_independence=round(expected, 2),
        fold_joint_excess=round(both_ok / expected, 2), hypergeometric_p=pval,
    )


def _correlation_eigenvalues(X, genes):
    """Eigenvalues of the m_eff x m_eff gene-gene correlation matrix, where m_eff excludes
    genes with zero variance in this cohort (a handful of genes are constant, e.g.
    unexpressed, within a given cohort; standardizing them to unit variance would be
    dividing by zero, and they carry no correlation information, so they are dropped
    rather than arbitrarily forced to variance 1). Returns (eigenvalues, n_samples,
    m_effective) with eigenvalues summing to m_effective, not necessarily len(genes).

    Uses the SVD-duality shortcut (eigenvalues of the smaller n x n Gram matrix,
    zero-padded to m_eff) only when n < m_eff; when n >= m_eff the m_eff x m_eff matrix is
    built and eigendecomposed directly, since the duality trick is invalid in that regime
    (it would otherwise return n > m_eff 'eigenvalues' for an m_eff x m_eff matrix, a real
    bug found and fixed during review round 2, 2026-08-14, after Codex and Fable
    independently flagged a Meff figure that did not reproduce from a fresh read of this
    function).
    """
    Xp = X.loc[genes].values.astype(float)
    sd_full = Xp.std(axis=1, ddof=0)
    keep = sd_full > 0
    Xp = Xp[keep]
    m_eff = Xp.shape[0]
    n = Xp.shape[1]
    mu = Xp.mean(axis=1, keepdims=True)
    sd = Xp.std(axis=1, keepdims=True, ddof=0)
    Z = (Xp - mu) / sd  # m_eff x n
    if n < m_eff:
        C = (Z.T @ Z) / n  # n x n
        eigs_small = np.linalg.eigvalsh(C)
        eigs = np.concatenate([eigs_small, np.zeros(m_eff - n)])
    else:
        R = (Z @ Z.T) / n  # m_eff x m_eff, exact
        eigs = np.linalg.eigvalsh(R)
    return np.clip(eigs, 0, None), n, m_eff


def _meff_cheverud_nyholt(eigs, m):
    """Cheverud (2001) / Nyholt (2004): Meff = 1 + (m-1)(1 - Var(lambda)/m)."""
    var_lambda = np.var(eigs, ddof=0)
    return 1 + (m - 1) * (1 - var_lambda / m)


def _meff_galwey(eigs):
    """Galwey (2009): Meff = (sum sqrt(lambda_i))^2 / sum(lambda_i) -- NOT the same
    estimator as Cheverud/Nyholt above; Galwey's own paper exists because the
    Cheverud/Nyholt estimator overstates the effective number of tests under correlation.
    An earlier version of this script computed only the Cheverud/Nyholt quantity while
    citing it as Galwey's -- caught in external review round 2 (Codex + Fable,
    2026-08-14) and corrected here; both estimators are now reported explicitly under
    their correct names."""
    return (np.sum(np.sqrt(eigs))) ** 2 / np.sum(eigs)


def panel_meff(cohorts, panel, stats):
    """Analysis 3: effective dimensionality of the 121-gene panel's expression, in each
    cohort separately (the panel is defined across cohorts, but 'how many independent axes
    does it span' is a within-cohort question). Reports BOTH the Cheverud/Nyholt and the
    Galwey (2009) estimators -- see docstrings above for why they are not interchangeable."""
    panel_121 = sorted(
        g for g in panel
        if stats["cptac"]["p"][g] < 0.05 and stats["tcga"]["p"][g] < 0.05
        and stats["gse68465"]["p"][g] < 0.05
        and stats["cptac"]["d"][g] == stats["tcga"]["d"][g] == stats["gse68465"]["d"][g]
        and stats["cptac"]["d"][g] != 0
    )
    m = len(panel_121)
    out = {"panel_size": m}
    for tag, (X, _) in cohorts.items():
        eigs, n, m_eff = _correlation_eigenvalues(X, panel_121)
        assert abs(eigs.sum() - m_eff) < 1e-6, f"eigenvalues should sum to m_eff={m_eff}, got {eigs.sum()}"
        meff_cn = _meff_cheverud_nyholt(eigs, m_eff)
        meff_galwey = _meff_galwey(eigs)
        top_eig = float(np.sort(eigs)[::-1][0])
        out[tag] = dict(
            n_samples=n, n_zero_variance_genes_dropped=m - m_eff,
            meff_cheverud_nyholt=round(float(meff_cn), 2),
            meff_cheverud_nyholt_pct=round(100 * meff_cn / m, 1),
            meff_galwey=round(float(meff_galwey), 2),
            meff_galwey_pct=round(100 * meff_galwey / m, 1),
            pc1_variance_fraction=round(top_eig / m_eff, 3),
        )
    return out


def stouffer_meta_analysis(cohorts, panel, stats):
    """Analysis 4: direction-signed Stouffer Z combining all 3 cohorts, BH-FDR, as the
    standard-meta-analysis baseline a statistical reviewer will ask for."""
    z_combined = {}
    for g in panel:
        zs = []
        for tag in ("cptac", "tcga", "gse68465"):
            p, d = stats[tag]["p"][g], stats[tag]["d"][g]
            if d == 0:
                zs.append(0.0)
                continue
            z_one_sided = sps.norm.isf(p / 2)  # two-sided naive p -> one-sided z magnitude
            zs.append(d * z_one_sided)
        z_combined[g] = sum(zs) / np.sqrt(len(zs))

    genes = panel
    zvals = np.array([z_combined[g] for g in genes])
    pvals = 2 * sps.norm.sf(np.abs(zvals))
    m = len(genes)
    order = np.argsort(pvals)
    ps_sorted = pvals[order]
    thresh = (np.arange(1, m + 1) / m) * 0.05
    passing = ps_sorted <= thresh
    n_survive = 0
    if passing.any():
        n_survive = int(np.max(np.where(passing)[0]) + 1)
    n_naive = int((pvals < 0.05).sum())
    return dict(n_genes=m, n_naive_p05=n_naive, n_survive_bh=n_survive,
                pct_survive_bh=round(100 * n_survive / m, 2))


def main():
    cohorts, panel = load_all()
    print("Computing per-gene stats for all cohorts (reused across all 4 analyses)...")
    stats = compute_full_stats(cohorts, panel)

    print("\n[1] Gene-resampling (hypergeometric) null...")
    r1 = gene_resampling_null(stats, panel)
    for k, v in r1.items():
        print(f"  {k}: observed={v['observed_naive_overlap']}, "
              f"expected={v['expected_under_independence']}, "
              f"fold={v['fold_over_gene_resampling_null']}x, p={v['hypergeometric_p']:.3e}, "
              f"triple-check={v['triple_concordant_check']}")
    with open(RESULTS / "phase5_gene_resampling_null.json", "w") as f:
        json.dump(r1, f, indent=2, default=float)

    print("\n[2] HPA joint-vs-marginal excess...")
    r2 = hpa_joint_excess(stats, panel)
    print(f"  {r2}")
    with open(RESULTS / "phase5_hpa_joint_excess.json", "w") as f:
        json.dump(r2, f, indent=2, default=float)

    print("\n[3] Panel effective dimensionality (Meff)...")
    r3 = panel_meff(cohorts, panel, stats)
    print(f"  {r3}")
    with open(RESULTS / "phase5_panel_meff.json", "w") as f:
        json.dump(r3, f, indent=2, default=float)

    print("\n[4] Stouffer meta-analysis baseline...")
    r4 = stouffer_meta_analysis(cohorts, panel, stats)
    print(f"  {r4}")
    with open(RESULTS / "phase5_stouffer_meta_analysis.json", "w") as f:
        json.dump(r4, f, indent=2, default=float)

    print("\nAll 4 analyses written to results/phase5_*.json")


if __name__ == "__main__":
    main()
