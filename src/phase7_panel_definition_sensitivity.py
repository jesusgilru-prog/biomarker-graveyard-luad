"""Phase 7, per peer-review simulation round 2 (Codex + Fable acting as referees for
Computers in Biology and Medicine, 2026-08-14): the remaining analytic condition for
Accept. The 121-gene panel depends on a specific definition (median-split log-rank,
nominal p<0.05, three-way direction concordance); this was sensitivity-checked for the
discovery-cohort COLLAPSE (Section 3.1, Cox continuous vs. log-rank, both give 0 BH-FDR
survivors in CPTAC) but never for the PANEL DEFINITION itself across all three cohorts.

Rewritten from an earlier, much slower version (killed by timeout) that recomputed each
gene's test statistic redundantly many times per configuration; this version computes
every (cohort, gene) statistic exactly ONCE per test type (log-rank, Cox) and reuses the
cached arrays for every threshold/config and every enrichment calculation.

1. Recomputes the three-way directional-concordance panel using the continuous Cox score
   test (survival_stats.cox_score_chi2) instead of median-split log-rank, at p<0.05.
2. Repeats the log-rank-based panel definition at two additional nominal thresholds
   (p<0.01, p<0.10).
3. For each resulting panel, recomputes the gene-resampling (hypergeometric) null
   enrichment and reports overlap with the original 121-gene log-rank/p<0.05 panel.

Run: python src/phase7_panel_definition_sensitivity.py
Writes: results/phase7_panel_definition_sensitivity.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_cptac_luad_discovery, load_tcga_luad_replication, load_gse68465_replication  # noqa: E402
from survival_stats import _logrank_components, cox_score_chi2  # noqa: E402


def _cox_score_and_direction(x, y):
    """Cox score test: chi2 from cox_score_chi2, direction from the sign of the score U."""
    time_, event = y[:, 0], y[:, 1]
    chi2 = cox_score_chi2(x, y)
    if chi2 == 0.0 or np.std(x) == 0:
        return 0.0, 0
    order = np.argsort(time_)
    t, d, xs = time_[order], event[order], x[order]
    event_times = np.unique(t[d == 1])
    at_risk_idx = np.searchsorted(t, event_times, side="left")
    U = 0.0
    for k, et in enumerate(event_times):
        risk_x = xs[at_risk_idx[k]:]
        s0 = risk_x.size
        if s0 <= 1:
            continue
        died_mask = (t == et) & (d == 1)
        dk = int(died_mask.sum())
        U += xs[died_mask].sum() - dk * (risk_x.sum() / s0)
    return chi2, (1 if U > 0 else (-1 if U < 0 else 0))


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"


def precompute_all_stats(cohorts, panel):
    """One pass over every (cohort, gene): logrank chi2/p/dir AND cox chi2/p/dir. Reused
    for every downstream configuration -- no redundant recomputation."""
    out = {}
    for tag, (X, y) in cohorts.items():
        n = len(panel)
        lr_p = np.empty(n)
        lr_d = np.empty(n, dtype=int)
        cx_p = np.empty(n)
        cx_d = np.empty(n, dtype=int)
        for i, g in enumerate(panel):
            x = X.loc[g].values
            c1, o1 = _logrank_components(x, y)
            lr_d[i] = 1 if o1 > 0 else (-1 if o1 < 0 else 0)
            lr_p[i] = sps.chi2.sf(c1, df=1)
            c2, d2 = _cox_score_and_direction(x, y)
            cx_d[i] = d2
            cx_p[i] = sps.chi2.sf(c2, df=1)
        out[tag] = dict(lr_p=lr_p, lr_d=lr_d, cx_p=cx_p, cx_d=cx_d)
    return out


def build_panel_mask(stats, tags, p_arr_key, d_arr_key, threshold):
    """Boolean mask over `panel` indices: p<threshold and same nonzero direction in all
    cohorts."""
    n = len(stats[tags[0]][p_arr_key])
    ok = np.ones(n, dtype=bool)
    dirs = np.zeros((len(tags), n), dtype=int)
    for j, tag in enumerate(tags):
        ok &= (stats[tag][p_arr_key] < threshold) & (stats[tag][d_arr_key] != 0)
        dirs[j] = stats[tag][d_arr_key]
    same_dir = np.all(dirs == dirs[0], axis=0)
    return ok & same_dir


def enrichment_for_config(stats, tags, p_key, d_key, threshold, M):
    """Hypergeometric enrichment per reference cohort, using cached stats only."""
    out = {}
    naive_mask = {t: (stats[t][p_key] < threshold) & (stats[t][d_key] != 0) for t in tags}
    for ref in tags:
        others = [t for t in tags if t != ref]
        o_ok = (stats[others[0]][p_key] < threshold) & (stats[others[0]][d_key] != 0)
        o_ok &= (stats[others[1]][p_key] < threshold) & (stats[others[1]][d_key] != 0)
        o_ok &= stats[others[0]][d_key] == stats[others[1]][d_key]
        n_naive = int(naive_mask[ref].sum())
        n_others = int(o_ok.sum())
        overlap = int((naive_mask[ref] & o_ok).sum())
        expected = n_naive * n_others / M if M else 0
        pval = (sps.hypergeom.sf(overlap - 1, M, n_others, n_naive)
                if expected > 0 else 1.0)
        out[ref] = dict(n_naive=n_naive, n_others_concordant=n_others, observed_overlap=overlap,
                         expected=round(expected, 2),
                         fold=round(overlap / expected, 2) if expected > 0 else None, p=pval)
    return out


def main():
    X1, y1 = load_cptac_luad_discovery()
    X2, y2 = load_tcga_luad_replication()
    X3, y3 = load_gse68465_replication()
    panel = sorted(set(X1.index) & set(X2.index) & set(X3.index))
    M = len(panel)
    ya1 = y1[["os_time_days", "os_event"]].values
    ya2 = y2[["os_time_days", "os_event"]].values
    ya3 = y3[["os_time_months", "os_event"]].values
    cohorts = {"cptac": (X1, ya1), "tcga": (X2, ya2), "gse68465": (X3, ya3)}
    tags = list(cohorts.keys())

    t0 = time.time()
    stats = precompute_all_stats(cohorts, panel)
    print(f"Precomputed all stats in {time.time()-t0:.1f}s")

    original_mask = build_panel_mask(stats, tags, "lr_p", "lr_d", 0.05)
    original_genes = {panel[i] for i in np.where(original_mask)[0]}
    print(f"Original (log-rank, p<0.05): {len(original_genes)} genes "
          f"({'matches manuscript' if len(original_genes) == 121 else 'MISMATCH'})")

    results = {"original_panel_size": len(original_genes)}

    print("\n[1] Cox continuous, p<0.05...")
    cox_mask = build_panel_mask(stats, tags, "cx_p", "cx_d", 0.05)
    cox_genes = {panel[i] for i in np.where(cox_mask)[0]}
    overlap_cox = cox_genes & original_genes
    enrich_cox = enrichment_for_config(stats, tags, "cx_p", "cx_d", 0.05, M)
    print(f"  panel: {len(cox_genes)}, overlap with original 121: {len(overlap_cox)}")
    print(f"  enrichment: {enrich_cox}")
    results["cox_p05"] = dict(panel_size=len(cox_genes), overlap_with_original=len(overlap_cox),
                               jaccard=round(len(overlap_cox) / len(cox_genes | original_genes), 3),
                               enrichment=enrich_cox)

    for thr, tag in [(0.01, "p01"), (0.10, "p10")]:
        print(f"\n[2] Log-rank, p<{thr}...")
        p_mask = build_panel_mask(stats, tags, "lr_p", "lr_d", thr)
        p_genes = {panel[i] for i in np.where(p_mask)[0]}
        overlap_p = p_genes & original_genes
        enrich_p = enrichment_for_config(stats, tags, "lr_p", "lr_d", thr, M)
        print(f"  panel: {len(p_genes)}, overlap with original 121: {len(overlap_p)}")
        print(f"  enrichment: {enrich_p}")
        union = p_genes | original_genes
        results[f"logrank_{tag}"] = dict(
            panel_size=len(p_genes), overlap_with_original=len(overlap_p),
            jaccard=round(len(overlap_p) / len(union), 3) if union else None,
            enrichment=enrich_p)

    with open(RESULTS / "phase7_panel_definition_sensitivity.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("\nWritten to results/phase7_panel_definition_sensitivity.json")


if __name__ == "__main__":
    main()
