"""Phase 2 fix, per external review (Fable/Gemini/Codex independently converged
on this): the original "10% replicate in both cohorts, 40x the 0.25% expected
by chance" claim had two real problems:

1. No direction-of-effect check: a gene could count as "replicated" with
   OPPOSITE direction in discovery vs a replication cohort (log-rank chi2 is
   direction-agnostic). Fixed here by requiring sign(logrank_direction) to
   match across discovery and both replication cohorts.

2. The 0.25% baseline assumed independence between the 949 genes (naive
   5%x5%). Genes are correlated (co-expression modules), so the true
   chance-level overlap rate is unknown and likely much higher than the
   independence assumption gives. Fixed here with an empirical permutation
   null: shuffle the outcome labels within each replication cohort (breaking
   any true association while preserving the real gene-gene correlation
   structure, since genes themselves are untouched), recompute naive p<0.05 +
   direction-concordance-with-discovery for all 949 genes in both permuted
   cohorts, and count the overlap. Repeat many times to get the empirical
   null distribution of "how many of the 949 would coincidentally look
   replicated by pure chance, given the real correlation structure."

Run: python src/phase3b_replication_empirical_null.py
Writes: results/replication_direction_aware.csv, results/replication_empirical_null.json
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
from data import load_cptac_luad_discovery, load_tcga_luad_replication, load_gse68465_replication  # noqa: E402
from survival_stats import logrank_chi2, logrank_direction, _logrank_components  # noqa: E402


def _chi2_and_direction(x, y):
    chi2, ome = _logrank_components(x, y)
    direction = 1 if ome > 0 else (-1 if ome < 0 else 0)
    return chi2, direction

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
SEED = 42
N_NULL_ITER = 1000


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


def main():
    disc = pd.read_csv(RESULTS / "discovery_naive_vs_calibrated.csv")
    naive_hits = disc[disc["naive_hit_p05"]].copy()
    genes = naive_hits["gene"].tolist()
    print(f"Discovery naive hits: {len(genes)}")

    X_disc, y_disc = load_cptac_luad_discovery()
    yarr_disc = y_disc[["os_time_days", "os_event"]].values
    _, dir_disc = naive_p_and_direction(genes, X_disc, yarr_disc)

    X_tcga, y_tcga = load_tcga_luad_replication()
    X_gse, y_gse = load_gse68465_replication()
    genes_tcga = [g for g in genes if g in X_tcga.index]
    genes_gse = [g for g in genes if g in X_gse.index]
    assert genes_tcga == genes and genes_gse == genes, "panel mismatch -- should not happen (shared-gene panel)"

    yarr_tcga = y_tcga[["os_time_days", "os_event"]].values
    yarr_gse = y_gse[["os_time_months", "os_event"]].values

    t0 = time.time()
    p_tcga, dir_tcga = naive_p_and_direction(genes, X_tcga, yarr_tcga)
    p_gse, dir_gse = naive_p_and_direction(genes, X_gse, yarr_gse)
    print(f"Observed naive p + direction computed in {time.time()-t0:.1f}s")

    sig_tcga = p_tcga < 0.05
    sig_gse = p_gse < 0.05
    concordant_tcga = (dir_tcga == dir_disc) & (dir_disc != 0)
    concordant_gse = (dir_gse == dir_disc) & (dir_disc != 0)

    replicates_tcga = sig_tcga & concordant_tcga
    replicates_gse = sig_gse & concordant_gse
    replicates_both = replicates_tcga & replicates_gse

    n_rep_tcga_naive = int(sig_tcga.sum())  # old, direction-blind count for comparison
    n_rep_tcga_dir = int(replicates_tcga.sum())
    n_rep_gse_naive = int(sig_gse.sum())
    n_rep_gse_dir = int(replicates_gse.sum())
    n_rep_both_naive = int((sig_tcga & sig_gse).sum())
    n_rep_both_dir = int(replicates_both.sum())

    print(f"TCGA: naive p<0.05 = {n_rep_tcga_naive}, direction-concordant = {n_rep_tcga_dir}")
    print(f"GSE68465: naive p<0.05 = {n_rep_gse_naive}, direction-concordant = {n_rep_gse_dir}")
    print(f"BOTH: naive p<0.05 (direction-blind) = {n_rep_both_naive}, "
          f"direction-concordant = {n_rep_both_dir}")

    out = naive_hits[["gene", "observed_logrank_chi2", "naive_p_value"]].copy()
    out["discovery_direction"] = dir_disc
    out["tcga_naive_p"] = p_tcga
    out["tcga_direction"] = dir_tcga
    out["tcga_replicates"] = replicates_tcga
    out["gse68465_naive_p"] = p_gse
    out["gse68465_direction"] = dir_gse
    out["gse68465_replicates"] = replicates_gse
    out["replicates_both_direction_concordant"] = replicates_both
    out = out.sort_values("naive_p_value")
    out.to_csv(RESULTS / "replication_direction_aware.csv", index=False)

    # ---- empirical null for the "both, direction-concordant" overlap count ----
    print(f"\nBuilding empirical null via {N_NULL_ITER} label permutations "
          f"(preserves real gene-gene correlation structure)...")
    rng = np.random.default_rng(SEED)
    n_tcga = yarr_tcga.shape[0]
    n_gse = yarr_gse.shape[0]
    X_tcga_vals = X_tcga.loc[genes].values
    X_gse_vals = X_gse.loc[genes].values

    t1 = time.time()
    null_counts = np.empty(N_NULL_ITER, dtype=int)
    for it in range(N_NULL_ITER):
        perm_tcga = yarr_tcga[rng.permutation(n_tcga)]
        perm_gse = yarr_gse[rng.permutation(n_gse)]
        cnt = 0
        for j in range(len(genes)):
            if dir_disc[j] == 0:
                continue
            chi2_t, dt = _chi2_and_direction(X_tcga_vals[j], perm_tcga)
            if dt != dir_disc[j] or sps.chi2.sf(chi2_t, df=1) >= 0.05:
                continue
            chi2_g, dg = _chi2_and_direction(X_gse_vals[j], perm_gse)
            if dg != dir_disc[j] or sps.chi2.sf(chi2_g, df=1) >= 0.05:
                continue
            cnt += 1
        null_counts[it] = cnt
        if (it + 1) % 100 == 0:
            print(f"  {it+1}/{N_NULL_ITER} null iterations, elapsed {time.time()-t1:.0f}s")

    observed = n_rep_both_dir
    null_mean = float(null_counts.mean())
    null_std = float(null_counts.std(ddof=1))
    empirical_p = float((np.sum(null_counts >= observed) + 1) / (N_NULL_ITER + 1))
    fold_enrichment = observed / null_mean if null_mean > 0 else float("inf")

    result = {
        "n_naive_discovery_hits": len(genes),
        "old_naive_both_direction_blind": n_rep_both_naive,
        "old_naive_pct_direction_blind": round(100 * n_rep_both_naive / len(genes), 1),
        "observed_both_direction_concordant": observed,
        "observed_pct_direction_concordant": round(100 * observed / len(genes), 1),
        "naive_independence_expected": round(len(genes) * 0.0025, 2),
        "naive_independence_fold": round(observed / (len(genes) * 0.0025), 1) if len(genes) else None,
        "empirical_null_mean": round(null_mean, 2),
        "empirical_null_std": round(null_std, 2),
        "empirical_fold_enrichment": round(fold_enrichment, 2),
        "empirical_p_value": empirical_p,
        "n_null_iterations": N_NULL_ITER,
        "random_state": SEED,
        "runtime_seconds": round(time.time() - t1, 1),
    }
    with open(RESULTS / "replication_empirical_null.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
