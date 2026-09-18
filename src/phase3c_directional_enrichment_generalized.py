"""Generalizes phase3b_replication_empirical_null.py to ANY cohort as
discovery (not just CPTAC), so the directional-concordance-enrichment result
-- the strongest, most defensible finding per the external novelty review
(2026-08-13: nobody else quantifies "real aggregate signal despite 0
individual FDR survivors" this way) -- can be shown to hold under the full
3-cohort role-swap design, not just one direction.

Same logic as phase3b: take the discovery cohort's naive hits (p<0.05,
computed on the shared 11,484-gene panel, already in
results/cohort_swap_<discovery>_naive_vs_calibrated.csv for tcga/gse68465,
or results/discovery_naive_vs_calibrated.csv for cptac), require direction
concordance (not just naive p<0.05) in BOTH other cohorts, and compare the
observed overlap to an empirical null built by permuting outcome labels
within each replication cohort (preserves real gene-gene correlation
structure -- this is what makes the null honest, per the Fase 2b external
review fix).

Run: python src/phase3c_directional_enrichment_generalized.py --discovery tcga
Run: python src/phase3c_directional_enrichment_generalized.py --discovery gse68465
(--discovery cptac reproduces phase3b's original result, kept for cross-check)

Writes: results/directional_enrichment_<discovery>.json,
        results/directional_enrichment_<discovery>_genes.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_cptac_luad_discovery, load_tcga_luad_replication, load_gse68465_replication  # noqa: E402
from survival_stats import _logrank_components  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
SEED = 42
N_NULL_ITER = 1000

COHORTS = {
    "cptac": (load_cptac_luad_discovery, "os_time_days",
              RESULTS / "discovery_naive_vs_calibrated.csv"),
    "tcga": (load_tcga_luad_replication, "os_time_days",
             RESULTS / "cohort_swap_tcga_naive_vs_calibrated.csv"),
    "gse68465": (load_gse68465_replication, "os_time_months",
                 RESULTS / "cohort_swap_gse68465_naive_vs_calibrated.csv"),
}


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--discovery", required=True, choices=list(COHORTS.keys()))
    ap.add_argument("--n-null-iter", type=int, default=N_NULL_ITER)
    args = ap.parse_args()
    disc_tag = args.discovery
    rep_tags = [t for t in COHORTS if t != disc_tag]
    n_null_iter = args.n_null_iter

    loader_disc, time_col_disc, hits_file = COHORTS[disc_tag]
    disc_df = pd.read_csv(hits_file)
    naive_hits = disc_df[disc_df["naive_p_value"] < 0.05].copy()
    genes = naive_hits["gene"].tolist()
    print(f"Discovery={disc_tag}: naive hits = {len(genes)}")

    X_disc, y_disc = loader_disc()
    yarr_disc = y_disc[[time_col_disc, "os_event"]].values
    _, dir_disc = naive_p_and_direction(genes, X_disc, yarr_disc)

    rep_data = {}
    for tag in rep_tags:
        loader, time_col, _ = COHORTS[tag]
        X, y = loader()
        yarr = y[[time_col, "os_event"]].values
        assert all(g in X.index for g in genes), f"panel mismatch in {tag} -- should not happen"
        rep_data[tag] = (X, yarr)

    t0 = time.time()
    sig, direction_ok, replicates = {}, {}, {}
    for tag in rep_tags:
        X, yarr = rep_data[tag]
        p, d = naive_p_and_direction(genes, X, yarr)
        sig[tag] = p < 0.05
        direction_ok[tag] = (d == dir_disc) & (dir_disc != 0)
        replicates[tag] = sig[tag] & direction_ok[tag]
        print(f"  {tag}: naive p<0.05 = {int(sig[tag].sum())}, "
              f"direction-concordant = {int(replicates[tag].sum())}")
    print(f"Observed naive p + direction computed in {time.time()-t0:.1f}s")

    replicates_both = replicates[rep_tags[0]] & replicates[rep_tags[1]]
    n_rep_both_naive = int((sig[rep_tags[0]] & sig[rep_tags[1]]).sum())
    n_rep_both_dir = int(replicates_both.sum())
    print(f"BOTH replication cohorts ({rep_tags[0]} & {rep_tags[1]}): "
          f"naive p<0.05 (direction-blind) = {n_rep_both_naive}, "
          f"direction-concordant = {n_rep_both_dir}")

    out = naive_hits[["gene"]].copy()
    out["discovery_direction"] = dir_disc
    for tag in rep_tags:
        out[f"{tag}_direction"] = None
    out.to_csv(RESULTS / f"directional_enrichment_{disc_tag}_genes.csv", index=False)

    # ---- empirical null: permute outcome labels within EACH replication cohort ----
    print(f"\nBuilding empirical null via {n_null_iter} label permutations "
          f"(preserves real gene-gene correlation structure)...")
    rng = np.random.default_rng(SEED)
    Xvals = {tag: rep_data[tag][0].loc[genes].values for tag in rep_tags}
    yvals = {tag: rep_data[tag][1] for tag in rep_tags}
    nvals = {tag: yvals[tag].shape[0] for tag in rep_tags}

    t1 = time.time()
    null_counts = np.empty(n_null_iter, dtype=int)
    for it in range(n_null_iter):
        perm = {tag: yvals[tag][rng.permutation(nvals[tag])] for tag in rep_tags}
        cnt = 0
        for j in range(len(genes)):
            if dir_disc[j] == 0:
                continue
            ok = True
            for tag in rep_tags:
                chi2_r, dr = _chi2_and_direction(Xvals[tag][j], perm[tag])
                if dr != dir_disc[j] or sps.chi2.sf(chi2_r, df=1) >= 0.05:
                    ok = False
                    break
            if ok:
                cnt += 1
        null_counts[it] = cnt
        if (it + 1) % 100 == 0:
            print(f"  {it+1}/{n_null_iter} null iterations, elapsed {time.time()-t1:.0f}s")

    observed = n_rep_both_dir
    null_mean = float(null_counts.mean())
    null_std = float(null_counts.std(ddof=1))
    empirical_p = float((np.sum(null_counts >= observed) + 1) / (n_null_iter + 1))
    fold_enrichment = observed / null_mean if null_mean > 0 else float("inf")

    result = {
        "discovery_cohort": disc_tag,
        "replication_cohorts": rep_tags,
        "n_naive_discovery_hits": len(genes),
        "old_naive_both_direction_blind": n_rep_both_naive,
        "old_naive_pct_direction_blind": round(100 * n_rep_both_naive / len(genes), 1) if genes else None,
        "observed_both_direction_concordant": observed,
        "observed_pct_direction_concordant": round(100 * observed / len(genes), 1) if genes else None,
        "empirical_null_mean": round(null_mean, 3),
        "empirical_null_std": round(null_std, 3),
        "empirical_fold_enrichment": round(fold_enrichment, 2),
        "empirical_p_value": empirical_p,
        "n_null_iterations": n_null_iter,
        "random_state": SEED,
        "runtime_seconds": round(time.time() - t1, 1),
    }
    with open(RESULTS / f"directional_enrichment_{disc_tag}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
