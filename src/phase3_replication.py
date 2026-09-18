"""Phase 2, step 2: external validation in the two independent replication
cohorts (TCGA-LUAD, GSE68465).

Primary analysis: of the genes that survived null calibration (BH-FDR<=0.05)
in the discovery cohort, how many replicate (survive their own BH-FDR<=0.05
in each replication cohort, corrected for exactly the number of survivors
being re-tested there)? With 0 discovery survivors (see
results/discovery_summary.json), this is trivially 0/0 -- reported as such,
not skipped, because "the calibration found nothing to replicate" is itself
part of the honest result.

Supplementary analysis (cheap, no permutation needed): of the 949 genes that
passed the NAIVE p<0.05 screen in discovery (i.e. what an uncalibrated
single-gene biomarker paper would report), how many are ALSO naively
significant (p<0.05, uncorrected, in that cohort alone) in each replication
cohort? This does not answer "does a real, calibration-surviving biomarker
replicate" (there are none to test) -- it answers a different, still useful
question: "how much of the naive-screen noise coincidentally lines up twice",
which is itself evidence for how thin naive replication is even before any
FDR correction is applied.

Run: python src/phase3_replication.py
Writes: results/replication_summary.json, results/naive_replication_949.csv
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
from data import load_tcga_luad_replication, load_gse68465_replication  # noqa: E402
from survival_stats import logrank_chi2  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
ALPHA = 0.05


def naive_p_for_genes(genes, X, yarr):
    X = X.loc[genes]
    observed = np.array([logrank_chi2(X.values[j], yarr) for j in range(X.shape[0])])
    return observed, sps.chi2.sf(observed, df=1)


def main():
    disc = pd.read_csv(RESULTS / "discovery_naive_vs_calibrated.csv")
    survivors = disc[disc["survives_bh"]]["gene"].tolist()
    naive_hits = disc[disc["naive_hit_p05"]]["gene"].tolist()
    print(f"Discovery: {len(naive_hits)} naive hits, {len(survivors)} BH-FDR survivors")

    X_tcga, y_tcga = load_tcga_luad_replication()
    X_gse, y_gse = load_gse68465_replication()
    yarr_tcga = y_tcga[["os_time_days", "os_event"]].values
    yarr_gse = y_gse[["os_time_months", "os_event"]].values

    # ---- primary: replicate the (empty) survivor set -----------------------
    replication_summary = {
        "n_discovery_survivors_tested": len(survivors),
        "n_replicate_in_tcga": None,
        "n_replicate_in_gse68465": None,
        "n_replicate_in_both": None,
        "pct_replicate_in_both": None,
        "note": None,
    }
    if not survivors:
        replication_summary["note"] = (
            "0 genes survived BH-FDR calibration in the discovery cohort, so there is "
            "nothing to replicate by the primary criterion -- reported as 0/0, not skipped."
        )
        replication_summary["n_replicate_in_tcga"] = 0
        replication_summary["n_replicate_in_gse68465"] = 0
        replication_summary["n_replicate_in_both"] = 0
        replication_summary["pct_replicate_in_both"] = 0.0

    # ---- supplementary: naive-to-naive replication of the 949 naive hits ---
    t0 = time.time()
    genes_tcga = [g for g in naive_hits if g in X_tcga.index]
    genes_gse = [g for g in naive_hits if g in X_gse.index]
    obs_tcga, p_tcga = naive_p_for_genes(genes_tcga, X_tcga, yarr_tcga)
    obs_gse, p_gse = naive_p_for_genes(genes_gse, X_gse, yarr_gse)
    print(f"Naive replication check done in {time.time()-t0:.1f}s")

    naive_df = disc[disc["naive_hit_p05"]][["gene", "observed_logrank_chi2", "naive_p_value"]].copy()
    naive_df = naive_df.rename(columns={"observed_logrank_chi2": "discovery_chi2",
                                         "naive_p_value": "discovery_naive_p"})
    naive_df = naive_df.set_index("gene")
    naive_df.loc[genes_tcga, "tcga_naive_p"] = p_tcga
    naive_df.loc[genes_gse, "gse68465_naive_p"] = p_gse
    naive_df["naive_replicates_tcga"] = naive_df["tcga_naive_p"] < 0.05
    naive_df["naive_replicates_gse68465"] = naive_df["gse68465_naive_p"] < 0.05
    naive_df["naive_replicates_both"] = (naive_df["naive_replicates_tcga"].fillna(False) &
                                          naive_df["naive_replicates_gse68465"].fillna(False))
    naive_df = naive_df.reset_index().sort_values("discovery_naive_p")
    naive_df.to_csv(RESULTS / "naive_replication_949.csv", index=False)

    n_hits = len(naive_hits)
    n_rep_tcga = int(naive_df["naive_replicates_tcga"].sum())
    n_rep_gse = int(naive_df["naive_replicates_gse68465"].sum())
    n_rep_both = int(naive_df["naive_replicates_both"].sum())
    n_rep_either = int((naive_df["naive_replicates_tcga"].fillna(False) |
                         naive_df["naive_replicates_gse68465"].fillna(False)).sum())

    replication_summary["supplementary_naive_to_naive"] = {
        "description": "Of the 949 naive (uncorrected p<0.05) discovery hits, how many are "
                        "ALSO naively significant (p<0.05, uncorrected) in each replication "
                        "cohort -- a weaker, non-calibrated check, reported for context.",
        "n_naive_discovery_hits": n_hits,
        "n_naive_replicate_tcga": n_rep_tcga,
        "n_naive_replicate_gse68465": n_rep_gse,
        "n_naive_replicate_both": n_rep_both,
        "n_naive_replicate_either": n_rep_either,
        "pct_naive_replicate_tcga": round(100 * n_rep_tcga / n_hits, 1),
        "pct_naive_replicate_gse68465": round(100 * n_rep_gse / n_hits, 1),
        "pct_naive_replicate_both": round(100 * n_rep_both / n_hits, 1),
        "expected_by_chance_alone_pct": 5.0 * 5.0 / 100,  # independent 5% x 5% if pure noise
    }

    with open(RESULTS / "replication_summary.json", "w") as f:
        json.dump(replication_summary, f, indent=2)
    print(json.dumps(replication_summary, indent=2))


if __name__ == "__main__":
    main()
