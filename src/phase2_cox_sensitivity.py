"""Sensitivity analysis requested by all three external reviewers: repeat the
discovery-cohort screen with the continuous Cox score test (cox_score_chi2)
instead of median-split log-rank, to check whether "0% survive BH-FDR" is a
consequence of dichotomization throwing away power, or holds regardless of
which reasonable statistic is used.

Same adaptive 3-stage design, same panel (11,484 shared genes), same seeds.

Run: python src/phase2_cox_sensitivity.py
Writes: results/discovery_cox_naive_vs_calibrated.csv,
        results/discovery_cox_summary.json
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
from survival_stats import cox_score_chi2  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nullcal_pkg" / "src"))
from nullcal import NullCalibration  # noqa: E402
from nullcal.corrections import benjamini_hochberg, benjamini_yekutieli  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
ALPHA = 0.05
N_PERM_STAGE_B = 20_000
N_PERM_STAGE_C = 1_000_000
TOP_K_STAGE_C = 20
SEED = 42


def shared_gene_panel():
    X1, _ = load_cptac_luad_discovery()
    X2, _ = load_tcga_luad_replication()
    X3, _ = load_gse68465_replication()
    return sorted(set(X1.index) & set(X2.index) & set(X3.index))


def main():
    panel = shared_gene_panel()
    X, y = load_cptac_luad_discovery()
    X = X.loc[panel]
    yarr = y[["os_time_days", "os_event"]].values
    print(f"Cox sensitivity: n={X.shape[1]}, events={int(yarr[:,1].sum())}, genes={len(panel)}")

    # standardize expression per gene (z-score) so the Cox coefficient/HR is
    # interpretable as "per 1 SD" -- does not affect the score test's p-value
    # (score test at beta=0 is invariant to linear rescaling of x), but keeps
    # HRs comparable across genes if reported later.
    Xvals = X.values
    Xvals = (Xvals - Xvals.mean(axis=1, keepdims=True)) / Xvals.std(axis=1, keepdims=True)

    t0 = time.time()
    observed = np.array([cox_score_chi2(Xvals[j], yarr) for j in range(Xvals.shape[0])])
    naive_p = sps.chi2.sf(observed, df=1)
    naive_hit = naive_p < 0.05
    n_naive = int(naive_hit.sum())
    print(f"Stage A: {n_naive}/{len(panel)} naive hits ({time.time()-t0:.1f}s)")

    escalate_idx = np.where(naive_hit)[0]
    escalate_genes = [panel[i] for i in escalate_idx]
    combined_p = naive_p.copy()

    if escalate_genes:
        t1 = time.time()
        nc = NullCalibration(cox_score_chi2, n_permutations=N_PERM_STAGE_B, method="permutation",
                              correction="none", alpha=ALPHA, random_state=SEED)
        res_b = nc.calibrate(Xvals[escalate_idx].T, yarr, names=escalate_genes)
        for i, idx in enumerate(escalate_idx):
            combined_p[idx] = res_b.p_value[i]
        print(f"Stage B: {N_PERM_STAGE_B} perms x {len(escalate_genes)} genes ({time.time()-t1:.1f}s)")

        order = np.argsort(combined_p[escalate_idx])
        topk_idx = escalate_idx[order[:min(TOP_K_STAGE_C, len(escalate_idx))]]
        topk_genes = [panel[i] for i in topk_idx]
        t2 = time.time()
        nc_c = NullCalibration(cox_score_chi2, n_permutations=N_PERM_STAGE_C, method="permutation",
                                correction="none", alpha=ALPHA, random_state=SEED)
        res_c = nc_c.calibrate(Xvals[topk_idx].T, yarr, names=topk_genes)
        for i, idx in enumerate(topk_idx):
            combined_p[idx] = res_c.p_value[i]
        print(f"Stage C: {N_PERM_STAGE_C} perms x {len(topk_genes)} genes ({time.time()-t2:.1f}s)")

    q_bh = benjamini_hochberg(combined_p)
    q_by = benjamini_yekutieli(combined_p)
    survives_bh = q_bh <= ALPHA
    survives_by = q_by <= ALPHA
    n_bh = int(survives_bh.sum())
    n_by = int(survives_by.sum())

    df = pd.DataFrame({
        "gene": panel, "observed_cox_score_chi2": observed, "naive_p_value": naive_p,
        "naive_hit_p05": naive_hit, "empirical_p_value": combined_p,
        "q_value_bh": q_bh, "survives_bh": survives_bh,
        "q_value_by": q_by, "survives_by": survives_by,
    }).sort_values("empirical_p_value")
    df.to_csv(RESULTS / "discovery_cox_naive_vs_calibrated.csv", index=False)

    summary = {
        "statistic": "cox_score_chi2 (continuous, z-scored expression)",
        "n_patients": int(X.shape[1]), "n_events": int(yarr[:, 1].sum()),
        "n_candidate_genes": len(panel), "n_naive_hits_p05": n_naive,
        "n_survive_bh_fdr05": n_bh, "n_survive_by_fdr05": n_by,
        "pct_naive_surviving_bh": round(100 * n_bh / n_naive, 2) if n_naive else None,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    with open(RESULTS / "discovery_cox_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
