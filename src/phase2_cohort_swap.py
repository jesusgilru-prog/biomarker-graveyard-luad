"""Power-vs-events sensitivity check requested independently by both Fable and
Codex: rerun the exact same naive-screen + adaptive-permutation-calibration
pipeline (src/phase2_discovery.py logic), but using each of the three cohorts
as "discovery" in turn. If the number of BH-FDR survivors increases with the
number of events (CPTAC 24 < TCGA ~101-180 < GSE68465 236), that is direct,
quantitative evidence that the discovery cohort's "0 survivors" result is a
function of event-limited power, not of biomarkers being fake -- exactly the
experiment both external reviewers proposed to settle question 3.

Uses the SAME 11,484-gene shared panel and the SAME adaptive 3-stage design
(naive screen -> escalate naive hits to 20,000 perms -> escalate top-20
boundary genes to 1,000,000 perms) for a fair comparison across cohorts.

Run: python src/phase2_cohort_swap.py --discovery tcga
Run: python src/phase2_cohort_swap.py --discovery gse68465
(--discovery cptac reproduces src/phase2_discovery.py's original result)

Writes: results/cohort_swap_<discovery>_summary.json,
        results/cohort_swap_<discovery>_naive_vs_calibrated.csv
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
from survival_stats import logrank_chi2  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nullcal_pkg" / "src"))
from nullcal import NullCalibration  # noqa: E402
from nullcal.corrections import benjamini_hochberg, benjamini_yekutieli  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
ALPHA = 0.05
TOP_K_STAGE_C = 20
SEED = 42

COHORTS = {
    "cptac": (load_cptac_luad_discovery, "os_time_days"),
    "tcga": (load_tcga_luad_replication, "os_time_days"),
    "gse68465": (load_gse68465_replication, "os_time_months"),
}


def shared_gene_panel():
    X1, _ = load_cptac_luad_discovery()
    X2, _ = load_tcga_luad_replication()
    X3, _ = load_gse68465_replication()
    return sorted(set(X1.index) & set(X2.index) & set(X3.index))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--discovery", required=True, choices=list(COHORTS.keys()))
    ap.add_argument("--n-perm-b", type=int, default=20_000)
    ap.add_argument("--n-perm-c", type=int, default=1_000_000)
    args = ap.parse_args()
    tag = args.discovery
    n_perm_b = args.n_perm_b
    n_perm_c = args.n_perm_c

    panel = shared_gene_panel()
    loader, time_col = COHORTS[tag]
    X, y = loader()
    X = X.loc[panel]
    yarr = y[[time_col, "os_event"]].values
    n_events = int(yarr[:, 1].sum())
    print(f"Discovery = {tag}: n={X.shape[1]}, events={n_events}, genes={len(panel)}")

    Xvals = X.values
    t0 = time.time()
    observed = np.array([logrank_chi2(Xvals[j], yarr) for j in range(Xvals.shape[0])])
    naive_p = sps.chi2.sf(observed, df=1)
    naive_hit = naive_p < 0.05
    n_naive = int(naive_hit.sum())
    print(f"Stage A: {n_naive}/{len(panel)} naive hits ({time.time()-t0:.1f}s)")

    escalate_idx = np.where(naive_hit)[0]
    escalate_genes = [panel[i] for i in escalate_idx]
    combined_p = naive_p.copy()

    if escalate_genes:
        t1 = time.time()
        nc = NullCalibration(logrank_chi2, n_permutations=n_perm_b, method="permutation",
                              correction="none", alpha=ALPHA, random_state=SEED)
        res_b = nc.calibrate(Xvals[escalate_idx].T, yarr, names=escalate_genes)
        for i, idx in enumerate(escalate_idx):
            combined_p[idx] = res_b.p_value[i]
        print(f"Stage B: {n_perm_b} perms x {len(escalate_genes)} genes ({time.time()-t1:.1f}s)")

        # Stage C: refine the top-K boundary genes with much higher resolution
        order = np.argsort(combined_p[escalate_idx])
        topk_idx = escalate_idx[order[:min(TOP_K_STAGE_C, len(escalate_idx))]]
        topk_genes = [panel[i] for i in topk_idx]
        t2 = time.time()
        nc_c = NullCalibration(logrank_chi2, n_permutations=n_perm_c, method="permutation",
                                correction="none", alpha=ALPHA, random_state=SEED)
        res_c = nc_c.calibrate(Xvals[topk_idx].T, yarr, names=topk_genes)
        for i, idx in enumerate(topk_idx):
            combined_p[idx] = res_c.p_value[i]
        print(f"Stage C: {n_perm_c} perms x {len(topk_genes)} genes ({time.time()-t2:.1f}s)")

    q_bh = benjamini_hochberg(combined_p)
    q_by = benjamini_yekutieli(combined_p)
    survives_bh = q_bh <= ALPHA
    survives_by = q_by <= ALPHA
    n_bh = int(survives_bh.sum())
    n_by = int(survives_by.sum())

    df = pd.DataFrame({
        "gene": panel, "observed_logrank_chi2": observed, "naive_p_value": naive_p,
        "naive_hit_p05": naive_hit, "empirical_p_value": combined_p,
        "q_value_bh": q_bh, "survives_bh": survives_bh,
        "q_value_by": q_by, "survives_by": survives_by,
    }).sort_values("empirical_p_value")
    df.to_csv(RESULTS / f"cohort_swap_{tag}_naive_vs_calibrated.csv", index=False)

    summary = {
        "discovery_cohort": tag, "n_patients": int(X.shape[1]), "n_events": n_events,
        "n_candidate_genes": len(panel), "n_naive_hits_p05": n_naive,
        "n_survive_bh_fdr05": n_bh, "n_survive_by_fdr05": n_by,
        "pct_naive_surviving_bh": round(100 * n_bh / n_naive, 2) if n_naive else None,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    with open(RESULTS / f"cohort_swap_{tag}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
