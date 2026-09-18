"""Phase 2, stage C: final high-resolution pass on the genes closest to the
BH decision boundary.

After stage B (20,000 permutations on the 949 naive hits), the *optimistic*
best case -- using each gene's naive asymptotic p-value as a lower bound on
what more permutations could reveal -- showed rank 3 sitting only 1.2x above
the BH(alpha=0.05) threshold. That is too close to leave unresolved at a
20,000-permutation floor (5e-5, itself already above the rank-1..3 thresholds
of ~4e-6..1.3e-5). This stage re-tests just the top 20 genes (by stage-B
empirical p) at n_permutations=1,000,000 (floor ~1e-6) to get a definitive
answer -- fast, because only 20 genes are involved (~1.6h).

This is the standard adaptive-permutation pattern taken one level further:
keep escalating precision only for candidates near the decision boundary,
not for the bulk that is already clearly non-significant.

Run: python src/phase2_discovery_stageC.py
Updates: results/discovery_naive_vs_calibrated.csv, results/discovery_summary.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_cptac_luad_discovery  # noqa: E402
from survival_stats import logrank_chi2  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nullcal_pkg" / "src"))
from nullcal import NullCalibration  # noqa: E402
from nullcal.corrections import benjamini_hochberg, benjamini_yekutieli  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
ALPHA = 0.05
N_PERMUTATIONS_STAGE_C = 1_000_000
TOP_K = 20
SEED = 42


def main():
    df = pd.read_csv(RESULTS / "discovery_naive_vs_calibrated.csv")
    top = df.sort_values("empirical_p_value").head(TOP_K)
    genes = top["gene"].tolist()
    print(f"Stage C: re-testing top {len(genes)} genes at {N_PERMUTATIONS_STAGE_C} permutations")
    print(genes)

    X, y = load_cptac_luad_discovery()
    X = X.loc[genes]
    yarr = y[["os_time_days", "os_event"]].values

    t0 = time.time()
    nc = NullCalibration(logrank_chi2, n_permutations=N_PERMUTATIONS_STAGE_C,
                          method="permutation", correction="none", alpha=ALPHA, random_state=SEED)
    res = nc.calibrate(X.values.T, yarr, names=genes)
    print(f"Stage C done in {time.time()-t0:.1f}s")

    stageC_p = dict(zip(genes, res.p_value))
    df = df.set_index("gene")
    for g, p in stageC_p.items():
        df.loc[g, "empirical_p_value"] = p
        df.loc[g, "n_permutations_used"] = N_PERMUTATIONS_STAGE_C
    df = df.reset_index()

    q_bh = benjamini_hochberg(df["empirical_p_value"].values)
    q_by = benjamini_yekutieli(df["empirical_p_value"].values)
    df["q_value_bh"] = q_bh
    df["q_value_by"] = q_by
    df["survives_bh"] = q_bh <= ALPHA
    df["survives_by"] = q_by <= ALPHA
    df = df.sort_values("empirical_p_value")
    df.to_csv(RESULTS / "discovery_naive_vs_calibrated.csv", index=False)

    n_naive = int(df["naive_hit_p05"].sum())
    n_bh = int(df["survives_bh"].sum())
    n_by = int(df["survives_by"].sum())

    with open(RESULTS / "discovery_summary.json") as f:
        summary = json.load(f)
    summary.update({
        "stage_C_top_k_genes": TOP_K,
        "stage_C_n_permutations": N_PERMUTATIONS_STAGE_C,
        "n_survive_bh_fdr05": n_bh,
        "n_survive_by_fdr05": n_by,
        "pct_naive_surviving_bh": round(100 * n_bh / n_naive, 3) if n_naive else None,
        "pct_naive_surviving_by": round(100 * n_by / n_naive, 3) if n_naive else None,
        "runtime_seconds_stageC": round(time.time() - t0, 1),
    })
    with open(RESULTS / "discovery_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(df.head(10)[["gene", "observed_logrank_chi2", "naive_p_value", "empirical_p_value",
                        "q_value_bh", "q_value_by", "survives_bh", "survives_by"]].to_string())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
