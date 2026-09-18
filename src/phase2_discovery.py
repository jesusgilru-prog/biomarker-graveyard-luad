"""Phase 2, step 1: naive screening vs null-calibrated screening on the
discovery cohort (CPTAC-LUAD).

Candidate panel: the 11,484 genes present in all three cohorts (discovery +
2 replications) -- restricting to this shared panel from the start is what
makes "does it replicate" a well-posed question later; it is documented in
data/DATASHEET.md, not a silent choice.

ADAPTIVE two-stage permutation design (a standard technique, e.g. PLINK's
--mperm adaptive mode / Besag & Clifford 1991), needed because a flat
n_permutations=1000 run showed the problem directly: only 38/11484 genes hit
the empirical-p floor (1/1001 ~ 0.001), while surviving BH-FDR at N=11,484
tests requires resolution far below that for genes ranked outside the top
~230 -- so a flat run under-resolves the very p-values the correction needs.
Running 20,000 permutations on all 11,484 genes would take ~18 hours; that is
wasteful, because the ~10,535 genes with naive (asymptotic chi2(1)) p >= 0.05
mathematically cannot survive BH/BY at alpha=0.05 (BH's threshold at the
largest possible rank N equals alpha itself, and here only 949/11484 = 8.3%
of tests are naively significant, so no p >= 0.05 test can ever be rejected).
n_permutations=20,000 (empirical-p floor ~5e-5) is enough resolution: BH's
step-up rule lets a *cluster* of k tied-at-the-floor genes survive together
as long as floor <= alpha*k/N, and the flat 1000-permutation run already
showed 38 genes saturating its (coarser) floor -- i.e. a tied cluster of that
size clears the 20,000-permutation floor with room to spare (need only
floor <= 0.05*38/11484 ~ 1.65e-4, and 5e-5 comfortably beats that).

So: Stage A computes the observed statistic and naive analytic p-value for
ALL 11,484 genes directly (cheap, no permutation). Stage B runs nullcal's
actual permutation calibration (still nullcal.NullCalibration.calibrate(),
not a reimplementation) at n_permutations=20,000 -- but ONLY on the 949 genes
that passed the naive p<0.05 screen. Their high-precision empirical p-values
replace the naive ones; the other 10,535 keep their (already sufficient,
guaranteed-non-survivor) naive p-value. BH/BY correction then runs once on
the combined 11,484-length p-value vector, which is what the correction
functions need to see the true multiple-testing burden.

Run: python src/phase2_discovery.py
Writes: results/discovery_naive_vs_calibrated.csv, results/discovery_summary.json
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
from survival_stats import logrank_chi2  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nullcal_pkg" / "src"))
from nullcal import NullCalibration  # noqa: E402
from nullcal.corrections import benjamini_hochberg, benjamini_yekutieli  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
ALPHA = 0.05
N_PERMUTATIONS_ESCALATED = 20_000
SEED = 42


def shared_gene_panel():
    X1, _ = load_cptac_luad_discovery()
    X2, _ = load_tcga_luad_replication()
    X3, _ = load_gse68465_replication()
    common = sorted(set(X1.index) & set(X2.index) & set(X3.index))
    return common


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    panel = shared_gene_panel()
    print(f"Candidate panel (genes shared by all 3 cohorts): {len(panel)}")

    X, y = load_cptac_luad_discovery()
    X = X.loc[panel]
    yarr = y[["os_time_days", "os_event"]].values
    Xvals = X.values  # genes x samples

    # ---- Stage A: observed statistic + naive analytic p-value, ALL genes ----
    t0 = time.time()
    observed = np.array([logrank_chi2(Xvals[j], yarr) for j in range(Xvals.shape[0])])
    naive_p = sps.chi2.sf(observed, df=1)
    naive_hit = naive_p < 0.05
    n_naive = int(naive_hit.sum())
    print(f"Stage A done in {time.time()-t0:.1f}s. Naive hits (p<0.05): {n_naive}/{len(panel)}")

    # ---- Stage B: high-resolution permutation calibration, naive hits only ----
    escalate_idx = np.where(naive_hit)[0]
    escalate_genes = [panel[i] for i in escalate_idx]
    X_esc = Xvals[escalate_idx]

    t1 = time.time()
    nc = NullCalibration(logrank_chi2, n_permutations=N_PERMUTATIONS_ESCALATED,
                          method="permutation", correction="none", alpha=ALPHA, random_state=SEED)
    res_esc = nc.calibrate(X_esc.T, yarr, names=escalate_genes)
    print(f"Stage B done in {time.time()-t1:.1f}s ({N_PERMUTATIONS_ESCALATED} perms x {n_naive} genes)")

    # ---- combine: escalated genes get their high-res empirical p; the rest
    # keep their naive analytic p (already >= 0.05, guaranteed non-survivors) --
    combined_p = naive_p.copy()
    empirical_p_full = naive_p.copy()  # for genes not escalated, report naive as-is
    for i, idx in enumerate(escalate_idx):
        combined_p[idx] = res_esc.p_value[i]
        empirical_p_full[idx] = res_esc.p_value[i]

    q_bh = benjamini_hochberg(combined_p)
    q_by = benjamini_yekutieli(combined_p)
    survives_bh = q_bh <= ALPHA
    survives_by = q_by <= ALPHA

    effect_above_null = np.full(len(panel), np.nan)
    null_mean = np.full(len(panel), np.nan)
    null_std = np.full(len(panel), np.nan)
    for i, idx in enumerate(escalate_idx):
        effect_above_null[idx] = res_esc.effect_above_null[i]
        null_mean[idx] = res_esc.null_mean[i]
        null_std[idx] = res_esc.null_std[i]

    df = pd.DataFrame({
        "gene": panel,
        "observed_logrank_chi2": observed,
        "naive_p_value": naive_p,
        "naive_hit_p05": naive_hit,
        "escalated_to_high_res_permutation": naive_hit,
        "n_permutations_used": np.where(naive_hit, N_PERMUTATIONS_ESCALATED, 0),
        "null_mean": null_mean,
        "null_std": null_std,
        "effect_above_null": effect_above_null,
        "empirical_p_value": empirical_p_full,
        "q_value_bh": q_bh,
        "survives_bh": survives_bh,
        "q_value_by": q_by,
        "survives_by": survives_by,
    }).sort_values("empirical_p_value")

    df.to_csv(RESULTS / "discovery_naive_vs_calibrated.csv", index=False)

    n_bh = int(survives_bh.sum())
    n_by = int(survives_by.sum())
    summary = {
        "cohort": "CPTAC-LUAD (discovery)",
        "n_patients": int(X.shape[1]),
        "n_events": int(yarr[:, 1].sum()),
        "n_candidate_genes": len(panel),
        "n_naive_hits_p05": n_naive,
        "n_escalated_to_high_res_permutation": n_naive,
        "n_permutations_escalated": N_PERMUTATIONS_ESCALATED,
        "random_state": SEED,
        "n_survive_bh_fdr05": n_bh,
        "n_survive_by_fdr05": n_by,
        "pct_naive_surviving_bh": round(100 * n_bh / n_naive, 2) if n_naive else None,
        "pct_naive_surviving_by": round(100 * n_by / n_naive, 2) if n_naive else None,
        "runtime_seconds_stageA": round(time.time() - t0, 1),
        "runtime_seconds_stageB": round(time.time() - t1, 1),
    }
    with open(RESULTS / "discovery_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
