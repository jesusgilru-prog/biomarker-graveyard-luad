"""SurvivalPFN (Qi et al., arXiv:2605.15488, checkpoint shi-ang/SurvivalPFN) evaluated on
the same 121-gene directional panel and same train/test split as phase8_ml_survival_panel.py.

Requested by the user after a 20-subagent literature/HuggingFace search (see workflow
"biomarker-graveyard-model-search") turned up SurvivalPFN as the cheapest real candidate
to try: a prior-data fitted network that does in-context inference with NO gradient
fine-tuning on the target dataset -- fit() just conditions on the training rows, predict()
does a single forward pass. No checkpoint has ever been evaluated with 121 continuous
covariates (published benchmarks go up to ~10); this script reports whatever the real
number is, including a plausible failure to add anything over the null.

Same design as phase8 (train on TCGA-LUAD, evaluate cross-cohort on CPTAC-LUAD and
GSE68465, panel vs N size-matched random-gene-panel null) so the result is directly
comparable to the classical-ML C-index already in the manuscript (up to 0.77).

Run with the SurvivalPFN repo's own conda env (survivalpfn_cpu), NOT the project's
"biomarker" env: SurvivalPFN pins torch==2.5.1 and lacks scikit-survival, so this script
only uses numpy/pandas/scipy helpers copied from phase8 (no sksurv import at all).

Run: /home/jesus/miniconda3/envs/survivalpfn_cpu/bin/python src/phase9_survivalpfn.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats as sps

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
sys.path.insert(0, "/home/jesus/SurvivalPFN")

from data import load_cptac_luad_discovery, load_tcga_luad_replication, load_gse68465_replication  # noqa: E402
from survival_stats import _logrank_components  # noqa: E402
from survivalpfn import SurvivalEstimator, harrell_concordance_index  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
LOGS = Path(__file__).resolve().parents[1] / "logs"
SEED = 42
N_NULL = 50


def chi2_dir(x, y):
    c, o = _logrank_components(x, y)
    return c, (1 if o > 0 else (-1 if o < 0 else 0))


def compute_panel(X1, y1, X2, y2, X3, y3, candidate_genes):
    ya1 = y1[["os_time_days", "os_event"]].values
    ya2 = y2[["os_time_days", "os_event"]].values
    ya3 = y3[["os_time_months", "os_event"]].values
    genes_3cohorts = []
    for g in candidate_genes:
        c1, d1 = chi2_dir(X1.loc[g].values, ya1)
        c2, d2 = chi2_dir(X2.loc[g].values, ya2)
        c3, d3 = chi2_dir(X3.loc[g].values, ya3)
        p1, p2, p3 = sps.chi2.sf(c1, 1), sps.chi2.sf(c2, 1), sps.chi2.sf(c3, 1)
        if p1 < 0.05 and p2 < 0.05 and p3 < 0.05 and d1 == d2 == d3 and d1 != 0:
            genes_3cohorts.append(g)
    return genes_3cohorts


def build_xy(X_genes_by_samples, y_clin, genes, time_col):
    Xf = X_genes_by_samples.loc[genes].T.values.astype(float)
    time_ = y_clin[time_col].values.astype(float)
    event = y_clin["os_event"].values.astype(bool)
    return Xf, time_, event


def zscore_fit(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return mu, sd


def zscore_apply(X, mu, sd):
    return (X - mu) / sd


def run_config(genes, X1, y1, X2, y2, X3, y3, estimator_kwargs):
    Xtr, time_tr, event_tr = build_xy(X2, y2, genes, "os_time_days")
    mu, sd = zscore_fit(Xtr)
    Xtr_z = zscore_apply(Xtr, mu, sd).astype(np.float32)

    # Same time>0 requirement note as phase8 (4/508 TCGA rows have os_time_days==0);
    # SurvivalPFN's positional/log-time handling is untested at this scale, so drop
    # them from training only, exactly like phase8 does for its own time-sensitive
    # models (IPCRidge/FastSurvivalSVM/FastKernelSurvivalSVM).
    pos = time_tr > 0
    Xtr_z, time_tr, event_tr = Xtr_z[pos], time_tr[pos].astype(np.float32), event_tr[pos].astype(np.float32)

    Xte1, time1, event1 = build_xy(X1, y1, genes, "os_time_days")
    Xte1_z = zscore_apply(Xte1, mu, sd).astype(np.float32)
    Xte3, time3, event3 = build_xy(X3, y3, genes, "os_time_months")
    Xte3_z = zscore_apply(Xte3, mu, sd).astype(np.float32)

    est = SurvivalEstimator(**estimator_kwargs)
    est.fit(X=Xtr_z, T=time_tr, delta=event_tr)

    out = {}
    for name, Xte, time_te, event_te in [("cptac", Xte1_z, time1.astype(np.float32), event1.astype(np.float32)),
                                          ("gse68465", Xte3_z, time3.astype(np.float32), event3.astype(np.float32))]:
        dist = est.predict_event_distribution(Xte)
        preds = dist.median().detach().cpu().numpy()
        c = harrell_concordance_index(preds=preds, T=time_te, delta=event_te)
        out[name] = float(c)
    return out


def main():
    t0 = time.time()
    print("Loading cohorts...", flush=True)
    X1, y1 = load_cptac_luad_discovery()
    X2, y2 = load_tcga_luad_replication()
    X3, y3 = load_gse68465_replication()
    candidate_genes = sorted(set(X1.index) & set(X2.index) & set(X3.index))
    print(f"Candidate universe: {len(candidate_genes)} genes", flush=True)

    panel = compute_panel(X1, y1, X2, y2, X3, y3, candidate_genes)
    print(f"Panel (should be 121): {len(panel)} genes {'OK' if len(panel) == 121 else 'MISMATCH'}", flush=True)

    estimator_kwargs = dict(device="cpu", model_path="shi-ang/SurvivalPFN")

    print(f"\nObserved panel, training on TCGA-LUAD, evaluating on CPTAC-LUAD + GSE68465...", flush=True)
    observed = run_config(panel, X1, y1, X2, y2, X3, y3, estimator_kwargs)
    print(f"Observed: {observed}", flush=True)

    rng = np.random.default_rng(SEED)
    other_genes = [g for g in candidate_genes if g not in panel]
    null_results = []
    for i in range(N_NULL):
        rand_panel = list(rng.choice(other_genes, size=len(panel), replace=False))
        res = run_config(rand_panel, X1, y1, X2, y2, X3, y3, estimator_kwargs)
        null_results.append(res)
        print(f"  null {i+1}/{N_NULL}: {res} ({time.time()-t0:.0f}s elapsed)", flush=True)

    summary = {
        "model": "SurvivalPFN (shi-ang/SurvivalPFN, arXiv:2605.15488)",
        "panel_size": len(panel),
        "n_null": N_NULL,
        "seed": SEED,
        "observed": observed,
        "null_cptac": [r["cptac"] for r in null_results],
        "null_gse68465": [r["gse68465"] for r in null_results],
    }
    for cohort in ["cptac", "gse68465"]:
        null_vals = np.array(summary[f"null_{cohort}"])
        summary[f"{cohort}_null_mean"] = float(null_vals.mean())
        summary[f"{cohort}_null_sd"] = float(null_vals.std())
        # (b+1)/(m+1) Phipson & Smyth estimator, same convention declared in the manuscript
        # Methods and used by phase8 -- NOT naive b/m, which can report an impossible p<0.001
        # with only m=50 nulls (bug found by external review, Kimi, 2026-08-19).
        b = int((null_vals >= observed[cohort]).sum())
        summary[f"{cohort}_empirical_p"] = float((b + 1) / (len(null_vals) + 1))

    out_path = RESULTS / "phase9_survivalpfn.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWritten: {out_path}")
    print(f"Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
