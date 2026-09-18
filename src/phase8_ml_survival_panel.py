"""Machine-learning survival analysis on the 121-gene directional panel.

Complementary to the univariate/directional-replication analysis (Sections 3.1-3.5 of
the manuscript): tests whether the panel carries genuine MULTIVARIABLE, jointly-learned
predictive signal, not just per-gene directional concordance. Uses the exact same raw
cohort loaders and the exact same 121-gene panel definition as
src/audit_reproducibility.py (median-split log-rank, p<0.05, concordant direction in all
three cohorts) -- no new gene selection, no leakage between panel definition and this
evaluation's training/test split.

Design (train on the largest, most-events cohort; evaluate cross-cohort, never
within-cohort, to avoid the small-n overfitting optimism a k-fold CV on n=106 would
give):
  - Train three model classes on TCGA-LUAD (n=508, 183 events) using the 121 panel
    genes as features (z-scored on the training cohort only, applied unchanged to the
    two held-out cohorts -- no leakage of test-cohort statistics into scaling):
      1. Random Survival Forest (sksurv.ensemble.RandomSurvivalForest)
      2. Gradient Boosting Survival Analysis (sksurv.ensemble.GradientBoostingSurvivalAnalysis)
      3. Elastic-net-penalized Cox (lifelines.CoxPHFitter, penalizer + l1_ratio)
  - Evaluate Harrell's concordance index on CPTAC-LUAD and GSE68465 (held out, never
    seen during training or feature scaling).
  - Null: repeat with N random size-121 gene panels drawn from the same 11,484-gene
    candidate universe (same train/test split, same scaling procedure, same model),
    to test whether the PANEL specifically carries more cross-cohort predictive signal
    than an arbitrary same-size gene set -- the same logic as the gene-resampling null
    already used for the univariate results (Section 3.2), applied here to a
    multivariable model instead of a single-gene test.

Run: python src/phase8_ml_survival_panel.py
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
from survival_stats import _logrank_components  # noqa: E402

from sksurv.ensemble import (RandomSurvivalForest, ExtraSurvivalTrees,
                              GradientBoostingSurvivalAnalysis,
                              ComponentwiseGradientBoostingSurvivalAnalysis)
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis, IPCRidge
from sksurv.svm import FastSurvivalSVM, FastKernelSurvivalSVM, MinlipSurvivalAnalysis
from sksurv.tree import SurvivalTree
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
from lifelines import CoxPHFitter

RESULTS = Path(__file__).resolve().parents[1] / "results"
LOGS = Path(__file__).resolve().parents[1] / "logs"

N_NULL = 50
SEED = 42
N_TREES = 300


def _make_models(rng_state):
    """11 distinct survival-model families, all appropriate for n~100-500,
    p~121 -- no deep-learning/foundation models (see manuscript Limitations:
    architecturally and by sample size inappropriate at this scale)."""
    return {
        "rsf": RandomSurvivalForest(n_estimators=N_TREES, min_samples_leaf=8,
                                     n_jobs=2, random_state=rng_state),
        "extra_trees": ExtraSurvivalTrees(n_estimators=N_TREES, min_samples_leaf=8,
                                           n_jobs=2, random_state=rng_state),
        "gbsa": GradientBoostingSurvivalAnalysis(n_estimators=150, learning_rate=0.05,
                                                  max_depth=2, subsample=0.8,
                                                  random_state=rng_state),
        "cw_gbsa": ComponentwiseGradientBoostingSurvivalAnalysis(
            n_estimators=150, learning_rate=0.1, random_state=rng_state),
        "survival_tree": SurvivalTree(min_samples_leaf=12, random_state=rng_state),
        "cox_ph": CoxPHSurvivalAnalysis(alpha=1.0),
        "coxnet": CoxnetSurvivalAnalysis(l1_ratio=0.5, alpha_min_ratio=0.01),
        "ipc_ridge": IPCRidge(alpha=1.0),
        "fast_svm": FastSurvivalSVM(max_iter=100, random_state=rng_state),
        "fast_kernel_svm": FastKernelSurvivalSVM(kernel="rbf", max_iter=100,
                                                  random_state=rng_state),
        "minlip": MinlipSurvivalAnalysis(),
    }


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
    """X_genes_by_samples: genes x samples. Returns (X samples x genes ndarray,
    sksurv structured y array)."""
    Xf = X_genes_by_samples.loc[genes].T.values.astype(float)
    time = y_clin[time_col].values.astype(float)
    event = y_clin["os_event"].values.astype(bool)
    y_struct = Surv.from_arrays(event=event, time=time)
    return Xf, y_struct, time, event


def zscore_fit(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return mu, sd


def zscore_apply(X, mu, sd):
    return (X - mu) / sd


def cindex(time, event, risk_scores):
    return concordance_index_censored(event, time, risk_scores)[0]


def eval_one(name, model, Xtr, ytr_struct, Xte, time_te, event_te):
    try:
        model.fit(Xtr, ytr_struct)
        risk = model.predict(Xte)
        return cindex(time_te, event_te, risk)
    except Exception as e:
        return None


def main():
    t0 = time.time()
    print("Loading cohorts...")
    X1, y1 = load_cptac_luad_discovery()
    X2, y2 = load_tcga_luad_replication()
    X3, y3 = load_gse68465_replication()
    candidate_genes = sorted(set(X1.index) & set(X2.index) & set(X3.index))
    print(f"Candidate universe: {len(candidate_genes)} genes")

    panel = compute_panel(X1, y1, X2, y2, X3, y3, candidate_genes)
    print(f"Panel (should be 121): {len(panel)} genes"
          f" {'OK' if len(panel) == 121 else 'MISMATCH'}")

    model_names = list(_make_models(SEED).keys())
    print(f"Models ({len(model_names)}): {model_names}")

    rng = np.random.default_rng(SEED)

    # 4/508 TCGA-LUAD training samples have os_time_days == 0 (real data: death or
    # censoring recorded on day of diagnosis). Harmless for every model here except
    # the three that require strictly positive time internally (log-time transform):
    # IPCRidge, FastSurvivalSVM, FastKernelSurvivalSVM. For those three only, fit on
    # the 504/508 samples with time > 0; every other model, and all evaluation
    # (test-cohort) data, is untouched.
    POSITIVE_TIME_ONLY = {"ipc_ridge", "fast_svm", "fast_kernel_svm"}

    def run_config(genes):
        Xtr, ytr_struct, time_tr, event_tr = build_xy(X2, y2, genes, "os_time_days")
        mu, sd = zscore_fit(Xtr)
        Xtr_z = zscore_apply(Xtr, mu, sd)

        pos_mask = time_tr > 0
        Xtr_z_pos = Xtr_z[pos_mask]
        ytr_struct_pos = Surv.from_arrays(event=event_tr[pos_mask], time=time_tr[pos_mask])

        Xte1, _, time1, event1 = build_xy(X1, y1, genes, "os_time_days")
        Xte1_z = zscore_apply(Xte1, mu, sd)
        Xte3, _, time3, event3 = build_xy(X3, y3, genes, "os_time_months")
        Xte3_z = zscore_apply(Xte3, mu, sd)

        out = {}
        for name, model in _make_models(SEED).items():
            tr_X = Xtr_z_pos if name in POSITIVE_TIME_ONLY else Xtr_z
            tr_y = ytr_struct_pos if name in POSITIVE_TIME_ONLY else ytr_struct
            out[f"{name}_cptac"] = eval_one(name, model, tr_X, tr_y, Xte1_z,
                                             time1, event1)
        for name, model in _make_models(SEED).items():
            tr_X = Xtr_z_pos if name in POSITIVE_TIME_ONLY else Xtr_z
            tr_y = ytr_struct_pos if name in POSITIVE_TIME_ONLY else ytr_struct
            out[f"{name}_gse68465"] = eval_one(name, model, tr_X, tr_y, Xte3_z,
                                                time3, event3)

        # elastic-net Cox via lifelines (independent implementation, kept from v1)
        try:
            import pandas as pd
            df = pd.DataFrame(Xtr_z, columns=[f"g{i}" for i in range(Xtr_z.shape[1])])
            df["time"], df["event"] = time_tr, event_tr
            cph = CoxPHFitter(penalizer=0.1, l1_ratio=0.5)
            cph.fit(df, duration_col="time", event_col="event")
            for suffix, Xte_z, t_te, e_te in [("cptac", Xte1_z, time1, event1),
                                                ("gse68465", Xte3_z, time3, event3)]:
                dfte = pd.DataFrame(Xte_z, columns=[f"g{i}" for i in range(Xte_z.shape[1])])
                risk = cph.predict_partial_hazard(dfte).values
                out[f"coxnet_lifelines_{suffix}"] = cindex(t_te, e_te, risk)
        except Exception:
            out["coxnet_lifelines_cptac"] = None
            out["coxnet_lifelines_gse68465"] = None
        return out

    print("\nObserved panel, training on TCGA-LUAD, evaluating on CPTAC-LUAD + GSE68465...")
    observed = run_config(panel)
    for k, v in observed.items():
        print(f"  {k}: {v}")

    print(f"\nRunning {N_NULL} size-matched random-gene-panel null iterations "
          f"({len(observed)} metrics/iteration, this takes a while)...")
    null_records = {k: [] for k in observed}
    other_genes = [g for g in candidate_genes]
    for i in range(N_NULL):
        rand_panel = list(rng.choice(other_genes, size=len(panel), replace=False))
        res = run_config(rand_panel)
        for k in null_records:
            v = res.get(k)
            if v is not None:
                null_records[k].append(v)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{N_NULL} done, {time.time() - t0:.0f}s elapsed")

    summary = {"panel_size": len(panel), "n_null": N_NULL, "seed": SEED,
               "n_trees": N_TREES, "models": model_names + ["coxnet_lifelines"],
               "observed": observed, "null": {}}
    for k, obs_v in observed.items():
        if obs_v is None or len(null_records[k]) < 5:
            continue
        nulls = np.array(null_records[k])
        p_emp = (np.sum(nulls >= obs_v) + 1) / (len(nulls) + 1)
        summary["null"][k] = {
            "null_mean": float(np.mean(nulls)), "null_sd": float(np.std(nulls)),
            "null_min": float(np.min(nulls)), "null_max": float(np.max(nulls)),
            "empirical_p": float(p_emp), "n_valid": len(nulls),
        }
        print(f"{k}: observed={obs_v:.4f}, null mean={np.mean(nulls):.4f} "
              f"(sd={np.std(nulls):.4f}), empirical p={p_emp:.4f}")

    RESULTS.mkdir(exist_ok=True)
    out_path = RESULTS / "phase8_ml_survival_panel.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWritten: {out_path}")
    print(f"Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
