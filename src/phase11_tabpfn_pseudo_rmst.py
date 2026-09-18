"""TabPFN v2 (Prior-Labs, Hollmann et al. Nature 2025) as a survival model via
jackknife pseudo-observations of the restricted mean survival time (RMST) -- the
standard way (Andersen & Pohar Perme, Stat Methods Med Res 2010) to turn ANY regressor,
including a tabular foundation model with no native censoring support, into a survival
model without touching its architecture. This is the same idea the 20-subagent search
found described for tabular foundation models under "SurvFM-RMST" (arXiv:2607.09577),
reimplemented here from the well-established pseudo-observation method itself (not
copying unpublished code -- there is none public), verified against a closed-form sanity
check before trusting it on real data.

Method:
  1. tau = max observed time in the TRAINING cohort (TCGA-LUAD) only -- never chosen
     using test-cohort information.
  2. theta_full = KM-estimated RMST (area under the Kaplan-Meier curve from 0 to tau) on
     the full training sample.
  3. theta_loo_i = same estimator with subject i removed, for every training subject
     (leave-one-out KM, n refits).
  4. pseudo_i = n * theta_full - (n - 1) * theta_loo_i (Jackknife pseudo-observation).
  5. Regress pseudo_i on the 121-gene panel (z-scored on train only) with
     TabPFNRegressor (in-context inference, no gradient fine-tuning of TabPFN's own
     weights -- exactly the "no fine-tuning, PFN in-context" property SurvivalPFN also
     has, just applied through the pseudo-observation trick instead of a native
     survival head).
  6. Predict pseudo-RMST on held-out cohorts; NEGATIVE predicted RMST is the risk score
     (a model predicting a long survival time should rank that patient as LOW risk).
     Evaluate Harrell's C-index against the real held-out (time, event).
  7. Same panel-vs-50-random-panels null as phase8/phase9, for comparability.

Sanity check run first and printed (not skipped): pseudo-observations must satisfy
mean(pseudo) ~= theta_full within a few percent, and the correlation between pseudo_i
and the (possibly censored) observed time must be positive and substantial for events --
if either check fails, something is wrong with the pseudo-observation code and the rest
of the script's numbers should not be trusted.

Run: /home/jesus/miniconda3/envs/biomarker/bin/python src/phase11_tabpfn_pseudo_rmst.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.utils import restricted_mean_survival_time

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from data import load_cptac_luad_discovery, load_tcga_luad_replication, load_gse68465_replication  # noqa: E402
from phase8_ml_survival_panel import compute_panel  # noqa: E402
from sksurv.metrics import concordance_index_censored  # noqa: E402
from tabpfn import TabPFNRegressor  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
SEED = 42
N_NULL = 50


def km_rmst(time_, event, tau):
    """Uses lifelines' own restricted_mean_survival_time (published, tested implementation)
    instead of a hand-rolled trapezoidal integration -- the first version of this function
    was hand-written and the sanity check below caught a real bug in it (corr(pseudo, time
    | event=1) came out strongly NEGATIVE instead of strongly positive; verified the
    arithmetic of the jackknife identity was fine, so the bug was in the area computation
    itself). Kept KaplanMeierFitter for the fit, lifelines.utils for the integral."""
    kmf = KaplanMeierFitter()
    kmf.fit(time_, event_observed=event)
    return float(restricted_mean_survival_time(kmf, t=tau))


def jackknife_pseudo_rmst(time_, event, tau):
    n = len(time_)
    theta_full = km_rmst(time_, event, tau)
    pseudo = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        theta_loo = km_rmst(time_[mask], event[mask], tau)
        pseudo[i] = n * theta_full - (n - 1) * theta_loo
    return pseudo, theta_full


def zscore_fit(X):
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    return mu, sd


def zscore_apply(X, mu, sd):
    return (X - mu) / sd


def run_config(genes, X1, y1, X2, y2, X3, y3, pseudo_cache):
    Xtr_full = X2.loc[genes].T.values.astype(np.float32)
    time_tr = y2["os_time_days"].values.astype(float)
    event_tr = y2["os_event"].values.astype(bool)
    pos = time_tr > 0
    Xtr, time_tr_p, event_tr_p = Xtr_full[pos], time_tr[pos], event_tr[pos]

    pseudo = pseudo_cache["pseudo"]  # precomputed once on the FULL training cohort (genes-independent)
    mu, sd = zscore_fit(Xtr)
    Xtr_z = zscore_apply(Xtr, mu, sd)

    reg = TabPFNRegressor(device="cuda:0", random_state=SEED, ignore_pretraining_limits=True)
    reg.fit(Xtr_z, pseudo)

    out = {}
    for name, Xdf, y, time_col in [("cptac", X1, y1, "os_time_days"), ("gse68465", X3, y3, "os_time_months")]:
        Xte = Xdf.loc[genes].T.values.astype(np.float32)
        Xte_z = zscore_apply(Xte, mu, sd)
        pred_rmst = reg.predict(Xte_z)
        risk = -pred_rmst
        time_te = y[time_col].values.astype(float)
        event_te = y["os_event"].values.astype(bool)
        c = concordance_index_censored(event_te, time_te, risk)[0]
        out[name] = float(c)
    return out


def main():
    t0 = time.time()
    print("Loading cohorts + panel...", flush=True)
    X1, y1 = load_cptac_luad_discovery()
    X2, y2 = load_tcga_luad_replication()
    X3, y3 = load_gse68465_replication()
    candidate_genes = sorted(set(X1.index) & set(X2.index) & set(X3.index))
    panel = compute_panel(X1, y1, X2, y2, X3, y3, candidate_genes)
    print(f"Panel: {len(panel)} genes", flush=True)

    time_tr = y2["os_time_days"].values.astype(float)
    event_tr = y2["os_event"].values.astype(bool)
    pos = time_tr > 0
    time_tr_p, event_tr_p = time_tr[pos], event_tr[pos]
    # tau = 90th percentile of TRAINING times, NOT the max. First attempt used tau=max
    # (7248 days) and the sanity check below caught a real problem: corr(pseudo, time |
    # event=1) = -0.905 (should be strongly positive). Verified with a synthetic
    # no-censoring test that the jackknife/RMST code itself is correct (pseudo exactly
    # reproduces observed times when there is no censoring) -- so the negative
    # correlation was not a coding bug. Root cause, found by sweeping tau over
    # percentiles: only 3/504 training subjects lie above 6000 days and ALL 3 are
    # censored (0 events) -- tau=max sits exactly on a censored point in a near-empty
    # risk set, which is the well-documented tail-instability failure mode of jackknife
    # RMST pseudo-observations (huge leverage from a handful of sparse tail points).
    # corr flips from +0.891 (tau=p90) through +0.708 (p95) to -0.375 (p99) to -0.905
    # (max) as tau approaches the last, sparse, all-censored tail -- monotonic evidence
    # for this mechanism, not noise. p90 is also the standard textbook recommendation
    # (tau well inside the region with adequate risk-set size), fixed BEFORE looking at
    # any test-cohort (CPTAC/GSE68465) result.
    tau = float(np.percentile(time_tr_p, 90))
    print(f"Computing jackknife pseudo-RMST on TCGA-LUAD training cohort (n={pos.sum()}, tau={tau:.0f} days, "
          f"= 90th percentile of training times, not the max -- see comment above)...", flush=True)
    pseudo, theta_full = jackknife_pseudo_rmst(time_tr_p, event_tr_p, tau)

    # Sanity checks -- must pass before trusting anything below.
    mean_pseudo = pseudo.mean()
    rel_err = abs(mean_pseudo - theta_full) / theta_full
    corr_events = np.corrcoef(pseudo[event_tr_p], time_tr_p[event_tr_p])[0, 1]
    print(f"SANITY: theta_full={theta_full:.1f}, mean(pseudo)={mean_pseudo:.1f}, "
          f"rel_err={rel_err:.4f} (should be small), "
          f"corr(pseudo,time | event=1)={corr_events:.3f} (should be strongly positive)", flush=True)
    if rel_err > 0.02 or corr_events < 0.5:
        print("SANITY CHECK FAILED -- aborting, do not trust downstream numbers.", flush=True)
        return

    pseudo_cache = {"pseudo": pseudo}

    print("\nObserved panel...", flush=True)
    observed = run_config(panel, X1, y1, X2, y2, X3, y3, pseudo_cache)
    print(f"Observed: {observed}", flush=True)

    rng = np.random.default_rng(SEED)
    other_genes = [g for g in candidate_genes if g not in panel]
    null_results = []
    for i in range(N_NULL):
        rand_panel = list(rng.choice(other_genes, size=len(panel), replace=False))
        res = run_config(rand_panel, X1, y1, X2, y2, X3, y3, pseudo_cache)
        null_results.append(res)
        print(f"  null {i+1}/{N_NULL}: {res} ({time.time()-t0:.0f}s elapsed)", flush=True)

    summary = {
        "model": "TabPFN v2 (Prior-Labs) regressor on jackknife pseudo-RMST (Andersen & Pohar Perme 2010)",
        "tau_days": tau, "theta_full_rmst_days": theta_full,
        "sanity_rel_err": rel_err, "sanity_corr_events": corr_events,
        "panel_size": len(panel), "n_null": N_NULL, "seed": SEED,
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

    out_path = RESULTS / "phase11_tabpfn_pseudo_rmst.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWritten: {out_path}")
    print(f"Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
