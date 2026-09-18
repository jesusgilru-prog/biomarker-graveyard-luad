"""Biomarker-graveyard demo — the first 'plato' on top of the 'receta'.

Simulates the situation of Moonshot #1: hundreds of candidate biomarkers screened
against an outcome, where only a handful carry real signal and the rest are noise.
It shows that:

  * naive screening (raw p < 0.05) declares a flood of false biomarkers,
  * Bonferroni alone (the 'lottery paper' warning) still lets some artifacts through
    when candidates are correlated,
  * null-signal calibration with FDR recovers the real ones while holding down the
    false-discovery rate — quantifying, on real data, what fraction of a published
    biomarker literature would survive an honest control.

Run:  python biomarker_graveyard_demo.py
Swap the `load_simulated_cohort` function for a real CPTAC/GEO/TCGA loader on your
own machine (those data servers are not reachable from every sandbox).
"""
from __future__ import annotations

import numpy as np

from nullcal import NullCalibration
from nullcal.statistics import abs_spearman


def load_simulated_cohort(n_patients=350, n_candidates=400, n_real=12,
                          block_size=20, seed=2024):
    """A cohort with correlated *noise* candidate biomarkers (as genes/proteins
    really are) plus a handful of genuinely associated ones.

    Key design point: the correlated noise blocks are **independent of the
    outcome**, so any association a screen finds among them is a true false
    positive. The real biomarkers are separate columns with a direct effect on
    the outcome. This is what lets null calibration visibly separate signal from
    artifact — if noise shared a latent factor with the outcome it would not be
    noise at all."""
    rng = np.random.default_rng(seed)
    X = np.empty((n_patients, n_candidates))
    # correlated NOISE blocks, unrelated to the outcome
    for start in range(0, n_candidates, block_size):
        latent = rng.normal(size=(n_patients, 1))
        end = min(start + block_size, n_candidates)
        X[:, start:end] = 0.7 * latent + 0.3 * rng.normal(size=(n_patients, end - start))
    # real biomarkers: overwrite a few columns with independent signals that
    # actually drive the outcome
    real_idx = rng.choice(n_candidates, size=n_real, replace=False)
    outcome = rng.normal(size=n_patients)
    for j in real_idx:
        sig = rng.normal(size=n_patients)          # standalone, not in any block
        X[:, j] = sig
        outcome += rng.uniform(0.35, 0.6) * sig
    names = [f"BM{j:03d}" for j in range(n_candidates)]
    return X, outcome, names, set(real_idx.tolist())


def main():
    X, y, names, real = load_simulated_cohort()
    n = len(names)

    # --- naive screening: raw correlation p-value < 0.05 ---------------------
    from scipy import stats as sps
    raw_p = np.array([sps.spearmanr(X[:, j], y).pvalue for j in range(n)])
    naive_hits = set(np.where(raw_p < 0.05)[0].tolist())

    # --- null-signal calibration with FDR ------------------------------------
    nc = NullCalibration(abs_spearman, n_permutations=1000,
                         correction="bh", alpha=0.05, random_state=7)
    res = nc.calibrate(X, y, names=names)
    survivors = {i for i, s in enumerate(res.survives) if s}

    def scorecard(label, hits):
        tp = len(hits & real)
        fp = len(hits - real)
        fdr = fp / max(len(hits), 1)
        print(f"{label:<28} declared={len(hits):>3}  "
              f"true={tp:>2}/{len(real)}  false={fp:>3}  FDR={fdr:5.1%}")

    print(f"\nCohort: {X.shape[0]} patients, {n} candidate biomarkers, "
          f"{len(real)} genuinely associated\n")
    scorecard("Naive p<0.05", naive_hits)
    scorecard("Null-calibrated (FDR)", survivors)

    frac = 100 * len(survivors & naive_hits) / max(len(naive_hits), 1)
    print(f"\nOf the {len(naive_hits)} 'biomarkers' the naive screen would publish, "
          f"only {len(survivors & naive_hits)} "
          f"({frac:.0f}%) survive null calibration.")
    print("That surviving fraction is exactly the headline number of the "
          "'biomarker graveyard' paper.\n")

    # optional figure ---------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        order = np.argsort(-res.effect_above_null)
        colors = ["#1e7d47" if res.survives[i] else "#c9ccd3" for i in order]
        edge = ["#0d5" if i in real else "none" for i in order]
        fig, ax = plt.subplots(figsize=(9, 4.2))
        ax.bar(range(n), res.effect_above_null[order], color=colors,
               edgecolor=edge, linewidth=0.8)
        ax.set_xlabel("candidate biomarkers (sorted by effect above null)")
        ax.set_ylabel("effect above null floor")
        ax.set_title("The biomarker graveyard: green = survives null calibration")
        ax.axhline(0, color="#333", linewidth=0.6)
        fig.tight_layout()
        fig.savefig("biomarker_graveyard.png", dpi=130)
        print("Figure written: biomarker_graveyard.png")
    except Exception as exc:  # pragma: no cover
        print(f"(figure skipped: {exc})")


if __name__ == "__main__":
    main()
