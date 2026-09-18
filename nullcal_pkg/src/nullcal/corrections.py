"""Multiple-testing corrections used by null-signal calibration.

Implemented from scratch (no statsmodels dependency) so the library stays light.
All functions take a 1-D array of p-values and return either adjusted p-values
or a boolean "reject" mask at a given error level.
"""
from __future__ import annotations

import numpy as np

__all__ = ["bonferroni", "benjamini_hochberg", "benjamini_yekutieli"]


def _as_pvals(pvalues):
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1:
        raise ValueError("pvalues must be 1-D")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("pvalues must lie in [0, 1]")
    return p


def bonferroni(pvalues):
    """Bonferroni-adjusted p-values (controls the family-wise error rate).

    Returns adjusted p-values clipped to 1.0. This is the *naive* correction
    that the lottery-paper finding shows spurious signals can still survive,
    which is exactly why null calibration is layered on top of it.
    """
    p = _as_pvals(pvalues)
    return np.minimum(p * p.size, 1.0)


def benjamini_hochberg(pvalues):
    """Benjamini-Hochberg adjusted p-values (controls the FDR under independence
    or positive dependence). Returns q-values in the original order."""
    p = _as_pvals(pvalues)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1)
    # enforce monotonicity from the largest rank downwards
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty(n, dtype=float)
    out[order] = q
    return out


def benjamini_yekutieli(pvalues):
    """Benjamini-Yekutieli adjusted p-values (controls the FDR under arbitrary
    dependence; more conservative than BH). Useful when candidate features are
    strongly correlated, as gene/protein biomarkers usually are."""
    p = _as_pvals(pvalues)
    n = p.size
    c_n = np.sum(1.0 / (np.arange(n) + 1))  # harmonic number
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n * c_n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty(n, dtype=float)
    out[order] = q
    return out
