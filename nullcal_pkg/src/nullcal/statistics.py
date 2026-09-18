"""Ready-made statistic functions to feed :class:`nullcal.NullCalibration`.

Each maps ``(x, y) -> float`` where larger means more signal. You can pass any
of these as ``statistic=`` or write your own (e.g. a cross-validated AUC).
"""
from __future__ import annotations

import numpy as np
from scipy import stats as _sps

__all__ = [
    "abs_pearson",
    "abs_spearman",
    "auc",
    "mutual_information_bin",
]


def abs_pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Absolute Pearson correlation. Robust to constant input (returns 0)."""
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    r, _ = _sps.pearsonr(x, y)
    return float(abs(r))


def abs_spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Absolute Spearman rank correlation (monotone, non-linear-friendly)."""
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    r, _ = _sps.spearmanr(x, y)
    if np.isnan(r):
        return 0.0
    return float(abs(r))


def auc(x: np.ndarray, y: np.ndarray) -> float:
    """Rank-based AUC of a single continuous score ``x`` against binary ``y``.

    Uses the Mann-Whitney U identity, so it needs no sklearn. Returns
    ``max(auc, 1 - auc)`` so that a perfectly *inverted* predictor also counts
    as signal (direction-agnostic, appropriate for screening)."""
    y = np.asarray(y)
    pos = x[y == 1]
    neg = x[y == 0]
    if pos.size == 0 or neg.size == 0:
        return 0.5
    u, _ = _sps.mannwhitneyu(pos, neg, alternative="two-sided")
    a = u / (pos.size * neg.size)
    return float(max(a, 1 - a))


def mutual_information_bin(x: np.ndarray, y: np.ndarray, bins: int = 8) -> float:
    """Mutual information between a binned continuous ``x`` and target ``y``.

    Cheap, dependency-light estimate via a 2-D histogram. Captures non-monotone
    dependence that correlation misses."""
    y = np.asarray(y)
    xb = np.digitize(x, np.quantile(x, np.linspace(0, 1, bins + 1)[1:-1]))
    if np.unique(y).size > bins:
        yb = np.digitize(y, np.quantile(y, np.linspace(0, 1, bins + 1)[1:-1]))
    else:
        yb = y.astype(int)
    c = np.histogram2d(xb, yb, bins=[np.unique(xb).size, np.unique(yb).size])[0]
    pxy = c / c.sum()
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    nz = pxy > 0
    mi = np.sum(pxy[nz] * np.log(pxy[nz] / (px @ py)[nz]))
    return float(max(mi, 0.0))
