"""Null-signal calibration — core.

The idea in one sentence: before believing that an observed association or model
performance is real, establish the *floor* that arises with no genuine signal,
and keep only what rises convincingly above that floor.

Three ways of building the null are provided:

* **permutation** — break the link between predictors and target by shuffling the
  target many times. The gold standard when you have a target.
* **synthetic** — replace the data with structure-free draws matching each
  column's marginal, so any "signal" found is by construction an artifact.
* **placebo** — for grouped / time-indexed data, relabel groups (e.g. shift a
  calendar, swap a region) to create controls that *should* show nothing.

The public entry point is :class:`NullCalibration`. It wraps a user-supplied
*statistic* function (anything that maps ``(X, y) -> float``, higher = more
signal: a correlation, an AUC, an R², a mutual information, ...) and reports for
each candidate an empirical p-value against the null, the effect above the null
floor, and — after multiple-testing correction — whether it *survives*.

This is a v0.1: small, dependency-light (numpy/scipy), and deterministic given a
seed. It is meant to be the reusable core that individual studies build on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from .corrections import benjamini_hochberg, benjamini_yekutieli, bonferroni

__all__ = ["NullCalibration", "CalibrationResult"]

_CORRECTIONS = {
    "bonferroni": bonferroni,
    "bh": benjamini_hochberg,
    "fdr_bh": benjamini_hochberg,
    "by": benjamini_yekutieli,
    "fdr_by": benjamini_yekutieli,
    "none": lambda p: np.asarray(p, dtype=float),
}


@dataclass
class CalibrationResult:
    """Outcome of calibrating one or more candidate features/models."""

    names: list
    observed: np.ndarray          # observed statistic per candidate
    null_mean: np.ndarray         # mean of the null distribution
    null_std: np.ndarray          # std of the null distribution
    effect_above_null: np.ndarray  # observed - null_mean
    z_above_null: np.ndarray      # (observed - null_mean) / null_std
    p_value: np.ndarray           # empirical one-sided p-value
    q_value: np.ndarray           # after multiple-testing correction
    survives: np.ndarray          # bool: q_value <= alpha
    n_permutations: int
    method: str
    correction: str
    alpha: float
    null_distribution: Optional[np.ndarray] = field(default=None, repr=False)

    def summary(self):
        """Return a plain list-of-dicts summary, sorted by q-value then effect."""
        rows = []
        for i, name in enumerate(self.names):
            rows.append(
                {
                    "feature": name,
                    "observed": float(self.observed[i]),
                    "null_mean": float(self.null_mean[i]),
                    "effect_above_null": float(self.effect_above_null[i]),
                    "z_above_null": float(self.z_above_null[i]),
                    "p_value": float(self.p_value[i]),
                    "q_value": float(self.q_value[i]),
                    "survives": bool(self.survives[i]),
                }
            )
        rows.sort(key=lambda r: (r["q_value"], -r["effect_above_null"]))
        return rows

    def n_survivors(self) -> int:
        return int(np.sum(self.survives))

    def survivor_fraction(self) -> float:
        return float(np.mean(self.survives)) if len(self.survives) else 0.0

    def to_frame(self):  # pragma: no cover - convenience only
        import pandas as pd

        return pd.DataFrame(self.summary())


def _empirical_pvalue(observed: float, null: np.ndarray) -> float:
    """One-sided (upper-tail) empirical p-value with the standard +1 correction,
    which keeps the p-value from ever being exactly zero (Phipson & Smyth 2010)."""
    b = int(np.sum(null >= observed))
    m = null.size
    return (b + 1) / (m + 1)


class NullCalibration:
    """Calibrate observed signal against an explicitly constructed null.

    Parameters
    ----------
    statistic:
        Callable ``(X_col, y) -> float`` where larger means more signal. ``X_col``
        is a 1-D array (one candidate feature) and ``y`` the target. Examples:
        absolute Pearson/Spearman correlation, AUC, mutual information.
    n_permutations:
        Size of the null distribution. 1000 is a reasonable default; use more for
        small p-values.
    method:
        ``"permutation"`` (shuffle the target), ``"synthetic"`` (structure-free
        surrogate features), or ``"placebo"`` (shuffle a supplied group label).
    correction:
        Multiple-testing correction across candidates: ``"bh"`` (FDR, default),
        ``"by"`` (FDR under dependence), ``"bonferroni"`` (FWER), or ``"none"``.
    alpha:
        Error level for the ``survives`` decision.
    random_state:
        Seed for reproducibility.
    """

    def __init__(
        self,
        statistic: Callable[[np.ndarray, np.ndarray], float],
        n_permutations: int = 1000,
        method: str = "permutation",
        correction: str = "bh",
        alpha: float = 0.05,
        random_state: Optional[int] = 42,
    ):
        if method not in {"permutation", "synthetic", "placebo"}:
            raise ValueError(f"unknown method: {method!r}")
        if correction not in _CORRECTIONS:
            raise ValueError(f"unknown correction: {correction!r}")
        self.statistic = statistic
        self.n_permutations = int(n_permutations)
        self.method = method
        self.correction = correction
        self.alpha = float(alpha)
        self.random_state = random_state

    # ------------------------------------------------------------------ nulls
    def _null_for_feature(self, x: np.ndarray, y: np.ndarray, rng, groups=None):
        stats = np.empty(self.n_permutations)
        n = y.shape[0]
        if self.method == "permutation":
            for k in range(self.n_permutations):
                stats[k] = self.statistic(x, y[rng.permutation(n)])
        elif self.method == "synthetic":
            # structure-free surrogate for x, matched to its marginal by resampling
            for k in range(self.n_permutations):
                x_surr = x[rng.integers(0, n, size=n)]
                x_surr = x_surr[rng.permutation(n)]
                stats[k] = self.statistic(x_surr, y)
        else:  # placebo: permute the target *within* each group, breaking any
            # cross-group structure while preserving within-group marginals.
            if groups is None:
                raise ValueError("method='placebo' requires groups=")
            groups = np.asarray(groups)
            idx_by_group = [np.where(groups == g)[0] for g in np.unique(groups)]
            for k in range(self.n_permutations):
                y_placebo = y.copy()
                for idx in idx_by_group:
                    y_placebo[idx] = y[idx][rng.permutation(idx.size)]
                stats[k] = self.statistic(x, y_placebo)
        return stats

    # ------------------------------------------------------------------- fit
    def calibrate(
        self,
        X,
        y,
        names: Optional[Sequence[str]] = None,
        groups=None,
        keep_null: bool = False,
    ) -> CalibrationResult:
        """Calibrate every column of ``X`` against the null.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)
        names : optional feature names
        groups : optional group labels (required for method='placebo')
        keep_null : store the full null matrix on the result (memory heavy)
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape
        if y.shape[0] != n_samples:
            raise ValueError("X and y have mismatched lengths")
        if names is None:
            names = [f"f{i}" for i in range(n_features)]
        names = list(names)

        rng = np.random.default_rng(self.random_state)

        observed = np.array([self.statistic(X[:, j], y) for j in range(n_features)])
        null_mean = np.empty(n_features)
        null_std = np.empty(n_features)
        pvals = np.empty(n_features)
        null_store = [] if keep_null else None

        for j in range(n_features):
            null = self._null_for_feature(X[:, j], y, rng, groups=groups)
            null_mean[j] = null.mean()
            null_std[j] = null.std(ddof=1) if null.size > 1 else 0.0
            pvals[j] = _empirical_pvalue(observed[j], null)
            if keep_null:
                null_store.append(null)

        effect = observed - null_mean
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(null_std > 0, effect / null_std, 0.0)
        qvals = _CORRECTIONS[self.correction](pvals)
        survives = qvals <= self.alpha

        return CalibrationResult(
            names=names,
            observed=observed,
            null_mean=null_mean,
            null_std=null_std,
            effect_above_null=effect,
            z_above_null=z,
            p_value=pvals,
            q_value=qvals,
            survives=survives,
            n_permutations=self.n_permutations,
            method=self.method,
            correction=self.correction,
            alpha=self.alpha,
            null_distribution=(np.array(null_store) if keep_null else None),
        )
