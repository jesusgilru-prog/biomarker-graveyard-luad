"""Survival statistics for feeding nullcal.NullCalibration.

nullcal expects a callable ``statistic(x, y) -> float`` (larger = more signal),
called once per candidate feature per permutation. Off-the-shelf survival
statistics (lifelines' CoxPHFitter, logrank_test) are iterative/object-heavy
and far too slow to call ~10 million times (thousands of genes x hundreds to
thousands of permutations x several cohorts).

This module supplies one purpose-built statistic: a closed-form vectorized
two-group log-rank chi-square (median-split dichotomization of expression,
then the standard Mantel-Haenszel log-rank formula). This is exactly the kind
of "write your own domain statistic" extension nullcal's own docstring invites
(see nullcal/statistics.py) -- it does NOT touch nullcal's permutation loop,
corrections, or p-value machinery, which stay untouched inside nullcal itself.

y is expected as an (n, 2) array: column 0 = time, column 1 = event (1=death,
0=censored). This is why it is passed to NullCalibration.calibrate as a 2-D
array rather than nullcal's usual 1-D target -- nullcal only ever does
``y[rng.permutation(n)]`` internally, which permutes whole (time, event) rows
together and keeps the pairing intact, so this works without modifying nullcal.
"""
from __future__ import annotations

import numpy as np


def _logrank_components(x: np.ndarray, y: np.ndarray):
    """Shared computation for logrank_chi2 / logrank_direction. Returns
    (chi2, o_minus_e) where o_minus_e's sign tells which group (high vs low
    expression) has MORE deaths than expected under the null: positive means
    high expression is associated with worse survival (more deaths than
    expected in the high-expression group), negative means high expression is
    protective."""
    time = y[:, 0]
    event = y[:, 1]
    n = time.shape[0]

    median = np.median(x)
    group1 = (x > median).astype(int)  # 1 = high expression
    if group1.sum() == 0 or group1.sum() == n:
        return 0.0, 0.0  # degenerate split (e.g. >half the values tied at median)

    order = np.argsort(time)
    t = time[order]
    d = event[order]
    g1 = group1[order]

    event_times = np.unique(t[d == 1])
    if event_times.size == 0:
        return 0.0, 0.0

    # at-risk counts just before each event time, via searchsorted on sorted t
    at_risk_idx = np.searchsorted(t, event_times, side="left")
    n_total = n - at_risk_idx
    n1_at_risk = np.array([np.sum(g1[at_risk_idx[i]:] == 1) for i in range(event_times.size)])

    d_total = np.array([np.sum((t == et) & (d == 1)) for et in event_times])
    d1 = np.array([np.sum((t == et) & (d == 1) & (g1 == 1)) for et in event_times])

    valid = n_total > 1
    n_total, n1_at_risk, d_total, d1 = (a[valid] for a in (n_total, n1_at_risk, d_total, d1))
    if n_total.size == 0:
        return 0.0, 0.0

    expected1 = d_total * n1_at_risk / n_total
    variance = (d_total * (n_total - d_total) * n1_at_risk * (n_total - n1_at_risk)
                / (n_total ** 2 * (n_total - 1)))
    var_sum = variance.sum()
    if var_sum <= 0:
        return 0.0, 0.0

    o_minus_e = float((d1 - expected1).sum())
    chi2 = (o_minus_e ** 2) / var_sum
    return float(chi2), o_minus_e


def logrank_chi2(x: np.ndarray, y: np.ndarray) -> float:
    """Two-group log-rank chi-square statistic, x dichotomized at its median.

    Standard Mantel-Haenszel closed-form (Peto & Peto 1972 / Mantel 1966):
    at each distinct event time, compare observed vs expected deaths in
    group 1 (high expression) against the risk-set-weighted null, sum across
    event times, and normalize by the pooled hypergeometric variance.

    Direction-agnostic (always >= 0) -- this is what nullcal.NullCalibration
    is fed as its `statistic=`, since it only needs "how much signal". For
    direction (does high expression mean better or worse survival), use
    :func:`logrank_direction`.
    """
    chi2, _ = _logrank_components(x, y)
    return chi2


def cox_score_chi2(x: np.ndarray, y: np.ndarray) -> float:
    """Continuous-covariate log-rank score with hypergeometric tie variance.

    The historical API name is retained for reproducibility. This is NOT
    the Breslow-ties score statistic. At each event time, conditional on d
    deaths in a risk set of size n, the null selects a uniform subset of d
    subjects without replacement. Its expression sum has mean d*mean(x)
    and variance d*(n-d)/(n-1)*Var(x), with population variance (ddof=0).
    Summing observed-minus-expected scores and these variances yields U^2/V.
    It is also the score statistic at beta=0 of the exact conditional
    partial likelihood. "Exact" here describes the tie likelihood, not the
    chi-square p-value approximation used by callers. Without tied deaths
    it agrees with the Breslow score; with binary x it is the log-rank test.
    Continuous expression enters directly, without median dichotomization.
    """
    time = y[:, 0]
    event = y[:, 1]
    n = time.shape[0]
    if np.std(x) == 0:
        return 0.0

    order = np.argsort(time)
    t = time[order]
    d = event[order]
    xs = x[order]

    event_times = np.unique(t[d == 1])
    if event_times.size == 0:
        return 0.0

    at_risk_idx = np.searchsorted(t, event_times, side="left")
    U = 0.0
    V = 0.0
    for k, et in enumerate(event_times):
        risk_x = xs[at_risk_idx[k]:]
        s0 = risk_x.size
        if s0 <= 1:
            continue
        s1 = risk_x.sum()
        s2 = (risk_x ** 2).sum()
        died_mask = (t == et) & (d == 1)
        dk = int(died_mask.sum())
        sum_x_deaths = xs[died_mask].sum()

        mean_risk = s1 / s0
        U += sum_x_deaths - dk * mean_risk
        var_risk = s2 / s0 - mean_risk ** 2
        V += dk * (s0 - dk) / (s0 - 1) * var_risk if s0 > 1 else 0.0

    if V <= 0:
        return 0.0
    return float((U ** 2) / V)


def logrank_direction(x: np.ndarray, y: np.ndarray) -> int:
    """Sign of the association: +1 if high expression associates with MORE
    deaths than expected (worse survival), -1 if fewer (protective), 0 if
    undefined (degenerate split or zero variance). Used to require the same
    direction of effect across cohorts before calling something "replicated"
    -- a gene that is bidirectionally "significant" (opposite signs in
    different cohorts) is not a replication, it is two unrelated artifacts."""
    _, o_minus_e = _logrank_components(x, y)
    if o_minus_e > 0:
        return 1
    if o_minus_e < 0:
        return -1
    return 0


def stratified_logrank_chi2(x: np.ndarray, y: np.ndarray, strata: np.ndarray) -> float:
    """Site-stratified (Mantel-Haenszel) two-group log-rank chi-square.

    Same median-split + O-E/variance machinery as ``_logrank_components``,
    but computed independently WITHIN each stratum (site) and then combined
    across strata: chi2 = (sum_s O-E_s)^2 / sum_s Var_s. This removes any
    confound where the stratifying variable (site) is associated with BOTH
    expression (batch effect) and outcome (differential event rate) --
    exactly the situation found in GSE68465 (mortality 37.5%-75.9% across
    its 4 profiling sites). Used together with within-stratum permutation
    (shuffle (time, event) pairs only within each site) so the null
    distribution also respects the site structure, not just the observed
    statistic.

    strata: (n,) array of site labels (any hashable), same order as x/y rows.
    """
    time = y[:, 0]
    event = y[:, 1]
    n = time.shape[0]

    median = np.median(x)
    group1 = (x > median).astype(int)
    if group1.sum() == 0 or group1.sum() == n:
        return 0.0

    total_oe = 0.0
    total_var = 0.0
    for s in np.unique(strata):
        mask = strata == s
        ts, ds, g1s = time[mask], event[mask], group1[mask]
        ns = ts.shape[0]
        if ns < 2 or g1s.sum() == 0 or g1s.sum() == ns:
            continue  # stratum degenerate (no events, or all-one-group) -- contributes nothing

        order = np.argsort(ts)
        t, d, g1 = ts[order], ds[order], g1s[order]
        event_times = np.unique(t[d == 1])
        if event_times.size == 0:
            continue

        at_risk_idx = np.searchsorted(t, event_times, side="left")
        n_total = ns - at_risk_idx
        n1_at_risk = np.array([np.sum(g1[at_risk_idx[i]:] == 1) for i in range(event_times.size)])
        d_total = np.array([np.sum((t == et) & (d == 1)) for et in event_times])
        d1 = np.array([np.sum((t == et) & (d == 1) & (g1 == 1)) for et in event_times])

        valid = n_total > 1
        n_total, n1_at_risk, d_total, d1 = (a[valid] for a in (n_total, n1_at_risk, d_total, d1))
        if n_total.size == 0:
            continue

        expected1 = d_total * n1_at_risk / n_total
        variance = (d_total * (n_total - d_total) * n1_at_risk * (n_total - n1_at_risk)
                    / (n_total ** 2 * (n_total - 1)))
        total_oe += float((d1 - expected1).sum())
        total_var += float(variance.sum())

    if total_var <= 0:
        return 0.0
    return (total_oe ** 2) / total_var
