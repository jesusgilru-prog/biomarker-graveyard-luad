import numpy as np
import pytest

from nullcal import NullCalibration
from nullcal.corrections import benjamini_hochberg, bonferroni
from nullcal.statistics import abs_pearson, abs_spearman, auc, mutual_information_bin


def make_data(n=400, n_features=60, n_real=5, strength=0.7, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    real = rng.choice(n_features, size=n_real, replace=False)
    y = rng.normal(size=n)
    for j in real:
        y += strength * X[:, j]
    return X, y, set(real.tolist())


def test_recovers_real_kills_noise():
    X, y, real = make_data()
    nc = NullCalibration(abs_pearson, n_permutations=400, correction="bh", random_state=1)
    res = nc.calibrate(X, y)
    survivors = {i for i, s in enumerate(res.survives) if s}
    # every real feature is recovered
    assert real.issubset(survivors)
    # and we do not drown in false positives (FDR control)
    false_pos = survivors - real
    assert len(false_pos) <= 3


def test_pure_noise_yields_few_survivors():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(300, 80))
    y = rng.normal(size=300)  # no signal at all
    nc = NullCalibration(abs_pearson, n_permutations=400, correction="bh", random_state=3)
    res = nc.calibrate(X, y)
    # under the null, FDR control should keep survivors very low
    assert res.n_survivors() <= 4
    assert res.survivor_fraction() < 0.1


def test_pvalue_never_zero():
    X, y, _ = make_data(n_features=10)
    nc = NullCalibration(abs_pearson, n_permutations=200, random_state=4)
    res = nc.calibrate(X, y)
    assert np.all(res.p_value > 0)
    assert np.all(res.p_value <= 1)


def test_bonferroni_more_conservative_than_bh():
    p = np.array([0.001, 0.01, 0.02, 0.04, 0.2, 0.5])
    assert np.all(bonferroni(p) >= benjamini_hochberg(p) - 1e-12)


def test_statistics_run():
    rng = np.random.default_rng(5)
    x = rng.normal(size=200)
    y_cont = x * 0.5 + rng.normal(size=200)
    y_bin = (x + rng.normal(size=200) > 0).astype(int)
    assert 0 <= abs_pearson(x, y_cont) <= 1
    assert 0 <= abs_spearman(x, y_cont) <= 1
    assert 0.5 <= auc(x, y_bin) <= 1
    assert mutual_information_bin(x, y_cont) >= 0


def test_placebo_requires_groups():
    X, y, _ = make_data(n_features=5)
    nc = NullCalibration(abs_pearson, n_permutations=50, method="placebo")
    with pytest.raises(ValueError):
        nc.calibrate(X, y)


def test_reproducible():
    X, y, _ = make_data()
    nc = NullCalibration(abs_pearson, n_permutations=200, random_state=7)
    r1 = nc.calibrate(X, y)
    r2 = nc.calibrate(X, y)
    assert np.allclose(r1.p_value, r2.p_value)
