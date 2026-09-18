"""nullcal — null-signal calibration for honest discovery.

Distinguish real signal from artifact by measuring the floor that arises with no
genuine structure, and keeping only what rises convincingly above it.

Quick start
-----------
>>> import numpy as np
>>> from nullcal import NullCalibration
>>> from nullcal.statistics import abs_pearson
>>> rng = np.random.default_rng(0)
>>> X = rng.normal(size=(300, 50))          # 50 candidate features
>>> y = X[:, 0] * 0.6 + rng.normal(size=300)  # only feature 0 is real
>>> nc = NullCalibration(abs_pearson, n_permutations=500, correction="bh")
>>> res = nc.calibrate(X, y)
>>> res.n_survivors()                        # should recover ~1
1
"""
from .core import CalibrationResult, NullCalibration
from . import corrections, statistics

__version__ = "0.1.0"
__all__ = [
    "NullCalibration",
    "CalibrationResult",
    "statistics",
    "corrections",
    "__version__",
]
