# nullcal — null-signal calibration for honest discovery

Distinguish **real signal from artifact** by measuring the floor that appears with
no genuine structure, and keeping only what rises convincingly above it.

This is the reusable engine behind the "biomarker graveyard" study and other
discovery work: before believing an association or a model score is real, you
build an explicit *null* and ask whether the observed signal clears it after
multiple-testing correction.

## Install

```bash
pip install -e .
# optional extras
pip install -e ".[pandas,test]"
```

Requires Python ≥3.9, numpy, scipy. (pandas optional, only for `.to_frame()`.)

## Quick start

```python
import numpy as np
from nullcal import NullCalibration
from nullcal.statistics import abs_spearman

rng = np.random.default_rng(0)
X = rng.normal(size=(300, 200))            # 200 candidate features
y = X[:, 0] * 0.6 + rng.normal(size=300)   # only feature 0 is real

nc = NullCalibration(
    statistic=abs_spearman,   # any (x, y) -> float, higher = more signal
    n_permutations=1000,
    method="permutation",     # or "synthetic" / "placebo"
    correction="bh",          # FDR (BH). Also: "by", "bonferroni", "none"
    alpha=0.05,
)
res = nc.calibrate(X, y)

print(res.n_survivors())          # how many clear the null after correction
for row in res.summary()[:5]:     # sorted by q-value then effect
    print(row)
```

## What it gives you, per candidate

- `observed` — the statistic on the real data
- `null_mean`, `null_std` — the calibrated null floor
- `effect_above_null`, `z_above_null` — signal above that floor
- `p_value` — empirical, one-sided, never exactly zero (Phipson–Smyth +1)
- `q_value` — after BH / BY / Bonferroni correction
- `survives` — boolean at your `alpha`

## The three nulls

| method        | what it breaks                         | when to use                              |
|---------------|----------------------------------------|------------------------------------------|
| `permutation` | link between features and target       | default; you have a target               |
| `synthetic`   | all structure (surrogate features)     | check a pipeline finds nothing on noise  |
| `placebo`     | cross-group structure (shuffle within groups) | time/grouped data; needs `groups=` |

## Multiple-testing corrections

`bonferroni` (FWER) is the *naive* control that the lottery-paper finding shows
spurious signals can still survive — which is exactly why null calibration is
layered on top. `bh` (FDR) is the default; `by` is the dependence-robust FDR,
appropriate when candidates are correlated (genes/proteins usually are).

## Ready-made statistics

`nullcal.statistics`: `abs_pearson`, `abs_spearman`, `auc` (rank-based,
direction-agnostic), `mutual_information_bin`. Or pass your own, e.g. a
cross-validated AUC.

## Example

`examples/biomarker_graveyard_demo.py` reproduces the core figure: hundreds of
correlated candidate biomarkers, only a handful real, and it shows how many the
naive screen would publish versus how many survive null calibration. Swap the
`load_simulated_cohort` function for a real CPTAC/GEO/TCGA loader on your machine.

## Tests

```bash
pytest tests/
```

Covers: recovery of real signal, FDR control on pure noise, non-zero p-values,
correction ordering, and reproducibility.

## Cite / status

v0.1.0 — MIT. Method: null-signal calibration (J. Gil Ruiz). This is the first
public version, meant to become the standard tool other studies build on.
