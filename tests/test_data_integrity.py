"""Minimal integrity tests for the three LUAD cohorts (Fase 1).

Run: pytest tests/ -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import (  # noqa: E402
    load_cptac_luad_discovery,
    load_tcga_luad_replication,
    load_gse68465_replication,
    hpa_lung_adenocarcinoma_claims,
)

COHORTS = {
    "cptac": (load_cptac_luad_discovery, 20, 5),
    "tcga": (load_tcga_luad_replication, 50, 20),
    "gse68465": (load_gse68465_replication, 100, 50),
}


@pytest.mark.parametrize("name", COHORTS.keys())
def test_cohort_shape_and_events(name):
    loader, min_n, min_events = COHORTS[name]
    X, y = loader()
    assert X.shape[1] == y.shape[0], f"{name}: X samples != y rows"
    assert X.shape[1] >= min_n, f"{name}: too few samples ({X.shape[1]})"
    assert X.shape[0] >= 1000, f"{name}: suspiciously few genes ({X.shape[0]})"
    assert set(y["os_event"].unique()) <= {0, 1}, f"{name}: os_event not binary"
    assert int(y["os_event"].sum()) >= min_events, f"{name}: too few events for screening"


@pytest.mark.parametrize("name", COHORTS.keys())
def test_no_missing_values(name):
    loader, _, _ = COHORTS[name]
    X, y = loader()
    assert X.isna().sum().sum() == 0, f"{name}: NaN in expression matrix"
    time_col = [c for c in y.columns if c.startswith("os_time")][0]
    assert y[time_col].isna().sum() == 0, f"{name}: NaN in survival time"
    assert y["os_event"].isna().sum() == 0, f"{name}: NaN in event indicator"


@pytest.mark.parametrize("name", COHORTS.keys())
def test_expression_is_log_transformed_and_finite(name):
    loader, _, _ = COHORTS[name]
    X, _ = loader()
    assert np.isfinite(X.values).all(), f"{name}: non-finite values in X"
    assert (X.values >= 0).all(), f"{name}: negative log2(x+1) values in X"


@pytest.mark.parametrize("name", COHORTS.keys())
def test_sample_ids_unique(name):
    loader, _, _ = COHORTS[name]
    X, _ = loader()
    assert X.columns.duplicated().sum() == 0, f"{name}: duplicate sample IDs"


@pytest.mark.parametrize("name", COHORTS.keys())
def test_gene_symbols_unique(name):
    loader, _, _ = COHORTS[name]
    X, _ = loader()
    assert X.index.duplicated().sum() == 0, f"{name}: duplicate gene symbols"


def test_gene_overlap_across_cohorts_is_substantial():
    X1, _ = load_cptac_luad_discovery()
    X2, _ = load_tcga_luad_replication()
    X3, _ = load_gse68465_replication()
    common = set(X1.index) & set(X2.index) & set(X3.index)
    assert len(common) >= 5000, f"too few genes shared by all 3 cohorts ({len(common)})"


def test_cptac_and_tcga_patients_are_disjoint():
    """CPTAC-LUAD uses its own prospective patient IDs, never TCGA barcodes."""
    _, y_cptac = load_cptac_luad_discovery()
    _, y_tcga = load_tcga_luad_replication()
    cptac_ids = set(y_cptac["case_id"])
    tcga_ids = set(y_tcga["submitter_id"])
    assert cptac_ids.isdisjoint(tcga_ids), "CPTAC and TCGA cohorts share patient IDs"
    assert not any(cid.startswith("TCGA-") for cid in cptac_ids), \
        "a CPTAC case_id looks like a TCGA barcode"


def test_hpa_claims_corpus_loads_and_has_expected_labels():
    hpa = hpa_lung_adenocarcinoma_claims()
    assert hpa.shape[0] > 1000
    assert set(hpa["call"].unique()) <= {
        "unprognostic", "potential prognostic favorable", "potential prognostic unfavorable",
        "validated prognostic favorable", "validated prognostic unfavorable",
    }
    assert hpa["p_value"].between(0, 1).all()
