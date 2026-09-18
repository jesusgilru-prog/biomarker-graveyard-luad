"""Reproducible loaders for The Biomarker Graveyard.

Cancer type: lung adenocarcinoma (LUAD). Chosen over the originally-planned breast
cohort because CPTAC-BRCA has only 2/105 death events (1.9%) in its available
follow-up window -- too few to screen anything against. CPTAC-LUAD has 24/106
(22.6%), and both independent replication cohorts (TCGA-LUAD, GSE68465) have
comparably real event rates. See data/DATASHEET.md for the full rationale.

Three independent cohorts, one endpoint (overall survival):

* DISCOVERY  -- CPTAC-LUAD (washu RNA-seq, FPKM), n=106, 24 events.
                Prospective CPTAC3 cohort. Patient IDs (format "C3L-xxxxx" /
                "C3N-xxxxx") are NOT TCGA barcodes ("TCGA-XX-XXXX") -- verified
                independent of TCGA-derived cohorts, see test_data_integrity.py.
* REPLICATION 1 -- TCGA-LUAD via the GDC REST API (open access, no token/DUA).
                Originally a reproducible random sample of 300 primary-tumor
                cases (seed=42, n=294 with usable survival) -- per external
                review, switched to the FULL available cohort (n=508 with
                usable survival, 183 events) since there was no statistical
                reason to discard ~40% of available replication power. The
                n=300 sample's results remain in results/ for provenance
                (this was the cohort used for the original discovery run).
* REPLICATION 2 -- GSE68465 (Shedden et al. 2008, GEO), Affymetrix HG-U133A
                microarray, n=442, 236 events. Different assay technology
                entirely (microarray vs RNA-seq), which is a deliberately
                strong replication test: a real biomarker's association should
                survive a platform change, not just a cohort change.

All expression loaders return genes-as-rows matrices indexed by HGNC gene
symbol, log2(x+1)-transformed (a fixed, data-independent transform -- no
leakage). All clinical loaders return one row per sample with `os_time` (in a
documented unit) and `os_event` (1=death, 0=censored).

Regenerating data/processed/ from data/raw/ requires re-running the fetch step
for TCGA (network calls to the GDC API); see REPRODUCE.md at the repo root.
"""
from __future__ import annotations

import io
import gzip
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"


def _log2p1(df: pd.DataFrame) -> pd.DataFrame:
    return np.log2(df.clip(lower=0) + 1)


# --------------------------------------------------------------------- CPTAC
def load_cptac_luad_discovery() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Discovery cohort. Returns (X genes x samples, y clinical) already aligned.

    X is log2(FPKM+1). y has columns: case_id, os_time_days, os_event.
    """
    expr = pd.read_csv(PROCESSED / "cptac_luad_fpkm.csv.gz", index_col=0)
    clin = pd.read_csv(PROCESSED / "cptac_luad_clinical.csv")
    clin = clin.set_index("case_id").reindex(expr.columns).rename_axis("case_id").reset_index()
    return _log2p1(expr), clin


# ------------------------------------------------------------- TCGA via GDC
def load_tcga_luad_replication() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replication cohort 1 -- FULL available TCGA-LUAD cohort (n=508, 183
    events). Returns (X genes x samples, y clinical).

    X is log2(TPM+1) from GDC STAR-Counts, protein-coding genes only.
    y has columns: submitter_id, os_time_days, os_event.
    """
    expr = pd.read_csv(PROCESSED / "tcga_luad_tpm_full.csv.gz", index_col=0)
    clin = pd.read_csv(PROCESSED / "tcga_luad_clinical_full.csv")
    clin = clin.set_index("submitter_id").reindex(expr.columns).rename_axis("submitter_id").reset_index()
    return _log2p1(expr), clin


def load_tcga_luad_replication_n300() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Original n=300-sample (seed=42) version of the TCGA-LUAD replication
    cohort, kept only for provenance/reproducing the first discovery run's
    exact numbers. Prefer load_tcga_luad_replication() (full cohort) for any
    new analysis."""
    expr = pd.read_csv(PROCESSED / "tcga_luad_tpm_n300.csv.gz", index_col=0)
    clin = pd.read_csv(PROCESSED / "tcga_luad_clinical_n300.csv")
    clin = clin.set_index("submitter_id").reindex(expr.columns).rename_axis("submitter_id").reset_index()
    return _log2p1(expr), clin


# ------------------------------------------------------------------ GSE68465
def load_gse68465_replication() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replication cohort 2. Returns (X genes x samples, y clinical).

    X is log2(microarray-intensity+1) (already-scanner-normalized MAS5-style
    intensities as deposited; NOT comparable in scale to RNA-seq TPM/FPKM --
    that is fine, because every statistic in this study is computed within one
    cohort at a time and compared only through p-values / survival status).
    y has columns: sample_id, os_time_months, os_event.
    """
    expr = pd.read_csv(PROCESSED / "gse68465_luad_expr.csv.gz", index_col=0)
    clin = pd.read_csv(PROCESSED / "gse68465_luad_clinical.csv")
    clin = clin.set_index("sample_id").reindex(expr.columns).rename_axis("sample_id").reset_index()
    return _log2p1(expr), clin


# -------------------------------------------------------------- HPA corpus
def load_hpa_prognostic_corpus() -> pd.DataFrame:
    """The 'claimed biomarker' corpus: HPA Pathology Atlas per-gene, per-cancer
    prognostic calls (Uhlén et al. 2017, Science). Returns a long-format frame:
    columns [gene, cancer_type, cohort, call, p_value]. `cohort` is 'TCGA' or
    'validation'. `call` is the HPA text label (e.g. 'prognostic favorable').
    """
    df = pd.read_csv(RAW / "hpa" / "proteinatlas.tsv", sep="\t", low_memory=False)
    prog_cols = [c for c in df.columns if c.startswith("Cancer prognostics")]
    long_rows = []
    for col in prog_cols:
        # "Cancer prognostics - Lung Adenocarcinoma (TCGA)" / "(validation)"
        cohort = "validation" if col.endswith("(validation)") else "TCGA"
        cancer_type = col.replace("Cancer prognostics - ", "")
        cancer_type = cancer_type.replace(" (TCGA)", "").replace(" (validation)", "")
        sub = df[["Gene", col]].dropna(subset=[col]).copy()
        sub.columns = ["gene", "raw"]
        sub["cancer_type"] = cancer_type
        sub["cohort"] = cohort
        # raw looks like "unprognostic (1.74e-1)" or "potential prognostic favorable (5.30e-6)"
        sub["call"] = sub["raw"].str.extract(r"^(.*)\s+\(")[0]
        sub["p_value"] = sub["raw"].str.extract(r"\(([\d.eE+-]+)\)")[0].astype(float)
        long_rows.append(sub.drop(columns=["raw"]))
    return pd.concat(long_rows, ignore_index=True)


def hpa_lung_adenocarcinoma_claims() -> pd.DataFrame:
    """Subset of the HPA corpus matching this study's cancer type (TCGA cohort
    only, i.e. the discovery-side claim, not HPA's own validation column)."""
    hpa = load_hpa_prognostic_corpus()
    return hpa[(hpa["cancer_type"] == "Lung Adenocarcinoma") & (hpa["cohort"] == "TCGA")]
