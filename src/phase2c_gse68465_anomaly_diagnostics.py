"""Promotes two ad hoc analyses from the Fase 2c investigation (originally run
interactively, see PROGRESS.md) to a versioned, independently re-runnable script with
archived output, per Fable's review round 1 finding (B14): the GSE68465
non-monotonic-FDR investigation (Section 3.4 of the manuscript) cited a per-site
mortality breakdown and a full-panel Galwey Meff that existed only as PROGRESS.md prose,
not as reproducible code + output.

1. Per-site mortality breakdown for GSE68465 (4 recruiting sites: MI, MSKCC, DFCI, HLM),
   extracted from the raw GEO series matrix's Sample_source_name field (not present in
   the already-processed clinical CSV).
2. Effective number of independent tests (Galwey 2009 Meff) over the FULL 11,484-gene
   shared panel, in each of the three cohorts -- distinct from the 121-gene panel-specific
   Meff computed in phase5_reviewer_robustness.py.

Run: python src/phase2c_gse68465_anomaly_diagnostics.py
Writes: results/phase2c_site_mortality.json, results/phase2c_full_panel_meff.json
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_cptac_luad_discovery, load_tcga_luad_replication, load_gse68465_replication  # noqa: E402
from phase5_reviewer_robustness import (_correlation_eigenvalues, _meff_cheverud_nyholt,  # noqa: E402
                                         _meff_galwey)

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
RAW = REPO / "data" / "raw" / "geo" / "GSE68465_series_matrix.txt.gz"


def site_mortality():
    with gzip.open(RAW, "rt") as f:
        lines = f.readlines()
    sample_ids = [x.strip('"\n') for x in
                  next(l for l in lines if l.startswith("!Sample_geo_accession")).strip().split("\t")[1:]]
    source = [x.strip('"\n') for x in
              next(l for l in lines if l.startswith("!Sample_source_name")).strip().split("\t")[1:]]
    site_map = dict(zip(sample_ids, source))

    _, clin = None, None
    clin = pd.read_csv(REPO / "data" / "processed" / "gse68465_luad_clinical.csv")
    clin["site"] = clin["sample_id"].map(site_map)
    grp = clin.groupby("site")["os_event"].agg(["mean", "count"])
    out = {
        site: {"mortality_rate": round(float(row["mean"]), 4), "n": int(row["count"])}
        for site, row in grp.iterrows()
    }
    out["mortality_range"] = [
        round(float(grp["mean"].min()), 4), round(float(grp["mean"].max()), 4)
    ]
    return out


def full_panel_meff():
    """Effective dimensionality of the FULL 11,484-gene candidate universe (distinct from
    panel_meff() in phase5_reviewer_robustness.py, which does the same computation
    restricted to the 121-gene panel). Reuses the corrected eigenvalue routine and reports
    both the Cheverud/Nyholt and Galwey (2009) estimators -- see phase5's docstrings for why
    an earlier version of this analysis, which reported only Cheverud/Nyholt under the
    Galwey label, was wrong (caught in external review round 2, 2026-08-14)."""
    X1, _ = load_cptac_luad_discovery()
    X2, _ = load_tcga_luad_replication()
    X3, _ = load_gse68465_replication()
    panel = sorted(set(X1.index) & set(X2.index) & set(X3.index))
    m = len(panel)
    out = {"panel_size": m}
    for tag, X in [("cptac", X1), ("tcga", X2), ("gse68465", X3)]:
        eigs, n, m_eff = _correlation_eigenvalues(X, panel)
        assert abs(eigs.sum() - m_eff) < 1e-3, f"eigenvalues should sum to m_eff={m_eff}, got {eigs.sum()}"
        meff_cn = _meff_cheverud_nyholt(eigs, m_eff)
        meff_galwey = _meff_galwey(eigs)
        out[tag] = dict(
            n_samples=n, n_zero_variance_genes_dropped=m - m_eff,
            meff_cheverud_nyholt=round(float(meff_cn), 1),
            meff_cheverud_nyholt_pct=round(100 * meff_cn / m, 2),
            meff_galwey=round(float(meff_galwey), 1),
            meff_galwey_pct=round(100 * meff_galwey / m, 2),
        )
    return out


def main():
    sm = site_mortality()
    print("Site mortality:", json.dumps(sm, indent=2))
    with open(RESULTS / "phase2c_site_mortality.json", "w") as f:
        json.dump(sm, f, indent=2)

    fm = full_panel_meff()
    print("Full-panel Meff:", json.dumps(fm, indent=2))
    with open(RESULTS / "phase2c_full_panel_meff.json", "w") as f:
        json.dump(fm, f, indent=2)


if __name__ == "__main__":
    main()
