"""Phase 6, per external peer-review simulation round 1 (Codex + Fable acting as referees
for Computers in Biology and Medicine, 2026-08-14): both reviewers independently flagged the
same blocking concern -- the 121-gene panel is enriched for cell-cycle/proliferation genes,
and proliferation covaries with tumor stage, so the cross-cohort directional concordance
could be a stage proxy rather than a stage-independent signal. Both reviewers verified stage
and age ARE publicly available for these cohorts (this was previously, incorrectly, declared
as "not available in the loaders" -- true of the loaders, not of the data).

Sources:
- CPTAC-LUAD: `cptac` package, source='mssm', columns 'tumor_stage_pathological' (Stage
  I-IV) and 'age'.
- TCGA-LUAD: GDC REST API, 'diagnoses.ajcc_pathologic_stage' (AJCC substages, e.g. 'Stage
  IIIA') and 'demographic.age_at_index'.
- GSE68465: raw GEO series matrix, 'disease_stage' (pathological T/N only, no M) and 'age'.

Because the three cohorts use different staging vocabularies (CPTAC/TCGA give AJCC stage
groups directly; GSE68465 gives only pT/pN), all three are collapsed to a common binary
early (I-II) vs. late (III-IV) grouping rather than attempting exact AJCC harmonization.
For GSE68465, late = N2/N3, or T4, or (T3 and N1) -- the standard AJCC 7th/8th edition rule
of thumb for stage III; early = everything else. This is an approximation, not an exact
re-staging, and is reported as such.

For each of the 121 panel genes, in each cohort with usable stage+age data, a Cox
proportional-hazards model (lifelines) is fit with standardized expression, stage (binary),
and age as covariates; the expression coefficient's sign and Wald p-value are compared to
the gene's original (unadjusted) direction from Section 3.2 of the manuscript.

Run: python src/phase6_stage_age_sensitivity.py
Writes: results/phase6_stage_age_sensitivity.json, results/phase6_stage_age_per_gene.csv
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import (load_cptac_luad_discovery, load_tcga_luad_replication,  # noqa: E402
                   load_gse68465_replication)
from survival_stats import _logrank_components  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"


def chi2_dir(x, y):
    c, o = _logrank_components(x, y)
    return c, (1 if o > 0 else (-1 if o < 0 else 0))


def get_panel_121():
    X1, y1 = load_cptac_luad_discovery()
    X2, y2 = load_tcga_luad_replication()
    X3, y3 = load_gse68465_replication()
    panel = sorted(set(X1.index) & set(X2.index) & set(X3.index))
    ya1 = y1[["os_time_days", "os_event"]].values
    ya2 = y2[["os_time_days", "os_event"]].values
    ya3 = y3[["os_time_months", "os_event"]].values
    genes = sorted(
        g for g in panel
        if sps.chi2.sf(chi2_dir(X1.loc[g].values, ya1)[0], 1) < 0.05
        and sps.chi2.sf(chi2_dir(X2.loc[g].values, ya2)[0], 1) < 0.05
        and sps.chi2.sf(chi2_dir(X3.loc[g].values, ya3)[0], 1) < 0.05
        and chi2_dir(X1.loc[g].values, ya1)[1] == chi2_dir(X2.loc[g].values, ya2)[1]
        == chi2_dir(X3.loc[g].values, ya3)[1] != 0
    )
    orig_dir = {g: chi2_dir(X1.loc[g].values, ya1)[1] for g in genes}
    return genes, orig_dir


def cptac_stage_age():
    import cptac
    l = cptac.Luad()
    clin = l.get_clinical(source="mssm")
    stage = clin["tumor_stage_pathological"]
    late = stage.isin(["Stage III", "Stage IV"])
    age = pd.to_numeric(clin["age"], errors="coerce")
    out = pd.DataFrame({"stage_late": late.astype(float), "age": age}, index=clin.index)
    out.index.name = "case_id"
    return out.dropna()


def tcga_stage_age():
    import requests
    r = requests.get(
        "https://api.gdc.cancer.gov/cases",
        params={
            "filters": json.dumps({"op": "=", "content": {
                "field": "project.project_id", "value": "TCGA-LUAD"}}),
            "fields": "diagnoses.ajcc_pathologic_stage,demographic.age_at_index,submitter_id",
            "size": 700, "format": "json",
        }, timeout=60,
    )
    rows = []
    for d in r.json()["data"]["hits"]:
        dx = d.get("diagnoses", [{}])
        stage = dx[0].get("ajcc_pathologic_stage") if dx else None
        age = d.get("demographic", {}).get("age_at_index")
        if stage is None or age is None:
            continue
        late = any(s in stage for s in ["III", "IV"])
        rows.append((d["submitter_id"], float(late), age))
    out = pd.DataFrame(rows, columns=["case_id", "stage_late", "age"]).set_index("case_id")
    return out


def _gse_late_from_pTN(stage_str):
    # stage_str like "pN2pT2" or "pNXpT1"
    import re
    m_n = re.search(r"pN(\d|X)", stage_str)
    m_t = re.search(r"pT(\d|X)", stage_str)
    if not m_n or not m_t or m_n.group(1) == "X" or m_t.group(1) == "X":
        return np.nan
    n, t = int(m_n.group(1)), int(m_t.group(1))
    late = (n >= 2) or (t == 4) or (t == 3 and n == 1)
    return float(late)


def gse68465_stage_age():
    with gzip.open(REPO / "data" / "raw" / "geo" / "GSE68465_series_matrix.txt.gz", "rt") as f:
        lines = f.readlines()
    sample_ids = [x.strip('"\n') for x in
                  next(l for l in lines if l.startswith("!Sample_geo_accession")).strip().split("\t")[1:]]
    stage_line = next(l for l in lines if l.startswith("!Sample_characteristics") and "disease_stage" in l)
    age_line = next(l for l in lines if l.startswith("!Sample_characteristics") and '"age:' in l)
    stages = [x.strip('"\n').replace("disease_stage: ", "") for x in stage_line.strip().split("\t")[1:]]
    ages = [x.strip('"\n').replace("age: ", "") for x in age_line.strip().split("\t")[1:]]
    rows = []
    for sid, st, ag in zip(sample_ids, stages, ages):
        late = _gse_late_from_pTN(st)
        try:
            age_val = float(ag)
        except ValueError:
            age_val = np.nan
        if np.isnan(late) or np.isnan(age_val):
            continue
        rows.append((sid, late, age_val))
    return pd.DataFrame(rows, columns=["sample_id", "stage_late", "age"]).set_index("sample_id")


def cox_adjusted_direction(expr, time, event, stage_late, age):
    df = pd.DataFrame({"expr": (expr - expr.mean()) / expr.std(), "time": time,
                        "event": event, "stage_late": stage_late, "age": age})
    cph = CoxPHFitter()
    try:
        cph.fit(df, duration_col="time", event_col="event")
        coef = cph.summary.loc["expr", "coef"]
        p = cph.summary.loc["expr", "p"]
        direction = 1 if coef > 0 else (-1 if coef < 0 else 0)
        return direction, p
    except Exception:
        return None, None


def main():
    genes, orig_dir = get_panel_121()
    print(f"Panel: {len(genes)} genes")

    X1, y1 = load_cptac_luad_discovery()
    X2, y2 = load_tcga_luad_replication()
    X3, y3 = load_gse68465_replication()

    cptac_meta = cptac_stage_age()
    tcga_meta = tcga_stage_age()
    gse_meta = gse68465_stage_age()

    print(f"CPTAC stage+age available: {len(cptac_meta)}/{y1.shape[0]}")
    print(f"TCGA stage+age available: {len(tcga_meta)}/{y2.shape[0]}")
    print(f"GSE68465 stage+age available: {len(gse_meta)}/{y3.shape[0]}")

    cohorts = {}
    y1c = y1.set_index("case_id")
    common1 = [c for c in cptac_meta.index if c in y1c.index and c in X1.columns]
    cohorts["cptac"] = (X1[common1], y1c.loc[common1, "os_time_days"].values,
                         y1c.loc[common1, "os_event"].values, cptac_meta.loc[common1])

    y2c = y2.set_index("submitter_id")
    common2 = [c for c in tcga_meta.index if c in y2c.index and c in X2.columns]
    cohorts["tcga"] = (X2[common2], y2c.loc[common2, "os_time_days"].values,
                        y2c.loc[common2, "os_event"].values, tcga_meta.loc[common2])

    y3c = y3.set_index("sample_id")
    common3 = [c for c in gse_meta.index if c in y3c.index and c in X3.columns]
    cohorts["gse68465"] = (X3[common3], y3c.loc[common3, "os_time_months"].values,
                            y3c.loc[common3, "os_event"].values, gse_meta.loc[common3])

    for tag, (X, t, e, meta) in cohorts.items():
        print(f"  {tag}: n={X.shape[1]} usable for adjustment")

    rows = []
    for g in genes:
        row = {"gene": g, "original_direction": orig_dir[g]}
        for tag, (X, t, e, meta) in cohorts.items():
            expr = X.loc[g].values
            d, p = cox_adjusted_direction(expr, t, e, meta["stage_late"].values, meta["age"].values)
            row[f"{tag}_adjusted_direction"] = d
            row[f"{tag}_adjusted_p"] = p
            row[f"{tag}_concordant_with_original"] = (d == orig_dir[g]) if d is not None else None
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "phase6_stage_age_per_gene.csv", index=False)

    summary = {"n_genes": len(genes)}
    for tag in cohorts:
        n_avail = df[f"{tag}_adjusted_direction"].notna().sum()
        n_concordant = df[f"{tag}_concordant_with_original"].sum()
        n_still_sig = ((df[f"{tag}_adjusted_p"] < 0.05) & df[f"{tag}_concordant_with_original"]).sum()
        summary[tag] = dict(
            n_samples=cohorts[tag][0].shape[1],
            n_genes_fit=int(n_avail),
            n_direction_concordant=int(n_concordant),
            pct_direction_concordant=round(100 * n_concordant / n_avail, 1) if n_avail else None,
            n_still_nominally_significant_and_concordant=int(n_still_sig),
        )
    with open(RESULTS / "phase6_stage_age_sensitivity.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
