"""Age, sex and stage of the ANALYZED survival cohorts (Table 1 of the Lung Cancer submission).

Sources (same as phase6_stage_age_sensitivity.py, plus sex):
- CPTAC-LUAD: data/raw/cptac/clinical_Pan-cancer.May2022.tsv.gz (consent/age, consent/sex,
  baseline/tumor_stage_pathological), restricted to the 106 analyzed cases.
- TCGA-LUAD: GDC REST API (demographic.age_at_index, demographic.sex_at_birth,
  diagnoses.ajcc_pathologic_stage), restricted to the 508 analyzed cases.
- GSE68465: raw GEO series matrix (age, Sex, disease_stage pT/pN), restricted to the 442 analyzed samples.
Late stage = III/IV (AJCC) for CPTAC/TCGA; for GSE68465 the T/N approximation of phase6.
Writes results/cohort_demographics_table1.json
"""
import gzip, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[1]


def _gse_late_from_pTN(stage_str):
    """Copied verbatim from phase6_stage_age_sensitivity.py (same T/N approximation)."""
    import re
    m_n = re.search(r"pN(\d|X)", stage_str)
    m_t = re.search(r"pT(\d|X)", stage_str)
    if not m_n or not m_t or m_n.group(1) == "X" or m_t.group(1) == "X":
        return np.nan
    n, t = int(m_n.group(1)), int(m_t.group(1))
    late = (n >= 2) or (t == 4) or (t == 3 and n == 1)
    return float(late)

P = REPO / "data" / "processed"


def summ(age, female, late):
    age = pd.to_numeric(age, errors="coerce")
    a = age.dropna(); f = female.dropna(); l = late.dropna()
    return {"n": int(len(age)), "age_n": int(len(a)), "age_median": float(a.median()),
            "age_q1": float(a.quantile(.25)), "age_q3": float(a.quantile(.75)),
            "sex_n": int(len(f)), "female_n": int(f.sum()), "female_pct": float(100 * f.mean()),
            "stage_n": int(len(l)), "late_n": int(l.sum()), "late_pct": float(100 * l.mean())}


def cptac():
    ids = pd.read_csv(P / "cptac_luad_clinical.csv").case_id
    c = pd.read_csv(REPO / "data/raw/cptac/clinical_Pan-cancer.May2022.tsv.gz", sep="\t", low_memory=False)
    c = c[c.tumor_code == "LUAD"].drop_duplicates("case_id").set_index("case_id").reindex(ids)
    sex = c["consent/sex"].str.lower()
    female = sex.map({"female": 1.0, "male": 0.0})
    st = c["baseline/tumor_stage_pathological"]
    late = st.where(st.notna() & st.astype(str).str.contains("Stage")).map(
        lambda s: float(any(k in str(s) for k in ["III", "IV"])) if isinstance(s, str) else np.nan)
    return summ(c["consent/age"], female, late)


def _ajcc_late(stg):
    """Recognized AJCC pathologic stage strings only (Stage 0/I/II/III/IV with A/B/C/1/2/3 substages);
    anything else (None, 'Not Reported', '[Discrepancy]', ...) is missing, not early."""
    if not isinstance(stg, str):
        return np.nan
    m = re.fullmatch(r"Stage (0|IV|III|II|I)[ABC]?[123]?", stg.strip())
    if not m:
        return np.nan
    return float(m.group(1) in ("III", "IV"))


def tcga():
    import requests
    ids = pd.read_csv(P / "tcga_luad_clinical_full.csv").submitter_id
    r = requests.get("https://api.gdc.cancer.gov/cases", params={
        "filters": json.dumps({"op": "=", "content": {"field": "project.project_id", "value": "TCGA-LUAD"}}),
        "fields": "submitter_id,demographic.age_at_index,demographic.sex_at_birth,diagnoses.ajcc_pathologic_stage",
        "size": 1000, "format": "json"}, timeout=120)
    js = r.json()
    (REPO / "results" / "tcga_luad_gdc_demographics_raw.json").write_text(json.dumps(js, indent=1))
    rows = {}
    for h in js["data"]["hits"]:
        dm = h.get("demographic", {}) or {}
        dx = (h.get("diagnoses") or [{}])[0]
        stg = dx.get("ajcc_pathologic_stage")
        g = (dm.get("sex_at_birth") or "").lower()
        rows[h["submitter_id"]] = (dm.get("age_at_index"), {"female": 1.0, "male": 0.0}.get(g, np.nan),
                                   _ajcc_late(stg))
    d = pd.DataFrame([rows.get(i, (np.nan, np.nan, np.nan)) for i in ids], columns=["age", "female", "late"])
    s = summ(d.age, d.female, d.late)
    s["missing_age"] = int(d.age.isna().sum()); s["missing_stage"] = int(d.late.isna().sum())
    s["missing_both"] = int((d.age.isna() & d.late.isna()).sum())
    s["unrecognized_stage_strings"] = sorted({str((h.get("diagnoses") or [{}])[0].get("ajcc_pathologic_stage"))
        for h in js["data"]["hits"] if h["submitter_id"] in set(ids) and np.isnan(_ajcc_late((h.get("diagnoses") or [{}])[0].get("ajcc_pathologic_stage")))})
    return s


def gse():
    ids = pd.read_csv(P / "gse68465_luad_clinical.csv").sample_id
    with gzip.open(REPO / "data/raw/geo/GSE68465_series_matrix.txt.gz", "rt") as f:
        lines = f.readlines()
    sid = [x.strip('"\n') for x in next(l for l in lines if l.startswith("!Sample_geo_accession")).strip().split("\t")[1:]]
    def row(key):
        l = next(l for l in lines if l.startswith("!Sample_characteristics") and f'"{key}:' in l)
        return [x.strip('"\n').split(": ", 1)[-1] for x in l.strip().split("\t")[1:]]
    d = pd.DataFrame({"age": row("age"), "sex": row("Sex"), "stage": row("disease_stage")}, index=sid).reindex(ids)
    female = d.sex.str.lower().map({"female": 1.0, "male": 0.0})
    late = d.stage.map(lambda s: _gse_late_from_pTN(s) if isinstance(s, str) else np.nan)
    return summ(d.age, female, late)


if __name__ == "__main__":
    out = {"CPTAC-LUAD": cptac(), "TCGA-LUAD": tcga(), "GSE68465": gse()}
    (REPO / "results" / "cohort_demographics_table1.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
