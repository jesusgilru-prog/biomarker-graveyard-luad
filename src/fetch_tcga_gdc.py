"""Fetch + build a TCGA replication cohort directly via the GDC REST API.

No `cptac` involved (it does not cover general TCGA) and no token/DUA needed:
this only touches "open access" files (processed RNA-seq gene counts), which
the GDC serves to anyone. Controlled-access data (raw sequence) would need a
dbGaP data use agreement; this study never requests it.

Steps:
  1. Query /cases for clinical fields, including `follow_ups.days_to_follow_up`
     (NOT `diagnoses.days_to_last_follow_up`, which is null for every TCGA-BRCA/
     LUAD case checked -- the real last-contact time for living patients lives
     in the separate `follow_ups` entity).
  2. Query /files for open-access "STAR - Counts" Gene Expression Quantification
     files restricted to Primary Tumor samples, one file per case.
  3. Reproducibly sample up to `n_sample` cases (seed=42) and bulk-download them
     via POST /data (one tar.gz).
  4. Parse each per-sample TSV, keep protein_coding genes, build a genes x
     samples TPM matrix, join to the survival table built in step 1.

Run: python src/fetch_tcga_gdc.py --project TCGA-LUAD --n-sample 300
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import shutil
import tarfile
from pathlib import Path

import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[1]
GDC = "https://api.gdc.cancer.gov"


def fetch_clinical(project: str) -> pd.DataFrame:
    fields = ["case_id", "submitter_id", "demographic.vital_status",
              "demographic.days_to_death", "follow_ups.days_to_follow_up"]
    params = {
        "filters": json.dumps({"op": "and", "content": [
            {"op": "in", "content": {"field": "project.project_id", "value": [project]}}]}),
        "fields": ",".join(fields), "format": "JSON", "size": "3000",
    }
    data = requests.get(f"{GDC}/cases", params=params, timeout=60).json()["data"]["hits"]
    rows = []
    for c in data:
        demo = c.get("demographic", {}) or {}
        fups = [f.get("days_to_follow_up") for f in (c.get("follow_ups") or [])
                if f.get("days_to_follow_up") is not None]
        last_fup = max(fups) if fups else None
        rows.append({"case_id": c["case_id"], "submitter_id": c["submitter_id"],
                     "vital_status": demo.get("vital_status"),
                     "days_to_death": demo.get("days_to_death"),
                     "days_to_last_follow_up": last_fup})
    clin = pd.DataFrame(rows)

    def os_time(row):
        if row["vital_status"] == "Dead":
            return row["days_to_death"] if pd.notna(row["days_to_death"]) else row["days_to_last_follow_up"]
        return row["days_to_last_follow_up"]

    clin["os_time_days"] = clin.apply(os_time, axis=1)
    clin["os_event"] = (clin["vital_status"] == "Dead").astype(int)
    return clin.dropna(subset=["os_time_days"])


def fetch_primary_tumor_files(project: str) -> dict:
    fields = ["file_id", "cases.case_id", "cases.submitter_id", "cases.samples.sample_type"]
    params = {
        "filters": json.dumps({"op": "and", "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": [project]}},
            {"op": "in", "content": {"field": "data_type", "value": ["Gene Expression Quantification"]}},
            {"op": "in", "content": {"field": "analysis.workflow_type", "value": ["STAR - Counts"]}},
            {"op": "in", "content": {"field": "access", "value": ["open"]}},
            {"op": "in", "content": {"field": "cases.samples.sample_type", "value": ["Primary Tumor"]}},
        ]}),
        "fields": ",".join(fields), "format": "JSON", "size": "2000",
    }
    data = requests.get(f"{GDC}/files", params=params, timeout=60).json()["data"]["hits"]
    seen = {}
    for d in data:
        cid = d["cases"][0]["case_id"]
        if cid not in seen:
            seen[cid] = {"file_id": d["file_id"], "submitter_id": d["cases"][0]["submitter_id"]}
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True, help="e.g. TCGA-LUAD, TCGA-BRCA")
    ap.add_argument("--n-sample", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tag = args.project.split("-")[-1].lower()
    raw_dir = REPO / "data" / "raw" / "gdc_tcga"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed = REPO / "data" / "processed"

    clin = fetch_clinical(args.project)
    print(f"{args.project}: {clin.shape[0]} cases with usable survival "
          f"({int(clin['os_event'].sum())} events)")

    files_by_case = fetch_primary_tumor_files(args.project)
    case_ids = sorted(files_by_case.keys())
    random.seed(args.seed)
    sampled = random.sample(case_ids, min(args.n_sample, len(case_ids)))
    file_ids = [files_by_case[c]["file_id"] for c in sampled]
    mapping = {files_by_case[c]["file_id"]: files_by_case[c]["submitter_id"] for c in sampled}

    r = requests.post(f"{GDC}/data", data=json.dumps({"ids": file_ids}),
                       headers={"Content-Type": "application/json"}, timeout=600, stream=True)
    tar_path = raw_dir / f"{tag}_sample{args.n_sample}.tar.gz"
    with open(tar_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)

    extract_dir = raw_dir / f"extracted_{tag}"
    with tarfile.open(tar_path) as tf:
        tf.extractall(extract_dir)

    records, gene_names = {}, None
    for file_id, submitter_id in mapping.items():
        tsvs = glob.glob(os.path.join(extract_dir, file_id, "*.tsv"))
        if not tsvs:
            continue
        df = pd.read_csv(tsvs[0], sep="\t", skiprows=1)
        df = df[df["gene_type"] == "protein_coding"].drop_duplicates(subset="gene_name")
        s = df.set_index("gene_name")["tpm_unstranded"]
        records[submitter_id] = s
        gene_names = set(s.index) if gene_names is None else (gene_names & set(s.index))
    shutil.rmtree(extract_dir)

    mat = pd.DataFrame(records).loc[sorted(gene_names)]
    clin_sub = clin[clin["submitter_id"].isin(mat.columns)].dropna(subset=["os_time_days"])
    keep = clin_sub["submitter_id"].tolist()
    mat = mat[keep]

    mat.to_csv(processed / f"tcga_{tag}_tpm_n{args.n_sample}.csv.gz", compression="gzip")
    clin_sub.to_csv(processed / f"tcga_{tag}_clinical_n{args.n_sample}.csv", index=False)
    print(f"Final: {mat.shape[1]} samples, {mat.shape[0]} protein-coding genes, "
          f"{int(clin_sub['os_event'].sum())} events.")


if __name__ == "__main__":
    main()
