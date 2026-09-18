"""Fetch + build the GSE68465 replication cohort (Shedden et al. 2008, GEO).

Downloads the series matrix directly from the GEO FTP mirror (no GEOparse --
in this environment GEOparse silently failed to parse this file's series
matrix format, returning 0 samples; a hand-rolled parser against the plain
text format is used instead and was verified against the raw file by eye).

Also downloads the GPL96 (Affymetrix HG-U133A) platform annotation to map
probe IDs to HGNC gene symbols, collapsing multiple probes per gene by
keeping the probe with the highest mean expression (standard practice).

Run: python src/fetch_gse68465.py
Writes: data/raw/geo/GSE68465_series_matrix.txt.gz, data/raw/geo/GPL96.txt,
        data/processed/gse68465_luad_expr.csv.gz,
        data/processed/gse68465_luad_clinical.csv
"""
from __future__ import annotations

import gzip
import io
from pathlib import Path

import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw" / "geo"
PROCESSED = REPO / "data" / "processed"

SERIES_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE68nnn/GSE68465/matrix/GSE68465_series_matrix.txt.gz"
GPL_URL = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL96&targ=self&form=text&view=full"


def _download(url: str, dest: Path):
    r = requests.get(url, timeout=120)
    dest.write_bytes(r.content)


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    series_path = RAW / "GSE68465_series_matrix.txt.gz"
    gpl_path = RAW / "GPL96.txt"
    if not series_path.exists():
        _download(SERIES_URL, series_path)
    if not gpl_path.exists():
        _download(GPL_URL, gpl_path)

    with gzip.open(series_path, "rt") as f:
        lines = f.readlines()

    sample_ids = [x.strip('"\n') for x in
                  next(l for l in lines if l.startswith("!Sample_geo_accession")).strip().split("\t")[1:]]

    def parse_char(line, key):
        vals = [x.strip('"\n') for x in line.strip().split("\t")[1:]]
        return {sid: v.split(":", 1)[1].strip() for sid, v in zip(sample_ids, vals) if v.startswith(key + ":")}

    vital, months_last = {}, {}
    for l in (l for l in lines if l.startswith("!Sample_characteristics_ch1")):
        vital.update(parse_char(l, "vital_status"))
        months_last.update(parse_char(l, "months_to_last_contact_or_death"))

    clin = pd.DataFrame({"sample_id": sample_ids})
    clin["os_event"] = clin["sample_id"].map(vital).map({"Dead": 1, "Alive": 0})
    clin["os_time_months"] = pd.to_numeric(clin["sample_id"].map(months_last), errors="coerce")
    clin = clin.dropna(subset=["os_event", "os_time_months"])

    start = next(i for i, l in enumerate(lines) if l.startswith("!series_matrix_table_begin"))
    end = next(i for i, l in enumerate(lines) if l.startswith("!series_matrix_table_end"))
    mat = pd.read_csv(io.StringIO("".join(lines[start + 1:end])), sep="\t")
    mat = mat.set_index(mat.columns[0])
    mat.columns = [c.strip('"') for c in mat.columns]

    with open(gpl_path) as f:
        glines = f.readlines()
    gstart = next(i for i, l in enumerate(glines) if l.startswith("!platform_table_begin"))
    gend = next(i for i, l in enumerate(glines) if l.startswith("!platform_table_end"))
    annot = pd.read_csv(io.StringIO("".join(glines[gstart + 1:gend])), sep="\t", low_memory=False)
    probe2gene = annot.set_index("ID")["Gene Symbol"].dropna()
    probe2gene = probe2gene[probe2gene != "---"]

    mat["gene_symbol"] = mat.index.map(probe2gene)
    mat = mat.dropna(subset=["gene_symbol"])
    mat["mean_expr"] = mat.drop(columns=["gene_symbol"]).mean(axis=1)
    mat = mat.sort_values("mean_expr", ascending=False).drop_duplicates(subset="gene_symbol", keep="first")
    gene_index = mat["gene_symbol"]
    mat = mat.drop(columns=["gene_symbol", "mean_expr"])
    mat.index = gene_index

    common = sorted(set(mat.columns) & set(clin["sample_id"]))
    clin = clin[clin["sample_id"].isin(common)]
    mat = mat[clin["sample_id"].tolist()]

    mat.to_csv(PROCESSED / "gse68465_luad_expr.csv.gz", compression="gzip")
    clin.to_csv(PROCESSED / "gse68465_luad_clinical.csv", index=False)
    print(f"GSE68465: {mat.shape[1]} samples, {mat.shape[0]} genes, "
          f"{int(clin['os_event'].sum())} events.")


if __name__ == "__main__":
    main()
