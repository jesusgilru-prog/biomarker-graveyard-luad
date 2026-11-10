"""Fetch + build the CPTAC-LUAD discovery cohort.

Downloads (via the `cptac` package, no credentials needed):
  - mssm-all_cancers clinical (pan-cancer, filtered to tumor_code == 'LUAD')
  - washu-luad transcriptomics (gene-level FPKM, tumor samples)

Run: python src/fetch_cptac_luad.py
Writes: data/raw/cptac/*, data/processed/cptac_luad_fpkm.csv.gz,
        data/processed/cptac_luad_clinical.csv
"""
from __future__ import annotations

import shutil
import warnings
from pathlib import Path

import cptac
import pandas as pd

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw" / "cptac"
PROCESSED = REPO / "data" / "processed"


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    cptac.download(cancer="all_cancers", source="mssm", dtype="clinical",
                    data_file="clinical_Pan-cancer.May2022.tsv.gz")
    cptac.download(cancer="luad", source="washu", dtype="transcriptomics",
                    data_file="LUAD_tumor_RNA-Seq_Expr_WashU_FPKM.tsv.gz")

    cptac_data_dir = Path(cptac.DATA_DIR)
    shutil.copy(cptac_data_dir / "mssm-all_cancers" / "clinical_Pan-cancer.May2022.tsv.gz",
                RAW / "clinical_Pan-cancer.May2022.tsv.gz")
    shutil.copy(cptac_data_dir / "washu-luad" / "LUAD_tumor_RNA-Seq_Expr_WashU_FPKM.tsv.gz",
                RAW / "LUAD_tumor_RNA-Seq_Expr_WashU_FPKM.tsv.gz")

    clin = pd.read_csv(RAW / "clinical_Pan-cancer.May2022.tsv.gz", sep="\t")
    luad = clin[clin["tumor_code"] == "LUAD"].copy()
    surv = luad[["case_id", "Overall survival, days", "Survival status (1, dead; 0, alive)"]].copy()
    surv.columns = ["case_id", "os_time_days", "os_event"]

    expr = pd.read_csv(RAW / "LUAD_tumor_RNA-Seq_Expr_WashU_FPKM.tsv.gz", sep="\t")
    # Duplicated symbols (399 in this release) map to distinct Ensembl IDs, often a near-silent retrocopy
    # next to the real gene. Keep the row with the highest mean expression (v1 kept the FIRST row; see
    # revision_gene_nov/analisis/PASO1_RESULTADO.md).
    expr = expr.dropna(subset=["gene_name"]).copy()
    expr["_m"] = expr.drop(columns=["gene_id", "gene_name"]).astype(float).mean(axis=1)
    expr = expr.sort_values("_m", ascending=False).drop_duplicates(subset="gene_name").drop(columns="_m")
    expr = expr.set_index("gene_name")
    expr = expr.drop(columns=["gene_id"])
    expr.columns = [c.replace("-T", "") for c in expr.columns]

    common = sorted(set(expr.columns) & set(surv["case_id"]))
    usable = surv[surv["case_id"].isin(common)].dropna(subset=["os_time_days"])
    keep = usable["case_id"].tolist()

    expr.loc[:, keep].to_csv(PROCESSED / "cptac_luad_fpkm.csv.gz", compression="gzip")
    usable.to_csv(PROCESSED / "cptac_luad_clinical.csv", index=False)
    print(f"CPTAC-LUAD discovery: {len(keep)} samples, "
          f"{int(usable['os_event'].sum())} events, {expr.shape[0]} genes.")


if __name__ == "__main__":
    main()
