"""Generates Supplementary Table S1: the 121-gene cross-cohort directional-replication
panel (Section 3.2 of the manuscript), with per-cohort test statistics, so the panel is
independently checkable gene-by-gene rather than only as a headline count.

Run: python src/generate_table_s1.py
Writes: results/table_s1_121_gene_panel.csv, paper/supplement_table_s1.tex
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_cptac_luad_discovery, load_tcga_luad_replication, load_gse68465_replication  # noqa: E402
from survival_stats import _logrank_components  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
PAPER = REPO / "paper"


def chi2_p_dir(x, y):
    c, o = _logrank_components(x, y)
    d = 1 if o > 0 else (-1 if o < 0 else 0)
    return c, sps.chi2.sf(c, df=1), d


def main():
    X1, y1 = load_cptac_luad_discovery()
    X2, y2 = load_tcga_luad_replication()
    X3, y3 = load_gse68465_replication()
    panel = sorted(set(X1.index) & set(X2.index) & set(X3.index))
    ya1 = y1[["os_time_days", "os_event"]].values
    ya2 = y2[["os_time_days", "os_event"]].values
    ya3 = y3[["os_time_months", "os_event"]].values

    rows = []
    for g in panel:
        c1, p1, d1 = chi2_p_dir(X1.loc[g].values, ya1)
        c2, p2, d2 = chi2_p_dir(X2.loc[g].values, ya2)
        c3, p3, d3 = chi2_p_dir(X3.loc[g].values, ya3)
        if p1 < 0.05 and p2 < 0.05 and p3 < 0.05 and d1 == d2 == d3 and d1 != 0:
            rows.append(dict(
                gene=g,
                direction=("unfavorable (high expression -> worse survival)" if d1 > 0
                           else "favorable (high expression -> better survival)"),
                cptac_chi2=c1, cptac_p=p1,
                tcga_chi2=c2, tcga_p=p2,
                gse68465_chi2=c3, gse68465_p=p3,
            ))

    df = pd.DataFrame(rows).sort_values("gene").reset_index(drop=True)
    assert len(df) == 121, f"expected 121 genes, got {len(df)}"
    df.to_csv(RESULTS / "table_s1_121_gene_panel.csv", index=False)
    print(f"{len(df)} genes written to results/table_s1_121_gene_panel.csv")

    # minimal LaTeX longtable for the supplement (gene + direction + 3 p-values only,
    # full chi2 statistics remain in the CSV for anyone who wants them)
    lines = [
        r"\documentclass{article}",
        r"\usepackage{longtable,booktabs,geometry}",
        r"\geometry{margin=2cm}",
        r"\begin{document}",
        r"\small",
        r"\begin{longtable}{lcccc}",
        r"\caption{Supplementary Table S1: the 121-gene cross-cohort directional-replication "
        r"panel. Direction: U = high expression associated with worse survival (unfavorable), "
        r"F = high expression associated with better survival (favorable). "
        r"$p$-values are naive (asymptotic $\chi^2_1$), not permutation-calibrated "
        r"(see main text Section 3.2 for the calibrated significance test at the panel level).}\\",
        r"\toprule",
        r"Gene & Dir. & CPTAC $p$ & TCGA $p$ & GSE68465 $p$ \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Gene & Dir. & CPTAC $p$ & TCGA $p$ & GSE68465 $p$ \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]
    for _, row in df.iterrows():
        d = "U" if "unfavorable" in row["direction"] else "F"
        lines.append(
            f"{row['gene']} & {d} & {row['cptac_p']:.2e} & {row['tcga_p']:.2e} & "
            f"{row['gse68465_p']:.2e} \\\\"
        )
    lines += [r"\end{longtable}", r"\end{document}"]
    with open(PAPER / "supplement_table_s1.tex", "w") as f:
        f.write("\n".join(lines))
    print("paper/supplement_table_s1.tex written")


if __name__ == "__main__":
    main()
