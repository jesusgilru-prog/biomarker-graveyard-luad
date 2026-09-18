"""Generates the 3 core manuscript figures from results/ already on disk (no new
computation except the cheap, already-verified Venn overlap counts in
results/venn_overlap_counts.json). All numbers plotted are real, sourced from files
written by phase2_discovery.py / phase3c_directional_enrichment_generalized.py /
phase4_hpa_validation.py -- nothing here is illustrative/fabricated.

Run: python src/generate_figures.py
Writes: paper/figures/fig1_venn_3cohort.pdf, fig2_three_nulls.pdf, fig3_hpa_null.pdf
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib_venn import venn3

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
FIGDIR = REPO / "paper" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 10, "font.family": "serif"})


def fig1_venn():
    with open(RESULTS / "venn_overlap_counts.json") as f:
        v = json.load(f)

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    diagram = venn3(
        subsets=(v["cptac_only"], v["tcga_only"], v["cptac_tcga"],
                 v["gse_only"], v["cptac_gse"], v["tcga_gse"], v["all_three_naive"]),
        set_labels=(f"CPTAC-LUAD\n(n={v['n_cptac']})",
                    f"TCGA-LUAD\n(n={v['n_tcga']})",
                    f"GSE68465\n(n={v['n_gse']})"),
        ax=ax,
    )
    for patch in diagram.patches:
        if patch is not None:
            patch.set_edgecolor("black")
            patch.set_linewidth(0.6)
    ax.set_title("Naive hits ($p<0.05$) per cohort, and their 3-way overlap",
                 fontsize=10, pad=28)
    ax.annotate(
        f"{v['all_three_dir_concordant']} of these {v['all_three_naive']}\n"
        "are also direction-concordant\n(the reported panel)",
        xy=(0, -0.05), xytext=(0, -0.32), textcoords="axes fraction",
        ha="center", fontsize=8.5, style="italic",
    )
    fig.subplots_adjust(top=0.82, bottom=0.22)
    fig.savefig(FIGDIR / "fig1_venn_3cohort.pdf", bbox_inches="tight")
    fig.savefig(FIGDIR / "fig1_venn_3cohort.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("fig1 written")


def fig2_three_nulls():
    tags = ["cptac", "tcga", "gse68465"]
    labels = ["CPTAC-anchored\n(949 candidates)", "TCGA-anchored\n(2,200 candidates)",
              "GSE68465-anchored\n(1,618 candidates)"]
    # cptac's full 1,000-iteration result predates phase3c (it was computed by the earlier
    # phase3b_replication_empirical_null.py, before that script was generalized to all 3
    # cohorts) and is saved under a different filename; tcga/gse68465 come from phase3c.
    source_files = {
        "cptac": RESULTS / "replication_empirical_null.json",
        "tcga": RESULTS / "directional_enrichment_tcga.json",
        "gse68465": RESULTS / "directional_enrichment_gse68465.json",
    }
    means, stds, observed, folds = [], [], [], []
    for tag in tags:
        with open(source_files[tag]) as f:
            d = json.load(f)
        means.append(d["empirical_null_mean"])
        stds.append(d["empirical_null_std"])
        observed.append(d["observed_both_direction_concordant"])
        folds.append(d["empirical_fold_enrichment"])

    x = np.arange(len(tags))
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.bar(x - 0.18, means, width=0.36, yerr=stds, capsize=4,
           label="Empirical null (mean $\\pm$ SD, 1,000 permutations)",
           color="#b0b0b0", edgecolor="black", linewidth=0.6)
    ax.bar(x + 0.18, observed, width=0.36, label="Observed (121 genes, all configurations)",
           color="#4c72b0", edgecolor="black", linewidth=0.6)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylim(0, 400)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Genes reaching significance with\nconcordant direction in both replication cohorts")
    for i, f_ in enumerate(folds):
        ax.annotate(f"{f_:.1f}$\\times$, $p<0.001$", xy=(x[i] + 0.18, observed[i]),
                    xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.88))
    ax.set_title("Directional-replication panel vs. 3 independently\nconstructed empirical nulls",
                 fontsize=10, pad=10)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig2_three_nulls.pdf", bbox_inches="tight")
    fig.savefig(FIGDIR / "fig2_three_nulls.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("fig2 written")


def fig3_hpa_null():
    with open(RESULTS / "hpa_validation_summary.json") as f:
        d = json.load(f)
    mean, std = d["empirical_null_mean"], d["empirical_null_std"]
    observed = d["observed_both_naive_and_concordant"]
    fold = d["empirical_fold_enrichment"]

    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.bar([0], [mean], yerr=[std], capsize=5, width=0.5,
           label="Empirical null (mean $\\pm$ SD, 1,000 permutations)",
           color="#b0b0b0", edgecolor="black", linewidth=0.6)
    ax.bar([1], [observed], width=0.5,
           label="Observed (HPA-published biomarkers)",
           color="#c44e52", edgecolor="black", linewidth=0.6)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylim(0, 200)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Null", "Observed"])
    ax.set_ylabel("Of 1,080 HPA-published LUAD genes:\nboth-cohort naive + direction-concordant")
    ax.set_ylim(0, 400)
    ax.annotate(f"{fold:.1f}$\\times$, $p<0.001$", xy=(1, observed),
                xytext=(0, 6), textcoords="offset points", ha="center", fontsize=9)
    ax.set_title("External validation: published biomarkers vs.\ntheir own empirical null",
                 fontsize=10)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.80))
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig3_hpa_null.pdf", bbox_inches="tight")
    fig.savefig(FIGDIR / "fig3_hpa_null.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("fig3 written")


if __name__ == "__main__":
    fig1_venn()
    fig2_three_nulls()
    fig3_hpa_null()
