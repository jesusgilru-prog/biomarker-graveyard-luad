"""Study design schematic (Figure 1 of the manuscript, pushing the current Figure 1
Venn diagram to Figure 2, etc.) -- requested after visual-format review: the manuscript
had no overview figure showing the full analytic pipeline, standard for this kind of
methods paper and expected by a reader/reviewer before the detailed per-section figures.

Run: python src/generate_figure_schematic.py
Writes: paper/figures/fig0_study_design.pdf / .png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MplPath

REPO = Path(__file__).resolve().parents[1]
FIGDIR = REPO / "paper" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 8.3, "font.family": "serif"})

COL_COHORT = "#dbe6f5"
COL_SCREEN = "#f5e6c8"
COL_NULL = "#e6d9f0"
COL_PANEL = "#4c72b0"
COL_DOWN = "#eaeaea"
EDGE = "black"


def box(ax, xy, w, h, text, facecolor, fontsize=7.6, fontweight="normal", textcolor="black"):
    x, y = xy
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                        linewidth=0.9, edgecolor=EDGE, facecolor=facecolor, zorder=2)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
             fontweight=fontweight, color=textcolor, zorder=3, linespacing=1.3)
    return (x + w / 2, y), (x + w / 2, y + h), (x, y + h / 2), (x + w, y + h / 2)


def arrow(ax, p_from, p_to, **kw):
    a = FancyArrowPatch(p_from, p_to, arrowstyle="-|>", mutation_scale=10,
                         linewidth=1.0, color="black", zorder=1, **kw)
    ax.add_patch(a)


def main():
    fig, ax = plt.subplots(figsize=(7.0, 8.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 15.2)
    ax.axis("off")

    # Row 1: three cohorts
    y1 = 13.4
    bw, bh = 2.9, 1.5
    c1 = box(ax, (0.3, y1), bw, bh,
              "CPTAC-LUAD\n$n$=106, 24 events\nRNA-seq", COL_COHORT)
    c2 = box(ax, (3.55, y1), bw, bh,
              "TCGA-LUAD\n$n$=508, 183 events\nRNA-seq", COL_COHORT)
    c3 = box(ax, (6.8, y1), bw, bh,
              "GSE68465\n$n$=442, 236 events\nmicroarray", COL_COHORT)

    # Row 2: naive screen + role swap
    y2 = 11.3
    s = box(ax, (1.2, y2), 7.6, 1.15,
             "Genome-wide naive screen (11,484 genes) $\\times$ 3 role-swapped\n"
             "discovery/replication configurations  --  Section 3.1", COL_SCREEN)
    for c in (c1, c2, c3):
        bottom_mid = c[0]
        arrow(ax, bottom_mid, (bottom_mid[0], y2 + 1.15))

    # bracket connecting screen box up to the three cohorts (visual only, via 3 short lines
    # already drawn above); connect screen box down to null branch
    top_mid_s = (5.0, y2)

    # Row 3: two complementary nulls
    y3 = 9.15
    n1 = box(ax, (0.3, y3), 4.3, 1.15,
              "Label-permutation null\n(destroys all cross-cohort signal)\n"
              "89--188$\\times$, $p<0.001$", COL_NULL)
    n2 = box(ax, (5.4, y3), 4.3, 1.15,
              "Gene-resampling null\n(hypergeometric, sharper test)\n"
              "2.7--3.0$\\times$, $p<10^{-28}$", COL_NULL)
    arrow(ax, top_mid_s, (2.45, y3 + 1.15))
    arrow(ax, top_mid_s, (7.55, y3 + 1.15))

    # Row 4: 121-gene panel
    y4 = 7.15
    p = box(ax, (2.3, y4), 5.4, 1.15,
             "121-gene directional panel\n(significant + concordant direction\n"
             "in all three cohorts simultaneously)  --  Section 3.2",
             COL_PANEL, fontweight="bold", textcolor="white")
    arrow(ax, (n1[0][0], y3), (5.0, y4 + 1.15))
    arrow(ax, (n2[0][0], y3), (5.0, y4 + 1.15))

    # Row 5: three downstream checks
    y5 = 4.7
    bw5 = 2.9
    d1 = box(ax, (0.3, y5), bw5, 1.65,
              "Panel definition\nsensitivity (§3.2)\nCox continuous, $p{<}0.01/0.10$:\n"
              "excess persists,\n2.1--6.6$\\times$", COL_DOWN, fontsize=7.0)
    d2 = box(ax, (3.55, y5), bw5, 1.65,
              "External validation (§3.3)\nHPA published corpus\n"
              "91/1,080 concordant,\n"
              "1.54$\\times$ joint excess,\n$p=5.7{\\times}10^{-8}$", COL_DOWN, fontsize=7.0)
    d3 = box(ax, (6.8, y5), bw5, 1.65,
              "Stage/age sensitivity\n(§3.5)\n120/121 direction-\n"
              "concordant after\nadjustment", COL_DOWN, fontsize=7.0)
    for d in (d1, d2, d3):
        arrow(ax, (d[0][0], y4), (d[0][0], y5 + 1.65))

    # Row 6: brief note (dashed, "open observation")
    y6 = 2.85
    note = box(ax, (2.3, y6), 5.4, 1.1,
                "Section 3.4 (open observation, not used to qualify\n"
                "the results above): non-monotonic FDR survival\n"
                "with cohort size -- investigated, unresolved",
                "white", fontsize=6.8)
    b = FancyBboxPatch((2.3, y6), 5.4, 1.1, boxstyle="round,pad=0.02,rounding_size=0.04",
                        linewidth=0.9, edgecolor="grey", facecolor="none",
                        linestyle=(0, (4, 2)), zorder=4)
    ax.add_patch(b)

    # bottom: conclusion
    y7 = 0.9
    concl = box(ax, (0.9, y7), 8.2, 1.35,
                 "Directional reproducibility across independently calibrated cohorts\n"
                 "recovers biological signal that gene-by-gene FDR correction destroys\n"
                 "almost entirely -- a triage principle, not a clinical biomarker panel",
                 "#f9f4e8", fontsize=7.6, fontweight="bold")
    arrow(ax, (5.0, y6), (5.0, y7 + 1.35))

    fig.tight_layout()
    fig.savefig(FIGDIR / "fig0_study_design.pdf", bbox_inches="tight")
    fig.savefig(FIGDIR / "fig0_study_design.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("fig0 written")


if __name__ == "__main__":
    main()
