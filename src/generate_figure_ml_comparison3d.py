"""3D bar comparison of all 15 machine-learning survival models (12 classical, Section
3.5, plus 3 foundation-model candidates, Section 3.6) across the two held-out cohorts.
Requested for visual diversity alongside the 3D PPI network figure; also fills a real gap
-- the ML comparison in the text (Sections 3.5-3.6) previously had no accompanying figure.
Reads every number directly from the result JSON files (never hand-transcribed), so a
change to any upstream script is reflected here automatically.

Run: python src/generate_figure_ml_comparison3d.py
Writes: paper/figures/fig6_ml_comparison_3d.pdf / .png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
FIGDIR = REPO / "paper" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 8.3, "font.family": "serif"})

CLASSICAL_LABELS = {
    "rsf": "RSF", "extra_trees": "Extra Trees", "gbsa": "GBSA", "cw_gbsa": "CW-GBSA",
    "survival_tree": "Surv. Tree", "cox_ph": "Cox PH", "coxnet": "Coxnet (sksurv)",
    "ipc_ridge": "IPCRidge", "fast_svm": "SVM (linear)", "fast_kernel_svm": "SVM (kernel)",
    "minlip": "Minlip", "coxnet_lifelines": "Coxnet (lifelines)",
}

COL_CLASSICAL = "#4c72b0"
COL_FOUNDATION = "#c44e52"


def main():
    phase8 = json.loads((RESULTS / "phase8_ml_survival_panel.json").read_text())
    obs8 = phase8["observed"]

    models, cptac_vals, gse_vals, is_foundation = [], [], [], []
    for key, label in CLASSICAL_LABELS.items():
        models.append(label)
        cptac_vals.append(obs8[f"{key}_cptac"])
        gse_vals.append(obs8[f"{key}_gse68465"])
        is_foundation.append(False)

    phase9 = json.loads((RESULTS / "phase9_survivalpfn.json").read_text())
    phase11 = json.loads((RESULTS / "phase11_tabpfn_pseudo_rmst.json").read_text())
    phase10 = json.loads((RESULTS / "phase10_bulkrnabert_embeddings.json").read_text())

    foundation_entries = [
        ("SurvivalPFN", phase9["observed"]["cptac"], phase9["observed"]["gse68465"]),
        ("TabPFN+RMST", phase11["observed"]["cptac"], phase11["observed"]["gse68465"]),
        ("BulkRNABert+panel", phase10["results"]["panel_plus_embedding"], None),
    ]
    for label, cptac_v, gse_v in foundation_entries:
        models.append(label)
        cptac_vals.append(cptac_v)
        gse_vals.append(gse_v if gse_v is not None else np.nan)  # BulkRNABert: GSE68465 excluded (Sec. 3.6)
        is_foundation.append(True)

    n = len(models)
    fig = plt.figure(figsize=(8.2, 5.8))
    ax = fig.add_subplot(111, projection="3d")

    x = np.arange(n) * 1.3
    dx = 0.85
    dy = 0.5
    y_positions = [0.0, 1.8]
    for cohort_idx, (vals, ypos) in enumerate(zip([cptac_vals, gse_vals], y_positions)):
        for i, v in enumerate(vals):
            if np.isnan(v):
                continue
            color = COL_FOUNDATION if is_foundation[i] else COL_CLASSICAL
            alpha = 0.95 if cohort_idx == 0 else 0.55
            ax.bar3d(x[i], ypos, 0, dx, dy, v, color=color, alpha=alpha,
                      edgecolor="black", linewidth=0.3, shade=True)

    ax.set_xticks(x + dx / 2)
    ax.set_xticklabels(models, rotation=60, ha="right", fontsize=6.3)
    ax.set_yticks([y + dy / 2 for y in y_positions])
    ax.set_yticklabels(["CPTAC-LUAD\n(solid)", "GSE68465\n(translucent)"], fontsize=7.2)
    ax.set_zlabel("C-index", fontsize=7.8)
    ax.set_zlim(0, 0.85)
    ax.view_init(elev=28, azim=-48)

    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor=COL_CLASSICAL, edgecolor="black", label="Classical ML (Sec. 3.5)"),
                     Patch(facecolor=COL_FOUNDATION, edgecolor="black", label="Foundation models (Sec. 3.6)")]
    ax.legend(handles=legend_elems, loc="upper left", fontsize=7.0, bbox_to_anchor=(0.0, 0.98))
    ax.set_title("C-index across all 15 machine-learning models tested,\nboth held-out cohorts "
                 "(BulkRNABert+panel: GSE68465 excluded, different assay platform)", fontsize=7.8)

    fig.tight_layout()
    fig.savefig(FIGDIR / "fig6_ml_comparison_3d.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIGDIR / "fig6_ml_comparison_3d.png", dpi=300, bbox_inches="tight")
    print(f"Written fig6_ml_comparison_3d.pdf/.png ({n} models)")


if __name__ == "__main__":
    main()
