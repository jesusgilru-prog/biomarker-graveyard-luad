"""Graph neural network (GNN) survival model over the 121-gene panel, using a real
protein-protein interaction network (STRING, https://string-db.org) as graph structure
-- a biologically-informed prior appropriate at this sample size, unlike sequence-scale
deep learning (see manuscript Limitations for why transformers/foundation models were
excluded).

Design: nodes = panel genes, edges = STRING interactions (default confidence
threshold >=0.4) for the observed panel, and the same STRING lookup repeated for each
random size-matched null panel (fair comparison: both observed and null use real
biological-network structure, not a structure learned from the same data being tested).
Node feature = per-patient standardized expression (1 scalar/node). A 2-layer GCN
produces a graph-level embedding (mean pooling), mapped to a single risk score, trained
with a Cox partial-likelihood loss (Breslow ties). Train on TCGA-LUAD, evaluate
concordance index on CPTAC-LUAD and GSE68465 (held out, never seen during training).

Run: python src/phase9_gnn_panel.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from scipy import stats as sps
from sksurv.metrics import concordance_index_censored
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_cptac_luad_discovery, load_tcga_luad_replication, load_gse68465_replication  # noqa: E402
from survival_stats import _logrank_components  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
STRING_URL = "https://string-db.org/api/tsv/network"
SEED = 42
N_NULL = 20
N_EPOCHS = 150
LR = 0.01
WEIGHT_DECAY = 5e-3


def chi2_dir(x, y):
    c, o = _logrank_components(x, y)
    return c, (1 if o > 0 else (-1 if o < 0 else 0))


def compute_panel(X1, y1, X2, y2, X3, y3, candidate_genes):
    ya1 = y1[["os_time_days", "os_event"]].values
    ya2 = y2[["os_time_days", "os_event"]].values
    ya3 = y3[["os_time_months", "os_event"]].values
    genes_3cohorts = []
    for g in candidate_genes:
        c1, d1 = chi2_dir(X1.loc[g].values, ya1)
        c2, d2 = chi2_dir(X2.loc[g].values, ya2)
        c3, d3 = chi2_dir(X3.loc[g].values, ya3)
        p1, p2, p3 = sps.chi2.sf(c1, 1), sps.chi2.sf(c2, 1), sps.chi2.sf(c3, 1)
        if p1 < 0.05 and p2 < 0.05 and p3 < 0.05 and d1 == d2 == d3 and d1 != 0:
            genes_3cohorts.append(g)
    return genes_3cohorts


def fetch_string_edges(genes, retries=3):
    """Real STRING PPI lookup. No fabricated edges -- genes with no STRING interaction
    above the default confidence threshold remain isolated nodes (self-loop only)."""
    for attempt in range(retries):
        try:
            params = {"identifiers": "%0d".join(genes), "species": 9606,
                      "caller_identity": "biomarker_graveyard_research"}
            r = requests.get(STRING_URL, params=params, timeout=60)
            r.raise_for_status()
            lines = r.text.strip().split("\n")
            if len(lines) < 1:
                return []
            header = lines[0].split("\t")
            ia, ib, isc = header.index("preferredName_A"), header.index("preferredName_B"), header.index("score")
            edges = []
            for line in lines[1:]:
                parts = line.split("\t")
                if len(parts) <= max(ia, ib, isc):
                    continue
                edges.append((parts[ia], parts[ib], float(parts[isc])))
            return edges
        except Exception as e:
            if attempt == retries - 1:
                print(f"  STRING fetch failed after {retries} tries: {e}")
                return []
            time.sleep(2)
    return []


def build_edge_index(genes, edges):
    idx = {g: i for i, g in enumerate(genes)}
    src, dst = [], []
    for a, b, score in edges:
        if a in idx and b in idx:
            src += [idx[a], idx[b]]
            dst += [idx[b], idx[a]]
    # self-loops for every node (including isolated ones)
    for i in range(len(genes)):
        src.append(i)
        dst.append(i)
    return torch.tensor([src, dst], dtype=torch.long)


class PanelGCN(nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        self.conv1 = GCNConv(1, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.lin = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index, batch):
        h = torch.relu(self.conv1(x, edge_index))
        h = self.dropout(h)
        h = torch.relu(self.conv2(h, edge_index))
        g = global_mean_pool(h, batch)
        return self.lin(g).squeeze(-1)


def cox_partial_loglik_loss(risk, time, event):
    """Breslow-tie Cox partial log-likelihood, negated (for minimization)."""
    order = torch.argsort(time, descending=True)
    risk, time, event = risk[order], time[order], event[order]
    log_cumsum = torch.logcumsumexp(risk, dim=0)
    diff = risk - log_cumsum
    return -torch.sum(diff * event) / (event.sum() + 1e-8)


def make_batch(X, edge_index):
    """X: n_samples x n_genes ndarray -> a torch_geometric Batch of n_samples graphs
    sharing the same edge_index, one scalar node feature each."""
    graphs = []
    for row in X:
        node_x = torch.tensor(row, dtype=torch.float32).unsqueeze(-1)
        graphs.append(Data(x=node_x, edge_index=edge_index))
    return Batch.from_data_list(graphs)


def build_xy(X_genes_by_samples, y_clin, genes, time_col):
    Xf = X_genes_by_samples.loc[genes].T.values.astype(float)
    time = y_clin[time_col].values.astype(float)
    event = y_clin["os_event"].values.astype(bool)
    return Xf, time, event


def zscore_fit(X):
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    return mu, sd


def train_and_eval(genes, edges, Xtr, ttr, etr, Xte1, t1, e1, Xte3, t3, e3, seed):
    torch.manual_seed(seed)
    edge_index = build_edge_index(genes, edges)
    mu, sd = zscore_fit(Xtr)
    Xtr_z = (Xtr - mu) / sd
    Xte1_z = (Xte1 - mu) / sd
    Xte3_z = (Xte3 - mu) / sd

    model = PanelGCN()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    time_t = torch.tensor(ttr, dtype=torch.float32)
    event_t = torch.tensor(etr, dtype=torch.float32)
    batch_tr = make_batch(Xtr_z, edge_index)

    model.train()
    for epoch in range(N_EPOCHS):
        opt.zero_grad()
        risk = model(batch_tr.x, batch_tr.edge_index, batch_tr.batch)
        loss = cox_partial_loglik_loss(risk, time_t, event_t)
        if torch.isnan(loss):
            return None, None
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        batch1 = make_batch(Xte1_z, edge_index)
        risk1 = model(batch1.x, batch1.edge_index, batch1.batch).numpy()
        batch3 = make_batch(Xte3_z, edge_index)
        risk3 = model(batch3.x, batch3.edge_index, batch3.batch).numpy()

    c1 = concordance_index_censored(e1, t1, risk1)[0]
    c3 = concordance_index_censored(e3, t3, risk3)[0]
    return c1, c3


def main():
    t0 = time.time()
    print("Loading cohorts...")
    X1, y1 = load_cptac_luad_discovery()
    X2, y2 = load_tcga_luad_replication()
    X3, y3 = load_gse68465_replication()
    candidate_genes = sorted(set(X1.index) & set(X2.index) & set(X3.index))
    panel = compute_panel(X1, y1, X2, y2, X3, y3, candidate_genes)
    print(f"Panel: {len(panel)} genes {'OK' if len(panel) == 121 else 'MISMATCH'}")

    Xtr, ttr, etr = build_xy(X2, y2, panel, "os_time_days")
    Xte1, t1, e1 = build_xy(X1, y1, panel, "os_time_days")
    Xte3, t3, e3 = build_xy(X3, y3, panel, "os_time_months")

    print("Fetching real STRING PPI network for the 121-gene panel...")
    edges = fetch_string_edges(panel)
    n_covered = len({g for a, b, s in edges for g in (a, b)})
    print(f"  {len(edges)} edges, {n_covered}/{len(panel)} genes with >=1 real interaction")

    print("\nTraining GCN on observed panel (TCGA-LUAD), evaluating CPTAC-LUAD + GSE68465...")
    c1_obs, c3_obs = train_and_eval(panel, edges, Xtr, ttr, etr, Xte1, t1, e1, Xte3, t3, e3, SEED)
    print(f"  observed: C-index CPTAC-LUAD={c1_obs:.4f}, GSE68465={c3_obs:.4f}")

    rng = np.random.default_rng(SEED)
    null1, null3 = [], []
    print(f"\nRunning {N_NULL} random size-matched gene-panel nulls "
          f"(each with its own real STRING lookup)...")
    for i in range(N_NULL):
        rand_panel = list(rng.choice(candidate_genes, size=len(panel), replace=False))
        rand_edges = fetch_string_edges(rand_panel)
        Xtr_r, _, _ = build_xy(X2, y2, rand_panel, "os_time_days")
        Xte1_r, _, _ = build_xy(X1, y1, rand_panel, "os_time_days")
        Xte3_r, _, _ = build_xy(X3, y3, rand_panel, "os_time_months")
        c1, c3 = train_and_eval(rand_panel, rand_edges, Xtr_r, ttr, etr, Xte1_r, t1, e1,
                                 Xte3_r, t3, e3, SEED)
        if c1 is not None:
            null1.append(c1)
            null3.append(c3)
        print(f"  {i + 1}/{N_NULL}: {len(rand_edges)} edges, "
              f"C-index CPTAC={c1}, GSE68465={c3}, {time.time() - t0:.0f}s elapsed")

    null1, null3 = np.array(null1), np.array(null3)
    p1 = (np.sum(null1 >= c1_obs) + 1) / (len(null1) + 1)
    p3 = (np.sum(null3 >= c3_obs) + 1) / (len(null3) + 1)

    summary = {
        "panel_size": len(panel), "n_null": N_NULL, "seed": SEED,
        "n_string_edges_observed": len(edges), "n_genes_with_string_edge": n_covered,
        "observed": {"cptac": c1_obs, "gse68465": c3_obs},
        "null": {
            "cptac": {"mean": float(null1.mean()), "sd": float(null1.std()),
                       "n_valid": len(null1), "empirical_p": float(p1)},
            "gse68465": {"mean": float(null3.mean()), "sd": float(null3.std()),
                          "n_valid": len(null3), "empirical_p": float(p3)},
        },
    }
    print(f"\nCPTAC-LUAD: observed={c1_obs:.4f}, null mean={null1.mean():.4f} "
          f"(sd={null1.std():.4f}), empirical p={p1:.4f}")
    print(f"GSE68465: observed={c3_obs:.4f}, null mean={null3.mean():.4f} "
          f"(sd={null3.std():.4f}), empirical p={p3:.4f}")

    RESULTS.mkdir(exist_ok=True)
    out_path = RESULTS / "phase9_gnn_panel.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWritten: {out_path}")
    print(f"Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
