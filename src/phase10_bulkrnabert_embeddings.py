"""BulkRNABert (InstaDeepAI, bioRxiv 10.1101/2024.06.18.599483) as a FROZEN embedding
extractor -- not fine-tuned -- feeding a 256-dim per-sample embedding into the same
classical survival models already used in phase8, to test whether it adds real signal
on top of the 121-gene panel. Requested by the user after the 20-subagent model search
flagged this as the cheapest real "feature extractor" candidate.

Important caveat, stated up front and re-verified below with real numbers rather than
assumed: BulkRNABert expects a FIXED, ordered vocabulary of 19,062 Ensembl gene IDs (its
own training vocabulary, see config.json / data/tcga_sample.csv on the model repo). Our
121-gene panel is only 0.6% of that vocabulary. Two things make this more tractable than
it first looks: (1) we do NOT restrict the model's input to the 121 genes -- we feed it
whatever of its own 19,062-gene vocabulary each cohort's FULL raw expression matrix
actually covers (not just the panel), so the embedding is computed from real, broad
transcriptome data, not from 121 genes zero-padded to 19,062; (2) real coverage, computed
against the model's own gene list via Ensembl-symbol mapping (mygene.info), is 87.0% for
TCGA-LUAD (RNA-seq) and 96.7% for CPTAC-LUAD (RNA-seq) -- both comfortably above the ~65%
floor past which the model's own paper says accuracy degrades. GSE68465 (microarray) is
only 60.8% covered and is intentionally EXCLUDED from this experiment for that reason (a
different platform on top of borderline coverage would confound any result).

Genes absent from a cohort's matrix are zero-filled (documented, not hidden) after
log10(1+x) transform, i.e. treated as the tokenizer's zero-expression bin -- the standard
missing-value convention for this kind of masked-omics model, not a a special imputation
scheme of our own invention.

Design: train on TCGA-LUAD, evaluate on CPTAC-LUAD (never seen by the embedding model in
any fine-tuning sense -- it is frozen and was never trained on either of our cohorts).
Three feature sets compared with the SAME model class (RandomSurvivalForest, same as
phase8) and the SAME z-scoring convention (fit on train only):
  (a) 121-gene panel alone (baseline, already reported in phase8: rsf_cptac=0.729)
  (b) BulkRNABert embedding (256-dim) alone
  (c) panel + embedding concatenated (377-dim)
  (d) SANITY CONTROL: panel + 256 IID standard-normal random features (same
      dimensionality as (c), zero real information) -- if (c) does not beat (d), the
      embedding is not adding anything a same-size noise vector wouldn't.

Run: /home/jesus/miniconda3/envs/biomarker/bin/python src/phase10_bulkrnabert_embeddings.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from data import load_cptac_luad_discovery, load_tcga_luad_replication  # noqa: E402
from phase8_ml_survival_panel import compute_panel  # noqa: E402
from sksurv.ensemble import RandomSurvivalForest  # noqa: E402
from sksurv.metrics import concordance_index_censored  # noqa: E402
from sksurv.util import Surv  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
LOGS = Path(__file__).resolve().parents[1] / "logs"
CKPT = "InstaDeepAI/BulkRNABert"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SEED = 42
N_NULL_NOISE = 20


def build_ensg_matrix(expr_symbols_by_samples_log2p1, ensg_order, symbol_by_ensg):
    """expr_symbols_by_samples_log2p1: genes(symbol)-as-rows x samples, log2(x+1).
    Returns raw-scale (samples x 19062) ndarray in the model's fixed ENSG order,
    zero-filled for genes absent from this cohort. Also returns coverage fraction."""
    raw = (2.0 ** expr_symbols_by_samples_log2p1) - 1.0  # invert data.py's _log2p1
    n_samples = raw.shape[1]
    out = np.zeros((n_samples, len(ensg_order)), dtype=np.float32)
    covered = 0
    have_symbols = set(raw.index)
    for j, ensg in enumerate(ensg_order):
        sym = symbol_by_ensg.get(ensg)
        if sym is not None and sym in have_symbols:
            out[:, j] = raw.loc[sym].values.astype(np.float32)
            covered += 1
    return out, covered / len(ensg_order)


@torch.no_grad()
def embed(raw_matrix, model, tokenizer, batch_size=1):
    """raw_matrix: samples x 19062 raw (non-log) values. Returns samples x 256 mean-pooled
    final-layer embeddings.

    batch_size=1 is deliberate, not a mistake: full self-attention over 19,062 gene
    positions makes the per-layer attention-weight tensor (batch * n_heads * 19062^2 *
    4 bytes) the memory bottleneck, not the model's own 6M parameters. batch=8 tried to
    allocate ~87GiB and OOM'd on a 96GB GPU (verified empirically, not estimated after
    the fact); batch=1 with bf16 autocast keeps it around ~6GB/layer, freed between
    layers since this runs under no_grad (no autograd graph retains prior layers)."""
    log_vals = np.log10(1.0 + raw_matrix)
    embeds = []
    for start in range(0, log_vals.shape[0], batch_size):
        batch = log_vals[start:start + batch_size]
        enc = tokenizer.batch_encode_plus(batch, return_tensors="pt")
        input_ids = enc["input_ids"].to(DEVICE)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(DEVICE.startswith("cuda"))):
            out = model(input_ids=input_ids)
            layer_emb = out[f"embeddings_{model.config.num_layers}"]  # (B, 19062, 256)
            pooled = layer_emb.mean(dim=1)  # mean-pool over genes
        embeds.append(pooled.float().cpu().numpy())
        if DEVICE.startswith("cuda"):
            torch.cuda.empty_cache()
        if start % 20 == 0:
            print(f"  embed batch {start}-{start+len(batch)}/{log_vals.shape[0]}", flush=True)
    return np.concatenate(embeds, axis=0)


def zscore_fit(X):
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    return mu, sd


def zscore_apply(X, mu, sd):
    return (X - mu) / sd


def cindex(time_, event, risk):
    return concordance_index_censored(event, time_, risk)[0]


def eval_feature_set(name, Xtr, ytr_time, ytr_event, Xte, yte_time, yte_event, rng_state):
    mu, sd = zscore_fit(Xtr)
    Xtr_z, Xte_z = zscore_apply(Xtr, mu, sd), zscore_apply(Xte, mu, sd)
    y_struct = Surv.from_arrays(event=ytr_event, time=ytr_time)
    model = RandomSurvivalForest(n_estimators=300, min_samples_leaf=8, n_jobs=2, random_state=rng_state)
    model.fit(Xtr_z, y_struct)
    risk = model.predict(Xte_z)
    c = cindex(yte_time, yte_event, risk)
    print(f"  {name}: C-index={c:.4f}", flush=True)
    return c


def main():
    t0 = time.time()
    print(f"Device: {DEVICE}", flush=True)

    print("Loading cohorts + panel...", flush=True)
    X1, y1 = load_cptac_luad_discovery()  # CPTAC = test
    X2, y2 = load_tcga_luad_replication()  # TCGA = train
    from data import load_gse68465_replication
    X3, y3 = load_gse68465_replication()
    candidate_genes = sorted(set(X1.index) & set(X2.index) & set(X3.index))
    panel = compute_panel(X1, y1, X2, y2, X3, y3, candidate_genes)
    print(f"Panel: {len(panel)} genes", flush=True)

    map_df = pd.read_csv("/tmp/claude-1000/-home-jesus/9090ab72-9425-421c-86c1-9a4a25079037/scratchpad/ensg_to_symbol_clean.csv")
    symbol_by_ensg = dict(zip(map_df["query"], map_df["symbol"]))
    with open("/tmp/claude-1000/-home-jesus/9090ab72-9425-421c-86c1-9a4a25079037/scratchpad/bulkrnabert_ensg_order.txt") as f:
        ensg_order = [line.strip() for line in f]

    print("Loading BulkRNABert (frozen, no fine-tuning)...", flush=True)
    config = AutoConfig.from_pretrained(CKPT, trust_remote_code=True)
    config.embeddings_layers_to_save = (config.num_layers,)
    tokenizer = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
    model = AutoModel.from_pretrained(CKPT, config=config, trust_remote_code=True).to(DEVICE).eval()

    raw_tcga, cov_tcga = build_ensg_matrix(X2.loc[[g for g in X2.index]], ensg_order, symbol_by_ensg)
    raw_cptac, cov_cptac = build_ensg_matrix(X1.loc[[g for g in X1.index]], ensg_order, symbol_by_ensg)
    print(f"Coverage: TCGA-LUAD {cov_tcga*100:.1f}%, CPTAC-LUAD {cov_cptac*100:.1f}% "
          f"(of BulkRNABert's {len(ensg_order)}-gene vocabulary)", flush=True)

    print("Embedding TCGA-LUAD...", flush=True)
    emb_tcga = embed(raw_tcga, model, tokenizer)
    print("Embedding CPTAC-LUAD...", flush=True)
    emb_cptac = embed(raw_cptac, model, tokenizer)
    np.savez(RESULTS / "phase10_bulkrnabert_embeddings.npz",
              emb_tcga=emb_tcga, emb_cptac=emb_cptac, coverage_tcga=cov_tcga, coverage_cptac=cov_cptac)

    panel_tcga = X2.loc[panel].T.values.astype(np.float32)
    panel_cptac = X1.loc[panel].T.values.astype(np.float32)
    ytr_time = y2["os_time_days"].values.astype(float)
    ytr_event = y2["os_event"].values.astype(bool)
    yte_time = y1["os_time_days"].values.astype(float)
    yte_event = y1["os_event"].values.astype(bool)
    pos = ytr_time > 0
    panel_tcga_p, emb_tcga_p = panel_tcga[pos], emb_tcga[pos]
    ytr_time_p, ytr_event_p = ytr_time[pos], ytr_event[pos]

    rng = np.random.default_rng(SEED)
    results = {}
    results["panel_only"] = eval_feature_set("panel_only (baseline, cf. phase8 rsf_cptac=0.729)",
                                               panel_tcga_p, ytr_time_p, ytr_event_p, panel_cptac, yte_time, yte_event, SEED)
    results["embedding_only"] = eval_feature_set("embedding_only",
                                                  emb_tcga_p, ytr_time_p, ytr_event_p, emb_cptac, yte_time, yte_event, SEED)
    results["panel_plus_embedding"] = eval_feature_set(
        "panel_plus_embedding",
        np.concatenate([panel_tcga_p, emb_tcga_p], axis=1), ytr_time_p, ytr_event_p,
        np.concatenate([panel_cptac, emb_cptac], axis=1), yte_time, yte_event, SEED)

    noise_ctrl_scores = []
    for i in range(N_NULL_NOISE):
        rs = np.random.default_rng(1000 + i)
        noise_tr = rs.standard_normal(size=(panel_tcga_p.shape[0], emb_tcga_p.shape[1])).astype(np.float32)
        noise_te = rs.standard_normal(size=(panel_cptac.shape[0], emb_tcga_p.shape[1])).astype(np.float32)
        c = eval_feature_set(f"panel_plus_random_noise[{i}]",
                              np.concatenate([panel_tcga_p, noise_tr], axis=1), ytr_time_p, ytr_event_p,
                              np.concatenate([panel_cptac, noise_te], axis=1), yte_time, yte_event, SEED)
        noise_ctrl_scores.append(c)

    summary = {
        "model": "InstaDeepAI/BulkRNABert (frozen feature extractor, not fine-tuned)",
        "coverage_tcga": cov_tcga, "coverage_cptac": cov_cptac,
        "panel_size": len(panel),
        "results": results,
        "panel_plus_random_noise_256d": {
            "n_reps": N_NULL_NOISE, "mean": float(np.mean(noise_ctrl_scores)), "sd": float(np.std(noise_ctrl_scores)),
            "scores": noise_ctrl_scores,
        },
    }
    out_path = RESULTS / "phase10_bulkrnabert_embeddings.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWritten: {out_path}")
    print(f"Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
