"""
fusion_reasoner_routed.py — CARGO fusion, LEVER D: per-DOC convex routing.

Motivation (from the dissim autopsy + local ceiling test, see memory):
  - The old fuse `z(s_op) + gamma_q*relu(z(gdoc))` is additive-only with a tiny gate, so on the
    dissimilar slice — where the operator BURIES 68% of golds past rank 100 with a very negative
    z-score — a small positive graph promotion cannot overcome it. Oracle routing (graph on dissim,
    operator on similar) DOUBLES dissim R@5 (2.9 -> 5.79) with zero head cost, so the signal is there.
  - BUT no query-side signal separates dissim from similar (every uncertainty/coverage/margin gap is
    ~0.06): "dissimilar" is a property of where the GOLD sits, not of query confidence. A per-QUERY
    router (the old gate) therefore cannot detect when to trust the graph.
  - Escape: route PER (query, doc). A doc that the graph scores high while the operator scores low is
    exactly an operator-buried / graph-confident candidate — surface it, regardless of slice. This
    needs no slice knowledge.

Fuse operator:
    zop, zg   = z(s_op), z(gdoc)                       # per-query standardized
    feat_bd   = [zop, zg, zg - zop]                    # per (query, doc)
    alpha_bd  = sigmoid( router(feat_bd) )             # in (0,1), per doc
    fused_doc = (1 - alpha_bd) * zop + alpha_bd * zg   # convex blend

Router last-layer bias is initialised very negative so alpha ~ alpha_init (default 0.02): at step 1 the
model IS the operator (protects same-domain/head exactly like the old gamma_init=0.01). Training (same
operator-hard-negative objective) can raise alpha only where it lowers the loss — i.e. where the graph
genuinely beats the operator for a doc. Operator scalars w=[w0,w1,w2], beta stay trainable.

Everything else (live operator recompute, hard-neg caches) matches fusion_reasoner.py.
"""
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from gfmrag.models.gfm_reasoner import GraphReasoner

W_INIT = (1.05, 1.05, 0.25)
BETA_INIT = 0.95


class RoutedFusionReasoner(nn.Module):
    def __init__(self, entity_model, feat_dim, alpha_init=0.02, router_hidden=16,
                 op_lr_scale=1.0, **kwargs):
        super().__init__()
        self.base = GraphReasoner(entity_model, feat_dim, **kwargs)

        # --- per-doc router: alpha_bd = sigmoid(router([zop, zg, zg-zop])) ---
        self.router = nn.Sequential(
            nn.Linear(3, router_hidden), nn.ReLU(), nn.Linear(router_hidden, 1)
        )
        nn.init.zeros_(self.router[-1].weight)
        # sigmoid(bias) = alpha_init  ->  bias = logit(alpha_init)
        a0 = min(max(alpha_init, 1e-4), 1 - 1e-4)
        nn.init.constant_(self.router[-1].bias, math.log(a0 / (1 - a0)))

        # --- trainable operator scalars (warm-started) ---
        self.op_lr_scale = float(op_lr_scale)
        self.register_buffer("w_init", torch.tensor(W_INIT, dtype=torch.float32))
        self.w_delta = nn.Parameter(torch.zeros(3))
        self.beta_delta = nn.Parameter(torch.zeros(()))

        # --- operator raw ingredients (CPU float16) ---
        self._dense: dict[str, torch.Tensor] = {}
        self._S: dict[str, torch.Tensor] = {}
        self._M: dict[str, torch.Tensor] = {}
        self._totS: dict[str, torch.Tensor] = {}
        self._row: dict[str, tuple] = {}
        for tag, ev in (("train", "OPERATOR_COMPONENTS"), ("test", "OPERATOR_COMPONENTS_TEST")):
            p = os.environ.get(ev)
            if not p:
                continue
            d = np.load(p, allow_pickle=True)
            self._dense[tag] = torch.from_numpy(np.asarray(d["dense"], dtype=np.float16))
            self._S[tag] = torch.from_numpy(np.asarray(d["S"], dtype=np.float16))
            self._M[tag] = torch.from_numpy(np.asarray(d["M"], dtype=np.float16))
            self._totS[tag] = torch.from_numpy(np.asarray(d["total_S"], dtype=np.float32))
            for i, q in enumerate(d["query_ids"]):
                self._row[str(q)] = (tag, i)
            print(f"[routed-fusion] loaded {ev}: dense/S/M {tuple(self._dense[tag].shape)} "
                  f"({len(d['query_ids'])} queries), alpha_init={alpha_init}")
        assert self._row, "no operator components: set OPERATOR_COMPONENTS / OPERATOR_COMPONENTS_TEST"

        self._raw_doc = None   # graph-alone doc scores (aux hard-neg loss)
        self._doc_ids = None
        self._s_op = None      # operator scores (detached) for hard-neg mining
        self._alpha_mean = None  # mean per-query alpha, cached for logging

    @staticmethod
    def _z(x):
        return (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-6)

    def _operator(self, ids, n_doc, device):
        order = [self._row[str(x.item() if hasattr(x, "item") else x)] for x in ids]
        tag = order[0][0]
        assert all(t == tag for t, _ in order), "batch mixes train/test operator tables"
        idx = torch.tensor([r for _, r in order], dtype=torch.long)
        dense = self._dense[tag].index_select(0, idx).to(device, torch.float32)
        S = self._S[tag].index_select(0, idx).to(device, torch.float32)
        M = self._M[tag].index_select(0, idx).to(device, torch.float32)
        totS = self._totS[tag].to(device)
        assert dense.shape[1] == n_doc, f"operator cols {dense.shape[1]} != {n_doc} doc nodes"
        w = self.w_init + self.op_lr_scale * self.w_delta
        beta = (BETA_INIT + self.op_lr_scale * self.beta_delta).clamp(min=0.05)
        dem = (totS.unsqueeze(0) - S).clamp(min=1e-6)
        degb = dem.pow(beta)
        return w[0] * self._z(dense) + w[1] * self._z(S / degb) + w[2] * self._z(M / degb)

    def forward(self, graph, batch, entities_weight=None):
        g = self.base(graph, batch, entities_weight)
        doc = graph.nodes_by_type["document"].to(g.device)
        gdoc = g.index_select(1, doc).float()               # [B, n_doc]

        s_op = self._operator(batch["id"], doc.numel(), g.device)
        zop, zg = self._z(s_op), self._z(gdoc)              # [B, n_doc]

        feat = torch.stack([zop, zg, zg - zop], dim=-1)     # [B, n_doc, 3]
        alpha = torch.sigmoid(self.router(feat)).squeeze(-1)  # [B, n_doc] per-doc convex weight
        fused = (1.0 - alpha) * zop + alpha * zg

        out = g.clone()
        out[:, doc] = fused.to(out.dtype)
        self._raw_doc = gdoc
        self._doc_ids = doc
        self._s_op = s_op.detach()
        self._alpha_mean = alpha.mean(-1).detach()
        return out
