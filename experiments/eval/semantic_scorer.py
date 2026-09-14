"""
semantic_scorer.py -- controlled comparison of semantic scorer architectures on
SIR-4. No graph, no reasoner, no fusion.

    arm "current"  s = w0 z(r_dir) + w1 z(S/p^beta) + w2 z(M/p^beta)
                   S = sum_j H_qdj, M = max_j H_qdj      (the handcrafted summaries)

    arm "attention"  s = sum_v a_v x_v,  a = softmax_v(MLP([x_v, t_v]))
                     over the direct view and every valid hypothetical answer, so
                     the weights sum to 1. The pooled score is a weighted average,
                     which keeps it on the same scale as the direct view no matter
                     how many hypothetical answers a query has.

    arm "gated"      s = g_dir x_dir + sum_j g_qdj x_qdj,  g = sigmoid(MLP([x, t]))
                     Independent gates that do not sum to 1, so several strong
                     views accumulate. Kept selectable via --arms; note the summed
                     views carry several times the spread of the direct view, so
                     this arm has to learn the channel balance the other two get
                     from normalisation.

    arm "dualsetmlp" h_j   = [x_j, MLP(x_j)]
                       u_all = sum_j h_j
                       u_sel = sum_j softmax_j(tau x_j) h_j
                       s     = linear([x_dir,u_all,u_sel]) + MLP([x_dir,u_all,u_sel])

                     This is the automatic raw-view scorer: additive pooling
                     preserves evidence accumulation, a learned positive
                     temperature provides monotonic selective pooling, and the
                     final residual MLP is not constrained to a weighted average.

All trained arms use the SAME selected loss, frozen encoder outputs, fit/dev
split, and development metric. In a comparison run, the only intended change is
the scorer architecture.

THE OBJECTIVE. `--loss operator` (DEFAULT) is operator_scorer.py's exactly, so the
`current` arm reproduces the scorer as it is fitted everywhere else in the project
and the only thing varying between arms is the architecture.

`--loss fixed` is the multi-gold correction. The operator objective's denominator
runs over the whole corpus INCLUDING the query's other golds; TOMATO has one gold
per query so that is harmless there, but SIR-4 matsci has 3.95, so every step
pushes gold 1 up by pushing golds 2..4 down. Both are available and every output
filename carries which one produced it, so the two can be compared rather than
argued about.

WHY THE HYPOTHETICAL VIEWS SHARE ONE SCALE. Standardising each view separately
sets every view's spread to exactly 1, which is precisely the quantity that says
whether a hypothetical answer discriminates between papers at all. A view that
gives every paper the same score would arrive at the gate looking as confident as
one that separates them. So the views are CENTRED individually and divided by a
single shared scale, which preserves their relative spreads.

Run (after the operator cell has produced the embedding caches):
    python3 eval/semantic_scorer.py --dataset sir4_matsci --model /content/qwen3
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np
# Torch at module level, unlike the rest of this file's function-local imports:
# DeepSetsScorer must inherit nn.Module at class-definition time so that
# train()/eval() actually gate its dropout. Every entry point here needs torch
# within seconds anyway.
import torch as _torch
import torch.nn as _nn

_ROOT = os.environ.get("SCIGRAPHIR_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, _ROOT)
from scigraphir_paths import (add_dataset_arg, banner, corpus_dir, emb_dir,  # noqa: E402
                         probes_path, set_dataset)

EPS = 1e-6            # popularity floor and normalisation floor, as specified
KS_REPORT = (1, 3, 5, 10, 25, 100)


def _op_module():
    """Load operator_scorer as a module so the embedding cache keys are IDENTICAL.

    The encoder, the query instruction, the instruction fingerprint and the
    filename layout all live there. Re-implementing any of them here would
    produce a cache MISS at best, and at worst a hit on a file built under a
    different instruction -- which is the same silent-wrong-baseline failure the
    rest of this pipeline is armoured against.
    """
    p = f"{_ROOT}/retriever/eval/operator_scorer.py"
    spec = importlib.util.spec_from_file_location("operator_scorer", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# --------------------------------------------------------------------------
# stage 1: raw semantic measurements, cached on disk
# --------------------------------------------------------------------------
def build_inputs(op, model_name, split, cache_root, force=False):
    """Materialise dense, H, mask and total_S for one split.

    H is [Q, Jmax, D] float16 in a MEMMAP, never a resident tensor: at CS scale it
    is 5,445 x 8 x 20,203 = 1.8 GB, which is fine on disk and fine to slice a
    minibatch out of, and not fine to hold on a GPU alongside activations.
    """
    doc_ids, corpus, queries, probes = op.load_split(split)
    D, Q = len(doc_ids), len(queries)
    qids = [q["id"] for q in queries]
    qkey = hashlib.md5("|".join(qids).encode()).hexdigest()[:8]
    dockey = hashlib.md5("|".join(doc_ids).encode()).hexdigest()[:8]
    slug = op.model_slug(model_name)
    qi = op.query_instruction(model_name)

    # THE CACHE KEY CARRIES EVERY INPUT THAT CHANGES THE NUMBERS. Dataset and
    # split alone are what the rest of this repo used to key on, and a matrix
    # from the wrong corpus passes a row-count check. Document ORDER is in here
    # too: the arrays are column-indexed by it, so a reordered corpus would
    # misalign every score with nothing to show for it.
    meta = {"dataset": os.path.basename(corpus_dir(split)).rsplit("_", 1)[0],
            "split": split, "encoder": model_name, "query_instruct": qi,
            "qkey": qkey, "dockey": dockey, "Q": Q, "D": D}
    os.makedirs(cache_root, exist_ok=True)
    tag = f"{split}_{qkey}_{dockey}{slug}"
    mpath = f"{cache_root}/semantic_inputs_{tag}.json"
    hpath = f"{cache_root}/semantic_H_{tag}.f16"

    if os.path.exists(mpath) and not force:
        got = json.load(open(mpath))
        stale = {k: (got.get(k), v) for k, v in meta.items() if got.get(k) != v}
        assert not stale, f"stale semantic cache {mpath}: {stale}"
        assert os.path.exists(hpath), f"manifest without matrix: {hpath}"
        H = np.memmap(hpath, np.float16, "r", shape=(Q, got["Jmax"], D))
        z = np.load(f"{cache_root}/semantic_side_{tag}.npz", allow_pickle=True)
        print(f"[sem] loaded cached inputs {tag}: H{H.shape} f16")
        return dict(H=H, dense=z["dense"], mask=z["mask"], total_S=z["total_S"],
                    doc_ids=doc_ids, queries=queries, meta=got)

    # Encode only on a genuine cache miss.  Importing sentence-transformers can
    # initialise a multi-gigabyte model stack, so a cache-only experiment must
    # not require it (and must not fail merely because that optional stack is
    # unavailable on the evaluation machine).
    dpath = f"{emb_dir()}/{split}_doc{slug}.npy"
    qpath = f"{emb_dir()}/{split}_query_{qkey}{slug}{op.qi_tag(qi)}.npy"
    ppath = f"{emb_dir()}/{split}_probe_{qkey}{slug}.npy"
    flat, own = [], []
    for i, q in enumerate(queries):
        for pr in probes.get(q["id"], []):
            flat.append(pr)
            own.append(i)
    if not all(os.path.exists(p) for p in (dpath, qpath, ppath)):
        from sentence_transformers import SentenceTransformer
        print(f"[sem] embeddings absent for {split}; encoding with {model_name}")
        enc = SentenceTransformer(model_name)
        enc.max_seq_length = 512
        op.cached_encode(enc, [corpus[d] for d in doc_ids], f"{split}_doc{slug}")
        op.cached_encode(enc, [q["question"] for q in queries],
                         f"{split}_query_{qkey}{slug}{op.qi_tag(qi)}", instruct=qi)
        op.cached_encode(enc, flat, f"{split}_probe_{qkey}{slug}")
    de = np.load(dpath).astype(np.float32)
    qe = np.load(qpath).astype(np.float32)
    pe = np.load(ppath).astype(np.float32)
    assert de.shape[0] == D and qe.shape[0] == Q and pe.shape[0] == len(flat), (
        f"embedding rows {de.shape[0]}/{qe.shape[0]}/{pe.shape[0]} != {D}/{Q}/{len(flat)}")

    own = np.asarray(own)
    counts = np.bincount(own, minlength=Q) if len(own) else np.zeros(Q, int)
    Jmax = int(counts.max()) if len(own) else 1
    meta["Jmax"] = Jmax
    print(f"[sem] {split}: Q={Q} D={D} Jmax={Jmax} "
          f"(views/query min {counts.min()} mean {counts.mean():.1f})")
    assert counts.min() > 0, "a query has no hypothetical answers; regenerate probes"

    # dense in row chunks -- one Q x D float32 is 440 MB at CS scale
    dense = np.empty((Q, D), np.float16)
    for i in range(0, Q, 512):
        dense[i:i + 512] = (qe[i:i + 512] @ de.T).astype(np.float16)

    H = np.memmap(hpath, np.float16, "w+", shape=(Q, Jmax, D))
    mask = np.zeros((Q, Jmax), bool)
    total_S = np.zeros(D, np.float64)
    t0 = time.time()
    for i in range(Q):
        idx = np.where(own == i)[0]
        h = np.clip(pe[idx] @ de.T, 0, None).astype(np.float32)      # [j_i, D]
        H[i, :len(idx)] = h.astype(np.float16)
        mask[i, :len(idx)] = True
        total_S += h.sum(0)                       # popularity over the WHOLE split
        if i % 500 == 0:
            print(f"  [sem] {split} {i}/{Q}  ({time.time() - t0:.0f}s)", flush=True)
    H.flush()

    np.savez_compressed(f"{cache_root}/semantic_side_{tag}.npz",
                        dense=dense, mask=mask, total_S=total_S.astype(np.float32))
    json.dump(meta, open(mpath, "w"), indent=1)
    print(f"[sem] wrote {hpath} ({os.path.getsize(hpath)/1e6:.0f} MB)")
    return dict(H=np.memmap(hpath, np.float16, "r", shape=(Q, Jmax, D)),
                dense=dense, mask=mask, total_S=total_S.astype(np.float32),
                doc_ids=doc_ids, queries=queries, meta=meta)


def gold_matrix(queries, doc_ids):
    """Padded gold indices + validity mask, and a per-query python list."""
    pos = {d: i for i, d in enumerate(doc_ids)}
    lists = []
    for q in queries:
        gs = [pos[g] for g in (q.get("supporting_documents") or []) if g in pos]
        lists.append(gs)
    G = max(1, max(len(g) for g in lists))
    idx = np.zeros((len(lists), G), np.int64)
    val = np.zeros((len(lists), G), np.float32)
    for i, gs in enumerate(lists):
        idx[i, :len(gs)] = gs
        val[i, :len(gs)] = 1.0
    return idx, val, lists


# --------------------------------------------------------------------------
# stage 2: the two architectures
# --------------------------------------------------------------------------
class Pop:
    """Per-document popularity for the anti-hub division, in one of three modes.

    loo        (total_S - S) per query: the historical estimate. TRANSDUCTIVE --
               a test paper's popularity is computed from the OTHER test queries'
               hypothetical answers, so it cannot run on one query alone.
    bank       mean ReLU cosine against the TRAIN-split answer bank. Query-
               independent: one [D] vector per corpus, computable for unseen
               papers from their embedding plus a frozen train artifact.
    predicted  a small MLP distilled from the bank targets. Fully local: needs
               only the paper's own embedding at inference.

    The object rides in the argument slot that used to carry total_S, so every
    scorer stays a pure function of (H, mask, dense, pop).
    """

    def __init__(self, mode, total_S=None, vec=None, predictor=None,
                 doc_emb=None, target=None):
        self.mode, self.total_S, self.vec = mode, total_S, vec
        # `joint` keeps the predictor LIVE inside the scoring path, so the
        # retrieval gradient reaches it. `predicted` freezes its output to a
        # vector instead. Same network, different training regime.
        self.predictor, self.doc_emb, self.target = predictor, doc_emb, target

    def parameters(self):
        """Predictor parameters, so the trainer can optimise them jointly."""
        return list(self.predictor.parameters()) if self.mode == "joint" else []

    def state(self):
        """Predictor weights, or None. Part of the checkpoint under joint training.

        The scorer and the predictor are ONE model: restoring the scorer to its
        best epoch while the predictor sits at whatever the last epoch left it
        would evaluate a pair that never existed during training.
        """
        if self.mode != "joint":
            return None
        return {k: v.detach().cpu().clone() for k, v in self.predictor.state_dict().items()}

    def load_state(self, st):
        if st is None or self.mode != "joint":
            return
        dev = next(self.predictor.parameters()).device
        self.predictor.load_state_dict({k: v.to(dev) for k, v in st.items()})

    def aux_loss(self):
        """Anchor p_hat to the bank targets in log space.

        Without it the retrieval loss is free to repurpose the predictor as extra
        scorer capacity -- it would stop meaning "how generally matchable is this
        paper" and start meaning "whatever lowers the ranking loss", which is a
        different model wearing the same name. The log keeps a handful of very
        popular papers from dominating.
        """
        if self.mode != "joint" or self.target is None:
            return 0.0
        p = self.predictor(self.doc_emb)
        return ((_torch.log(EPS + p) - _torch.log(EPS + self.target)) ** 2).mean()

    @staticmethod
    def from_data(data, device, variant=""):
        """`variant` selects an alternative popularity stored on the same split.

        The mlp arm carries its own LEARNED popularity as part of the proposal, so
        one run needs two sources live at once: the baseline's leave-one-out and
        the predictor's. They are stored side by side rather than in two runs.
        """
        vk = f"pop_vec{variant}"
        mk = f"pop_mode{variant}"
        if data.get(mk, "loo") == "loo":
            return Pop("loo", total_S=_torch.as_tensor(data["total_S"], device=device))
        return Pop(data[mk], vec=_torch.as_tensor(data[vk], device=device).float())

    def per_query(self, S):
        """[B, D] popularity, given the query's own answer-sum S [B, D]."""
        if self.mode == "loo":
            return (self.total_S.unsqueeze(0) - S).clamp_min(EPS)
        if self.mode == "joint":
            # Recomputed every step and DIFFERENTIABLE: this is the whole point.
            return self.predictor(self.doc_emb).clamp_min(EPS).unsqueeze(0).expand_as(S)
        # Query-independent: the same vector for every query. Includes the
        # query's own answers when the query came from the train split, which is
        # the definition of the bank -- one contribution among thousands.
        return self.vec.unsqueeze(0).clamp_min(EPS).expand_as(S)


def _views(H, mask, dense, pop, beta):
    """Popularity adjustment + the two normalisations. Returns x_dir, x_hyp, Ht.

    H     [B, J, D] float32   raw ReLU'd hypothetical-answer matches
    mask  [B, J]    float32   1 for a real answer, 0 for padding
    dense [B, D]    float32   direct question-paper cosine
    """
    import torch
    m3 = mask.unsqueeze(-1)                                   # [B, J, 1]
    S = (H * m3).sum(1)                                       # [B, D]
    p = pop.per_query(S)
    Ht = H / p.unsqueeze(1).pow(beta)
    Ht = Ht * m3                                              # padding contributes 0

    D = H.shape[-1]
    nj = mask.sum(1).clamp_min(1.0)                           # [B] real answers
    mu = Ht.sum(-1, keepdim=True) / D                         # [B, J, 1]
    c = (Ht - mu) * m3                                        # centre each answer
    # ONE shared scale across this query's valid answers, so a view that barely
    # separates papers stays narrow instead of being inflated to spread 1.
    sig = torch.sqrt((c * c).sum((1, 2)) / (nj * D) + EPS)    # [B]
    x_hyp = c / (sig.view(-1, 1, 1) + EPS)
    x_dir = (dense - dense.mean(1, keepdim=True)) / (dense.std(1, keepdim=True) + EPS)
    return x_dir, x_hyp * m3, Ht


def _set_views(H, mask, dense, pop, beta):
    """Return direct and hypothetical-answer inputs for learned set pooling.

    Unlike ``_views``, this uses one mean and one scale for the complete
    hypothetical-answer set of a query.  Consequently an answer that barely
    separates papers stays weak, and relative offsets between answers are not
    erased before the set network sees them.
    """
    import torch
    m3 = mask.unsqueeze(-1)
    S = (H * m3).sum(1)
    p = pop.per_query(S)
    adjusted = H / p.unsqueeze(1).pow(beta)
    adjusted = adjusted * m3

    D = H.shape[-1]
    n = (mask.sum(1) * D).clamp_min(1.0)
    mu = adjusted.sum((1, 2)) / n
    centred = (adjusted - mu.view(-1, 1, 1)) * m3
    scale = torch.sqrt((centred * centred).sum((1, 2)) / n + EPS)
    x_hyp = centred / (scale.view(-1, 1, 1) + EPS)
    x_dir = (dense - dense.mean(1, keepdim=True)) / (dense.std(1, keepdim=True) + EPS)
    return x_dir, x_hyp * m3


class MatchabilityPredictor(_nn.Module):
    """Query-Independent Document Matchability Estimator.

        p_hat(d) = softplus( g_eta(E(d)) ),   g_eta: dim -> hidden -> 1

    Distils the train-answer bank into a function of the paper embedding alone,
    so popularity at inference needs nothing but the paper itself: no other test
    queries, no bank matmul, no lookup table. Fitted on log targets so a few
    extremely popular papers cannot dominate the loss.
    """

    def __init__(self, dim, hidden=64):
        super().__init__()
        self.net = _nn.Sequential(_nn.Linear(dim, hidden), _nn.GELU(),
                                  _nn.Linear(hidden, 1))

    def forward(self, E):
        return _nn.functional.softplus(self.net(E)).squeeze(-1)


def bank_popularity(bank_emb, doc_emb, chunk=4096):
    """p_bank[d] = mean over bank answers of ReLU(cos(h, d)). Pure numpy, chunked."""
    out = np.zeros(doc_emb.shape[0], np.float64)
    for i in range(0, bank_emb.shape[0], chunk):
        out += np.clip(bank_emb[i:i + chunk] @ doc_emb.T, 0, None).sum(0)
    return (out / max(bank_emb.shape[0], 1)).astype(np.float32)


def fit_matchability(doc_emb, targets, device, seed=0, hidden=64, lr=1e-3,
                     weight_decay=1e-2, epochs=200, patience=10):
    """Fit p_hat to the bank targets with log-MSE, early-stopped on held-out docs."""
    N = doc_emb.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    n_val = max(1, N // 10)
    va, fi = perm[:n_val], perm[n_val:]
    X = _torch.as_tensor(doc_emb, device=device)
    y = _torch.log(EPS + _torch.as_tensor(targets, device=device))
    m = MatchabilityPredictor(doc_emb.shape[1], hidden).to(device)
    groups = [{"params": [p for p in m.parameters() if p.ndim >= 2],
               "weight_decay": weight_decay},
              {"params": [p for p in m.parameters() if p.ndim < 2],
               "weight_decay": 0.0}]
    opt = _torch.optim.AdamW(groups, lr=lr)
    best, best_sd, stale = float("inf"), None, 0
    for ep in range(epochs):
        m.train()
        for bi in range(0, len(fi), 1024):
            sel = _torch.as_tensor(fi[bi:bi + 1024], device=device)
            loss = ((_torch.log(EPS + m(X[sel])) - y[sel]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with _torch.no_grad():
            v = float(((_torch.log(EPS + m(X[va])) - y[va]) ** 2).mean())
        if v < best - 1e-6:
            best, best_sd, stale = v, {k: t.clone() for k, t in m.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= patience:
                break
    m.load_state_dict(best_sd)
    m.eval()
    with _torch.no_grad():
        pred = m(X).cpu().numpy()
        corr = float(np.corrcoef(np.log(EPS + pred), np.log(EPS + targets))[0, 1])
    print(f"[matchability] fitted on {len(fi)} docs, held-out {n_val}: "
          f"log-MSE {best:.4f}, log-corr {corr:.3f} (ep {ep + 1})")
    return m, {"heldout_logmse": round(best, 6), "log_corr": round(corr, 4),
               "epochs_run": ep + 1, "hidden": hidden}


class CurrentScorer:
    """w0 z(dense) + w1 z(S/p^b) + w2 z(M/p^b) -- the handcrafted sum and max."""

    name = "current"

    def __init__(self, device):
        import torch
        import torch.nn as nn
        self.w = nn.Parameter(torch.tensor([1.0, 1.0, 0.2], device=device))
        self.logbeta = nn.Parameter(torch.zeros((), device=device))

    def parameters(self):
        return [self.w, self.logbeta]

    def state(self):
        return {"w": self.w.detach().cpu().tolist(),
                "beta": float(self.logbeta.detach().exp().cpu())}

    def load(self, st):
        import torch
        with torch.no_grad():
            self.w.copy_(torch.tensor(st["w"], device=self.w.device))
            self.logbeta.copy_(torch.tensor(math.log(st["beta"]), device=self.w.device))

    def __call__(self, H, mask, dense, pop):
        import torch
        beta = self.logbeta.exp()
        m3 = mask.unsqueeze(-1)
        S_raw = (H * m3).sum(1)
        degb = pop.per_query(S_raw).pow(beta)
        # max over REAL answers only: padding is 0 and every H is >= 0, so a
        # padded slot would silently act as a floor of zero on an all-zero row.
        M_raw = H.masked_fill(m3 == 0, -1.0).max(1).values.clamp_min(0.0)

        def z(x):
            return (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + EPS)

        return (self.w[0] * z(dense) + self.w[1] * z(S_raw / degb)
                + self.w[2] * z(M_raw / degb))


class DenseScorer:
    """The dense baseline: rank by the query-document cosine alone.

    No hypothetical answers, no popularity term, no parameters, no training. It is
    here so the table shows what the hypothetical-answer machinery buys over plain
    query-document similarity in the SAME encoder space -- without it, a reader
    cannot tell whether `current` and `attention` are close to each other because
    both are good or because neither adds anything over the encoder.
    """

    name = "dense"

    def __init__(self, device):
        pass

    def parameters(self):
        return []

    def state(self):
        return {"note": "no parameters; ranks by cos(E(q), E(d))"}

    def load(self, st):
        pass

    def __call__(self, H, mask, dense, pop):
        return (dense - dense.mean(1, keepdim=True)) / (dense.std(1, keepdim=True) + EPS)

class DeepSetsScorer(_nn.Module):
    """Compact Deep Sets, ~62 parameters. No sum, no max, no handcrafted summaries.

        z_qdj = phi([x_hyp_qdj, x_dir_qd])        phi: 2 -> 4 -> 4
        u_qd  = masked mean_j z_qdj               over VALID (and kept) answers
        s     = rho([x_dir_qd, u_qd])             rho: 5 -> 4 -> 1

    Pairing each answer with the direct score inside phi lets the network encode
    interactions ("a strong answer AND a weak direct match") before pooling, and
    the masked MEAN keeps u on one scale regardless of how many answers a query
    has -- the 610-parameter version summed, which tied its output scale to J.

    REGULARISATION IS THE POINT of this version; the large one overfit (best
    train loss of all arms, worse test).
      * 15% whole-answer dropout, training only: an entire hypothetical answer is
        dropped for a query, with the SAME mask for every paper of that query, so
        the model cannot rely on any single answer existing. The masked mean
        renormalises over the kept answers, so no 1/(1-p) scaling is needed.
      * weight decay handled by train_arm (matrices only), dev early stopping,
        and three seeds handled by the --seeds loop in main.

    THIS CLASS INHERITS nn.Module AND THE TRAINERS CALL train()/eval(). The other
    scorer classes are plain objects, so dropout inserted there would stay active
    during evaluation; hasattr guards in train_arm/dev_ndcg/score_rows make the
    mode switch a no-op for them and real for this one.

    If dropout removes every answer of a query (possible at J=1), u is zero and
    the score falls back to a function of x_dir alone, which is the right
    degradation. Permutation invariant: shared phi, symmetric mean.
    """

    name = "deepsets"

    def __init__(self, device, hidden=4, p_drop=0.15):
        super().__init__()
        self.p_drop = float(p_drop)
        self.phi = _nn.Sequential(_nn.Linear(2, hidden), _nn.GELU(),
                                  _nn.Linear(hidden, hidden))
        self.rho = _nn.Sequential(_nn.Linear(hidden + 1, hidden), _nn.GELU(),
                                  _nn.Linear(hidden, 1))
        self.logbeta = _nn.Parameter(_torch.zeros(()))
        self.to(device)

    def state(self):
        return {"sd": {k: v.detach().cpu().tolist() for k, v in self.state_dict().items()},
                "beta": float(self.logbeta.detach().exp().cpu()),
                "p_drop": self.p_drop}

    def load(self, st):
        import torch
        dev = self.logbeta.device
        self.load_state_dict({k: torch.tensor(v, device=dev) for k, v in st["sd"].items()})

    def forward(self, H, mask, dense, pop):
        import torch
        x_dir, x_hyp, _ = _views(H, mask, dense, pop, self.logbeta.exp())
        B, J, D = x_hyp.shape
        m = mask
        if self.training and self.p_drop > 0:
            # One mask per (query, answer), shared across all D papers: the unit
            # being dropped is the ANSWER, not a (paper, answer) cell.
            keep = (torch.rand(B, J, device=m.device) >= self.p_drop).float()
            m = m * keep
        inp = torch.stack([x_hyp, x_dir.unsqueeze(1).expand(B, J, D)], -1)  # [B,J,D,2]
        z = self.phi(inp) * m.view(B, J, 1, 1)                              # [B,J,D,h]
        u = z.sum(1) / m.sum(1).clamp(min=1.0).view(B, 1, 1)                # masked mean
        feat = torch.cat([x_dir.unsqueeze(-1), u], -1)                      # [B,D,h+1]
        return self.rho(feat).squeeze(-1)


class SetMLPScorer(_nn.Module):
    """A small, automatic, permutation-invariant semantic scorer.

    For every adjusted hypothetical-answer match x_j, one shared MLP produces a
    learned representation.  Summing these representations is the standard
    Deep Sets invariant; it is not a precomputed semantic feature.  A final
    residual MLP maps the pooled set and direct query match to one paper score.

    The first per-view channel is an identity path.  It gives gradients a stable
    route and makes the initial model a strong direct-plus-pooled retriever.  All
    output weights, the nonlinear set features, and beta remain trainable.  No
    maximum, top-k statistic, rank position, or manually weighted operator input
    is computed.
    """

    name = "setmlp"

    def __init__(self, device, hidden=8, p_drop=0.0):
        super().__init__()
        hidden = int(hidden)
        assert hidden >= 2
        self.hidden = hidden
        self.p_drop = float(p_drop)
        # h-1 nonlinear channels plus one unmodified identity channel.
        self.phi = _nn.Sequential(
            _nn.Linear(1, hidden), _nn.GELU(), _nn.Linear(hidden, hidden - 1))
        self.linear = _nn.Linear(hidden + 1, 1, bias=False)
        self.residual = _nn.Sequential(
            _nn.Linear(hidden + 1, hidden), _nn.GELU(),
            _nn.Dropout(0.10), _nn.Linear(hidden, 1))
        self.logbeta = _nn.Parameter(_torch.zeros(()))

        # Stable residual learning: epoch zero is direct + the identity set
        # channel.  This is an initialisation only; every coefficient is learned.
        with _torch.no_grad():
            self.linear.weight.zero_()
            self.linear.weight[0, 0] = 1.0       # direct query view
            self.linear.weight[0, 1] = 1.0       # pooled identity view
            self.residual[-1].weight.zero_()
            self.residual[-1].bias.zero_()
        self.to(device)

    def state(self):
        return {"sd": {k: v.detach().cpu().tolist() for k, v in self.state_dict().items()},
                "beta": float(self.logbeta.detach().exp().cpu()),
                "hidden": self.hidden, "p_drop": self.p_drop}

    def load(self, st):
        dev = self.logbeta.device
        self.load_state_dict({k: _torch.tensor(v, device=dev) for k, v in st["sd"].items()})

    def forward(self, H, mask, dense, pop):
        import torch
        x_dir, x_hyp = _set_views(H, mask, dense, pop, self.logbeta.exp())
        B, J, D = x_hyp.shape
        m = mask
        if self.training and self.p_drop > 0:
            keep = (torch.rand(B, J, device=m.device) >= self.p_drop).float()
            # Never erase the entire set: keep the first valid answer if a rare
            # all-dropped row occurs.  The rule is independent of paper scores.
            empty = (keep * m).sum(1) == 0
            if empty.any():
                first = m.float().argmax(1)
                keep[empty, first[empty]] = 1.0
            m = m * keep

        nonlinear = self.phi(x_hyp.unsqueeze(-1))
        per_view = torch.cat([x_hyp.unsqueeze(-1), nonlinear], -1)
        pooled = (per_view * m.view(B, J, 1, 1)).sum(1)              # [B,D,h]

        # Calibrate every learned pooled channel across this query's candidate
        # corpus.  This prevents a high-variance channel from winning merely by
        # scale and gives the final MLP comparable inputs.
        pooled = ((pooled - pooled.mean(1, keepdim=True)) /
                  (pooled.std(1, keepdim=True) + EPS))
        feat = torch.cat([x_dir.unsqueeze(-1), pooled], -1)
        return (self.linear(feat) + self.residual(feat)).squeeze(-1)


class DualSetMLPScorer(_nn.Module):
    """Set MLP with complementary additive and learned-selective pooling.

    ``u_all`` accumulates evidence from every answer. ``u_sel`` uses a learned
    positive temperature and a softmax *inside the set representation* to focus
    on informative answers.  The final score is unconstrained: unlike view-attention as the
    scorer itself, it is not a convex average and therefore does not cap
    evidence accumulation.  Both pools are permutation invariant and neither
    computes a handcrafted maximum.
    """

    name = "dualsetmlp"

    def __init__(self, device, hidden=8, p_drop=0.0):
        super().__init__()
        hidden = int(hidden)
        assert hidden >= 2
        self.hidden = hidden
        self.p_drop = float(p_drop)
        self.phi = _nn.Sequential(
            _nn.Linear(1, hidden), _nn.GELU(), _nn.Linear(hidden, hidden - 1))
        # A single positive temperature is enough to learn the continuum from
        # broad averaging (small tau) to strongest-view selection (large tau).
        # Monotonic selection generalises better than another free MLP here.
        self.logtau = _nn.Parameter(_torch.tensor(math.log(5.0)))
        width = 1 + 2 * hidden
        self.linear = _nn.Linear(width, 1, bias=False)
        self.residual = _nn.Sequential(
            _nn.Linear(width, hidden), _nn.GELU(), _nn.Dropout(0.10),
            _nn.Linear(hidden, 1))
        self.logbeta = _nn.Parameter(_torch.zeros(()))
        with _torch.no_grad():
            self.linear.weight.zero_()
            self.linear.weight[0, 0] = 1.0
            self.linear.weight[0, 1] = 1.0
            self.residual[-1].weight.zero_()
            self.residual[-1].bias.zero_()
        self.to(device)

    def state(self):
        return {"sd": {k: v.detach().cpu().tolist() for k, v in self.state_dict().items()},
                "beta": float(self.logbeta.detach().exp().cpu()),
                "hidden": self.hidden, "p_drop": self.p_drop}

    def load(self, st):
        dev = self.logbeta.device
        self.load_state_dict({k: _torch.tensor(v, device=dev) for k, v in st["sd"].items()})

    def forward(self, H, mask, dense, pop):
        import torch
        x_dir, x_hyp = _set_views(H, mask, dense, pop, self.logbeta.exp())
        B, J, D = x_hyp.shape
        m = mask
        if self.training and self.p_drop > 0:
            keep = (torch.rand(B, J, device=m.device) >= self.p_drop).float()
            empty = (keep * m).sum(1) == 0
            if empty.any():
                first = m.float().argmax(1)
                keep[empty, first[empty]] = 1.0
            m = m * keep

        nonlinear = self.phi(x_hyp.unsqueeze(-1))
        per_view = torch.cat([x_hyp.unsqueeze(-1), nonlinear], -1)
        valid = m.view(B, J, 1)
        u_all = (per_view * valid.unsqueeze(-1)).sum(1)

        logits = self.logtau.exp() * x_hyp
        logits = logits.masked_fill(valid == 0, -float("inf"))
        attn = torch.softmax(logits, 1)
        u_sel = (attn.unsqueeze(-1) * per_view).sum(1)

        def corpus_z(u):
            return (u - u.mean(1, keepdim=True)) / (u.std(1, keepdim=True) + EPS)

        feat = torch.cat([x_dir.unsqueeze(-1), corpus_z(u_all), corpus_z(u_sel)], -1)
        return (self.linear(feat) + self.residual(feat)).squeeze(-1)


class AttentionScorer:
    """The thesis method (Ch.6 Eq 6.10-6.12), exactly as written.

        l_v = f_phi([x_v, t_v])         one shared MLP, t=0 direct, t=1 hypothetical
        a_v = softmax over ALL J+1 views (padded answers masked to -inf)
        s   = sum_v a_v x_v

    No temperature, no warm start, no separate channels: this arm exists to test
    the written method as written. Views are normalised by _views(), i.e. the
    shared-scale refinement (centre each answer, one scale per query) that the
    method adopted in place of per-view standardisation.

    Known structural properties, stated so the result is readable:
      * the score is a weighted average, so it cannot exceed the paper's own
        largest view -- evidence cannot accumulate across views;
      * averaging J partly-independent views shrinks the hypothetical mass's
        spread well below x_dir's, so the direct view tends to dominate.

    Permutation invariant: one MLP scores every view, softmax and the weighted
    sum are symmetric. The direct view is always valid, so no row is fully masked.
    """

    name = "attention"

    def __init__(self, device, hidden=8):
        import torch
        import torch.nn as nn
        self.mlp = nn.Sequential(nn.Linear(2, hidden), nn.GELU(),
                                 nn.Linear(hidden, 1)).to(device)
        self.logbeta = nn.Parameter(torch.zeros((), device=device))

    def parameters(self):
        return list(self.mlp.parameters()) + [self.logbeta]

    def state(self):
        return {"mlp": {k: v.detach().cpu().tolist() for k, v in self.mlp.state_dict().items()},
                "beta": float(self.logbeta.detach().exp().cpu())}

    def load(self, st):
        import torch
        dev = self.logbeta.device
        self.mlp.load_state_dict({k: torch.tensor(v, device=dev)
                                  for k, v in st["mlp"].items()})
        with torch.no_grad():
            self.logbeta.copy_(torch.tensor(math.log(st["beta"]), device=dev))

    def __call__(self, H, mask, dense, pop):
        import torch
        x_dir, x_hyp, _ = _views(H, mask, dense, pop, self.logbeta.exp())
        B, J, D = x_hyp.shape
        dev = mask.device
        views = torch.cat([x_dir.unsqueeze(1), x_hyp], 1)            # [B, 1+J, D]
        vmask = torch.cat([torch.ones(B, 1, device=dev), mask], 1)
        t = torch.cat([torch.zeros(1, device=dev),
                       torch.ones(J, device=dev)]).view(1, 1 + J, 1)
        inp = torch.stack([views, t.expand(B, 1 + J, D)], -1)        # [B, 1+J, D, 2]
        logit = self.mlp(inp).squeeze(-1)                            # [B, 1+J, D]
        logit = logit.masked_fill(vmask.unsqueeze(-1) == 0, -float("inf"))
        a = torch.softmax(logit, 1)
        return (a * views).sum(1)


class SortedMLPScorer:
    """One MLP on the SORTED vector of view scores. No sum, no max, no attention.

        input  = [ x_dir , sort_desc(x_hyp_1..J) padded to jmax , n_valid/jmax ]
        s      = MLP(input)

    NOTHING IS HANDCRAFTED. Sorting is not a summary statistic, it is a
    canonical ordering: it makes the input permutation invariant by construction
    while throwing away nothing. The MLP then learns whatever function of the
    score distribution it wants -- how much the best answer counts, whether the
    second one matters, whether the gap between them matters, how many answers
    need to fire. `sum` and `max` are not computed anywhere.

    IT GENERALISES SUMMARY-BASED SCORING, WHICH IS NOT THE SAME AS CONTAINING IT.
    `current` is w0 x_dir + w1 (sum_j x_hyp) + w2 (max_j x_hyp), and on this input
    the sum is the sum of the sorted coordinates, the max is the first sorted
    coordinate, and x_dir is coordinate 0 -- so a linear readout gets close. But
    `_views` centres each hypothetical answer SEPARATELY before the shared scale,
    so the max channel here is the largest CENTRED value, which is not the same
    quantity `current` computes from raw matches. The honest claim is that this
    arm exposes the complete ordered match profile instead of two fixed summaries
    of it, not that it reproduces the baseline exactly.

    NO SPREAD PROBLEM. Every coordinate gets its own free weight, so the model can
    set any balance between the direct match and the answers. Both previous arms
    failed on exactly this: gated summed the answers and they overwhelmed x_dir,
    attention averaged them and x_dir overwhelmed them.

    Padded slots sort to the end and are zeroed, and the valid count is supplied as
    a feature, so a query with three answers and one with eight are distinguishable
    without the padding masquerading as evidence.
    """

    name = "mlp"

    def __init__(self, device, jmax, hidden=16):
        import torch
        import torch.nn as nn
        self.jmax = int(jmax)
        self.net = nn.Sequential(nn.Linear(self.jmax + 2, hidden), nn.GELU(),
                                 nn.Linear(hidden, 1)).to(device)
        self.logbeta = nn.Parameter(torch.zeros((), device=device))

    def parameters(self):
        return list(self.net.parameters()) + [self.logbeta]

    def state(self):
        return {"net": {k: v.detach().cpu().tolist() for k, v in self.net.state_dict().items()},
                "jmax": self.jmax,
                "beta": float(self.logbeta.detach().exp().cpu())}

    def load(self, st):
        import torch
        dev = self.logbeta.device
        self.net.load_state_dict({k: torch.tensor(v, device=dev)
                                  for k, v in st["net"].items()})
        with torch.no_grad():
            self.logbeta.copy_(torch.tensor(math.log(st["beta"]), device=dev))

    def __call__(self, H, mask, dense, pop):
        import torch
        x_dir, x_hyp, _ = _views(H, mask, dense, pop, self.logbeta.exp())
        B, J, D = x_hyp.shape
        dev = x_hyp.device

        # Invalid answers sort to the end, then get zeroed by position. Comparing
        # against the fill value to find them would be fragile; the count is exact.
        fill = torch.finfo(x_hyp.dtype).min / 4
        xs, _ = torch.sort(x_hyp.masked_fill(mask.unsqueeze(-1) == 0, fill),
                           dim=1, descending=True)
        nv = mask.sum(1)                                            # [B]
        xs = xs * (torch.arange(J, device=dev).view(1, J) < nv.view(B, 1)).float().unsqueeze(-1)

        # One fixed input width, so a split whose Jmax differs cannot change the
        # layer shape. Truncating keeps the HIGHEST scores, which is the useful end.
        if J < self.jmax:
            xs = torch.cat([xs, torch.zeros(B, self.jmax - J, D, device=dev)], 1)
        elif J > self.jmax:
            xs = xs[:, :self.jmax]

        feat = torch.cat([x_dir.unsqueeze(-1),                       # [B, D, 1]
                          xs.permute(0, 2, 1),                       # [B, D, jmax]
                          (nv.view(B, 1, 1) / self.jmax).expand(B, D, 1)], -1)
        return self.net(feat).squeeze(-1)                            # [B, D]


class GatedScorer:
    """Adaptive gated view pooling. Independent sigmoid gates, no softmax."""

    name = "gated"

    def __init__(self, device, hidden=8):
        import torch.nn as nn
        self.mlp = nn.Sequential(nn.Linear(2, hidden), nn.GELU(),
                                 nn.Linear(hidden, 1)).to(device)
        import torch
        self.logbeta = nn.Parameter(torch.zeros((), device=device))

    def parameters(self):
        return list(self.mlp.parameters()) + [self.logbeta]

    def state(self):
        return {"mlp": {k: v.detach().cpu().tolist() for k, v in self.mlp.state_dict().items()},
                "beta": float(self.logbeta.detach().exp().cpu())}

    def load(self, st):
        import torch
        self.mlp.load_state_dict({k: torch.tensor(v, device=self.logbeta.device)
                                  for k, v in st["mlp"].items()})
        with torch.no_grad():
            self.logbeta.copy_(torch.tensor(math.log(st["beta"]),
                                            device=self.logbeta.device))

    def __call__(self, H, mask, dense, pop):
        import torch
        x_dir, x_hyp, _ = _views(H, mask, dense, pop, self.logbeta.exp())
        B, J, D = x_hyp.shape
        views = torch.cat([x_dir.unsqueeze(1), x_hyp], 1)            # [B, 1+J, D]
        vmask = torch.cat([torch.ones(B, 1, device=mask.device), mask], 1)
        t = torch.cat([torch.zeros(1, device=mask.device),
                       torch.ones(J, device=mask.device)]).view(1, 1 + J, 1)
        inp = torch.stack([views, t.expand(B, 1 + J, D)], -1)        # [B, 1+J, D, 2]
        g = torch.sigmoid(self.mlp(inp).squeeze(-1))                 # [B, 1+J, D]
        # SUM, not a weighted average: gates do not sum to 1, so a paper that
        # matches several views well accumulates past one that spikes on a single
        # view. A softmax here would cap every score at its own largest view.
        return (g * views * vmask.unsqueeze(-1)).sum(1)


# --------------------------------------------------------------------------
# stage 3: the corrected multi-gold loss, and dev nDCG@10
# --------------------------------------------------------------------------
def operator_loss(scores, gold_idx, gold_val):
    """operator_scorer.py's objective, reproduced exactly.

        -torch.log_softmax(S_op, 1)[qi, gi].mean()

    i.e. the mean over all (query, gold) PAIRS of -log_softmax(s)[gold], with the
    denominator running over the whole corpus. THIS IS THE DEFAULT, so the
    `current` arm here reproduces the operator as it is actually fitted everywhere
    else in the project and the comparison changes only the architecture.

    It differs from `multigold_loss` in two ways at once, which is worth knowing
    when reading a --loss ablation:
      1. a query's other golds stay inside each gold's denominator;
      2. weighting is per (query, gold) pair, so a 10-gold query counts ten times
         as much as a 1-gold one.
    """
    import torch
    lp = torch.log_softmax(scores, 1).gather(1, gold_idx)
    return -(lp * gold_val).sum() / gold_val.sum().clamp_min(1.0)


def multigold_loss(scores, gold_idx, gold_val):
    """-log( exp(s_g) / (exp(s_g) + sum_{d not gold} exp(s_d)) ), averaged.

    Golds are removed from EACH OTHER's denominator, then averaged within a query
    and equally across queries so a 10-gold query does not outweigh a 1-gold one.
    """
    import torch
    # scatter_ADD, not scatter_. Padded gold slots carry index 0, so a query whose
    # first real gold IS document 0 would have two writes racing at the same cell
    # and the -inf could be overwritten by the padding's value, silently leaving a
    # gold in its own denominator. Accumulating avoids the collision entirely.
    acc = torch.zeros_like(scores)
    acc.scatter_add_(1, gold_idx, gold_val)
    neg = scores.masked_fill(acc > 0, -float("inf"))
    neg_lse = torch.logsumexp(neg, 1)                            # [B]
    sg = scores.gather(1, gold_idx)                              # [B, G]
    per_gold = torch.logaddexp(sg, neg_lse.unsqueeze(1)) - sg
    per_query = (per_gold * gold_val).sum(1) / gold_val.sum(1).clamp_min(1.0)
    return per_query.mean()


def ndcg_at(order, golds, k):
    g = set(golds)
    if not g:
        return 0.0
    dcg = sum(1 / math.log2(r + 2) for r, d in enumerate(order[:k]) if d in g)
    idcg = sum(1 / math.log2(r + 2) for r in range(min(len(g), k)))
    return dcg / idcg if idcg else 0.0


def batches(n, bs, shuffle=False, rng=None):
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


def score_rows(model, data, rows, device, qbatch, torch, pop=None):
    """Score a set of queries against the whole corpus, in query minibatches."""
    if hasattr(model, "eval"):
        model.eval()          # dropout must NOT fire at evaluation time
    out = np.empty((len(rows), data["H"].shape[-1]), np.float32)
    pop = Pop.from_data(data, device) if pop is None else pop
    for s in range(0, len(rows), qbatch):
        sel = rows[s:s + qbatch]
        H = torch.as_tensor(np.asarray(data["H"][sel], np.float32), device=device)
        mk = torch.as_tensor(data["mask"][sel].astype(np.float32), device=device)
        dn = torch.as_tensor(np.asarray(data["dense"][sel], np.float32), device=device)
        out[s:s + qbatch] = model(H, mk, dn, pop).detach().float().cpu().numpy()
    return out


def dev_ndcg_multi(model, data, rows, gold_lists, device, qbatch, torch, ks=(10, 100),
                   pop=None):
    """nDCG at several cutoffs from ONE scoring pass over the dev queries.

    Scoring the dev set is the expensive part of an epoch (every query against the
    whole corpus), so computing @10 and @100 with two calls would double it for a
    number that comes free from the same ranking.
    """
    with torch.no_grad():
        sc = score_rows(model, data, rows, device, qbatch, torch, pop)
    acc = {k: [] for k in ks}
    kmax = max(ks)
    for i, r in enumerate(rows):
        if not gold_lists[r]:
            continue
        order = list(np.argsort(-sc[i])[:kmax])
        for k in ks:
            acc[k].append(ndcg_at(order[:k], gold_lists[r], k))
    return tuple(float(np.mean(acc[k])) if acc[k] else 0.0 for k in ks)


def dev_ndcg(model, data, rows, gold_lists, device, qbatch, torch, k=10, pop=None):
    with torch.no_grad():
        sc = score_rows(model, data, rows, device, qbatch, torch, pop)
    vals = []
    for i, r in enumerate(rows):
        if gold_lists[r]:
            vals.append(ndcg_at(list(np.argsort(-sc[i])[:k]), gold_lists[r], k))
    return float(np.mean(vals)) if vals else 0.0


def train_fixed(model, tr, fit_rows, gold_idx, gold_val, device, a, torch,
                n_epochs, lr=None, seed=None):
    """Train for exactly n_epochs on fit_rows. No dev split, no early stopping.

    Used for the FINAL fit of a k-fold run, where the epoch count has already been
    chosen by the folds and there is no held-out data left to select on -- which is
    the point: every train query is in this fit.
    """
    pop = Pop.from_data(tr, device) if pop is None else pop
    # Under joint training the popularity predictor is part of the optimised
    # model, so its matrices belong in the decayed group like any other.
    trainable = list(model.parameters()) + pop.parameters()
    groups = [g for g in (
        {"params": [p for p in trainable if p.ndim >= 2],
         "weight_decay": a.weight_decay},
        {"params": [p for p in trainable if p.ndim < 2],
         "weight_decay": 0.0}) if g["params"]]
    opt = torch.optim.AdamW(groups, lr=a.lr if lr is None else lr)
    gi = torch.as_tensor(gold_idx, device=device)
    gv = torch.as_tensor(gold_val, device=device)
    rng = np.random.default_rng(a.seed if seed is None else seed)
    lossfn = operator_loss if a.loss == "operator" else multigold_loss
    hist = []
    for ep in range(n_epochs):
        if hasattr(model, "train"):
            model.train()
        losses = []
        for bi in batches(len(fit_rows), a.qbatch, True, rng):
            sel = fit_rows[bi]
            H = torch.as_tensor(np.asarray(tr["H"][sel], np.float32), device=device)
            mk = torch.as_tensor(tr["mask"][sel].astype(np.float32), device=device)
            dn = torch.as_tensor(np.asarray(tr["dense"][sel], np.float32), device=device)
            loss = lossfn(model(H, mk, dn, pop), gi[sel], gv[sel])
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.detach()))
        hist.append({"epoch": ep + 1, "train_loss": float(np.mean(losses))})
        print(f"  [{model.name}] final ep{ep + 1:3d}/{n_epochs} "
              f"train {np.mean(losses):.4f}", flush=True)
    if hasattr(model, "eval"):
        model.eval()
    return hist


def kfold_epochs(build, tr, all_rows, gold_idx, gold_val, gold_lists, device, a,
                 torch, lr=None, seed=None):
    """Choose the epoch count by K-fold CV over EVERY train query.

    Each query serves in dev exactly once, so the selection signal spans the whole
    train split rather than one 300-query slice, and every query still contributes
    to fitting in K-1 of the K folds. Returns the median selected epoch, which is
    then used for a final fit on all of the data.
    """
    K = a.kfold
    rng = np.random.default_rng(a.seed if seed is None else seed)
    order = all_rows.copy()
    rng.shuffle(order)
    folds = np.array_split(order, K)
    picked = []
    for k in range(K):
        dev_k = folds[k]
        fit_k = np.concatenate([folds[j] for j in range(K) if j != k])
        m = build()
        _, _, hist, _, _ = train_arm(m, tr, fit_k, dev_k, gold_idx, gold_val,
                                     gold_lists, device, a, torch, lr=lr, seed=seed)
        key = "dev_loss" if a.select_on == "loss" else "dev_ndcg@10"
        best_ep = (min(hist, key=lambda r: r[key]) if a.select_on == "loss"
                   else max(hist, key=lambda r: r[key]))["epoch"]
        picked.append(best_ep)
        print(f"  [fold {k + 1}/{K}] fit {len(fit_k)} dev {len(dev_k)} "
              f"-> epoch {best_ep}")
    E = int(np.median(picked))
    print(f"  [kfold] selected epochs per fold {picked} -> median {E}")
    return max(E, 1), picked


def dev_loss(model, data, rows, gold_idx, gold_val, device, qbatch, torch, lossfn,
             pop=None):
    """The training objective evaluated on held-out dev queries.

    WHY SELECT ON THIS RATHER THAN dev nDCG@10. nDCG@10 with binary relevance is a
    STEP function: a query's score moves only when a gold crosses a rank boundary
    inside the top 10, so most parameter updates change it by exactly zero and the
    rest change it in jumps. Taking the best over ~30 epochs x 3 seeds of a chunky
    signal on the same 300 queries selects the checkpoint that got luckiest on
    those queries, not the one that generalises -- which is why the physics run
    inverted: the arm with the WORST dev nDCG won on test, and the arm with the
    best dev nDCG lost. The loss is continuous, moves every step, and is the
    quantity actually being optimised. nDCG is still computed and reported.
    """
    if hasattr(model, "eval"):
        model.eval()
    pop = Pop.from_data(data, device) if pop is None else pop
    gi = torch.as_tensor(gold_idx, device=device)
    gv = torch.as_tensor(gold_val, device=device)
    tot, n = 0.0, 0
    with torch.no_grad():
        for s in range(0, len(rows), qbatch):
            sel = rows[s:s + qbatch]
            H = torch.as_tensor(np.asarray(data["H"][sel], np.float32), device=device)
            mk = torch.as_tensor(data["mask"][sel].astype(np.float32), device=device)
            dn = torch.as_tensor(np.asarray(data["dense"][sel], np.float32), device=device)
            # The loss fns average within a call, so weight by rows to recover the
            # overall mean when the last minibatch is short.
            tot += float(lossfn(model(H, mk, dn, pop), gi[sel], gv[sel])) * len(sel)
            n += len(sel)
    return tot / max(n, 1)


def train_arm(model, tr, fit_rows, dev_rows, gold_idx, gold_val, gold_lists,
              device, a, torch, lr=None, seed=None, pop=None):
    # WEIGHT DECAY ON WEIGHT MATRICES ONLY (ndim >= 2). Biases and the scalar
    # calibration parameters are excluded deliberately: decaying them is not
    # regularisation, it is a prior. log_tau -> 0 means tau -> 1, logbeta -> 0
    # means beta -> 1, and shrinking `current`'s w toward 0 flattens the corpus
    # softmax and inflates the loss the optimiser is trying to reduce. The stated
    # purpose is to discourage sharp MLP functions, and this is the scoping that
    # actually does that and nothing else.
    pop = Pop.from_data(tr, device) if pop is None else pop
    # Under joint training the popularity predictor is part of the optimised
    # model, so its matrices belong in the decayed group like any other.
    trainable = list(model.parameters()) + pop.parameters()
    groups = [g for g in (
        {"params": [p for p in trainable if p.ndim >= 2],
         "weight_decay": a.weight_decay},
        {"params": [p for p in trainable if p.ndim < 2],
         "weight_decay": 0.0}) if g["params"]]
    opt = torch.optim.AdamW(groups, lr=a.lr if lr is None else lr)
    gi = torch.as_tensor(gold_idx, device=device)
    gv = torch.as_tensor(gold_val, device=device)
    rng = np.random.default_rng(a.seed if seed is None else seed)
    lossfn = operator_loss if a.loss == "operator" else multigold_loss
    on_loss = a.select_on == "loss"
    sel_k = 100 if a.select_on == "ndcg100" else 10
    # Lower is better for loss, higher for nDCG, so track the score to BEAT in the
    # sign the criterion wants and compare one way.
    best_sel, best_state, stale, hist = float("inf"), None, 0, []
    best_nd, best_pop = 0.0, None
    for ep in range(a.epochs):
        # Real for nn.Module arms (dropout on/off), no-op for the plain classes.
        if hasattr(model, "train"):
            model.train()
        losses = []
        for bi in batches(len(fit_rows), a.qbatch, True, rng):
            sel = fit_rows[bi]
            H = torch.as_tensor(np.asarray(tr["H"][sel], np.float32), device=device)
            mk = torch.as_tensor(tr["mask"][sel].astype(np.float32), device=device)
            dn = torch.as_tensor(np.asarray(tr["dense"][sel], np.float32), device=device)
            loss = lossfn(model(H, mk, dn, pop), gi[sel], gv[sel])
            aux = pop.aux_loss()
            if not isinstance(aux, float):
                loss = loss + a.pop_lambda * aux
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        if hasattr(model, "eval"):
            model.eval()
        dl = dev_loss(model, tr, dev_rows, gold_idx, gold_val, device, a.qbatch,
                      torch, lossfn, pop)
        nd, nd100 = dev_ndcg_multi(model, tr, dev_rows, gold_lists, device,
                                   a.qbatch, torch, (10, 100), pop)
        hist.append({"epoch": ep + 1, "train_loss": float(np.mean(losses)),
                     "dev_loss": dl, "dev_ndcg@10": nd, "dev_ndcg@100": nd100})
        # all three minimised after the sign flip
        sel_now = dl if on_loss else -(nd100 if sel_k == 100 else nd)
        flag = ""
        if sel_now < best_sel - 1e-9:
            best_sel, best_nd, stale, flag = sel_now, nd, 0, " *"
            best_state = json.loads(json.dumps(model.state()))
            best_pop = pop.state()          # None unless the predictor is joint
        else:
            stale += 1
        print(f"  [{model.name}] ep{ep + 1:3d} train {np.mean(losses):.4f} "
              f"dev_loss {dl:.4f} nDCG@10 {nd:.4f} nDCG@100 {nd100:.4f} "
              f"beta {math.exp(float(model.logbeta.detach())):.3f}{flag}", flush=True)
        if stale >= a.patience:
            print(f"  [{model.name}] early stop: {stale} evals without a "
                  f"dev {a.select_on} gain")
            break
    model.load(best_state)
    # pop_tr and pop_te SHARE one predictor object, so restoring it here also
    # restores the popularity the test split will be scored with.
    pop.load_state(best_pop)
    # Return the nDCG AT THE SELECTED CHECKPOINT, not the best nDCG seen. Those
    # differ under loss selection, and reporting the max would reintroduce exactly
    # the optimistic bias this change removes.
    return best_nd, best_state, hist, best_sel, best_pop


def distill_operator(model, teacher, tr, fit_rows, device, a, torch, seed=None):
    """Initialise a raw-view neural scorer from the fitted operator ranking.

    This is representation distillation, not an inference-time ensemble: the
    teacher is used only on source-training queries.  The student never receives
    the operator's sum or maximum as input and the teacher is absent at test.
    """
    groups = [g for g in (
        {"params": [p for p in model.parameters() if p.ndim >= 2],
         "weight_decay": a.weight_decay},
        {"params": [p for p in model.parameters() if p.ndim < 2],
         "weight_decay": 0.0}) if g["params"]]
    opt = torch.optim.AdamW(groups, lr=a.distill_lr)
    pop = Pop.from_data(tr, device)
    rng = np.random.default_rng(a.seed if seed is None else seed)
    teacher_was_training = getattr(teacher, "training", False)
    if hasattr(teacher, "eval"):
        teacher.eval()
    if hasattr(model, "eval"):
        model.eval()  # deterministic teacher matching; gradients remain enabled
    for ep in range(a.distill_epochs):
        losses = []
        for bi in batches(len(fit_rows), a.qbatch, True, rng):
            sel = fit_rows[bi]
            H = torch.as_tensor(np.asarray(tr["H"][sel], np.float32), device=device)
            mk = torch.as_tensor(tr["mask"][sel].astype(np.float32), device=device)
            dn = torch.as_tensor(np.asarray(tr["dense"][sel], np.float32), device=device)
            with torch.no_grad():
                target = teacher(H, mk, dn, pop)
                target = ((target - target.mean(1, keepdim=True)) /
                          (target.std(1, keepdim=True) + EPS))
            pred = model(H, mk, dn, pop)
            pred = ((pred - pred.mean(1, keepdim=True)) /
                    (pred.std(1, keepdim=True) + EPS))
            loss = torch.mean((pred - target) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        print(f"  [distill] ep{ep + 1:3d} mse {np.mean(losses):.6f}", flush=True)
    if hasattr(teacher, "train") and teacher_was_training:
        teacher.train()


def write_predictions(scores, data, path, topk=100):
    recs = []
    for i, q in enumerate(data["queries"]):
        order = np.argsort(-scores[i])[:topk]
        recs.append({"id": q["id"], "stratum": q.get("stratum"),
                     "supporting_documents": q.get("supporting_documents", []),
                     "predictions": {"document": [[data["doc_ids"][j], float(scores[i, j])]
                                                  for j in order]}})
    json.dump(recs, open(path, "w"))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="BAAI/bge-large-en-v1.5")
    # 0 = every train query left after the dev slice. The old default of 2,500
    # came from operator_scorer.py, where it fitted FOUR parameters and more data
    # bought nothing. The learned arms here have 34-200 parameters and their
    # measured failure mode is overfitting, so capping the fit set is backwards:
    # it discarded 2,645 CS queries and 6,869 TOMATO queries for no reason.
    ap.add_argument("--train_fit", type=int, default=0,
                    help="fit queries after the dev slice; 0 = all remaining")
    # 300, not operator_scorer.py's 600. Dev is sliced FIRST, so on a small domain
    # it decides how much fit data is left: matsci has 1,304 train queries, where
    # 600 leaves 704 and 300 leaves 1,004. 300 still gives a stable nDCG@10.
    ap.add_argument("--dev", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2,
                    help="applied to MLP weight MATRICES only; see train_arm")
    # Per-arm override, unset by default so every arm trains at --lr. Kept
    # because attention peaked at epoch 1 under 1e-3 while the others peaked at
    # 4-6, so a separate rate may be wanted again.
    ap.add_argument("--lr_attention", type=float, default=None)
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--qbatch", type=int, default=8, help="queries per minibatch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--arms", default="dense,current,attention,deepsets,mlp",
                    help="any of dense, current, attention, deepsets, setmlp, "
                         "dualsetmlp, mlp, gated")
    ap.add_argument("--mlp_hidden", type=int, default=16)
    ap.add_argument("--ds_hidden", type=int, default=4,
                    help="compact deepsets phi/rho width (spec: 4)")
    ap.add_argument("--ds_dropout", type=float, default=0.15,
                    help="whole-answer dropout in the deepsets arm, training only")
    ap.add_argument("--set_hidden", type=int, default=8,
                    help="identity-preserving set-MLP width")
    ap.add_argument("--set_dropout", type=float, default=0.0,
                    help="whole-answer dropout in the setmlp arm, training only")
    ap.add_argument("--distill_operator", default=None,
                    help="fitted current-scorer JSON used only to initialise a set MLP")
    ap.add_argument("--distill_epochs", type=int, default=0)
    ap.add_argument("--distill_lr", type=float, default=1e-3)
    ap.add_argument("--seeds", default="0",
                    help="comma-separated training seeds; each trained arm runs once "
                         "per seed and keeps the best dev checkpoint. The fit/dev "
                         "SPLIT stays fixed by --seed, so seeds vary only init, "
                         "batch order and dropout.")
    # DEFAULT IS THE OPERATOR'S OWN OBJECTIVE, so the `current` arm reproduces the
    # scorer as it is fitted elsewhere in the project and the only thing varying
    # between arms is the architecture. `fixed` is the multi-gold correction:
    # SIR-4 averages ~4 golds per query and the operator objective keeps them in
    # each other's denominator, so it pushes a query's own labels apart. Run both
    # to measure that rather than assume it.
    ap.add_argument("--loss", default="fixed", choices=["operator", "fixed"])
    # DEPLOYABILITY OF THE POPULARITY TERM. `loo` needs the OTHER test queries'
    # hypothetical answers, so it cannot score one query alone. `bank` replaces it
    # with the train-answer bank (query-independent, one matmul per corpus at
    # indexing time). `predicted` distils that bank into an MLP of the paper
    # embedding, so inference is fully local. Default stays loo so every prior
    # number reproduces.
    ap.add_argument("--popularity", default="loo", choices=["loo", "bank", "predicted"])
    # The mlp arm's popularity is PART OF THE ARM, not a run-level axis: the whole
    # proposal is a scorer that needs nothing but the paper embedding at inference.
    # Set 0 to make it share --popularity with the other arms instead.
    ap.add_argument("--mlp_popularity", type=int, default=1,
                    help="1 = the mlp arm uses its own learned popularity predictor")
    # TWO-STAGE vs JOINT. Two-stage fits the predictor to the bank targets, freezes
    # it, then trains the scorer -- so a win is attributable to the scorer and the
    # predictor keeps meaning "general matchability". Joint lets the retrieval
    # gradient reach the predictor as well, anchored by --pop_lambda on the same
    # log-MSE target; more expressive, less interpretable, and the predictor can
    # drift into being extra scorer capacity if lambda is too small.
    ap.add_argument("--mlp_pop_joint", type=int, default=0,
                    help="1 = train the popularity predictor jointly with the scorer")
    ap.add_argument("--pop_lambda", type=float, default=1.0,
                    help="weight on the matchability anchor under joint training")
    # Checkpoint/early-stopping criterion. `loss` is the default because dev
    # nDCG@10 is a step function and selecting its max over ~90 evaluations on one
    # 300-query slice picks luck, not generalisation. `ndcg` reproduces the old
    # behaviour.
    # SELECT ON THE METRIC, NOT THE LOSS. Measured on physics: dev loss and dev
    # nDCG@10 move in OPPOSITE directions on the same held-out queries (attention
    # seed 2, ep18 -> ep29: loss 5.8507 -> 5.6360 while nDCG 0.4329 -> 0.3976).
    # The loss is InfoNCE over the whole corpus, so it is rewarded for separating
    # the gold from negatives at rank 2000 that no metric sees; beta collapses
    # (0.89 -> 0.37), switching off the popularity discount that suppresses
    # generically attractive papers at the TOP. Global separation improves, head
    # precision degrades. nDCG@10 is a chunky step function, which is the real
    # problem, but the cure is a SMOOTHER VERSION OF THE METRIC:
    #   ndcg    nDCG@10, the reported metric, chunky but aligned
    #   ndcg100 nDCG@100, responds to a gold moving anywhere in the top 100, so
    #           many more events per epoch, still head-weighted by the log discount
    #   loss    kept only to reproduce the runs that exposed this
    ap.add_argument("--select_on", default="loss",
                    choices=["ndcg", "ndcg100", "loss"],
                    help="ndcg = nDCG@10 (the reported metric); ndcg100 = nDCG@100 "
                         "(same metric, more resolution per epoch); loss = dev loss")
    # 0 = the historical single dev slice. K >= 2 runs K-fold CV over EVERY train
    # query to choose the epoch count, then refits on ALL of them -- so no query is
    # permanently held out of training, and the selection signal is the whole train
    # split instead of a 300-query sample.
    ap.add_argument("--kfold", type=int, default=0)
    ap.add_argument("--force_build", action="store_true")
    ap.add_argument("--cache_only", action="store_true",
                    help="fail rather than load an embedding model if an array is missing")
    ap.add_argument("--selection_only", action="store_true",
                    help="train and select on development data without scoring test")
    add_dataset_arg(ap)
    a = ap.parse_args()
    set_dataset(a.dataset)
    print(banner())

    import torch
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(a.seed)
    op = _op_module()
    out = a.out or f"{_ROOT}/experiments/results/semantic_{a.dataset}"
    os.makedirs(out, exist_ok=True)
    cache = f"{_ROOT}/outputs/caches/semantic/{a.dataset}"
    print(f"device={device} encoder={a.model}\nout={out}")

    if a.cache_only:
        # build_inputs reaches the encoder only when one of these arrays is
        # absent.  Check here so a cache-only run fails before importing any
        # embedding library or constructing a model.
        for split in ("train", "test"):
            _, _, queries, _ = op.load_split(split)
            qkey = hashlib.md5("|".join(q["id"] for q in queries).encode()).hexdigest()[:8]
            slug = op.model_slug(a.model)
            qi = op.query_instruction(a.model)
            required = [f"{emb_dir()}/{split}_doc{slug}.npy",
                        f"{emb_dir()}/{split}_query_{qkey}{slug}{op.qi_tag(qi)}.npy",
                        f"{emb_dir()}/{split}_probe_{qkey}{slug}.npy"]
            missing = [p for p in required if not os.path.exists(p)]
            assert not missing, "cache-only run is missing:\n  " + "\n  ".join(missing)
    tr = build_inputs(op, a.model, "train", cache, a.force_build)
    te = build_inputs(op, a.model, "test", cache, a.force_build)

    arms_req = [x.strip() for x in a.arms.split(",") if x.strip()]
    # THE MLP ARM CARRIES ITS OWN LEARNED POPULARITY. It is not a separate axis:
    # the proposal is one deployable scorer -- learned pooling AND a popularity
    # discount predicted from the paper embedding alone, with no dependence on the
    # other test queries. `current` keeps the leave-one-out term it has always
    # used, so the comparison is proposal-vs-baseline as each is actually meant to
    # be deployed. Both sources are computed once and stored side by side.
    MLP_ARMS = {"mlp": "joint", "mlp2s": "_pred", "mlpbank": "_bank"}
    if any(x in arms_req for x in MLP_ARMS) or a.popularity != "loo":
        # TRAIN targets are already on disk: total_S is the sum over every train
        # answer, so dividing by the answer count IS the bank mean for the train
        # corpus. Only the TEST corpus needs a fresh matmul, against the SAME
        # train bank -- no test query ever contributes to any popularity value.
        slug = op.model_slug(a.model)
        n_bank = int(tr["mask"].sum())
        p_tr = (tr["total_S"] / max(n_bank, 1)).astype(np.float32)
        bank_tr = p_tr.copy()          # the anchor target, before any prediction
        bank_emb = np.load(f"{emb_dir()}/train_probe_{tr['meta']['qkey']}{slug}.npy"
                           ).astype(np.float32)
        assert bank_emb.shape[0] == n_bank, (
            f"train answer bank {bank_emb.shape[0]} rows != {n_bank} valid answers")
        de_te = np.load(f"{emb_dir()}/test_doc{slug}.npy").astype(np.float32)
        assert de_te.shape[0] == len(te["doc_ids"]), "test doc embedding misaligned"
        p_te = bank_popularity(bank_emb, de_te)
        print(f"[pop] bank of {n_bank} train answers; train p in "
              f"[{p_tr.min():.4f}, {p_tr.max():.4f}], test p in "
              f"[{p_te.min():.4f}, {p_te.max():.4f}]")
        # KEEP THE BANK VECTORS. They are the anchor targets AND the `mlpbank`
        # arm's popularity, so overwriting them with predictions below would lose
        # the one configuration measured to beat the baseline.
        bank_te = p_te.copy()
        tr["pop_mode_bank"] = te["pop_mode_bank"] = "bank"
        tr["pop_vec_bank"], te["pop_vec_bank"] = bank_tr, bank_te
        pop_info = {"mode": a.popularity, "bank_answers": n_bank}
        _fitted, pop_joint = False, None
        if a.popularity == "predicted" or a.mlp_popularity:
            _fitted = True
            de_tr = np.load(f"{emb_dir()}/train_doc{slug}.npy").astype(np.float32)
            assert de_tr.shape[0] == len(tr["doc_ids"]), "train doc embedding misaligned"
            gm, fit_info = fit_matchability(de_tr, p_tr, device, seed=a.seed)
            pop_info["fit"] = fit_info
            with _torch.no_grad():
                # The scorer sees PREDICTED popularity on BOTH splits, so its
                # training-time input distribution matches inference exactly.
                p_tr = gm(_torch.as_tensor(de_tr, device=device)).cpu().numpy()
                p_te = gm(_torch.as_tensor(de_te, device=device)).cpu().numpy()
            tr["pop_mode_pred"] = te["pop_mode_pred"] = "predicted"
            tr["pop_vec_pred"], te["pop_vec_pred"] = p_tr, p_te
            _torch.save(gm.state_dict(),
                        f"{out}/matchability_{a.dataset}{slug}.pt")
            # Joint mode rebuilds the predictor per seed, warm-started from this
            # two-stage fit, so it begins at "general matchability" rather than at
            # noise and any drift is attributable to the retrieval gradient.
            pop_joint = {"init": {k: v.detach().cpu() for k, v in gm.state_dict().items()},
                         "dim": de_tr.shape[1], "hidden": fit_info["hidden"],
                         "de_tr": de_tr, "de_te": de_te,
                         "target_tr": bank_tr}
        if a.popularity != "loo":
            tr["pop_mode"] = te["pop_mode"] = a.popularity
            tr["pop_vec"], te["pop_vec"] = p_tr, p_te
        # pop_joint only exists if a predictor was fitted above; the joint arm is
        # unavailable otherwise and falls back to the frozen vectors.
        pop_joint = pop_joint if (a.mlp_pop_joint and _fitted) else None
    else:
        pop_info, pop_joint = {"mode": "loo"}, None
    gidx, gval, glists = gold_matrix(tr["queries"], tr["doc_ids"])
    te_glists = None if a.selection_only else gold_matrix(te["queries"], te["doc_ids"])[2]

    # SAME SLICING RULE AS operator_scorer.py: dev first, then fit from what is
    # left. matsci train holds 1,304 queries, so a 2,500-query fit set does not
    # exist and the request is silently truncated -- print what actually happened.
    Q = len(tr["queries"])
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(Q)
    dev_rows = perm[:a.dev]
    fit_rows = perm[a.dev:] if a.train_fit <= 0 else perm[a.dev:a.dev + a.train_fit]
    all_rows = perm                      # every train query, for --kfold
    # Under --kfold the single fit/dev split is unused: the folds provide both, and
    # the final model refits on all_rows. So an empty fit_rows is only a problem in
    # the non-kfold path.
    assert a.kfold >= 2 or len(fit_rows) > 0, (
        f"no fit queries left: {Q} train queries, --dev {a.dev}")
    if a.kfold < 2 and ((a.train_fit > 0 and len(fit_rows) < a.train_fit)
                        or len(dev_rows) < a.dev):
        print(f"[split] REQUESTED fit={a.train_fit} dev={a.dev} but the train split "
              f"holds {Q} queries -> using fit={len(fit_rows)} dev={len(dev_rows)}")
    # ONE input width across splits: the two splits can have different Jmax and a
    # layer shaped for one would fail on the other, after training.
    JFIX = max(tr["H"].shape[1], te["H"].shape[1])
    print(f"[split] jmax(train)={tr['H'].shape[1]} jmax(test)={te['H'].shape[1]} -> JFIX={JFIX}")
    if a.kfold >= 2:
        print(f"[split] {a.kfold}-fold CV over ALL {Q} train queries: each fold fits "
              f"on ~{Q - Q // a.kfold} and devs on ~{Q // a.kfold}; the final model "
              f"refits on all {Q}. test={len(te['queries'])}")
    else:
        print(f"[split] fit={len(fit_rows)} dev={len(dev_rows)} of {Q} train queries "
              f"(seed {a.seed}); test={len(te['queries'])}")

    # Filenames are per-arm AND per-loss and never reused, so a rerun adds files
    # rather than silently replacing another configuration's predictions. Without
    # the loss in the name a legacy run would overwrite the corrected one and the
    # two would be indistinguishable afterwards.
    LSUF = "_operatorloss" if a.loss == "operator" else "_fixedloss"
    # bank/predicted results must never overwrite the loo ones they are read against.
    LSUF += "" if a.popularity == "loo" else f"_{a.popularity}pop"
    TAG = {k: f"{k}{LSUF}" for k in
           ("current", "attention", "deepsets", "setmlp", "dualsetmlp", "gated")}
    TAG["dense"] = "dense"          # untrained, so no loss belongs in its name
    for k in ("mlp", "mlp2s", "mlpbank"):
        TAG[k] = f"{k}{LSUF}"
    seeds = [int(x) for x in str(a.seeds).split(",") if x.strip() != ""]
    results, summary = {}, {}
    for arm in [x.strip() for x in a.arms.split(",") if x.strip()]:
        assert arm in TAG, f"unknown arm {arm!r}; choose from {sorted(TAG)}"
        print(f"\n=== arm: {arm} ===")

        def build():
            return (CurrentScorer(device) if arm == "current"
                    else AttentionScorer(device, a.hidden) if arm == "attention"
                    else DeepSetsScorer(device, a.ds_hidden, a.ds_dropout) if arm == "deepsets"
                    else SetMLPScorer(device, a.set_hidden, a.set_dropout) if arm == "setmlp"
                    else DualSetMLPScorer(device, a.set_hidden, a.set_dropout) if arm == "dualsetmlp"
                    else DenseScorer(device) if arm == "dense"
                    else SortedMLPScorer(device, JFIX, a.mlp_hidden) if arm in MLP_ARMS
                    else GatedScorer(device, a.hidden))

        # THE MLP ARM READS ITS OWN POPULARITY. Its learned predictor is part of
        # the proposal, not a run-level axis, so it is selected per arm here while
        # every other arm keeps whatever --popularity chose.
        # EACH MLP VARIANT DIFFERS ONLY IN ITS POPULARITY SOURCE, so one run
        # measures all three against the same baseline on the same split:
        #   mlp      joint  -- predictor trained with the scorer (the proposal)
        #   mlp2s    _pred  -- predictor fitted to the bank, then frozen
        #   mlpbank  _bank  -- the measured bank mean, nothing learned
        want = MLP_ARMS.get(arm, "")
        joint_here = want == "joint" and pop_joint is not None
        vk = "" if joint_here else (want if f"pop_vec{want}" in tr else "")

        def make_pops():
            """Fresh Pop pair. Under joint mode the predictor is rebuilt per seed."""
            if not joint_here:
                return Pop.from_data(tr, device, vk), Pop.from_data(te, device, vk)
            gj = MatchabilityPredictor(pop_joint["dim"], pop_joint["hidden"]).to(device)
            gj.load_state_dict({k: v.to(device) for k, v in pop_joint["init"].items()})
            dtr = _torch.as_tensor(pop_joint["de_tr"], device=device)
            dte = _torch.as_tensor(pop_joint["de_te"], device=device)
            tgt = _torch.as_tensor(pop_joint["target_tr"], device=device)
            # ONE predictor shared by both splits: the test popularity must come
            # from the network that training produced, not a second copy.
            return (Pop("joint", predictor=gj, doc_emb=dtr, target=tgt),
                    Pop("joint", predictor=gj, doc_emb=dte))

        pop_tr, pop_te = make_pops()

        model = build()
        npar = sum(p.numel() for p in model.parameters())
        print(f"  parameters: {npar}"
              + ("  popularity: predictor, JOINT with the scorer" if joint_here
                 else "  popularity: predictor, frozen (two-stage)" if vk == "_pred"
                 else "  popularity: train-answer bank (measured, not learned)" if vk == "_bank"
                 else f"  popularity: {tr.get('pop_mode', 'loo')}"))
        t0 = time.time()
        per_seed = {}
        # BEFORE the branch. An untrained arm never enters the seed loop, so an
        # initialisation inside it leaves this name unbound for `dense` and the
        # predictor-saving step below crashes on the very first arm.
        best_pop_seed = None
        if npar == 0:
            # Nothing to fit. Its dev score is still recorded so the trained arms
            # can be read against a fixed reference on the same queries.
            print("  no parameters: scoring directly, no training")
            best = dev_ndcg(model, tr, dev_rows, glists, device, a.qbatch, torch,
                            pop=pop_tr)
            state, hist = model.state(), []
            print(f"  [dense] dev nDCG@10 {best:.4f}")
        else:
            lr = a.lr_attention if (arm == "attention" and a.lr_attention) else a.lr
            ndec = sum(p.numel() for p in model.parameters() if p.ndim >= 2)
            print(f"  lr: {lr}  weight_decay: {a.weight_decay} on {ndec} of {npar} params"
                  + (f"  seeds: {seeds}" if len(seeds) > 1 else ""))
            # SEEDS VARY TRAINING ONLY (init, batch order, dropout). The fit/dev
            # split is pinned by --seed, so every seed and every arm sees the same
            # queries and the best-dev selection is a fair comparison.
            # SEEDS ARE SELECTED ON THE SAME CRITERION AS EPOCHS. Picking the
            # best-of-3 by dev nDCG while epochs are picked by dev loss would put
            # the discarded step-function signal straight back in, one level up.
            best_sel_seed, best, state, hist = float("inf"), -1.0, None, []
            for sd in seeds:
                torch.manual_seed(sd)
                # Rebuild for EVERY seed, including a one-seed invocation such
                # as --seeds 2.  Otherwise that invocation silently retains the
                # model constructed under --seed (normally zero).
                model = build()
                # REBUILD THE PREDICTOR TOO. Left outside the loop it would carry
                # seed 0's trained weights into seed 1, so the seeds would not be
                # independent and later ones would start pre-trained.
                pop_tr, pop_te = make_pops()
                if len(seeds) > 1:
                    print(f"  -- seed {sd} --")
                if arm in ("setmlp", "dualsetmlp") and a.distill_epochs > 0:
                    assert a.distill_operator and os.path.exists(a.distill_operator), (
                        "--distill_epochs requires an existing --distill_operator JSON")
                    teacher = CurrentScorer(device)
                    teacher.load(json.load(open(a.distill_operator)))
                    print(f"  distilling fitted operator for {a.distill_epochs} epochs; "
                          "teacher is not used at inference")
                    distill_operator(model, teacher, tr, fit_rows, device, a, torch, seed=sd)
                if a.kfold >= 2:
                    # Folds choose the epoch count; the final fit then uses EVERY
                    # train query. There is no held-out data left to select on,
                    # which is the intended trade: all the data trains the model.
                    E, picked = kfold_epochs(build, tr, all_rows, gidx, gval, glists,
                                             device, a, torch, lr=lr, seed=sd)
                    model = build()
                    h = train_fixed(model, tr, all_rows, gidx, gval, device, a,
                                    torch, E, lr=lr, seed=sd)
                    st = json.loads(json.dumps(model.state()))
                    b = dev_ndcg(model, tr, all_rows, glists, device, a.qbatch, torch)
                    bsel, bpop = float(h[-1]["train_loss"]), pop_tr.state()
                    per_seed[sd] = {"kfold_epochs": picked, "epochs_used": E,
                                    "train_ndcg@10_insample": round(b, 4)}
                else:
                    b, st, h, bsel, bpop = train_arm(model, tr, fit_rows, dev_rows,
                                                     gidx, gval, glists, device, a,
                                                     torch, lr=lr, seed=sd, pop=pop_tr)
                    # Name the field after what was actually selected on: bsel is
                    # the loss under --select_on loss and MINUS the nDCG otherwise,
                    # so a fixed "dev_loss" label printed the nDCG twice.
                    per_seed[sd] = {"dev_ndcg@10": round(b, 4),
                                    f"selected_on_{a.select_on}":
                                        round(bsel if a.select_on == "loss" else -bsel, 4)}
                if bsel < best_sel_seed - 1e-9:
                    best_sel_seed, best, state, hist = bsel, b, st, h
                    best_pop_seed, best_pop_te = bpop, pop_te
            if len(seeds) > 1:
                print(f"  per-seed: {per_seed}")
                if a.kfold >= 2:
                    # NO HELD-OUT DATA REMAINS, so a seed cannot be selected on
                    # merit. Keeping the lowest FINAL TRAINING loss is a tie-break,
                    # not model selection, and it is reported as such.
                    print("  kfold: every train query is in the final fit, so no "
                          "held-out selection is possible; seeds differ only by "
                          "init/order and the lowest final train loss is kept")
                else:
                    print(f"  kept the seed with the best dev "
                          f"{'loss' if a.select_on == 'loss' else 'nDCG@10'}")
            model.load(state)
            if best_pop_seed is not None:
                # Restore the WINNING seed's predictor into the Pop that test
                # scoring uses, so both halves come from one checkpoint.
                best_pop_te.load_state(best_pop_seed)
                pop_te = best_pop_te
        tag = TAG[arm]
        json.dump(state, open(f"{out}/params_semantic_{tag}_{a.dataset}.json", "w"), indent=1)
        # THE JOINT PREDICTOR IS HALF THE MODEL, SO IT HAS TO BE HALF THE CHECKPOINT.
        # Only the pre-joint two-stage fit was ever written to disk, so once the
        # process exited the trained popularity was gone and the params json on disk
        # described a scorer paired with a predictor that no longer existed. Anything
        # reloading this arm -- the graph fusion warm start, a rerun, a transfer --
        # would silently get the wrong half. Written next to the scorer under a
        # matching name so the pair cannot be separated by accident.
        if best_pop_seed is not None:
            _torch.save(best_pop_seed, f"{out}/popnet_semantic_{tag}_{a.dataset}.pt")
            print(f"[{arm}] saved joint popularity predictor -> "
                  f"popnet_semantic_{tag}_{a.dataset}.pt")
        pred, test_ndcg = None, None
        if not a.selection_only:
            # pop_te, NOT the run-level default. Without it a joint run scored
            # test with leave-one-out popularity and the trained predictor was
            # never used at inference at all.
            sc = score_rows(model, te, np.arange(len(te["queries"])),
                            device, a.qbatch, torch, pop=pop_te)
            pred = write_predictions(
                sc, te, f"{out}/predictions_semantic_{tag}_{a.dataset}_test.json")
            test_ndcg = float(np.mean(
                [ndcg_at(list(np.argsort(-sc[i])[:10]), te_glists[i], 10)
                 for i in range(len(te_glists)) if te_glists[i]]))
        results[arm] = {"best_dev_ndcg@10": best, "params": state, "history": hist,
                        "per_seed_dev_ndcg@10": per_seed,
                        "predictions": pred, "minutes": round((time.time() - t0) / 60, 2)}
        summary[arm] = {"dev_ndcg@10": round(best, 4),
                        "test_ndcg@10_quick": (None if test_ndcg is None
                                                else round(test_ndcg, 4))}
        destination = "test deliberately not scored" if pred is None else os.path.basename(pred)
        print(f"[{arm}] best dev nDCG@10 {best:.4f} -> {destination}")

    meta = {"dataset": a.dataset, "encoder": a.model, "device": device,
            "hyperparameters": {k: getattr(a, k) for k in
                                ("train_fit", "dev", "epochs", "patience", "lr",
                                 "hidden", "ds_hidden", "ds_dropout", "mlp_hidden",
                                 "set_hidden", "set_dropout",
                                 "distill_operator", "distill_epochs", "distill_lr",
                                 "qbatch", "seed", "seeds", "loss", "popularity",
                                 "select_on", "kfold", "mlp_popularity",
                                 "mlp_pop_joint", "pop_lambda",
                                 "lr_attention", "weight_decay")},
            "popularity": pop_info,
            "actual_split": {"fit": int(len(fit_rows)), "dev": int(len(dev_rows)),
                             "train_queries": int(Q), "test_queries": len(te["queries"])},
            "loss": ("operator objective (operator_scorer.py): full-corpus denominator "
                     "including the query's other golds, weighted per (query, gold) pair"
                     if a.loss == "operator" else
                     "multi-gold corrected: other golds excluded from each denominator, "
                     "averaged within query then equally across queries"),
            "arms": results, "quick_summary": summary}
    mp = f"{out}/semantic_comparison_{a.dataset}{LSUF}.json"
    json.dump(meta, open(mp, "w"), indent=1)
    print(f"\nwrote {mp}")
    print("\nquick development selection" +
          (" (test not scored):" if a.selection_only else "/test nDCG@10:"))
    for k, v in summary.items():
        test_text = "not scored" if v["test_ndcg@10_quick"] is None else f"{v['test_ndcg@10_quick']:.4f}"
        print(f"  {k:10} dev {v['dev_ndcg@10']:.4f}  test {test_text}")


if __name__ == "__main__":
    main()
