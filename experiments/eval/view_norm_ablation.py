"""
view_norm_ablation.py -- does per-view standardisation of the hypothetical-answer
scores change the sorted-MLP scorer (the thesis semantic scorer)?

    shared   (as shipped) centre each answer separately, divide ALL of a query's
             answers by ONE shared scale. Keeps each answer's relative spread.
    perview  centre each answer AND divide by its OWN scale, so every answer has
             spread 1. This is the "per-solution normaliser" that PMI proper would
             apply and that the shipped design deliberately dropped.

Everything else is semantic_scorer.py unchanged: same arm (mlp), same loss, same
popularity predictor (joint, as in the thesis), same fit/dev split (--seed 0),
same selection rule. Each training seed is run on its own so the comparison is
paired per seed and reported as mean +/- sd over seeds, not best-of-3.

    python3 eval/view_norm_ablation.py --dataset tomato --seeds 0,1,2
    python3 eval/view_norm_ablation.py --dataset tomato --seeds 0,1,2 --analyze_only
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np

_ROOT = os.environ.get("SCIGRAPHIR_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, f"{_ROOT}/experiments/eval")

_spec = importlib.util.spec_from_file_location(
    "semantic_scorer", f"{_ROOT}/experiments/eval/semantic_scorer.py")
ss = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ss)
import score_sir4  # noqa: E402  (per-query metrics, same code as every results table)

EPS = ss.EPS
_views_shared = ss._views


def _views_perview(H, mask, dense, pop, beta):
    """Identical to semantic_scorer._views except for the scale: one per answer."""
    import torch
    m3 = mask.unsqueeze(-1)
    S = (H * m3).sum(1)
    p = pop.per_query(S)
    Ht = H / p.unsqueeze(1).pow(beta)
    Ht = Ht * m3
    D = H.shape[-1]
    mu = Ht.sum(-1, keepdim=True) / D                          # [B, J, 1]
    c = (Ht - mu) * m3
    sig = torch.sqrt((c * c).sum(-1, keepdim=True) / D + EPS)  # [B, J, 1]  per view
    x_hyp = c / (sig + EPS)
    x_dir = (dense - dense.mean(1, keepdim=True)) / (dense.std(1, keepdim=True) + EPS)
    return x_dir, x_hyp * m3, Ht


VARIANTS = {"shared": _views_shared, "perview": _views_perview}


def run_one(variant, seed, a, out):
    ss._views = VARIANTS[variant]
    argv = ["semantic_scorer.py", "--dataset", a.dataset, "--arms", "mlp",
            "--seeds", str(seed), "--seed", str(a.split_seed),
            "--epochs", str(a.epochs), "--patience", str(a.patience),
            "--lr", str(a.lr), "--mlp_hidden", str(a.mlp_hidden),
            "--qbatch", str(a.qbatch), "--loss", a.loss,
            "--select_on", a.select_on, "--mlp_popularity", "1",
            "--mlp_pop_joint", str(a.joint), "--cache_only", "--out", out]
    print(f"\n##### {variant} seed {seed}: {' '.join(argv[1:])}", flush=True)
    old = sys.argv
    sys.argv = argv
    t0 = time.time()
    try:
        ss.main()
    finally:
        sys.argv = old
    print(f"##### {variant} seed {seed} done in {(time.time() - t0) / 60:.1f} min", flush=True)


def per_query_metrics(pred_path, queries, bge_rank):
    out = {}
    for rec in json.load(open(pred_path)):
        q = queries[rec["id"]]
        m = score_sir4.score_one(score_sir4.ranked_docs(rec), set(q["supporting_documents"]), [])
        names = ["all", q.get("stratum")]
        if rec["id"] in bge_rank:
            sim = "dissimilar" if bge_rank[rec["id"]] > 100 else "similar"
            names += [sim, f"{q.get('stratum')}+{sim}"]
        m["slices"] = names
        out[rec["id"]] = m
    return out


def paired_bootstrap(x, y, n=5000, seed=0):
    """Mean difference y-x with a percentile 95% CI over queries (paired)."""
    rng = np.random.default_rng(seed)
    d = np.asarray(y) - np.asarray(x)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def analyze(a, out_root):
    qpath = f"{ss.corpus_dir('test')}/raw/test.json"
    queries = {q["id"]: q for q in json.load(open(qpath))}
    bge_rank = {}
    if a.bge and os.path.exists(a.bge):
        for rec in json.load(open(a.bge)):
            q = queries.get(rec["id"])
            if q:
                bge_rank[rec["id"]] = score_sir4.best_gold_rank(
                    score_sir4.ranked_docs(rec), set(q["supporting_documents"]))
    seeds = [int(s) for s in a.seeds.split(",")]
    metrics = ["ndcg@5", "recall@3", "recall@5", "recall@10", "recall@25", "recall@100", "mrr"]
    slices = ["all", "same", "cross"] + (["similar", "dissimilar", "same+dissimilar",
                                          "cross+dissimilar"] if bge_rank else [])
    pq = {}      # (variant, seed) -> {qid: metrics}
    betas, epochs = {}, {}
    for v in VARIANTS:
        for s in seeds:
            d = f"{out_root}/{v}_seed{s}"
            pred = f"{d}/predictions_semantic_mlp_fixedloss_{a.dataset}_test.json"
            if not os.path.exists(pred):
                print(f"missing {pred}; skipping")
                continue
            pq[(v, s)] = per_query_metrics(pred, queries, bge_rank)
            comp = json.load(open(f"{d}/semantic_comparison_{a.dataset}_fixedloss.json"))
            arm = comp["arms"]["mlp"]
            betas[(v, s)] = arm["params"]["beta"]
            epochs[(v, s)] = len(arm["history"])
    if not pq:
        return
    lines = [f"# view normalisation ablation: sorted-MLP scorer, {a.dataset}",
             "", f"fit/dev split seed {a.split_seed}; training seeds {seeds}; "
             f"select_on {a.select_on}; loss {a.loss}; joint popularity {a.joint}",
             f"beta (learned popularity exponent): "
             + ", ".join(f"{v}/s{s}={betas[(v, s)]:.3f}" for (v, s) in sorted(betas)),
             f"epochs run: " + ", ".join(f"{v}/s{s}={epochs[(v, s)]}" for (v, s) in sorted(epochs)),
             ""]
    summary = {}
    for sl in slices:
        qids = [qid for qid, m in next(iter(pq.values())).items() if sl in m["slices"]]
        if not qids:
            continue
        lines += [f"## [{sl}]  n={len(qids)}", "",
                  "| metric | shared (mean +/- sd over seeds) | perview (mean +/- sd) | "
                  "delta perview-shared, paired over queries, mean of seeds [95% CI] |",
                  "|---|--:|--:|--:|"]
        summary[sl] = {"n": len(qids)}
        for met in metrics:
            vals = {v: [] for v in VARIANTS}
            per_seed_delta = []
            for s in seeds:
                if ("shared", s) not in pq or ("perview", s) not in pq:
                    continue
                xs = np.array([pq[("shared", s)][q][met] for q in qids])
                ys = np.array([pq[("perview", s)][q][met] for q in qids])
                vals["shared"].append(xs.mean()); vals["perview"].append(ys.mean())
                per_seed_delta.append(ys - xs)
            if not per_seed_delta:
                continue
            # paired over queries, averaging the per-query difference across seeds first
            d_q = np.mean(np.stack(per_seed_delta), 0)
            dm, lo, hi = paired_bootstrap(np.zeros_like(d_q), d_q)
            sh, pv = np.array(vals["shared"]), np.array(vals["perview"])
            lines.append(f"| {met} | {sh.mean():.4f} +/- {sh.std():.4f} | "
                         f"{pv.mean():.4f} +/- {pv.std():.4f} | "
                         f"{dm:+.4f} [{lo:+.4f}, {hi:+.4f}] |")
            summary[sl][met] = {"shared": sh.tolist(), "perview": pv.tolist(),
                                "delta": dm, "ci95": [lo, hi]}
        lines.append("")
    md = "\n".join(lines)
    open(f"{out_root}/view_norm_{a.dataset}.md", "w").write(md)
    json.dump({"betas": {f"{v}/s{s}": b for (v, s), b in betas.items()},
               "epochs": {f"{v}/s{s}": e for (v, s), e in epochs.items()},
               "summary": summary}, open(f"{out_root}/view_norm_{a.dataset}.json", "w"), indent=1)
    print(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tomato")
    ap.add_argument("--seeds", default="0,1,2", help="training seeds, each run separately")
    ap.add_argument("--split_seed", type=int, default=0, help="fit/dev split seed (pinned)")
    ap.add_argument("--variants", default="shared,perview")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--mlp_hidden", type=int, default=16)
    ap.add_argument("--qbatch", type=int, default=8)
    ap.add_argument("--loss", default="fixed", choices=["fixed", "operator"])
    ap.add_argument("--select_on", default="ndcg100", choices=["ndcg", "ndcg100", "loss"])
    ap.add_argument("--joint", type=int, default=1, help="1 = popularity predictor trained jointly (thesis)")
    ap.add_argument("--bge", default=None, help="BGE predictions for the similar/dissimilar slices")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cpu", type=int, default=1, help="1 = force CPU (9x faster than MPS for this model)")
    ap.add_argument("--analyze_only", action="store_true")
    a = ap.parse_args()
    ss.set_dataset(a.dataset)
    out_root = a.out or f"{_ROOT}/experiments/results/view_norm_{a.dataset}"
    os.makedirs(out_root, exist_ok=True)
    if a.cpu:
        import torch
        torch.backends.mps.is_available = lambda: False
    if not a.analyze_only:
        for v in [x.strip() for x in a.variants.split(",") if x.strip()]:
            assert v in VARIANTS, v
            for s in [int(x) for x in a.seeds.split(",")]:
                d = f"{out_root}/{v}_seed{s}"
                if os.path.exists(f"{d}/predictions_semantic_mlp_fixedloss_{a.dataset}_test.json"):
                    print(f"skip {v} seed {s}: predictions exist")
                    continue
                run_one(v, s, a, d)
    analyze(a, out_root)


if __name__ == "__main__":
    main()
