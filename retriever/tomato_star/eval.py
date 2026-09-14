"""Evaluation: nDCG@10 and Recall@k, broken down by stratum (same / cross / all)."""
from __future__ import annotations
import numpy as np
from .config import K_VALUES


def ndcg10(order: list[str], gold: set[str]) -> float:
    if not gold:
        return 0.0
    dcg  = sum(1.0 / np.log2(r + 2) for r, d in enumerate(order[:10]) if d in gold)
    idcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(gold), 10)))
    return dcg / idcg if idcg else 0.0


def _recall_at_k(order: list[str], gold: set[str], k: int) -> float:
    return len(set(order[:k]) & gold) / len(gold) if gold else 0.0


def evaluate(rankings: dict[str, list[str]], queries: list[dict],
             golds: dict[str, list[str]], ks=K_VALUES) -> dict:
    """rankings: query_id -> [corpus_key ...] best-first.
    Returns {stratum: {R@k, nDCG@10, n}} for stratum in same/cross/all.
    """
    buckets = {"same": [], "cross": [], "method": [], "all": []}
    for q in queries:
        qid = q["query_id"]
        if qid not in rankings:
            continue
        g = set(golds.get(qid, []))
        if not g:
            continue
        order = rankings[qid]
        rec = {k: _recall_at_k(order, g, k) for k in ks}
        nd  = ndcg10(order, g)
        row = (rec, nd)
        buckets["all"].append(row)                       # every query with a gold (nothing dropped)
        if q.get("stratum") in ("same", "cross", "method"):
            buckets[q["stratum"]].append(row)

    out = {}
    for name, rows in buckets.items():
        if not rows:
            out[name] = {**{f"R@{k}": float("nan") for k in ks}, "nDCG@10": float("nan"), "n": 0}
            continue
        out[name] = {**{f"R@{k}": 100 * np.mean([r[0][k] for r in rows]) for k in ks},
                     "nDCG@10": 100 * np.mean([r[1] for r in rows]),
                     "n": len(rows)}
    return out


def print_table(name: str, metrics: dict, ks=K_VALUES) -> None:
    print(f"\n{name}")
    head = "            " + "".join(f"{'R@'+str(k):>8}" for k in ks) + f"{'nDCG@10':>9}{'n':>7}"
    print(head)
    for s in ("same", "cross", "method", "all"):
        m = metrics[s]
        if m["n"] == 0:
            continue
        print(f"  {s:9s}" + "".join(f"{m[f'R@{k}']:8.1f}" for k in ks) +
              f"{m['nDCG@10']:9.1f}{m['n']:7d}")
