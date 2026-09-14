"""
Unified retrieval scorer — cross / same / all  nDCG@10 + Recall@{1,5,10,25}.

Works on ANY predictions file in the shared schema (so GFM-RAG, BM25, BGE are comparable):
  JSON list; each entry has:
    stratum                : "cross" | "same"
    supporting_documents   : [gold_doc_id, ...]                (the gold inspiration(s))
    predictions.document   : [[doc_id, score], ...]            (ranked, best first)

GFM-RAG's predictions_tomato_test.json already has this shape; bm25.py / bge.py emit it too.

Usage:  python score.py predictions_tomato_test.json --name "GFM-RAG (trained)"
"""

import argparse
import json
import math
from collections import defaultdict

KS = (1, 5, 10, 25)
NDCG_K = 10


def score_entry(gold: list, ranked: list) -> dict:
    gold = set(gold)
    rel = [1 if d in gold else 0 for d in ranked]
    out = {}
    for k in KS:  # recall@k = fraction of gold found in top-k (= hit@k when single gold)
        out[f"R@{k}"] = (sum(rel[:k]) / len(gold)) if gold else 0.0
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel[:NDCG_K]))
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(gold), NDCG_K))) or 1.0
    out[f"nDCG@{NDCG_K}"] = dcg / idcg
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("preds")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    data = json.load(open(args.preds))
    agg = defaultdict(lambda: defaultdict(list))
    skipped = 0
    for e in data:
        gold = e.get("supporting_documents") or []
        ranked = [d[0] for d in e.get("predictions", {}).get("document", [])]
        st = e.get("stratum", "?")
        if not gold:
            skipped += 1
            continue
        m = score_entry(gold, ranked)
        for k, v in m.items():
            agg[st][k].append(v)
            agg["all"][k].append(v)

    metrics = ["R@1", "R@5", "R@10", "R@25", f"nDCG@{NDCG_K}"]
    name = args.name or args.preds
    print(f"\n=== {name}  ({len(data)} queries, {skipped} skipped for no gold) ===")
    print(f"{'stratum':8} " + " ".join(f"{m:>9}" for m in metrics) + "   n")
    for st in ["same", "cross", "all"]:
        if st not in agg:
            continue
        row = agg[st]
        n = len(row[metrics[0]])
        print(f"{st:8} " + " ".join(f"{100*sum(row[m])/n:9.1f}" for m in metrics) + f"   {n}")
    # surface any off-stratum (not same/cross) entries that would otherwise hide in 'all'
    if "all" in agg:
        known = sum(len(agg[s][metrics[0]]) for s in ("same", "cross") if s in agg)
        if known != len(agg["all"][metrics[0]]):
            print(f"  [warn] {len(agg['all'][metrics[0]]) - known} queries had an off-stratum label (not same/cross)")


if __name__ == "__main__":
    main()
