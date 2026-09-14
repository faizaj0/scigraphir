"""Reciprocal-rank fusion of two or more prediction files (score.py schema), aligned by query id."""
import argparse, json
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("preds", nargs="+")
ap.add_argument("--out", required=True)
ap.add_argument("--k", type=int, default=60)
ap.add_argument("--topk", type=int, default=100)
a = ap.parse_args()

lists = [json.load(open(p)) for p in a.preds]
byid = [{e["id"]: e for e in L} for L in lists]
common = set(byid[0])
for b in byid[1:]:
    common &= set(b)
print(f"{len(common)} common query ids across {len(a.preds)} files "
      f"(sizes: {[len(b) for b in byid]})")

out = []
for qid in common:
    base = byid[0][qid]
    fused = defaultdict(float)
    for b in byid:
        ranked = [d[0] for d in b[qid]["predictions"]["document"]]
        for rank, doc in enumerate(ranked):
            fused[doc] += 1.0 / (a.k + rank + 1)
    top = sorted(fused.items(), key=lambda x: -x[1])[:a.topk]
    out.append({"id": qid, "stratum": base.get("stratum"),
                "supporting_documents": base.get("supporting_documents", []),
                "predictions": {"document": [[d, s] for d, s in top]}})
json.dump(out, open(a.out, "w"))
print("wrote", a.out, len(out))
