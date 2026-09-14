"""Add bge_top1 + ours_top1 to inputs_500.jsonl from the Colab export (join by query text).

API-free.  The Colab export (downstream_top1_colab.json) is a list of records:
    {query, bge_top1:{title,abstract,is_gold}, ours_top1:{title,abstract,is_gold}}
where `query` == research_question + " " + background_survey.  We join each
inputs_500 record to its Colab record by that text and attach the two arms.

Run:
  cd TOMATO-Star
  python -m analysis.downstream.augment_bge_ours \
      --colab outputs/caches/ours/downstream_top1_colab.json
"""
from __future__ import annotations

import argparse
import json

from analysis.downstream._common import INPUTS_PATH

norm = lambda s: " ".join((s or "").lower().split())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--colab", required=True)
    a = ap.parse_args()

    colab = json.load(open(a.colab))
    by_full, by_rq = {}, {}
    for r in colab:
        by_full[norm(r["query"])] = {"bge": r["bge_top1"], "ours": r["ours_top1"]}
        by_rq[norm(r["query"].split("?")[0])] = {"bge": r["bge_top1"], "ours": r["ours_top1"]}

    def wrap(d):
        return {"key": "(colab)", "title": d["title"],
                "abstract": d["abstract"], "is_gold": d["is_gold"]}
    empty = {"key": "(unmatched)", "title": "", "abstract": "", "is_gold": False}

    recs = [json.loads(l) for l in open(INPUTS_PATH)]
    nf = nr = miss = 0
    for rec in recs:
        hit = by_full.get(norm(rec["research_question"] + " " + rec["background_survey"]))
        if hit:
            nf += 1
        else:
            hit = by_rq.get(norm(rec["research_question"].split("?")[0]))
            if hit:
                nr += 1
        if hit:
            rec["bge_top1"] = wrap(hit["bge"]); rec["ours_top1"] = wrap(hit["ours"])
        else:
            miss += 1; rec["bge_top1"] = dict(empty); rec["ours_top1"] = dict(empty)

    with open(INPUTS_PATH, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    tot = len(recs)
    bh = sum(int(r["bge_top1"]["is_gold"]) for r in recs)
    oh = sum(int(r["ours_top1"]["is_gold"]) for r in recs)
    print(f"joined: full-match {nf}, rq-fallback {nr}, MISSED {miss} / {tot}")
    print(f"top1==gold:  BGE {bh}/{tot} ({100*bh/tot:.1f}%)   ours {oh}/{tot} ({100*oh/tot:.1f}%)")


if __name__ == "__main__":
    main()
