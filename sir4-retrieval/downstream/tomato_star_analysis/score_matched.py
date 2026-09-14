"""Matched-Score rubric: score each composed delta vs the GT delta hypothesis.

Reference-based.  Reuses the real MOOSE-Star rubric (scoring_utils): three dimensions
Motivation / Mechanism / Methodology, 0-4 each, total 0-12, recall-style (what % of GT
content is correctly covered).  Judge = gpt-4o-mini at temperature 0.

Needs OPENAI_API_KEY.  Resumable.

Run (per arm):
  cd TOMATO-Star
  python -m analysis.downstream.score_matched --arm none       --workers 20
  python -m analysis.downstream.score_matched --arm bm25       --workers 20
  python -m analysis.downstream.score_matched --arm moose_chem --workers 20
  python -m analysis.downstream.score_matched --arm oracle     --workers 20

Output: results/downstream/matched_{arm}.jsonl
  {query_id, arm, stratum, scores:{motivation,mechanism,methodology}, total, failed}
"""
from __future__ import annotations

import argparse

from analysis.downstream._common import (
    OUT_DIR,
    INPUTS_PATH,
    RERANKER_PROMPT_TEMPLATE,
    SCORING_RUBRIC,
    JsonlWriter,
    chat,
    done_ids,
    index_by,
    parse_scores,
    read_jsonl,
    run_concurrent,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["none", "random", "bm25", "bge", "qwen3", "reasonir",
                             "moose_chem", "ours", "scigraphir", "oracle"])
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--tag", default="",
                    help="namespace outputs by judge (e.g. gpt-4o); default "
                         "keeps the original matched_{arm}.jsonl filenames")
    args = ap.parse_args()

    suffix = f"__{args.tag}" if args.tag else ""
    comp_path = OUT_DIR / f"compositions_{args.arm}.jsonl"   # judge-independent
    out_path = OUT_DIR / f"matched_{args.arm}{suffix}.jsonl"

    inputs = index_by(read_jsonl(INPUTS_PATH))
    comps = read_jsonl(comp_path)
    if not comps:
        raise SystemExit(f"No compositions at {comp_path}. Run compose first.")

    already = done_ids(out_path)
    todo = [c for c in comps if c["query_id"] not in already]
    print(f"arm={args.arm}  comps={len(comps)}  done={len(already)}  todo={len(todo)}")
    if not todo:
        print("Nothing to do.")
        return

    writer = JsonlWriter(out_path)

    def work(c: dict) -> dict:
        gt = inputs[c["query_id"]]["gt_delta"]
        gen = c.get("delta_hypothesis") or ""
        if not gen.strip():
            return {"query_id": c["query_id"], "arm": args.arm,
                    "stratum": c["stratum"],
                    "scores": {"motivation": 0, "mechanism": 0, "methodology": 0},
                    "total": 0, "failed": False}
        prompt = RERANKER_PROMPT_TEMPLATE.format(
            gt_hypothesis=gt, generated_hypothesis=gen,
            scoring_rubric=SCORING_RUBRIC)
        resp = chat(prompt, model=args.model, temperature=0.0,
                    max_tokens=args.max_tokens)
        sc = parse_scores(resp)
        if sc is None:
            return {"query_id": c["query_id"], "arm": args.arm,
                    "stratum": c["stratum"], "scores": None,
                    "total": None, "failed": True}
        total = sc["motivation"] + sc["mechanism"] + sc["methodology"]
        return {"query_id": c["query_id"], "arm": args.arm,
                "stratum": c["stratum"], "scores": sc,
                "total": total, "failed": False}

    n = run_concurrent(todo, work, writer, workers=args.workers,
                       desc=f"score-{args.arm}")
    writer.close()
    print(f"\nWrote {n} scores -> {out_path}")


if __name__ == "__main__":
    main()
