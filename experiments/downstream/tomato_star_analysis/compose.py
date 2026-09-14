"""Compose a delta hypothesis from the TOP-1 retrieved inspiration, per arm.

Fixed composer = gpt-4o-mini, using the real MOOSE-Star composition prompt.
The ONLY thing that differs between arms is the retrieved inspiration fed in, so any
downstream quality difference is attributable to retrieval.

Arms (floor -> ceiling):
  none        no inspiration content (empty title/abstract, same prompt) -- closed-book floor
  bm25        BM25 top-1
  moose_chem  MOOSE-Chem top-1
  oracle      the query's GOLD inspiration (best possible retrieval) -- ceiling
              (requires the gold_top1 field; re-run build_inputs to add it)

Needs OPENAI_API_KEY in the environment.  Resumable (skips query_ids already written).

Run (one arm at a time):
  cd TOMATO-Star
  python -m analysis.downstream.compose --arm none        --workers 20
  python -m analysis.downstream.compose --arm bm25        --workers 20
  python -m analysis.downstream.compose --arm moose_chem  --workers 20
  python -m analysis.downstream.compose --arm oracle      --workers 20

Output: results/downstream/compositions_{arm}.jsonl
  {query_id, arm, stratum, insp_key, is_gold, delta_hypothesis, raw_response}
"""
from __future__ import annotations

import argparse

from analysis.downstream._common import (
    OUT_DIR,
    INPUTS_PATH,
    JsonlWriter,
    build_compose_prompt,
    chat,
    done_ids,
    extract_delta,
    read_jsonl,
    run_concurrent,
)

# Retrieved-inspiration arms map to a field in the input record; "none" feeds
# empty inspiration content (closed-book floor) through the identical prompt.
ARM_FIELD = {"random": "random_top1",
             "bm25": "bm25_top1", "bge": "bge_top1", "qwen3": "qwen3_top1",
             "reasonir": "reasonir_top1", "moose_chem": "moose_chem_top1",
             "lattice": "lattice_top1",
             "ours": "ours_top1", "scigraphir": "scigraphir_top1",
             "oracle": "gold_top1"}
ARMS = list(ARM_FIELD) + ["none"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--temperature", type=float, default=0.6,
                    help="Generation temperature (MOOSE-Star used 0.6).")
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--no-reuse", action="store_true",
                    help="always call the API; default reuses an existing "
                         "composition of the same (query, inspiration title) "
                         "from any other arm, so identical retrievals share "
                         "one composition and the arm comparison is paired")
    args = ap.parse_args()

    out_path = OUT_DIR / f"compositions_{args.arm}.jsonl"

    rows = read_jsonl(INPUTS_PATH)
    if not rows:
        raise SystemExit(f"No inputs at {INPUTS_PATH}. Run build_inputs first.")

    field = ARM_FIELD.get(args.arm)
    if field is not None and field not in rows[0]:
        raise SystemExit(
            f"Input records lack '{field}' (needed for arm '{args.arm}'). "
            f"Re-run:  python -m analysis.downstream.build_inputs")

    already = done_ids(out_path)
    todo = [r for r in rows if r["query_id"] not in already]
    print(f"arm={args.arm}  total={len(rows)}  done={len(already)}  todo={len(todo)}")
    if not todo:
        print("Nothing to do.")
        return

    writer = JsonlWriter(out_path)

    # ---- reuse identical (query, inspiration) compositions from other arms ----
    if field is not None and not args.no_reuse:
        norm = lambda t: " ".join((t or "").lower().split())
        pool = {}
        by_qid = {r["query_id"]: r for r in rows}
        for other in ARMS:
            if other in (args.arm, "none"):
                continue
            ofield = ARM_FIELD[other]
            for c in read_jsonl(OUT_DIR / f"compositions_{other}.jsonl"):
                t = c.get("insp_title")
                if t is None:            # older compositions: title lives in inputs
                    t = (by_qid.get(c["query_id"], {}).get(ofield) or {}).get("title")
                t = norm(t)
                if not t:
                    continue
                pool.setdefault((c["query_id"], t), (other, c))
        reused, still = 0, []
        for r in todo:
            insp = r[field]
            hit = pool.get((r["query_id"], norm(insp["title"]))) if insp["title"] else None
            if hit:
                other, c = hit
                writer.write({
                    "query_id": r["query_id"], "arm": args.arm,
                    "stratum": r["stratum"], "field": r.get("field"),
                    "insp_key": insp["key"],
                    "insp_title": insp["title"], "is_gold": insp["is_gold"],
                    "delta_hypothesis": c["delta_hypothesis"],
                    "raw_response": c["raw_response"], "reused_from": other})
                reused += 1
            else:
                still.append(r)
        print(f"  reused {reused} compositions from other arms; {len(still)} to generate")
        todo = still

    def work(r: dict) -> dict:
        if args.arm == "none":
            insp_key, is_gold, title, abstract = "__none__", False, "", ""
        else:
            insp = r[field]
            insp_key, is_gold = insp["key"], insp["is_gold"]
            title, abstract = insp["title"], insp["abstract"]
        prompt = build_compose_prompt(
            r["research_question"], r["background_survey"],
            r["prev_hypothesis"], title, abstract,
        )
        resp = chat(prompt, model=args.model,
                    temperature=args.temperature, max_tokens=args.max_tokens)
        return {
            "query_id": r["query_id"],
            "arm": args.arm,
            "stratum": r["stratum"],
            "field": r.get("field"),
            "insp_key": insp_key,
            "insp_title": title,
            "is_gold": is_gold,
            "delta_hypothesis": extract_delta(resp),
            "raw_response": resp,
        }

    n = run_concurrent(todo, work, writer, workers=args.workers, desc=args.arm)
    writer.close()
    print(f"\nWrote {n} new compositions -> {out_path}")


if __name__ == "__main__":
    main()
