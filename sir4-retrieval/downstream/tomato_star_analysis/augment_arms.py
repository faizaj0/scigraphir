"""Attach retriever arms to inputs_500.jsonl from saved per-query prediction files.

API-free.  Adds `{arm}_top1 = {key, title, abstract, is_gold}` for each --arm given,
using the same first-resolvable-key rule as build_inputs (a ranked key that has no
title/abstract in the corpus is skipped to the next one).

Accepted prediction formats (auto-detected):
  A. CARGO/SciGraphIR list: [{id, supporting_documents, predictions:{document:[[key,score],...]}}]
     (ranked by score, descending)
  B. MOOSE-Chem style dict: {query_id: [key, key, ...]}  (already ranked)

Run (from TOMATO-Star):
  python -m analysis.downstream.augment_arms \
      --arm qwen3=/Users/faizajalil/Desktop/CARGO/kg-construction/eval/predictions_qwen3.json \
      --arm scigraphir=/Users/faizajalil/Desktop/CARGO/tmp/qual_examples/predictions_tomato_ccmp.json \
      --arm reasonir=outputs/caches/reasonir/predictions_reasonir_tomato.json

Re-running with the same arm overwrites that arm's field only.
"""
from __future__ import annotations

import argparse
import json

from analysis.downstream._common import INPUTS_PATH, read_jsonl
from analysis.downstream.build_inputs import _corpus_meta_and_records, _top1


def _load_rankings(path: str) -> dict[str, list[str]]:
    d = json.load(open(path))
    if isinstance(d, dict):
        return {q: [k.strip().lower() for k in ks] for q, ks in d.items()}
    out = {}
    for r in d:
        docs = sorted(r["predictions"]["document"], key=lambda x: -x[1])
        out[r["id"]] = [k.strip().lower() for k, _ in docs]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", default=[],
                    help="name=path/to/predictions.json (repeatable)")
    ap.add_argument("--random", type=int, default=None, metavar="SEED",
                    help="also write random_top1: a uniformly random corpus doc "
                         "(non-empty text, never the gold) per query")
    args = ap.parse_args()
    if not args.arm and args.random is None:
        ap.error("give at least one --arm or --random SEED")

    recs = read_jsonl(INPUTS_PATH)
    if not recs:
        raise SystemExit(f"No inputs at {INPUTS_PATH}. Run build_inputs first.")
    meta, _ = _corpus_meta_and_records()

    for spec in args.arm:
        name, path = spec.split("=", 1)
        field = f"{name}_top1"
        rank = _load_rankings(path)
        missing = hit = unres = 0
        for r in recs:
            ks = rank.get(r["query_id"])
            if ks is None:
                missing += 1
                r[field] = {"key": "(missing)", "title": "", "abstract": "",
                            "is_gold": False}
                continue
            top = _top1(ks, meta, r["gold_key"])
            if top is None:
                unres += 1
                top = {"key": "(unresolved)", "title": "", "abstract": "",
                       "is_gold": False}
            r[field] = top
            hit += int(top["is_gold"])
        n = len(recs)
        print(f"{name:<12} top1==gold {hit}/{n} ({100*hit/n:.1f}%)   "
              f"missing queries {missing}   unresolved {unres}")
        if missing:
            print(f"   WARNING: {missing} queries absent from {path}")

    if args.random is not None:
        import random
        rng = random.Random(args.random)
        keys = sorted(k for k, (t, a) in meta.items() if (t or a))
        for r in recs:
            k = rng.choice(keys)
            while k == r["gold_key"]:
                k = rng.choice(keys)
            t, a = meta[k]
            r["random_top1"] = {"key": k, "title": t, "abstract": a, "is_gold": False}
        print(f"{'random':<12} seed {args.random}: one random non-gold doc per query "
              f"from {len(keys)} candidates")

    with INPUTS_PATH.open("w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Updated {INPUTS_PATH}")


if __name__ == "__main__":
    main()
