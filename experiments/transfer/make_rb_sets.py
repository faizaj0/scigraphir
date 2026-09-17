"""
make_rb_sets.py -- write ResearchBench's sets.json so CompleteGoldSet@k can be scored.

SIR-4 ships a validated decomposition per query: several accepted sets, each a group
of papers that together constitute one solution. ResearchBench ships no such thing.
What it ships is the list of inspirations cited by one paper, which is ONE realised
solution, so its gold family has size one:

    M_RB(q)   = {S1}
    M_SIR4(q) = {S1, ..., Sm}

That makes CompleteGoldSet@k the same object as CompleteSet@k at family size one,
which is why the metric transfers at all.

THIS IS AN INTERPRETATION AND MUST BE LABELLED AS ONE. ResearchBench never validated
its golds as a set; it recorded which papers were cited as inspirations. Treating
them as a decomposition is our reading, defensible because they are the inspirations
behind one specific paper, but it is not ResearchBench's own claim and the write-up
has to say so. Reporting it as `CompleteSet@k` alongside SIR-4's would imply a
validation that does not exist.

Usage
-----
    python3 transfer/make_rb_sets.py
    python3 transfer/make_rb_sets.py --out /tmp/rb_sets.json
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(ROOT)
RB = f"{REPO_ROOT}/retriever/data/researchbench_test"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default=f"{RB}/raw/test.json")
    ap.add_argument("--out", default=f"{RB}/sets.json")
    a = ap.parse_args()

    queries = json.load(open(a.queries))
    sets, sizes, empty = {}, Counter(), 0
    for q in queries:
        golds = list(dict.fromkeys(q.get("supporting_documents") or []))  # order-stable dedup
        if not golds:
            empty += 1
            continue
        sets[q["id"]] = {"sets": [golds]}
        sizes[len(golds)] += 1

    json.dump(sets, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")
    print(f"  {len(sets):,} queries, one set each"
          + (f"   ({empty} queries had no golds and were omitted)" if empty else ""))
    print("  set size: " + "  ".join(f"{k}:{v}" for k, v in sorted(sizes.items())))
    tot = sum(k * v for k, v in sizes.items())
    print(f"  {tot:,} gold rows, mean set size {tot / max(len(sets), 1):.2f}")
    # A single-paper "set" makes CompleteGoldSet@k identical to hits@k for that query,
    # which is not wrong but is worth knowing when reading the column.
    if sizes[1]:
        print(f"  note: {sizes[1]} queries have a one-paper set, where "
              f"CompleteGoldSet@k degenerates to hits@k")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
