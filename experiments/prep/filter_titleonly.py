"""
filter_titleonly.py -- drop queries whose gold documents have no abstract.

WHY. Roughly 11% of cs_test documents and 20% of cs_train documents carry only a
title in the `documents.json` text field (median ~50 chars against a corpus
median of ~1,200). The length distribution is cleanly bimodal -- p10 = 100 chars,
p15 = 468 -- so any threshold in 150..400 selects the same set. Requested to cut
training time: dropping ~50% of train queries halves the steps per epoch.

WHAT IT DOES NOT DO. It does not touch `documents.json`. The corpus stays whole,
so the golds of dropped queries remain as distractors and per-query difficulty is
unchanged. Only the query list shrinks.

WHAT TO KNOW BEFORE USING IT ON `test`. Dropping test queries saves no training
time (test is inference over 1,052 queries) and it is not stratum-neutral: 43.3%
of `cross` test queries have a title-only gold against 30.4% of `same`, so the
cross slice shrinks 1.4x faster than the same slice and the surviving cross
number is measured on the easier half. The manifest records the before/after
stratum counts so this is stated rather than discovered later.

IDEMPOTENT. The untouched query list is preserved once as `{split}.json.full`
and every run filters from THAT, so re-running with a different --min-chars
re-derives from the original rather than compounding.

Usage
-----
    python3 prep/filter_titleonly.py --dataset sir4_cs --splits train
    python3 prep/filter_titleonly.py --dataset sir4_cs --splits train,test
    python3 prep/filter_titleonly.py --dataset sir4_cs --splits train --dry-run
    python3 prep/filter_titleonly.py --dataset sir4_cs --splits train --restore
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                                  # experiments/
REPO_ROOT = os.path.dirname(ROOT)                                 # $SCIGRAPHIR_ROOT
sys.path.insert(0, REPO_ROOT)
from scigraphir_paths import banner, corpus_dir, set_dataset       # noqa: E402


def mirrors(dataset: str, split: str) -> list[str]:
    """Every raw/ dir holding this split's query file.

    stage_sir4.py writes the staged corpus TWICE: once under experiments/data
    (our copy) and once under retriever/data (where the pipeline reads).
    Filtering only one leaves the two disagreeing, which is invisible until a
    downstream count looks wrong.
    """
    out = [f"{corpus_dir(split)}/raw"]
    if dataset.startswith("sir4_"):
        dom = dataset[len("sir4_"):]
        mine = f"{ROOT}/data/{dom}/tomato_{split}/raw"
        if os.path.isdir(mine):
            out.append(mine)
    return out


def source_path(raw: str, split: str) -> str:
    """Path to read from: the .full backup once it exists, else the live file."""
    full = f"{raw}/{split}.json.full"
    return full if os.path.exists(full) else f"{raw}/{split}.json"


def strata(queries: list) -> dict:
    return dict(sorted(Counter(q.get("stratum") or "?" for q in queries).items()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=os.environ.get("SCIGRAPHIR_DATASET", "tomato"))
    ap.add_argument("--splits", default="train",
                    help="comma separated; 'train' or 'train,test'")
    ap.add_argument("--min-chars", type=int, default=200,
                    help="a gold shorter than this counts as title-only (default 200)")
    ap.add_argument("--mode", default="any", choices=["any", "all"],
                    help="'any': drop if ANY gold is title-only (default). "
                         "'all': drop only if EVERY gold is title-only (far milder)")
    ap.add_argument("--dry-run", action="store_true", help="report, write nothing")
    ap.add_argument("--restore", action="store_true",
                    help="put the untouched query lists back and exit")
    a = ap.parse_args()

    if a.dataset == "tomato":
        print("refusing to run against tomato; pass --dataset sir4_cs", file=sys.stderr)
        return 2
    set_dataset(a.dataset)
    print(banner())

    splits = [s.strip() for s in a.splits.split(",") if s.strip()]

    if a.restore:
        for split in splits:
            for raw in mirrors(a.dataset, split):
                full = f"{raw}/{split}.json.full"
                if os.path.exists(full):
                    shutil.copy(full, f"{raw}/{split}.json")
                    print(f"  restored {raw}/{split}.json")
                else:
                    print(f"  no backup for {raw}/{split}.json, nothing to restore")
        return 0

    report = {"dataset": a.dataset, "min_chars": a.min_chars, "mode": a.mode, "splits": {}}

    for split in splits:
        raw0 = mirrors(a.dataset, split)[0]
        corpus = json.load(open(f"{raw0}/documents.json"))
        queries = json.load(open(source_path(raw0, split)))

        short = {d for d, t in corpus.items() if len((t or "").strip()) < a.min_chars}

        keep, drop = [], []
        for q in queries:
            gold = q.get("supporting_documents") or []
            n = sum(1 for g in gold if g in short)
            bad = (n == len(gold) and gold) if a.mode == "all" else (n > 0)
            (drop if bad else keep).append(q)

        # Golds that are no longer any surviving query's target. NOT removed --
        # they stay in documents.json as distractors -- but reported, because it
        # is the number that decides whether shrinking the corpus is worth it.
        kept_gold = {g for q in keep for g in (q.get("supporting_documents") or [])}
        orphan = len(set(corpus) - kept_gold)

        s_before, s_after = strata(queries), strata(keep)
        print(f"\n=== {a.dataset}/{split} ===")
        print(f"  corpus {len(corpus):,} docs, {len(short):,} under {a.min_chars} chars "
              f"({len(short)/max(len(corpus),1):.1%})")
        print(f"  queries {len(queries):,} -> {len(keep):,}   dropped {len(drop):,} "
              f"({len(drop)/max(len(queries),1):.1%})")
        for st in sorted(set(s_before) | set(s_after)):
            b, k = s_before.get(st, 0), s_after.get(st, 0)
            print(f"    {st:8} {b:6,} -> {k:6,}   ({(b-k)/max(b,1):.1%} dropped)")
        print(f"  documents no longer gold for any kept query: {orphan:,} "
              f"(left in the corpus as distractors)")

        report["splits"][split] = {
            "documents": len(corpus), "short_documents": len(short),
            "queries_before": len(queries), "queries_after": len(keep),
            "dropped": len(drop),
            "strata_before": s_before, "strata_after": s_after,
            "orphaned_documents": orphan,
            "dropped_ids": [q["id"] for q in drop],
        }

        if a.dry_run:
            continue
        if not keep:
            print(f"  refusing to write an empty query list for {split}", file=sys.stderr)
            return 1

        for raw in mirrors(a.dataset, split):
            full = f"{raw}/{split}.json.full"
            if not os.path.exists(full):
                shutil.copy(f"{raw}/{split}.json", full)     # once, before the first write
            json.dump(keep, open(f"{raw}/{split}.json", "w"))
            print(f"  wrote {raw}/{split}.json  ({len(keep):,} queries; "
                  f"original kept at {os.path.basename(full)})")

    if a.dry_run:
        print("\ndry run, nothing written")
        return 0

    out = f"{ROOT}/data/filter_titleonly_{a.dataset}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(report, open(out, "w"), indent=1)
    print(f"\nmanifest -> {out}")
    print("NOTE: build the graphs AFTER this. A graph built before the filter "
          "still carries the dropped queries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
