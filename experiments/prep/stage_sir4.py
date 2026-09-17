"""
Stage SIR-4 benchmark exports for the retrieval pipeline.

Reads sir-4/data/benchmark/<export>/raw/documents.json and eval.json, validates
the corpus/query references, and writes the retrieval schema under
experiments/data/<domain>/ and retriever/data/sir4_<domain>_<split>/.
Activation points experiments/data/active at the selected domain for bundling.
Dataset-specific paths keep domains and their caches separate. MANIFEST.json
records the source exports and staging counts.

From the repository root:
    python experiments/prep/stage_sir4.py --domain cs
    python experiments/prep/stage_sir4.py --domain cs --verify-only
    python experiments/prep/stage_sir4.py --domain biology --activate
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                                  # experiments/
REPO_ROOT = os.path.dirname(ROOT)                                 # $SCIGRAPHIR_ROOT
BENCH = f"{REPO_ROOT}/sir-4/data/benchmark"
# Where the pipeline scripts resolve corpora (scigraphir_paths.corpus_dir).
KG_DATA = f"{REPO_ROOT}/retriever/data"

# domain -> (train export dir, test export dir)
DOMAINS = {
    "cs":      ("cs_train_final",      "cs_test_final"),
    "biology": ("biology_train_low",   "biology_test_low"),
    "physics": ("physics_train_low",   "physics_test_low"),
    "matsci":  ("matsci_train_low",    "matsci_test_low"),
}


def load_export(export_dir: str) -> tuple[dict, list]:
    """Return (corpus, queries) from a SIR-4 benchmark export directory."""
    corpus = json.load(open(f"{export_dir}/raw/documents.json"))
    rows = json.load(open(f"{export_dir}/eval.json"))
    return corpus, rows


def to_query_rows(rows: list) -> list:
    """SIR-4 eval rows -> the loader's query schema.

    The export already carries id / question / answer / supporting_documents /
    stratum, which is exactly the schema. `quartet` (the whole decomposition
    record, ~8KB per row) is dropped: it is not read downstream and it would
    multiply the file size by ~40x for nothing.
    """
    out = []
    for r in rows:
        out.append({
            "id": r["id"],
            "question": r["question"],
            "answer": r.get("answer", ""),
            "supporting_documents": r["supporting_documents"],
            "stratum": r.get("stratum"),
        })
    return out


def verify(corpus: dict, queries: list, tag: str) -> list[str]:
    """Return a list of problems. Empty list = clean."""
    problems = []

    ids = [q["id"] for q in queries]
    if len(ids) != len(set(ids)):
        problems.append(f"{tag}: {len(ids) - len(set(ids))} duplicate query ids")

    # Every gold must be IN the corpus, or the query is unanswerable and will
    # silently score 0 for every system -- a floor that looks like a hard slice.
    missing_gold = [q["id"] for q in queries
                    if not any(g in corpus for g in q["supporting_documents"])]
    if missing_gold:
        problems.append(f"{tag}: {len(missing_gold)} queries whose gold is not in the corpus "
                        f"(e.g. {missing_gold[:3]})")

    partial = sum(1 for q in queries
                  if any(g not in corpus for g in q["supporting_documents"])
                  and any(g in corpus for g in q["supporting_documents"]))
    if partial:
        problems.append(f"{tag}: {partial} queries with SOME gold missing from the corpus")

    empty_q = [q["id"] for q in queries if not (q["question"] or "").strip()]
    if empty_q:
        problems.append(f"{tag}: {len(empty_q)} empty questions")

    empty_d = [d for d, t in corpus.items() if not (t or "").strip()]
    if empty_d:
        problems.append(f"{tag}: {len(empty_d)} empty documents")

    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="cs", choices=sorted(DOMAINS))
    ap.add_argument("--out", default=f"{ROOT}/data", help="staging root")
    ap.add_argument("--verify-only", action="store_true",
                    help="check the export and report; write nothing")
    ap.add_argument("--activate", action="store_true",
                    help="also point data/active at this domain (default: on for a fresh stage)")
    a = ap.parse_args()

    train_dir, test_dir = DOMAINS[a.domain]
    dom_root = f"{a.out}/{a.domain}"

    problems, stats = [], {}
    staged = {}
    for split, export in (("train", train_dir), ("test", test_dir)):
        src = f"{BENCH}/{export}"
        if not os.path.isdir(src):
            print(f"MISSING export: {src}", file=sys.stderr)
            return 2
        corpus, rows = load_export(src)
        queries = to_query_rows(rows)
        problems += verify(corpus, queries, f"{a.domain}/{split}")
        golds = {g for q in queries for g in q["supporting_documents"]}
        stats[split] = {
            "export": export,
            "documents": len(corpus),
            "queries": len(queries),
            "distinct_golds": len(golds),
            "gold_coverage": round(len(golds & set(corpus)) / max(len(golds), 1), 4),
            "strata": dict(sorted(
                {s: sum(1 for q in queries if q.get("stratum") == s)
                 for s in {q.get("stratum") for q in queries}}.items(),
                key=lambda kv: -kv[1])),
        }
        staged[split] = (corpus, queries)

    print(f"\n=== SIR-4 -> staging  domain={a.domain} ===")
    for split, s in stats.items():
        print(f"  {split:5} {s['documents']:7,} docs  {s['queries']:6,} queries  "
              f"{s['distinct_golds']:6,} distinct golds  gold-in-corpus {s['gold_coverage']:.1%}  "
              f"strata {s['strata']}")

    # train/test corpora overlap: not a fault, but it decides whether "unseen
    # document" claims are defensible, so it is reported every run.
    ctr, cte = set(staged["train"][0]), set(staged["test"][0])
    print(f"  corpus overlap: {len(ctr & cte):,} docs shared "
          f"({100 * len(ctr & cte) / len(cte):.1f}% of test)")
    qtr = {q['id'] for q in staged['train'][1]}
    qte = {q['id'] for q in staged['test'][1]}
    print(f"  query id overlap: {len(qtr & qte)}")

    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print("  -", p)
    else:
        print("\nno problems found")

    if a.verify_only:
        return 1 if problems else 0
    if problems:
        print("\nrefusing to stage with problems above; fix or rerun with --verify-only to inspect",
              file=sys.stderr)
        return 1

    # Written TWICE, on purpose:
    #   1. experiments/data/<domain>/  -- our own copy, survives anything done
    #      to retriever/data, and is what prep/make_smoke.py carves from.
    #   2. retriever/data/sir4_<domain>_<split>/  -- where scigraphir_paths
    #      resolves `corpus_dir()`, i.e. where the pipeline scripts actually
    #      look. Staging only to (1) leaves every script reporting "no such
    #      file" at the top of a paid run.
    dataset = f"sir4_{a.domain}"
    for split, (corpus, queries) in staged.items():
        for raw in (f"{dom_root}/tomato_{split}/raw",
                    f"{KG_DATA}/{dataset}_{split}/raw"):
            os.makedirs(raw, exist_ok=True)
            json.dump(corpus, open(f"{raw}/documents.json", "w"))
            json.dump(queries, open(f"{raw}/{split}.json", "w"))
        print(f"  {split:5} -> {dom_root}/tomato_{split}/raw")
        print(f"        -> {KG_DATA}/{dataset}_{split}/raw   (what the scripts read)")

    json.dump({"domain": a.domain, "source": BENCH, "splits": stats},
              open(f"{dom_root}/MANIFEST.json", "w"), indent=1)

    # `active` is what the bundle/Colab step reads. A symlink, so switching
    # domains is atomic and there is never a half-copied tree.
    link = f"{a.out}/active"
    if a.activate or not os.path.exists(link):
        if os.path.islink(link):
            os.unlink(link)
        elif os.path.exists(link):
            shutil.rmtree(link)
        os.symlink(dom_root, link)
        print(f"  active -> {a.domain}")
    else:
        cur = os.path.realpath(link)
        if cur != os.path.realpath(dom_root):
            print(f"  NOTE: active still points at {os.path.basename(cur)}; "
                  f"pass --activate to switch to {a.domain}")

    print(f"\nstaged to {dom_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
