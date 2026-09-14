"""
sir4_to_lattice.py -- materialise one SIR-4 field as a LATTICE dataset + patch the relevance prompt.

LATTICE (llm-guided-hierarchical-search/src/run.py) reads data/<dataset>/<subset>/
documents.jsonl ({"id","content"}) and examples.jsonl ({"query","gold_ids","excluded_ids"})
and evaluates the FIRST --num_eval_samples rows of examples.jsonl. So the examples file IS
the subset, written in manifest order; strata.json is row-aligned like TOMATO's
strata_500.json. Extra keys (query_id, stratum) ride along for scoring.

Dataset = "SIR-4", subset = field. Doc ids become d_NNNN (LATTICE parses "[i] id" and
"] " out of ids, and DOIs carry "/" and "."); id_map.json maps back to DOIs.

RELEVANCE PROMPT. src/prompts.py picks the relevance definition by --subset name and
silently falls back to the StackExchange wording for unknown names. --patch-prompts adds
the four SIR-4 field names as aliases of the TOMATO inspiration definition (idempotent).

Usage
-----
    python3 llm_baselines/sir4_to_lattice.py --field cs --patch-prompts
    python3 llm_baselines/sir4_to_lattice.py --field cs --subset none        # every query
    # -> <LATTICE_DIR>/data/SIR-4/cs/{documents.jsonl, examples.jsonl, strata.json, id_map.json}
"""
from __future__ import annotations

import argparse
import json
import os

from sir4_llm_data import FIELDS, LATTICE_DIR, default_subset, load_corpus, load_queries, load_subset

PATCH_MARK = "# SIR-4 fields (added by sir4_to_lattice.py)"
# inside get_relevance_definition(), which is indented 4 spaces; goes right after the last alias
ANCHOR = "    RELEVANCE_DEFINITIONS['inspiration'] = RELEVANCE_DEFINITIONS['full']\n"
PATCH = (f"    {PATCH_MARK}\n    for _f in {FIELDS + ['mir']!r}:\n"
         "        RELEVANCE_DEFINITIONS[_f] = RELEVANCE_DEFINITIONS['full']\n")


def patch_prompts(lattice_dir: str) -> None:
    p = f"{lattice_dir}/src/prompts.py"
    src = open(p).read()
    if PATCH_MARK in src:
        print("prompts.py already patched")
        return
    assert ANCHOR in src, "prompts.py changed; add the SIR-4 field aliases by hand inside get_relevance_definition()"
    open(p, "w").write(src.replace(ANCHOR, ANCHOR + PATCH, 1))
    print(f"patched {p}: SIR-4 fields now use the inspiration relevance definition")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", required=True)
    ap.add_argument("--subset", default="default", help="manifest path, 'default', or 'none'")
    ap.add_argument("--lattice-root", default=LATTICE_DIR)
    ap.add_argument("--patch-prompts", action="store_true")
    a = ap.parse_args()

    if a.patch_prompts:
        patch_prompts(a.lattice_root)

    corpus = load_corpus(a.field)
    subset = None if a.subset == "none" else (default_subset(a.field) if a.subset == "default" else a.subset)
    queries = load_subset(subset, a.field, load_queries(a.field))

    dois = list(corpus)
    width = max(4, len(str(len(dois))))
    doi2id = {d: f"d_{i:0{width}d}" for i, d in enumerate(dois)}
    out = f"{a.lattice_root}/data/SIR-4/{a.field}"
    os.makedirs(out, exist_ok=True)

    with open(f"{out}/documents.jsonl", "w") as f:
        for d in dois:
            title, abstract = corpus[d]
            content = f"Title: {title}\n\nAbstract: {abstract}" if abstract else f"Title: {title}"
            f.write(json.dumps({"id": doi2id[d], "content": content}, ensure_ascii=False) + "\n")
    strata = []
    with open(f"{out}/examples.jsonl", "w") as f:
        for q in queries:
            golds = [doi2id[g] for g in q["golds"]]
            f.write(json.dumps({"query": q["question"], "gold_ids": golds, "excluded_ids": [],
                                "query_id": q["query_id"], "stratum": q["stratum"]}, ensure_ascii=False) + "\n")
            strata.append(q["stratum"])
    json.dump({"order": "examples.jsonl row order", "strata": strata, "query_ids": [q["query_id"] for q in queries]},
              open(f"{out}/strata.json", "w"))
    json.dump({v: k for k, v in doi2id.items()}, open(f"{out}/id_map.json", "w"), indent=1)
    print(f"SIR-4/{a.field}: {len(dois)} docs, {len(queries)} examples ({strata.count('same')} same / "
          f"{strata.count('cross')} cross) -> {out}")
    print(f"  run.py needs --dataset SIR-4 --subset {a.field} --num_eval_samples {len(queries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
