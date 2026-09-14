"""
build_moose_star_tree_sir4.py -- SPECTER2 k-means tree for MOOSE-Star on one SIR-4 field.

MOOSE-Star's Preprocessing/hierarchical_search/build_hierarchical_tree.py reads a directory
of {"inspiration": [{found_title, found_abstract}, ...]} files, dedups by lowercased title,
DROPS any paper without an abstract, embeds with SPECTER2 and builds a branching-15 tree.
No LLM calls. This script writes that input for a field (title-only docs get the
placeholder abstract "No abstract available." so the tree holds the whole corpus, as the
other methods see it) and calls the builder into its own directory, so the TOMATO tree's
embedding cache is never touched.

Needs torch + transformers + adapters (SPECTER2) in the environment; ~4k abstracts embed in
a few minutes on CPU. If that stack is not installed locally, run the same two steps in
Colab (moose_star_msir.ipynb has the pip cell) and copy outputs/moose_star/<field>/tree back.

Usage
-----
    python3 llm_baselines/build_moose_star_tree_sir4.py --field matsci
    python3 llm_baselines/build_moose_star_tree_sir4.py --field cs --device mps
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from sir4_llm_data import MOOSE_STAR_DIR, OUT_ROOT, load_corpus

PLACEHOLDER = "No abstract available."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", required=True)
    ap.add_argument("--branching-factor", type=int, default=15)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--input-only", action="store_true", help="write the input dir and verify it loads; no embedding")
    a = ap.parse_args()

    corpus = load_corpus(a.field)
    root = f"{OUT_ROOT}/moose_star/{a.field}"
    sft = f"{root}/sft_qa_input"
    tree = f"{root}/tree"
    os.makedirs(sft, exist_ok=True)
    insp = [{"found_title": t, "found_abstract": ab or PLACEHOLDER, "found_doi": doi}
            for doi, (t, ab) in corpus.items()]
    # one file; the builder only needs the "inspiration" list (filename year is unused here)
    json.dump({"inspiration": insp}, open(f"{sft}/2025_sir4_{a.field}.json", "w"), ensure_ascii=False)
    json.dump({t.lower(): doi for doi, (t, _) in corpus.items()}, open(f"{root}/title_to_doi.json", "w"))
    print(f"{a.field}: wrote {len(insp)} papers ({sum(1 for _, ab in corpus.values() if not ab)} with placeholder abstract)")
    if a.input_only:
        sys.path.insert(0, MOOSE_STAR_DIR)
        from Preprocessing.hierarchical_search.build_hierarchical_tree import load_inspirations_from_sft_qa_dir
        papers = load_inspirations_from_sft_qa_dir(sft)
        assert len(papers) == len(corpus), f"builder loaded {len(papers)} of {len(corpus)} docs"
        print(f"  builder loads all {len(papers)} docs; tree build skipped (--input-only)")
        return 0

    cmd = [sys.executable, f"{MOOSE_STAR_DIR}/Preprocessing/hierarchical_search/build_hierarchical_tree.py",
           "--sft_qa_dir", sft, "--output_dir", tree, "--branching_factor", str(a.branching_factor),
           "--batch_size", str(a.batch_size), "--seed", "42"]
    if a.device:
        cmd += ["--device", a.device]
    print(" ".join(cmd))
    return subprocess.call(cmd, cwd=MOOSE_STAR_DIR)


if __name__ == "__main__":
    raise SystemExit(main())
