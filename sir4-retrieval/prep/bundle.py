"""
bundle.py -- zip the code and staged data the Colab run needs, ready for Drive.

THE ZIP MIRRORS THE REPO. Every entry's path inside the zip is its path relative
to the CARGO root, so on Colab you unzip to /content/cargo, export
CARGO_ROOT=/content/cargo, and every script resolves exactly as it does locally.
No copying files into place, and no second layout to keep in sync.

GRAPHS ARE NOW INCLUDED, which reverses this file's original policy. The old
rationale was that Colab should rebuild them from the frame caches, so a stale
graph could never be paired with fresh frames. That was right when the build was
unsettled. It is wrong now: the graphs are built at a threshold chosen from
measured evidence (tau_canon 0.95), audited, and verified to rebuild
byte-identically. Rebuilding on Colab would run a different torch and
sentence-transformers against the same inputs and could silently diverge from
the artefact that was actually audited. Ship the audited one.

The frame, probe and embedding caches are included when they exist, because they
are the expensive artefacts: re-extracting 24,384 document frames because a zip
was thin is the single worst way to lose money in this pipeline.

Usage
-----
    python3 prep/bundle.py --dataset sir4_cs_smoke      # rehearsal bundle
    python3 prep/bundle.py --dataset sir4_cs            # the real one
"""
from __future__ import annotations

import argparse
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CARGO = os.path.dirname(ROOT)
sys.path.insert(0, CARGO)


# Directories never walked when packaging a CODE tree. `cache/` and `graph/` are
# outputs, and the caches we DO want are added separately and dataset-scoped;
# walking them here would sweep up every other corpus's artefacts too.
SKIP_DIRS = {"__pycache__", ".git", ".ipynb_checkpoints", "cache", "graph",
             "outputs", "data", "visuals", "audits", ".venv", "node_modules"}

# ALLOWLIST, not a denylist. A denylist was tried first and leaked twice: the
# initial version had no filter at all and tried to zip 5.3 GB, then a version
# excluding .npy/.csv/.log still pulled 1.2 GB of experiment output because
# those trees are full of multi-megabyte .json and .jsonl (a single
# tabc_smt_static/graphs/*/train.json is 47 MB). These directories mix source
# with years of run artefacts, so enumerating what is NOT code is a losing game.
# Enumerate what IS.
CODE_EXT = (".py", ".yaml", ".yml", ".md", ".sh", ".ipynb", ".toml", ".cfg", ".txt")


def _skip_file(f: str) -> bool:
    return not f.endswith(CODE_EXT)


def add_tree(z: zipfile.ZipFile, src: str, arc: str, seen: dict,
             code: bool = True) -> tuple[int, int]:
    """Add a file or tree. `code=False` packs everything (used for data/caches)."""
    n = b = 0
    if not os.path.exists(src):
        return 0, 0
    if os.path.isfile(src):
        z.write(src, arc)
        return 1, os.path.getsize(src)
    for root, dirs, files in os.walk(src):
        if code:
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        else:
            dirs[:] = [d for d in dirs if d not in {"__pycache__", ".git"}]
        for f in files:
            if code and _skip_file(f):
                continue
            p = os.path.join(root, f)
            z.write(p, os.path.join(arc, os.path.relpath(p, src)))
            n += 1
            b += os.path.getsize(p)
    seen[arc] = (n, b)
    return n, b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sir4_cs")
    ap.add_argument("--out", default=None)
    # The semantic-scorer experiments read the corpus, the answers and the code.
    # They never touch a graph. TOMATO's two graphs are 174 MB of the 229 MB a
    # full bundle would carry, so shipping them to run a scorer comparison is
    # three quarters of the upload wasted.
    # THE MIDDLE SETTING, AND THE ONE A GRAPH RUN ACTUALLY WANTS. --slim drops the graphs
    # (so it cannot train) and --no flag at all ships emb_dir(), which on TOMATO is
    # UNSCOPED and therefore resolves to the shared outputs/caches/op_emb: 1.5 GB of every
    # corpus's embeddings, almost none of it TOMATO's, to carry a cache that cell 5b
    # regenerates on the GPU in minutes. Measured on this tree: 2.1 GB with it, ~600 MB
    # without, for a bundle that trains identically.
    ap.add_argument("--no-emb", action="store_true",
                    help="ship the graphs and corpora but NOT the embedding cache. The "
                         "Colab run re-encodes in Phase 3 anyway, so this only changes "
                         "upload time. Ignored under --slim, which drops it already.")
    ap.add_argument("--slim", action="store_true",
                    help="ship only what a semantic-scorer run reads: corpus raw/, "
                         "hypothetical answers, code. No graphs, no frames, no "
                         "embedding cache, no operator components.")
    a = ap.parse_args()

    import cargo_paths as cp
    cp.set_dataset(a.dataset)
    out = a.out or f"{ROOT}/{a.dataset}_bundle.zip"

    kg, v16 = f"{CARGO}/kg-construction", f"{CARGO}/kg-construction-v16"

    def arc(src: str) -> str:
        """Path inside the zip = path relative to the CARGO root.

        This is the whole portability trick: unzip to /content/cargo, set
        CARGO_ROOT=/content/cargo, and cargo_paths resolves every corpus, cache
        and graph without a single path being special-cased for Colab.
        """
        return os.path.relpath(src, CARGO)

    CODE = [f"{CARGO}/cargo_paths.py", f"{CARGO}/cargo_ann.py", v16,
            f"{kg}/construct_v2", f"{kg}/eval",
            f"{kg}/experiments/probe_greasoner",
            # transfer/ holds make_rb_sets.py and paired_bootstrap.py, which the
            # ResearchBench notebook shells out to on Colab. Omitting it made the
            # notebook resolve every path correctly against the local repo and then
            # fail on Colab, which is the worst possible place to find out.
            f"{ROOT}/eval", f"{ROOT}/prep", f"{ROOT}/transfer", f"{ROOT}/PLAN.md"]

    # SOME CORPORA ARE EVALUATION-ONLY. ResearchBench has no train split and never
    # will: it is the held-out benchmark for the transfer experiment. Asking for its
    # train artefacts would print four MISSING lines for files that are absent by
    # design, and a warning that cries wolf is worse than no warning at all -- the
    # reader learns to skim past it and misses the one that matters.
    SPLITS = [s for s in ("train", "test")
              if os.path.exists(f"{cp.corpus_dir(s)}/raw/documents.json")]
    if not SPLITS:
        print(f"no corpus found for {a.dataset!r}; nothing to bundle")
        return 1
    if SPLITS == ["test"]:
        print(f"note: {a.dataset} is evaluation-only (no train split); "
              f"bundling the test split alone")

    # SLIM SHIPS raw/ ONLY. A corpus directory also accumulates derived graph
    # artefacts next to the raw files -- TOMATO's holds a 275 MB
    # operator_components.npz and a 58 MB processed/ tree -- none of which a
    # semantic-scorer run opens. raw/ is documents.json plus the split's queries.
    DATA = [f"{cp.corpus_dir(s)}/raw" if a.slim else cp.corpus_dir(s) for s in SPLITS]

    # Built graphs. Needed by BOTH phases, not just training:
    # precompute_operator_components reads nodes.csv to order its columns to the
    # graph's document nodes, so shipping the graph is what keeps the operator
    # components aligned with the model that consumes them.
    if not a.slim:
        for split in SPLITS:
            DATA.append(f"{cp.graph_dir(split)}/processed/stage1")
    else:
        print("[slim] graphs omitted; this bundle cannot train the graph reasoner")

    # Expensive caches, only if already built.
    for split in SPLITS:
        if not a.slim:
            # Extracted frames are graph-construction input; the scorer never
            # reads them.
            for side in ("doc", "query"):
                DATA.append(cp.frames_path(side, split))
        DATA.append(cp.probes_path(split))
    # OPTIONAL: the BGE embedding cache. Colab's cell 5b regenerates it, so its
    # absence is normal for any domain whose operator was never fit locally.
    # Reporting it as MISSING sent the reader chasing a non-problem, which is
    # worse than not reporting it -- a warning that cries wolf gets ignored when
    # something real is missing.
    OPTIONAL = {cp.emb_dir()}
    # SLIM ALSO DROPS THE EMBEDDING CACHE. For TOMATO this is not a small saving:
    # emb_dir() is UNSCOPED on the legacy dataset, so it resolves to the shared
    # outputs/caches/op_emb holding every corpus's embeddings plus the concept
    # matrices -- 1.6 GB of a 2.1 GB bundle, almost none of it TOMATO's. A slim
    # bundle is for encoding on Colab, so the local cache is not wanted anyway.
    if not a.slim and not a.no_emb:
        DATA.append(cp.emb_dir())
    elif a.no_emb:
        print(f"[no-emb] embedding cache omitted ({cp.emb_dir()}); Phase 3 re-encodes it")
    else:
        print("[slim] embedding cache omitted; the run encodes on the target machine")

    # sets.json defines CompleteSet@k, the one metric no other benchmark in the
    # lineage can compute. Without it score_sir4.py silently reports 0.
    #
    # The export directory name is NOT derivable from the domain: cs is
    # `cs_test_final` while the others are `*_test_low`. Guessing a suffix would
    # have shipped every non-CS bundle without its sets file, so the mapping is
    # imported from the one place that owns it.
    sys.path.insert(0, HERE)
    from stage_sir4 import DOMAINS
    dom = a.dataset.replace("sir4_", "").replace("_smoke", "")
    if dom in DOMAINS:
        DATA.append(f"{CARGO}/quartet/data/benchmark/{DOMAINS[dom][1]}/sets.json")
    else:
        print(f"note: {dom!r} is not a known domain; no sets.json will be bundled")

    seen: dict = {}
    tot_n = tot_b = 0
    missing = []
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for src, is_code in [(s, True) for s in CODE] + [(s, False) for s in DATA]:
            n, b = add_tree(z, src, arc(src), seen, code=is_code)
            tot_n += n
            tot_b += b
            opt = src in OPTIONAL
            if not n and not is_code and not opt:
                missing.append(arc(src))
            note = "" if n else ("   (absent; Colab regenerates it)" if opt
                                 else "   (absent, skipped)")
            print(f"{'  ' if n else '--'} {arc(src):58} {n:6,} files  {b/1e6:8.1f} MB" + note)

    print(f"\n{tot_n:,} files, {tot_b/1e6:.1f} MB raw -> {os.path.getsize(out)/1e6:.1f} MB zipped")
    print(f"wrote {out}")

    # A thin bundle is only discovered on Colab, an upload and a runtime later.
    # Name what is missing here instead.
    if missing:
        print("\nMISSING (the Colab run will fail on these):")
        for m in missing:
            print("  -", m)
    print("\nUpload to Drive at:  MyDrive/cargo-gfmrag/")
    print("On Colab:  unzip to /content/cargo  then  export CARGO_ROOT=/content/cargo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
