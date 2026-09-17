#!/usr/bin/env python3
"""
make_release_data.py -- package the SIR-4 data a reader needs to reproduce the thesis
results, as one zip that unpacks onto the repository root.

The benchmark is included under sir-4/dataset/; prepare_data.py unpacks it.
This optional reproduction bundle also carries cached method inputs. For each dataset:

  retriever/data/<dataset>_<split>/raw/                staged corpus and queries
  sciafford/cache/<dataset>/frames_*.jsonl             LLM affordance representation extractions (Stage 1 and 3)
  retriever/probes/cache/<dataset>/probes_*.jsonl      hypothetical answers
  sir-4/data/benchmark/<export>/                   SIR-4 export: sets.json (CompleteSet@k),
                                                       manifest.json, and for the test split
                                                       eval.json / eval_primary.json

With the caches present, `sciafford/build_greasoner_dataset.py` rebuilds the graphs without
any LLM call, and every notebook finds its inputs where `scigraphir_paths.py` expects them.
Built graphs are not included by default (about 355 MB for SIR-4; they rebuild
byte-identically from the caches); pass --include-graphs to add them.

`--source` is the directory that holds the data. It is this repository once the data is in
place, or the original development workspace, whose older directory names are mapped onto
the repository layout automatically.

    python3 experiments/prep/make_release_data.py --source /path/to/workspace
    python3 experiments/prep/make_release_data.py --source /path/to/workspace --include-graphs

Unpack at the repository root:  unzip scigraphir-data-sir4.zip -d /path/to/scigraphir
"""
from __future__ import annotations

import argparse
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

SIR4_EXPORTS = {  # dataset -> (train export, test export); mirrors stage_sir4.DOMAINS
    "sir4_cs": ("cs_train_final", "cs_test_final"),
    "sir4_biology": ("biology_train_low", "biology_test_low"),
    "sir4_physics": ("physics_train_low", "physics_test_low"),
    "sir4_matsci": ("matsci_train_low", "matsci_test_low"),
}

# Repository layout -> layout of the original development workspace (longest prefix first).
LEGACY = [
    ('retriever/probes/', "kg-construction/construct_v2/"),
    ("sciafford/", "kg-construction-v16/"),
    ("retriever/", "kg-construction/"),
    ("sir-4/data/", "quartet/data.nosync/"),
    ("sir-4/", "quartet/"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=REPO, help="directory that holds the data (default: this repository)")
    ap.add_argument("--datasets", default=",".join(SIR4_EXPORTS),
                    help="comma-separated dataset names as used by --dataset elsewhere")
    ap.add_argument("--out", default=None, help="zip path (default <repo>/scigraphir-data-<name>.zip)")
    ap.add_argument("--include-graphs", action="store_true", help="also add the built *_SciAfford graphs")
    ap.add_argument("--full-train-export", action="store_true",
                    help="also add the large eval.json / eval_primary.json of the SIR-4 train exports")
    a = ap.parse_args()

    source = os.path.abspath(os.path.expanduser(a.source))
    legacy = os.path.isdir(os.path.join(source, "kg-construction"))
    os.environ["SCIGRAPHIR_ROOT"] = REPO          # resolve repository-layout paths
    sys.path.insert(0, REPO)
    import scigraphir_paths as sp

    def rel(path: str) -> str:
        return os.path.relpath(path, REPO)

    def at_source(repo_rel: str) -> str:
        if legacy:
            for new, old in LEGACY:
                if repo_rel.startswith(new):
                    repo_rel = old + repo_rel[len(new):]
                    break
        return os.path.join(source, repo_rel)

    datasets = [d for d in a.datasets.split(",") if d]
    name = "sir4" if all(d.startswith("sir4_") for d in datasets) else "-".join(datasets)
    out = a.out or os.path.join(REPO, f"scigraphir-data-{name}.zip")

    wanted: list[str] = []          # repository-relative paths
    for ds in datasets:
        sp.set_dataset(ds)
        for split in ("train", "test"):
            raw = os.path.join(sp.corpus_dir(split), "raw")
            wanted += [rel(os.path.join(raw, "documents.json")), rel(os.path.join(raw, f"{split}.json"))]
            wanted += [rel(sp.affordances_path("doc", split)), rel(sp.affordances_path("query", split)), rel(sp.answers_path(split))]
            if a.include_graphs:
                s1 = os.path.join(sp.graph_dir(split), "processed", "stage1")
                wanted += [rel(os.path.join(s1, f)) for f in ("nodes.csv", "edges.csv", "relations.csv", f"{split}.json")]
        if ds in SIR4_EXPORTS:
            train_x, test_x = SIR4_EXPORTS[ds]
            bench = "sir-4/data/benchmark"
            wanted += [f"{bench}/{test_x}/{f}" for f in ("sets.json", "manifest.json", "eval.json", "eval_primary.json")]
            wanted += [f"{bench}/{train_x}/{f}" for f in ("sets.json", "manifest.json")]
            if a.full_train_export:
                wanted += [f"{bench}/{train_x}/{f}" for f in ("eval.json", "eval_primary.json")]

    missing = [r for r in wanted if not os.path.exists(at_source(r))]
    if missing:
        print(f"MISSING under {source} (fix before packaging, or drop the dataset):")
        for r in missing:
            print("  -", at_source(r))
        return 1

    total = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for r in wanted:
            src = at_source(r)
            z.write(src, r)
            b = os.path.getsize(src)
            total += b
            print(f"  {r:80} {b / 1e6:8.1f} MB")
    print(f"\n{len(wanted)} files, {total / 1e6:.1f} MB raw -> {os.path.getsize(out) / 1e6:.1f} MB zipped")
    print(f"wrote {out}")
    print("Attach it to a GitHub release; readers unzip it at the repository root.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
