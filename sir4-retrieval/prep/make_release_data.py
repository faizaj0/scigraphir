#!/usr/bin/env python3
"""
make_release_data.py -- package the SIR-4 data a reader needs to reproduce the thesis
results, as one zip that unpacks onto the repository root.

The repository holds code only. This zip carries, for each requested dataset:

  kg-construction/data/<dataset>_<split>/raw/            staged corpus and queries
  kg-construction-v16/cache/<dataset>/frames_*.jsonl     LLM frame extractions (Stage 1 and 3)
  kg-construction/construct_v2/cache/<dataset>/probes_*  hypothetical answers
  quartet/data/benchmark/<export>/                       SIR-4 export: sets.json (CompleteSet@k),
                                                         manifest.json, and for the test split
                                                         eval.json / eval_primary.json

With the caches present, `build_greasoner_dataset.py` rebuilds the graphs without any LLM
call, and every notebook finds its inputs where `cargo_paths.py` expects them. Built graphs
are not included by default (about 355 MB for SIR-4; they rebuild byte-identically from the
caches); pass --include-graphs to add them.

    python3 sir4-retrieval/prep/make_release_data.py --source ~/Desktop/CARGO
    python3 sir4-retrieval/prep/make_release_data.py --source ~/Desktop/CARGO --include-graphs

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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=os.environ.get("CARGO_ROOT") or REPO,
                    help="root that holds the data (the workspace, or this repo once data is in place)")
    ap.add_argument("--datasets", default=",".join(SIR4_EXPORTS),
                    help="comma-separated dataset names as used by --dataset elsewhere")
    ap.add_argument("--out", default=None, help="zip path (default <repo>/scigraphir-data-<name>.zip)")
    ap.add_argument("--include-graphs", action="store_true", help="also add the built *_v16sc graphs")
    ap.add_argument("--full-train-export", action="store_true",
                    help="also add the large eval.json / eval_primary.json of the SIR-4 train exports")
    a = ap.parse_args()

    source = os.path.abspath(os.path.expanduser(a.source))
    os.environ["CARGO_ROOT"] = source          # cargo_paths reads it at import time
    sys.path.insert(0, REPO)
    import cargo_paths as cp

    datasets = [d for d in a.datasets.split(",") if d]
    name = "sir4" if all(d.startswith("sir4_") for d in datasets) else "-".join(datasets)
    out = a.out or os.path.join(REPO, f"scigraphir-data-{name}.zip")

    wanted: list[str] = []
    for ds in datasets:
        cp.set_dataset(ds)
        for split in ("train", "test"):
            raw = os.path.join(cp.corpus_dir(split), "raw")
            wanted += [os.path.join(raw, "documents.json"), os.path.join(raw, f"{split}.json")]
            wanted += [cp.frames_path("doc", split), cp.frames_path("query", split), cp.probes_path(split)]
            if a.include_graphs:
                s1 = os.path.join(cp.graph_dir(split), "processed", "stage1")
                wanted += [os.path.join(s1, f) for f in ("nodes.csv", "edges.csv", "relations.csv", f"{split}.json")]
        if ds in SIR4_EXPORTS:
            train_x, test_x = SIR4_EXPORTS[ds]
            bench = os.path.join(source, "quartet", "data", "benchmark")
            wanted += [os.path.join(bench, test_x, f) for f in ("sets.json", "manifest.json", "eval.json", "eval_primary.json")]
            wanted += [os.path.join(bench, train_x, f) for f in ("sets.json", "manifest.json")]
            if a.full_train_export:
                wanted += [os.path.join(bench, train_x, f) for f in ("eval.json", "eval_primary.json")]

    missing = [p for p in wanted if not os.path.exists(p)]
    if missing:
        print("MISSING (fix before packaging, or drop the dataset):")
        for p in missing:
            print("  -", os.path.relpath(p, source))
        return 1

    total = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in wanted:
            arc = os.path.relpath(os.path.realpath(p), os.path.realpath(source))
            if arc.startswith("quartet/data.nosync/"):       # the workspace keeps quartet/data as a symlink
                arc = arc.replace("quartet/data.nosync/", "quartet/data/", 1)
            z.write(p, arc)
            b = os.path.getsize(p)
            total += b
            print(f"  {arc:80} {b / 1e6:8.1f} MB")
    print(f"\n{len(wanted)} files, {total / 1e6:.1f} MB raw -> {os.path.getsize(out) / 1e6:.1f} MB zipped")
    print(f"wrote {out}")
    print("Attach it to a GitHub release; readers unzip it at the repository root.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
