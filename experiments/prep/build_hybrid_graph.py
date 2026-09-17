#!/usr/bin/env python3
"""
build_hybrid_graph.py -- build the default SciAfford graph: affordance structure,
OpenIE query entity seeds, entity-to-paper mention edges and paper-to-affordance links.
The saved `_hyb` suffix identifies the default full-method graph.

Writes retriever/data/<dataset>_<split>_<suffix>/{processed/stage1/{nodes,edges,relations}.csv,
processed/stage1/<split>.json, raw/documents.json} for train and test, in the exact layout the engine
loads. Nothing in the source graphs is touched.

What is added to the affordance component (walk-prior gate in results/qualitative/walk_prior_variants.md):
  entities   OpenIE entity nodes that are a SEED of at least one query of that split (start_nodes.entity)
             and have OpenIE degree <= --cap. Non-seed entities never receive mass, so they are left out
             (--all-entities keeps every entity under the cap; ~3x more nodes).
  mentions   entity --mentioned_in--> paper, copied from the OpenIE graph for the kept entities.
  shortcuts  paper --paper_<rel>--> affordance representation for the function / limitation / mechanism affordance representations the paper owns
             through its methods (achieves, overcomes, limited_by, works_via), its tasks (limited_by) and
             its findings (concerns, explains). New relation names, so their embeddings are their own.
  seeds      each query's start_nodes.entity becomes the kept OpenIE entity seeds (plus whatever the
             affordance representation file already listed, which the loader ignores when the node is absent).

    python3 prep/build_hybrid_graph.py --dataset tomato [--suffix hyb] [--cap 30] [--all-entities]
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import shutil
import sys

csv.field_size_limit(10 ** 9)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "retriever", "data")
SHORTCUT = {"contributes": ("achieves", "overcomes", "limited_by", "works_via"),
            "addresses": ("limited_by",),
            "reports": ("concerns", "explains")}


def read_csv(path):
    with open(path, newline="") as fh:
        r = csv.reader(fh); header = next(r); rows = list(r)
    return header, rows


def build_split(ds, split, suffix, cap, all_entities):
    fdir, odir = f"{DATA}/{ds}_{split}_v16sc", f"{DATA}/{ds}_{split}"
    out = f"{DATA}/{ds}_{split}_{suffix}"
    fs1, os1, out1 = f"{fdir}/processed/stage1", f"{odir}/processed/stage1", f"{out}/processed/stage1"
    for p in (f"{fs1}/nodes.csv", f"{fs1}/edges.csv", f"{fs1}/{split}.json", f"{os1}/nodes.csv", f"{os1}/edges.csv", f"{os1}/{split}.json"):
        assert os.path.exists(p), f"missing {p}"
    _, fnodes = read_csv(f"{fs1}/nodes.csv"); _, fedges = read_csv(f"{fs1}/edges.csv")
    _, onodes = read_csv(f"{os1}/nodes.csv"); _, oedges = read_csv(f"{os1}/edges.csv")
    ftype = {r[0]: r[1] for r in fnodes}; otype = {r[0]: r[1] for r in onodes}
    fqs = json.load(open(f"{fs1}/{split}.json")); oqs = {q["id"]: q for q in json.load(open(f"{os1}/{split}.json"))}

    # --- entities: seeds of this split (or all), degree-capped in the OpenIE graph
    odeg = collections.Counter()
    for r in oedges: odeg[r[0]] += 1; odeg[r[2]] += 1
    seed_ents = {n for q in oqs.values() for n in q["start_nodes"].get("entity", [])}
    keep = {n for n, t in otype.items() if t == "entity" and odeg[n] <= cap and (all_entities or n in seed_ents) and n not in ftype}
    mentions = [(r[0], "mentioned_in", r[2]) for r in oedges if r[1] == "is_mentioned_in" and r[0] in keep and r[2] in ftype]
    touched = {a for a, _, _ in mentions}
    keep &= touched                                   # an entity with no paper edge is a dead node

    # --- shortcuts: paper -> affordance representation the paper owns two hops away
    owns = collections.defaultdict(set)               # method/task/finding -> [(rel, affordance representation)]
    for a, rel, b, *_ in fedges:
        if rel in ("achieves", "overcomes", "limited_by", "works_via", "concerns", "explains"):
            owns[a].add((rel, b))
    shortcuts = set()
    for a, rel, b, *_ in fedges:
        if ftype.get(a) == "document" and rel in SHORTCUT:
            for r2, f in owns.get(b, ()):
                if r2 in SHORTCUT[rel]:
                    shortcuts.add((a, f"paper_{r2}", f))
    shortcuts = sorted(shortcuts)

    # --- write
    os.makedirs(out1, exist_ok=True); os.makedirs(f"{out}/raw", exist_ok=True)
    with open(f"{out1}/nodes.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["name", "type", "attributes"])
        for r in fnodes: w.writerow(r)
        for n in sorted(keep): w.writerow([n, "entity", "{}"])
    rels = collections.OrderedDict()
    with open(f"{out1}/edges.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["source", "relation", "target", "attributes"])
        for r in fedges: w.writerow(r); rels[r[1]] = 1
        for a, rel, b in mentions: w.writerow([a, rel, b, "{}"]); rels[rel] = 1
        for a, rel, b in shortcuts: w.writerow([a, rel, b, "{}"]); rels[rel] = 1
    with open(f"{out1}/relations.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["name", "attributes"])
        for rel in rels: w.writerow([rel, "{}"])
    n_seed = 0
    for q in fqs:
        ents = [n for n in oqs.get(q["id"], {}).get("start_nodes", {}).get("entity", []) if n in keep]
        old = [n for n in q["start_nodes"].get("entity", []) if n in ftype]     # SciAfford graph entity seeds that exist
        q["start_nodes"]["entity"] = sorted(set(old) | set(ents)); n_seed += len(q["start_nodes"]["entity"])
    json.dump(fqs, open(f"{out1}/{split}.json", "w"))
    _docs = f"{fdir}/raw/documents.json"
    if not os.path.exists(_docs): _docs = f"{odir}/raw/documents.json"     # MIR keeps the corpus only in the OpenIE dir
    shutil.copy(_docs, f"{out}/raw/documents.json")
    info = {"split": split, "frame_nodes": len(fnodes), "frame_edges": len(fedges), "entities_added": len(keep),
            "mention_edges": len(mentions), "shortcut_edges": len(shortcuts), "queries": len(fqs),
            "entity_seeds_per_query": round(n_seed / max(1, len(fqs)), 2), "cap": cap, "all_entities": all_entities,
            "relations": list(rels)}
    json.dump(info, open(f"{out}/BUILD.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in info.items() if k != "relations"}))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tomato")
    ap.add_argument("--suffix", default="hyb")
    ap.add_argument("--cap", type=int, default=30)
    ap.add_argument("--all-entities", action="store_true")
    ap.add_argument("--splits", default="train,test")
    a = ap.parse_args()
    for s in a.splits.split(","):
        build_split(a.dataset, s, a.suffix, a.cap, a.all_entities)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
