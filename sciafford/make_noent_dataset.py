"""
make_noent_dataset.py -- produce the `--no_entity_seeds` variant of a SciAfford dataset
WITHOUT rebuilding the graph.

WHY THIS IS NOT A REBUILD. In build_greasoner_dataset.py the node map (`new_type`)
and the edge set (`G`) are finalised at lines 214-236. Every read of
`a.entity_seeds` happens at line 249 or later, entirely inside the query-seeding
block. So nodes.csv, edges.csv and relations.csv are byte-identical with or
without the flag, and only `{split}.json`'s `start_nodes` differs. Re-running the
builder would spend an hour of embedding to reproduce files it cannot change.

That also means the node ORDER is unchanged, so operator_components.npz and
operator_scores.npz stay aligned and are reused as-is. Realigning them is the
expensive part of a real rebuild (see the 2026-07-16 soft-only migration); this
change does not trigger it.

WHY HARDLINKS. The two .npz files are 605 MB of the 819 MB total and are
identical between the two datasets. `cp -al` links rather than copies them, so
the variant costs ~60 MB of real disk instead of 819 MB. The split JSON is
UNLINKED before being rewritten: opening a hardlink with "w" truncates the
shared inode and would corrupt the source dataset.

Usage
-----
    python3 make_noent_dataset.py --dataset tomato
    python3 make_noent_dataset.py --dataset tomato --check   # verify only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = os.environ.get("SCIGRAPHIR_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "retriever", "data")
GRAPH_FILES = ("nodes.csv", "edges.csv", "relations.csv")


def sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def strip_split(src_dir: str, dst_dir: str, split: str) -> dict:
    """Rewrite {split}.json without the `entity` seed channel."""
    src = f"{src_dir}/processed/stage1/{split}.json"
    dst = f"{dst_dir}/processed/stage1/{split}.json"
    rows = json.load(open(src))

    before = after = 0
    zero = []
    for r in rows:
        sn = r["start_nodes"]
        before += sum(len(v) for v in sn.values())
        sn.pop("entity", None)
        n = sum(len(v) for v in sn.values())
        after += n
        if n == 0:
            zero.append(r["id"])

    # UNLINK FIRST. dst is currently a hardlink to src; opening it "w" would
    # truncate the source dataset's file, not create a new one.
    if os.path.exists(dst):
        os.unlink(dst)
    json.dump(rows, open(dst, "w"))

    return {"queries": len(rows),
            "seeds_before": before / len(rows),
            "seeds_after": after / len(rows),
            "zero_seed_ids": zero}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tomato")
    ap.add_argument("--suffix", default="_noent")
    ap.add_argument("--check", action="store_true",
                    help="verify an existing variant instead of creating one")
    a = ap.parse_args()

    pairs = [(f"{a.dataset}_train_v16sc", "train"), (f"{a.dataset}_test_v16sc", "test")]
    ok = True

    for name, split in pairs:
        src_dir, dst_dir = f"{DATA}/{name}", f"{DATA}/{name}{a.suffix}"
        print(f"\n=== {name} -> {name}{a.suffix} ===")

        if not os.path.isdir(src_dir):
            print(f"  MISSING source {src_dir}", file=sys.stderr)
            return 2

        if not a.check:
            if os.path.exists(dst_dir):
                print(f"  destination exists, refusing to overwrite: {dst_dir}",
                      file=sys.stderr)
                return 2
            # -a preserves structure, -l hardlinks regular files.
            subprocess.run(["cp", "-al", src_dir, dst_dir], check=True)
            stats = strip_split(src_dir, dst_dir, split)
            print(f"  queries {stats['queries']}  "
                  f"seeds/query {stats['seeds_before']:.2f} -> {stats['seeds_after']:.2f}")
            if stats["zero_seed_ids"]:
                print(f"  WARNING {len(stats['zero_seed_ids'])} queries now have NO seeds: "
                      f"{stats['zero_seed_ids'][:5]}", file=sys.stderr)
                ok = False
            else:
                print("  zero-seed queries: 0")

        # VERIFY the graph really is untouched. This is the whole premise.
        for f in GRAPH_FILES:
            s, d = (f"{src_dir}/processed/stage1/{f}", f"{dst_dir}/processed/stage1/{f}")
            if not os.path.exists(d):
                print(f"  MISSING {d}", file=sys.stderr); ok = False; continue
            same = sha(s) == sha(d)
            linked = os.stat(s).st_ino == os.stat(d).st_ino
            print(f"  {f:16s} identical={same}  hardlinked={linked}")
            ok &= same

        # The handcrafted scorer caches must still be present and shared.
        for f in ("operator_components.npz", "operator_scores.npz"):
            s, d = f"{src_dir}/{f}", f"{dst_dir}/{f}"
            if os.path.exists(s):
                linked = os.path.exists(d) and os.stat(s).st_ino == os.stat(d).st_ino
                print(f"  {f:24s} present={os.path.exists(d)}  hardlinked={linked}")
                ok &= os.path.exists(d)

        # And the split file must NOT be shared, or we edited the source.
        sp = f"{dst_dir}/processed/stage1/{split}.json"
        if os.path.exists(sp):
            shared = os.stat(sp).st_ino == os.stat(f"{src_dir}/processed/stage1/{split}.json").st_ino
            print(f"  {split}.json shared-inode={shared} (must be False)")
            ok &= not shared
            got = json.load(open(sp))
            has_ent = any("entity" in r["start_nodes"] for r in got)
            print(f"  {split}.json still has entity channel: {has_ent} (must be False)")
            ok &= not has_ent

    print("\nOK" if ok else "\nFAILED", file=sys.stderr if not ok else sys.stdout)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
