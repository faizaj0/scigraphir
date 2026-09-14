#!/usr/bin/env python3
"""Keep the title cache across a Stage 4 change that did not alter MATCHING.

`matcher_fingerprint()` hashes the source of every matching function, so ANY edit
to one of them invalidates every stored verdict. That is the right default: a
changed threshold or a changed regex means the cached answer is no longer the
answer the current code would give.

It is the wrong default for a change that only alters WHICH SERVICE IS ASKED
FIRST. `--s2-first` reorders two lookups; both still score candidates with the
same `similarity` and hold them to the same `accepted()` bar, so a verdict
already in the cache is exactly what the new code would produce. Discarding
14,041 of them would cost hours of Semantic Scholar's 1 req/s for no change in
the data.

So this exists, and it is deliberately NOT automatic. Run it only when you can
say why the edit cannot change a verdict. If in doubt, do not run it: refetching
is slow and cheap, whereas a stale verdict is fast and wrong.

    python3 build/restamp_cache.py --why "s2-first is route order only"

Refuses to run when the fingerprint already matches, so it cannot quietly paper
over a real matching change made later.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "data" / "_cache" / "resolve_titles.json"


def load_stage4():
    spec = importlib.util.spec_from_file_location("S4", Path(__file__).parent / "04_resolve.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, default=DEFAULT)
    ap.add_argument("--why", required=True,
                    help="one sentence on why the edit cannot change a verdict. "
                         "Stored in the cache file, so a later reader can judge it")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if not a.cache.exists():
        sys.exit(f"no cache at {a.cache}")
    m = load_stage4()
    want = m.matcher_fingerprint()
    data = json.loads(a.cache.read_text())
    have = data.get("_fp")
    entries = data.get("entries") or {}

    print(f"  cache      {a.cache}")
    print(f"  entries    {len(entries):,}")
    print(f"  stamped    {have}")
    print(f"  code is    {want}")

    if have == want:
        print("\n  already current; nothing to do (and nothing was written)")
        return
    if data.get("_v") != m.CACHE_VERSION:
        sys.exit(f"\n  cache is schema v{data.get('_v')}, code is v{m.CACHE_VERSION}. "
                 f"That is a real format change, not a route change. Refusing.")
    if a.dry_run:
        print(f"\n  --dry-run: would restamp {len(entries):,} entries to {want}")
        return

    # The old file is kept, not overwritten: if the justification turns out to be
    # wrong, the only way back is the untouched original.
    backup = a.cache.with_suffix(f".json.pre-{have}")
    shutil.copy2(a.cache, backup)
    data["_fp"] = want
    data.setdefault("_restamps", []).append({"from": have, "to": want, "why": a.why})
    tmp = a.cache.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(a.cache)
    print(f"\n  restamped {len(entries):,} entries -> {want}")
    print(f"  reason recorded: {a.why}")
    print(f"  previous file kept at {backup.name}")


if __name__ == "__main__":
    main()
