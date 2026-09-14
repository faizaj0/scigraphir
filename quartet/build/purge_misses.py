#!/usr/bin/env python3
"""Drop cached NON-RESULTS so a rerun actually retries them.

THE TRAP THIS EXISTS FOR.

Stage 4's title cache stores every lookup, including the ones that found nothing.
That is the right default -- a title no index can match costs a request every
time you ask, and asking again usually gets the same silence -- but it makes one
kind of rerun silently useless:

    a run resolves badly because of HOW it was configured,
    you fix the configuration and rerun,
    and every failure is served from cache, so nothing is retried.

That is exactly what happened to physics_train. A run with --s2-first resolved
51.3%, cached 9,730 `s2_match` misses, and the rerun without that flag finished
in 3 minutes at 99% cache hit having retried none of them. The fix was not the
configuration; it was that the cache had already answered.

WHICH MISSES ARE WORTH RETRYING.

Not all of them. A miss carries the route that produced it:

    s2_match / oa_title   a title search found nothing. A DIFFERENT service, or
                          the same one later, may well find it. RETRY.
    doi                   an identifier that resolved to nothing. Usually a real
                          gap in the index, occasionally transient. RETRY.
    citation_string       Stage 4 REFUSED to search, because the string is a
                          journal locator with no title in it. Deterministic:
                          the same string gets the same refusal forever, so
                          retrying costs a request and changes nothing. KEEP.
                          These are handled by the Crossref route instead, which
                          reads the locator rather than searching for a title.

So the default drops the first three and keeps `citation_string`.

    python3 build/purge_misses.py                       # see what it would do
    python3 build/purge_misses.py --write
    python3 build/purge_misses.py --write --routes s2_match

Resolved entries are never touched, whatever the flags: this only ever removes
answers of the form "nothing found".
"""
from __future__ import annotations

import argparse
import collections
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "data" / "_cache" / "resolve_titles.json"

# Deterministic refusals: the same input always produces them, so a retry is a
# request spent to be told the same thing.
KEEP = {"citation_string", "empty"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, default=DEFAULT)
    ap.add_argument("--routes", nargs="*", default=None,
                    help=f"routes to purge. Default: every miss route except "
                         f"{', '.join(sorted(KEEP))}")
    ap.add_argument("--write", action="store_true",
                    help="actually modify the cache. Without it, reports only")
    a = ap.parse_args()

    if not a.cache.exists():
        sys.exit(f"no cache at {a.cache}")
    data = json.loads(a.cache.read_text())
    entries = data.get("entries") or {}

    miss = {k: v for k, v in entries.items() if v.get("match_quality") == "none"}
    by_route = collections.Counter(v.get("route") for v in miss.values())
    targets = set(a.routes) if a.routes else {r for r in by_route if r not in KEEP}

    print(f"  cache          {a.cache}")
    print(f"  entries        {len(entries):,}")
    print(f"  resolved       {len(entries) - len(miss):,}   (never touched)")
    print(f"  misses         {len(miss):,}\n")
    drop = 0
    for route, n in by_route.most_common():
        mark = "PURGE" if route in targets else "keep "
        drop += n if route in targets else 0
        print(f"    {mark}  {str(route):<18}{n:>8,}")
    print(f"\n  would remove   {drop:,} miss(es); {len(entries) - drop:,} entries remain")

    if not drop:
        print("  nothing to do")
        return
    if not a.write:
        print("\n  dry run. Add --write to apply.")
        print("  Then rerun Stage 4 -- those titles will be looked up again.")
        return

    backup = a.cache.with_suffix(".json.pre-purge")
    shutil.copy2(a.cache, backup)
    data["entries"] = {k: v for k, v in entries.items()
                       if not (v.get("match_quality") == "none"
                               and v.get("route") in targets)}
    tmp = a.cache.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(a.cache)
    print(f"\n  removed {drop:,}; {len(data['entries']):,} entries remain")
    print(f"  previous file kept at {backup.name}")
    print(f"\n  Stage 4 will now retry those {drop:,} titles. At OpenAlex's "
          f"$0.001 per\n  title search that is about ${drop * 0.001:.2f}.")


if __name__ == "__main__":
    main()
