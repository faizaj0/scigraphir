#!/usr/bin/env python3
"""Put `openalex_references` back on decomposed records, from Stage 2. No API calls.

WHY THIS EXISTS.

Stage 4 has a shortcut: an inspiration whose title matches the citing paper's own
reference list needs no OpenAlex title search, because that list already carries
the DOI. A title search is the one request in Stage 4 that cannot be batched, so
the shortcut is the difference between ~16,600 individual requests and ~1,600.

The shortcut could never fire. Stage 3's success path built a fresh record
instead of extending the paper, so `openalex_references` survived on the papers
that were DROPPED and was stripped from the ones that went on to be resolved.
Stage 3 is fixed, but a decomposed file already on disk cost real money and must
not be regenerated to recover a field Stage 2 still has.

So: join on DOI, offline, free. Measured on 400 biology papers, 92.4% of
inspirations become matchable, which is a 90% cut in Stage 4's request count.

    python3 build/patch_refs.py --decomposed data/03_decomposed/biology_train_low.jsonl

Idempotent: a record that already has a non-empty list is left alone, so running
it twice is a no-op and running it on a file from the FIXED Stage 3 does nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def stage2_for(decomposed: Path, explicit: Path | None) -> Path:
    """Guess the Stage 2 file from the decomposed name, unless told.

    `biology_train_low.jsonl` -> `biology_train.jsonl`. The tag is whatever the
    pipeline appended, and Stage 2 output is never tagged: it is shared by every
    run of that split, which is exactly why the join is possible at all.
    """
    if explicit:
        return explicit
    parts = decomposed.stem.split("_")
    for n in (2, 1):                       # domain_split, then domain alone
        cand = ROOT / "data" / "02_fulltext" / ("_".join(parts[:n]) + ".jsonl")
        if cand.exists():
            return cand
    sys.exit(f"cannot find the Stage 2 file for {decomposed.name}; pass --stage2")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decomposed", type=Path, required=True)
    ap.add_argument("--stage2", type=Path, default=None,
                    help="override the guessed data/02_fulltext/<domain>_<split>.jsonl")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change, write nothing")
    a = ap.parse_args()

    dec = a.decomposed
    if not dec.exists():
        sys.exit(f"no such file: {dec}")
    src = stage2_for(dec, a.stage2)
    print(f"[io] decomposed {dec}\n[io] stage 2    {src}")

    # Pass 1: which DOIs need the field. Only decomposed records are resolved,
    # so a dropped paper's reference list is not worth the memory.
    need: set[str] = set()
    total = have = 0
    with open(dec) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("decomposed"):
                continue
            total += 1
            if r.get("openalex_references"):
                have += 1
            elif r.get("doi"):
                need.add(r["doi"])
    print(f"  decomposed records {total:,}   already have refs {have:,}   "
          f"need them {len(need):,}")
    if not need:
        print("  nothing to do")
        return

    # Pass 2: pull just those reference lists, trimmed to what Stage 4 reads.
    refs: dict[str, list] = {}
    with open(src) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            d = r.get("doi")
            if d not in need:
                continue
            refs[d] = [{"title": x["title"], "doi": x["doi"]}
                       for x in (r.get("openalex_references") or [])
                       if x.get("title") and x.get("doi")]
            if len(refs) == len(need):
                break
    found = sum(1 for v in refs.values() if v)
    print(f"  stage 2 supplied a usable list for {found:,}/{len(need):,}")

    if a.dry_run:
        print("  --dry-run: nothing written")
        return

    # Written whole then renamed, so an interrupted patch cannot leave a JSONL
    # truncated mid-line over a file that took $79 of API spend to produce.
    tmp = dec.with_suffix(dec.suffix + ".patched")
    patched = 0
    with open(dec) as fin, open(tmp, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            r = json.loads(line)
            if (r.get("decomposed") and not r.get("openalex_references")
                    and refs.get(r.get("doi"))):
                r["openalex_references"] = refs[r["doi"]]
                patched += 1
            fout.write(json.dumps(r) + "\n")
    tmp.replace(dec)
    print(f"  patched {patched:,} record(s) -> {dec}")
    print(f"\n  Stage 4 will now report 'stage 2b supplied a DOI for N titles'. "
          f"If it does not,\n  the join found nothing and the run will be as "
          f"expensive as before.")


if __name__ == "__main__":
    main()
