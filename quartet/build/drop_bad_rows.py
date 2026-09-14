"""
Delete Stage 2 rows that fail an audit check, so the fetcher re-does only those.

Stage 2 resumes by DOI: any paper missing from the output file is fetched again. So
removing the bad rows and rerunning the ordinary Stage 2 command repairs the corpus
without touching the papers that were already extracted correctly.

The alternative is deleting the whole file. On the run this was written for that meant
refetching 15,242 papers to repair 3,280, about 90 minutes of pointless work.

Reruns nothing itself. It rewrites the files and prints the Stage 2 command to run next.

Usage:
  python build/audit_fulltext.py --domain all --split both    # see what is wrong
  python build/drop_bad_rows.py  --domain all --split both --dry-run
  python build/drop_bad_rows.py  --domain all --split both
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "domains.yaml"
OUT = ROOT / "data" / "02_fulltext"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_fulltext import (  # noqa: E402  - same directory, shared definitions
    CAP, RE_ABSTRACT_TAG, RE_BIBENV, RE_FOOTER, RE_MARKUP_OPEN, RE_NUM_SOUP, RE_PGF,
    RE_REFLIST, RE_REFLIST_IN_BODY, RE_TABLE_CELLS, RE_TABLE_GLYPH, dup_fraction)


def why_bad(row: dict) -> str | None:
    """The first fatal audit check this row fails, or None if it is clean.

    Imported from audit_fulltext rather than restated, so a row can never be judged
    clean here and dirty there.
    """
    t = row.get("fulltext", "")
    if not t:
        return "empty"
    if len(RE_REFLIST_IN_BODY.findall(t)) >= 8:
        return "reference list inside the body"
    if len(RE_TABLE_CELLS.findall(t)) + len(RE_TABLE_GLYPH.findall(t)) >= 20:
        return "table flattened into prose"
    if len(RE_REFLIST.findall(t[-8000:])) >= 5:
        return "bibliography pasted into text"
    if RE_FOOTER.search(t):
        return "LaTeXML page footer"
    if len(RE_PGF.findall(t)) > 3:
        return "TikZ/pgf commands"
    if RE_MARKUP_OPEN.match(t[:60]):
        return "raw markup at the start"
    if RE_ABSTRACT_TAG.search(t[:2000]):
        return "abstract tag leaked"
    if RE_BIBENV.search(t):
        return "thebibliography in text"
    if len("".join(RE_NUM_SOUP.findall(t))) > 0.15 * len(t):
        return "numeric soup"
    if row.get("fulltext_chars", 0) > CAP:
        return "longer than the cap"
    if dup_fraction(t) > 0.15:
        return "duplicated text"
    return None


def refuse_if_fetching() -> None:
    """Abort if Stage 2 is running. This is not a nicety, it is data loss.

    Stage 2 holds each output file open in append mode for the whole run. This script
    rewrites the file and os.replace()s it, which unlinks the inode Stage 2 is holding.
    The fetcher does not notice: it keeps writing happily into a file with zero links,
    so every paper it fetches from that moment on is discarded, silently, until the run
    ends. Observed live -- the fetcher held inode 214796182 while the directory entry
    pointed at 214901342.

    Rewriting in place instead of renaming would be worse, not better: it would trade
    silent loss for interleaved half-written lines.
    """
    try:
        out = subprocess.run(["pgrep", "-f", "02_fulltext.py"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return                       # cannot check; the warning in --help has to do
    pids = [p for p in out.stdout.split() if p.strip()]
    if pids:
        sys.exit(
            f"REFUSING TO RUN: 02_fulltext.py is running (pid {', '.join(pids)}).\n"
            f"This script replaces the files that process has open, which makes every\n"
            f"paper it fetches afterwards vanish. Stop it first:\n"
            f"    pkill -f 02_fulltext.py\n"
            f"then rerun this, then start the fetch again.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, help="domain id, comma list, or 'all'")
    ap.add_argument("--split", required=True, choices=["train", "test", "both"])
    ap.add_argument("--dry-run", action="store_true", help="report, change nothing")
    a = ap.parse_args()

    if not a.dry_run:
        refuse_if_fetching()

    known = [d["id"] for d in yaml.safe_load(open(CONFIG))["domains"]]
    domains = known if a.domain == "all" else [d.strip() for d in a.domain.split(",")]
    splits = ["train", "test"] if a.split == "both" else [a.split]

    total_kept = total_dropped = 0
    touched: set[str] = set()
    for d in domains:
        for s in splits:
            path = OUT / f"{d}_{s}.jsonl"
            if not path.exists():
                continue
            kept, reasons = [], {}
            for line in open(path):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    reasons["unparseable"] = reasons.get("unparseable", 0) + 1
                    continue
                why = why_bad(row)
                if why:
                    reasons[why] = reasons.get(why, 0) + 1
                else:
                    kept.append(line)
            dropped = sum(reasons.values())
            total_kept += len(kept)
            total_dropped += dropped
            print(f"{path.name:<24} keep {len(kept):>6,}   drop {dropped:>6,}"
                  + (f"   {reasons}" if reasons else ""))
            if dropped and not a.dry_run:
                # Write and rename, so an interrupted run cannot leave a truncated file.
                # Stage 1 lost 19,051 papers to exactly that.
                tmp = path.with_suffix(".jsonl.partial")
                with open(tmp, "w") as fh:
                    fh.writelines(kept)
                os.replace(tmp, path)
                touched.add(d)

    print(f"\nkept {total_kept:,}, dropped {total_dropped:,}")
    if a.dry_run:
        print("dry run: nothing was written")
    elif touched:
        print("\nNow rerun Stage 2. Resume refetches exactly the rows removed:")
        for d in sorted(touched):
            print(f"  python3 build/02_fulltext.py --domain {d} --split {a.split} "
                  f"--headroom 1.4 --workers 12")


if __name__ == "__main__":
    main()
