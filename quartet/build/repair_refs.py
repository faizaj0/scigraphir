"""
Re-clean the stored bibliographies in place. No refetching.

Stage 2 stores each reference as a string that should be the TITLE. On the ar5iv route
it often is not: 39% of physics and 40% of materials science references arrived as the
author list, with the title either buried behind it or absent entirely.

    "G. Pan, J. Ding, Y. Du, D.-J. Lee, and Y. Lu, A DFT accurate machine learning..."
                                                   ^ the title starts here

That matters because Stage 4 looks each string up in Semantic Scholar to turn it into a
gold document. An author list never resolves, so the gold is dropped with no error
anywhere -- the benchmark just quietly gets smaller, and unevenly by domain, since the
rate is 40% in physics and 3% in biology.

The repair works on the stored data because the author run is still sitting in the
string: a better stripper applied to it recovers the title. Measured over 124,300
references, this takes the author-list rate from 20.6% to 1.2%.

Usage:
  python build/repair_refs.py --domain all --split both --dry-run
  python build/repair_refs.py --domain all --split both
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "domains.yaml"
OUT = ROOT / "data" / "02_fulltext"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("_ft", Path(__file__).parent / "02_fulltext.py")
_ft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ft)
strip_authors = _ft.strip_authors

# Same detector the audit uses: two or more "Surname, I." or "I. Surname" groups with no
# title-like text in front of them.
AUTHORY = re.compile(
    r"^[^a-z]{0,4}(?:[A-Z][A-Za-z'`\-]+,?\s+(?:[A-Z]\.\s*){1,3}[;,&]?\s*){2,}"
    r"|^(?:[A-Z]\.(?:[\s-]?[A-Z]\.)*\s+[A-Z][A-Za-z'`\-]+,?\s*(?:and\s*)?){2,}")


def refuse_if_fetching() -> None:
    """Abort while Stage 2 runs. It holds these files open in append mode, so replacing
    them makes every paper it fetches afterwards vanish into an unlinked inode."""
    try:
        out = subprocess.run(["pgrep", "-f", "02_fulltext.py"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return
    pids = [p for p in out.stdout.split() if p.strip()]
    if pids:
        sys.exit(f"REFUSING TO RUN: 02_fulltext.py is running (pid {', '.join(pids)}).\n"
                 f"Stop it first:  pkill -f 02_fulltext.py")


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

    print(f"{'file':<24}{'refs':>9}{'author-list':>13}{'after':>8}{'changed':>10}")
    t_refs = t_before = t_after = t_changed = 0
    samples: list[tuple[str, str]] = []
    for d in domains:
        for s in splits:
            path = OUT / f"{d}_{s}.jsonl"
            if not path.exists():
                continue
            lines, refs, before, after, changed = [], 0, 0, 0, 0
            for line in open(path):
                if not line.strip():
                    continue
                row = json.loads(line)
                bib = row.get("bibliography") or {}
                new = {}
                for k, v in bib.items():
                    refs += 1
                    was_bad = bool(AUTHORY.match(v))
                    before += was_bad
                    fixed = strip_authors(v)
                    # Never let a repair empty a reference out: a short leftover means the
                    # stripper misfired, and a wrong title is worse than a messy one.
                    if len(fixed) < 12:
                        fixed = v
                    if fixed != v:
                        changed += 1
                        if was_bad and len(samples) < 6:
                            samples.append((v, fixed))
                    after += bool(AUTHORY.match(fixed))
                    new[k] = fixed
                row["bibliography"] = new
                lines.append(json.dumps(row) + "\n")
            if not refs:
                continue
            t_refs += refs; t_before += before; t_after += after; t_changed += changed
            print(f"{path.name:<24}{refs:>9,}{100*before/refs:>12.0f}%{100*after/refs:>7.0f}%"
                  f"{changed:>10,}")
            if not a.dry_run:
                # Write then rename: an interrupted run cannot leave a truncated file.
                tmp = path.with_suffix(".jsonl.partial")
                with open(tmp, "w") as fh:
                    fh.writelines(lines)
                os.replace(tmp, path)

    if not t_refs:
        print("nothing to do")
        return
    print(f"\n{t_refs:,} references: author-list rate {100*t_before/t_refs:.1f}% -> "
          f"{100*t_after/t_refs:.1f}%, {t_changed:,} strings rewritten")
    for old, new in random.Random(0).sample(samples, min(4, len(samples))):
        print(f"\n  before: {old[:100]}")
        print(f"  after : {new[:100]}")
    if a.dry_run:
        print("\ndry run: nothing was written")


if __name__ == "__main__":
    main()
