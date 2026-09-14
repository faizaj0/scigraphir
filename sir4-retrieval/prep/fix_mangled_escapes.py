"""
fix_mangled_escapes.py -- repair mangled \\uXXXX escapes in stage1 graph CSVs.

WHAT IS WRONG. Some stage1 CSVs carry control bytes where a \\uXXXX escape was unescaped
two hex digits at a time instead of four:

    "\\u00b7"  MIDDLE DOT      became  b"\\x00" + b"b7"
    "\\u00e9"  e-acute         became  b"\\x00" + b"e9"
    "\\u03b1"  GREEK ALPHA     became  b"\\x03" + b"b1"

So a node named  "time complexity of o(|v|^{1/2} · |e|)"  is stored with a NUL in the
middle of it. Two consequences, one loud and one silent:

  LOUD    csv.DictReader raises "_csv.Error: line contains NUL" on any file where the
          mangled codepoint happened to start with 00. That is why an audit pass fails on
          sir4_cs train and sir4_physics test and passes on the others.
  SILENT  every other mangled codepoint (\\x03 for alpha, and so on) reads back fine and
          leaves a corrupt node NAME in the graph. The same concept written correctly
          elsewhere cannot merge with it, so canonicalisation silently misses a pair.

THE REPAIR IS MOSTLY RIGHT, NOT EXACT. Read this before trusting it.

Where all four hex digits survived, B*256 + int(hex, 16) reconstructs the original exactly:

    b"$\\x03b3$-ray"                -> "$\\u03b3$-ray"    = gamma-ray            CORRECT
    b"intensity of \\x03c3 and \\x03c0" -> sigma and pi                              CORRECT

Where the escape lost a digit BEFORE this corruption, the same rule produces a plausible
but wrong character. Measured case, sir4_physics test:

    bytes  n \\x00 e e l          the physics term is "Neel" (n + U+00E9 + e + l)
    rule   \\x00 + "ee" -> U+00EE  giving "nil", not "neel"

There is no automatic way to tell the two apart: U+00EE is as plausible a codepoint as
U+00E9. So --repair improves the majority (the Greek and mathematical-alphanumeric sites,
which are most of them) and gets a handful wrong. --strip invents nothing but leaves those
names misspelled by a deletion instead. Neither is correct.

THE ONLY CORRECT FIX is to regenerate stage1 with the upstream unescaping bug fixed, so the
escapes are decoded four hex digits at a time. This script does not do that and does not know
where that bug lives; it exists so the audit cell is runnable and so the damage is counted.

Dry run by default. --repair or --strip writes, keeping a .bak beside each file it changes.

Usage
-----
    python3 prep/fix_mangled_escapes.py                      # dry run, every sir4_* graph
    python3 prep/fix_mangled_escapes.py --repair             # substitute, .bak kept
    python3 prep/fix_mangled_escapes.py --strip              # delete control bytes only
    python3 prep/fix_mangled_escapes.py --glob 'tomato_*'    # a different corpus
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
CARGO = os.path.dirname(os.path.dirname(HERE))
DATA = f"{CARGO}/kg-construction/data"

# tab, newline, carriage return are legitimate; everything else below 0x20 is not.
CTRL = bytes(b for b in range(32) if b not in (9, 10, 13))
# a control byte followed by exactly two lowercase-or-digit hex characters
MANGLED = re.compile(b"([" + re.escape(CTRL) + b"])([0-9a-fA-F]{2})")


def repair(raw: bytes) -> tuple[bytes, int, int]:
    """Return (repaired, n_repaired, n_left). Leaves unreconstructable control bytes."""
    n = [0]

    def sub(m: re.Match) -> bytes:
        cp = m.group(1)[0] * 256 + int(m.group(2), 16)
        n[0] += 1
        return chr(cp).encode("utf-8")

    out = MANGLED.sub(sub, raw)
    left = sum(1 for b in out if b in CTRL)
    return out, n[0], left


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="sir4_*_v16sc",
                    help="dataset directory glob under kg-construction/data")
    ap.add_argument("--repair", action="store_true",
                    help="substitute the reconstructed codepoint. Right for the intact "
                         "escapes, wrong for the ones that had already lost a digit.")
    ap.add_argument("--strip", action="store_true",
                    help="delete the control bytes instead. Invents no character, but "
                         "leaves those names misspelled by a deletion.")
    a = ap.parse_args()

    paths = sorted(p for p in glob.glob(f"{DATA}/{a.glob}/processed/stage1/*.csv")
                   if "smoke" not in p)
    if not paths:
        print(f"no CSVs matched {DATA}/{a.glob}/processed/stage1/*.csv")
        return 1

    tot_fix = tot_left = tot_files = 0
    for p in paths:
        raw = open(p, "rb").read()
        if not any(b in CTRL for b in raw):
            continue
        out, fixed, left = repair(raw)
        tot_files += 1
        tot_fix += fixed
        tot_left += left
        rel = os.path.relpath(p, DATA)
        print(f"{rel:56s} repaired={fixed:4d}  unreconstructable={left:3d}")
        # one worked example per file, so the change is visible and not taken on trust
        m = MANGLED.search(raw)
        if m:
            ls = raw.rfind(b"\n", 0, m.start()) + 1
            le = raw.find(b"\n", m.start())
            before = raw[ls:le][:110]
            after = repair(raw[ls:le])[0][:110]
            print(f"    before: {before!r}")
            print(f"    after : {after!r}")
        if a.repair or a.strip:
            shutil.copy(p, p + ".bak")
            open(p, "wb").write(out if a.repair else
                                bytes(b for b in raw if b not in CTRL))

    print(f"\n{tot_files} file(s), {tot_fix} escape(s) repairable, "
          f"{tot_left} control byte(s) with nothing to reconstruct them from")
    if not (a.repair or a.strip):
        print("DRY RUN -- nothing written. --repair substitutes, --strip deletes; read the\n"
              "module docstring first, neither is a correct fix.")
    else:
        print("written. Rebuild the affected bundles before the next Colab run:")
        print("  python3 prep/bundle.py --dataset sir4_<domain>")
    # The graphs already indexed from these CSVs are unaffected until they are rebuilt: the
    # stage2 cache is keyed on a fingerprint of the config, not of the CSV contents, so a
    # repaired CSV does NOT invalidate an existing graph.pt. Delete the stage2 directory for
    # any dataset whose names must actually change.
    print("\nNOTE: stage2 caches are keyed on the config fingerprint, not on CSV contents,")
    print("      so an existing graph.pt will NOT rebuild by itself. Remove")
    print("      <dataset>/processed/stage2/ for any graph whose node names must change.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
