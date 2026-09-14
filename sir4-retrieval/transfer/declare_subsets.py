"""
declare_subsets.py -- freeze the evaluation subsets BEFORE any model is run.

WHY THIS IS A SCRIPT AND NOT A FLAG. The strict zero-shot subset decides which
queries the headline transfer number is computed over. If it can be chosen after
seeing results, it is not a subset, it is a knob. This script writes the query ids
to a file once; every downstream tool reads that file and cannot redefine it.

WHY NOT EXCLUDE BY DISCIPLINE LABEL. That was the original plan: drop ResearchBench's
Biology, Cell Biology and Chemistry buckets, keep 990 queries. Measured against the
queries' own text, the buckets do not describe content:

    Physics  -> 45% chemistry/materials, 10% physics
    Math     -> no dominant subject at all
    Law      -> 36% business, 24% biomedical

and 125 biomedical queries survive inside the supposedly-clean 990, which is 12.6% of
it. Excluding a bucket named "Biology" does not exclude biology.

WHAT WE EXCLUDE INSTEAD. The thing that actually matters is overlap with the training
distribution. TOMATO-Star is 89.5% biomedical by keyword majority over its 7,000
training documents, and the primary SIR-4 comparator is the Biology checkpoint. So the
strict subset excludes queries whose OWN TEXT reads biomedical, using the same frozen
lexicon that produced the 89.5% figure. Same instrument on both sides of the
comparison.

The classifier is deliberately crude. A crude rule fixed in advance is worth more here
than a good one chosen afterwards, and every borderline call it makes is inspectable
in the emitted file.

Usage
-----
    python3 transfer/declare_subsets.py            # writes subsets.json, refuses to overwrite
    python3 transfer/declare_subsets.py --force    # re-declare (records why in the file)
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CARGO = os.path.dirname(ROOT)
RB = f"{CARGO}/kg-construction/data/researchbench_test"

# FROZEN. This is the lexicon that measured TOMATO-Star at 89.5% biomedical. Editing
# it after results exist would silently redefine the headline denominator, so if it
# ever must change, change it here, delete subsets.json, and say so in the write-up.
BIOMED = (r"patient|clinical|neuro|brain|cortex|gene|protein|cell |tumor|cancer|"
          r"disease|in vivo|in vitro|receptor|synap|therap|diagnos|medical|drug|"
          r"health|biolog|antibod|enzyme|metabol|immun")
OTHER = {
    "chem/mat":      r"catalys|polymer|alloy|crystal|molecul|synthesis|nanoparticl|"
                     r"electrode|semiconductor|thin film|adsorpt|photocatal|corrosion",
    "physics/astro": r"quantum|photon|particle|relativ|thermodynam|superconduct|plasma|"
                     r"astro|cosmo|galax|pulsar|stellar|orbit|turbulen",
    "cs/ml":         r"machine learning|deep learning|neural network|algorithm|dataset|"
                     r"classif|internet of things|blockchain|software",
    "business/econ": r"firm|market|investor|shareholder|supply chain|financ|econom|"
                     r"governance|consumer|management",
    "earth/env":     r"climate|soil|wastewater|emission|ecosystem|pollut|groundwater|"
                     r"atmospher|seismic",
}


def query_text(q: dict) -> str:
    """The question only. NOT the answer or the background survey: those quote the
    gold papers, so classifying on them would leak the labels into the subset."""
    return ((q.get("research_question") or "") + " " + (q.get("question") or ""))[:1500]


def label(t: str) -> tuple[bool, str]:
    t = t.lower()
    bio = len(re.findall(BIOMED, t))
    others = {k: len(re.findall(p, t)) for k, p in OTHER.items()}
    top, n = max(others.items(), key=lambda kv: kv[1])
    if bio == 0 and n == 0:
        return False, "none"
    return (bio >= n, "biomed" if bio >= n else top)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default=f"{RB}/raw/test.json")
    ap.add_argument("--out", default=f"{RB}/subsets.json")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    if os.path.exists(a.out) and not a.force:
        cur = json.load(open(a.out))
        print(f"{a.out} already exists and will NOT be overwritten.")
        for k, v in cur.get("subsets", {}).items():
            print(f"  {k:22} {len(v):5} queries")
        print("\nThat is the point: the subsets are declared once. --force to redeclare.")
        return 0

    queries = json.load(open(a.queries))
    strict, excluded, comp = [], [], collections.Counter()
    by_bucket = collections.defaultdict(collections.Counter)
    for q in queries:
        is_bio, lab = label(query_text(q))
        comp[lab] += 1
        by_bucket[q.get("discipline", "?")][lab] += 1
        (excluded if is_bio else strict).append(q["id"])

    # The label-based subset is recorded too, purely so the write-up can quantify how
    # much the two definitions disagree rather than asserting that they do.
    LABEL_EXCLUDE = ("Biology", "Cell Biology", "Chemistry")
    label_based = [q["id"] for q in queries if q.get("discipline") not in LABEL_EXCLUDE]

    out = {
        "declared_by": "transfer/declare_subsets.py",
        "rationale": ("strict_zeroshot excludes queries whose own question text reads "
                      "biomedical, because TOMATO-Star is 89.5% biomedical and the "
                      "primary SIR-4 comparator is the Biology checkpoint. Excluding by "
                      "ResearchBench discipline label does not work: its buckets do not "
                      "describe content."),
        "classifier": {"biomed_pattern": BIOMED, "other_patterns": OTHER,
                       "field_used": "research_question + question (never answer)"},
        "subsets": {"all": [q["id"] for q in queries],
                    "strict_zeroshot": strict,
                    "label_based_990": label_based},
    }
    json.dump(out, open(a.out, "w"), indent=1)

    n = len(queries)
    print(f"wrote {a.out}\n")
    print(f"  all                 {n:5}")
    print(f"  strict_zeroshot     {len(strict):5}   ({len(excluded)} excluded as biomedical)")
    print(f"  label_based_990     {len(label_based):5}   (recorded for comparison only)")
    ov = len(set(strict) & set(label_based))
    print(f"\n  the two definitions agree on {ov} queries; "
          f"{len(set(label_based) - set(strict))} queries are in the label-based subset "
          f"but read biomedical")
    print("\n  content label over all queries:")
    for k, v in comp.most_common():
        print(f"    {k:16} {v:5}  {v / n * 100:5.1f}%")
    print("\n  biomedical share per ResearchBench bucket:")
    for d, c in sorted(by_bucket.items(), key=lambda kv: -kv[1]["biomed"] / max(sum(kv[1].values()), 1)):
        tot = sum(c.values())
        print(f"    {d:24} {c['biomed']:4}/{tot:4}  {c['biomed'] / tot * 100:5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
