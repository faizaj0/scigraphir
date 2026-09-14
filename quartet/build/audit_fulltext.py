"""
Audit Stage 2 output.

Every check here corresponds to a bug that actually shipped at some point, so a clean
report means those specific failures are absent -- not that the text is good in general.
Run it after a smoke test and again after the full fetch.

Usage:
  python build/audit_fulltext.py --domain cs --split both
  python build/audit_fulltext.py --domain all --split both
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import statistics as st
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "domains.yaml"
OUT = ROOT / "data" / "02_fulltext"


def _cap() -> int:
    """The cap Stage 2 actually used, from the manifest it writes beside the data.

    Hardcoding 150,000 here reported a false failure against a run made in a shell with
    QUARTET_MAX_CHARS=200000 exported: the data was correct and the auditor was wrong.
    """
    try:
        return int(json.loads((OUT / "manifest.json").read_text())["max_chars"])
    except (OSError, ValueError, KeyError):
        import os
        return int(os.environ.get("QUARTET_MAX_CHARS", "150000"))


CAP = _cap()

# Each pattern is a known failure mode, named for what went wrong.
RE_FOOTER = re.compile(r"by L a T e XML|by LaTeXML|Generated on \w{3} \w{3}")
RE_PGF = re.compile(r"pgfsys@|lxSVG@|\\pgf")
RE_MARKUP_OPEN = re.compile(r'^\s*(?:class=|id=|<|ltx_)')
RE_REFLIST = re.compile(
    r"External Links|Cited by:|arXiv preprint|In Proceedings of|pages \d+--\d+")
# "Does the stored text reach the end of the paper?" Papers do not all end in a section
# called Conclusion: maths ends on a theorem or Final Remarks, empirical papers on
# Discussion or Limitations, others on Future Work or an Outlook. Matching the single
# word scored those as truncated and put the figure ~15 points low.
RE_CONCLUSION = re.compile(
    r"\bconclusion|\bconcluding\b|\bsummary\b|\bdiscussion\b|final remarks?"
    r"|future work|\boutlook\b|\blimitations\b|acknowledg", re.I)
RE_ABSTRACT_TAG = re.compile(r"ltx_abstract|<abstract>")
RE_BIBENV = re.compile(r"thebibliography|\\bibitem")
# Runs of bare numbers: the pgfplots coordinate dump, and JATS table cells.
RE_NUM_SOUP = re.compile(r"(?:-?\d[\d.eE+-]*[\s,;]+){12,}")
# A journal reference list sitting inside the body. Some publishers file it as
# <sec sec-type="ref-list"> under <body> rather than in <back>, and it reached 35% of
# biology papers. This is a leakage check, not a tidiness one: the extractor is supposed
# to infer inspirations and only then ground them in the reference list, so a body that
# already spells out every cited title hands it the answers.
#
# Three formats, not one. Keying only on "2021;42(5):373-498" and "doi:10." passed
# PMC8967256 as clean while it carried a 48,055-character reference list, because that
# journal numbers its entries and writes plain author initials. A detector that
# recognises one house style silently certifies every other house style.
RE_REFLIST_IN_BODY = re.compile(
    r"\d{4};\d+\s*\(\d+\)\s*:\s*\d+-\d+"           # Eur Heart J. 2021;42(5):373-498
    r"|doi:\s*10\.\d{4,}"                          # doi: 10.1093/eurheartj/ehaa612
    r"|\b\d{1,3}\.\s+[A-Z][a-z]+\s+[A-Z]{1,3},"    # "12. Smith AB, Jones CD,"
    r"|\b[A-Z][a-z]+\s+[A-Z]{1,3},\s*[A-Z][a-z]+\s+[A-Z]{1,3}\.\s+[A-Z]")
# A results table flattened into prose. Rows of decimals, or runs of tick marks from a
# comparison grid. NOT "x" or "root": those appear constantly in ordinary maths writing,
# and including them made this check fire on "a k x k neighbourhood window".
RE_TABLE_CELLS = re.compile(r"(?:[-+]?\d+\.\d+\s+){6,}")
RE_TABLE_GLYPH = re.compile(r"[○●✓✗]")
# A reference stored as an AUTHOR LIST rather than a title. Stage 4 looks each of these
# up in Semantic Scholar to build the gold set, and an author list never resolves, so the
# gold is dropped with no error anywhere. It was 39% of physics and 40% of materials
# science references before repair_refs.py, against 3% of biology -- an uneven loss that
# would have shrunk exactly the domains the cross-domain comparison depends on.
RE_AUTHOR_LIST = re.compile(
    r"^[^a-z]{0,4}(?:[A-Z][A-Za-z\'`\-]+,?\s+(?:[A-Z]\.\s*){1,3}[;,&]?\s*){2,}"
    r"|^(?:[A-Z]\.(?:[\s-]?[A-Z]\.)*\s+[A-Z][A-Za-z\'`\-]+,?\s*(?:and\s*)?){2,}")


def dup_fraction(text: str, window: int = 300) -> float:
    """Share of the text made of 300-char blocks seen earlier in the same paper.

    Catches both duplication modes: the pgf command loop (a short slice repeated
    hundreds of times) and the PMC nested-section bug (whole subsections emitted twice).
    Stride is half the window so an offset repeat is still caught.
    """
    if len(text) < 4 * window:
        return 0.0
    seen: set[str] = set()
    dup = total = 0
    for i in range(0, len(text) - window, window // 2):
        block = text[i: i + window]
        total += 1
        if block in seen:
            dup += 1
        seen.add(block)
    return dup / max(total, 1)


def audit(path: Path) -> dict | None:
    if not path.exists():
        return None
    rows = []
    for line in open(path):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    if not rows:
        return None

    n = len(rows)
    lens = [r.get("fulltext_chars", 0) for r in rows]
    refs = [len(r.get("bibliography") or {}) for r in rows]
    texts = [r.get("fulltext", "") for r in rows]

    # Truncated papers cannot show a conclusion, so scoring them as failures would
    # understate coverage. Report the two groups separately.
    untrunc = [r for r in rows if "fulltext_chars_original" not in r]

    return {
        "n": n,
        "routes": collections.Counter(r.get("fulltext_via", "?") for r in rows),
        "chars_median": int(st.median(lens)),
        "chars_max": max(lens),
        "over_cap": sum(1 for x in lens if x > CAP),
        "truncated": n - len(untrunc),
        "refs_median": int(st.median(refs)),
        "refs_zero": sum(1 for x in refs if x == 0),
        "footer": sum(1 for t in texts if RE_FOOTER.search(t)),
        "pgf": sum(1 for t in texts if len(RE_PGF.findall(t)) > 3),
        "markup_open": sum(1 for t in texts if RE_MARKUP_OPEN.match(t[:60])),
        "abstract_tag": sum(1 for t in texts if RE_ABSTRACT_TAG.search(t[:2000])),
        "reflist": sum(1 for t in texts if len(RE_REFLIST.findall(t[-8000:])) >= 5),
        "bibenv": sum(1 for t in texts if RE_BIBENV.search(t)),
        "numsoup": sum(1 for t in texts
                       if len("".join(RE_NUM_SOUP.findall(t))) > 0.15 * max(len(t), 1)),
        "reflist_body": sum(1 for t in texts if len(RE_REFLIST_IN_BODY.findall(t)) >= 8),
        "table_prose": sum(1 for t in texts if len(RE_TABLE_CELLS.findall(t))
                           + len(RE_TABLE_GLYPH.findall(t)) >= 20),
        "dup": sum(1 for t in texts if dup_fraction(t) > 0.15),
        "authorlist": round(100 * sum(1 for r in rows for v in (r.get("bibliography") or {}).values()
                                      if RE_AUTHOR_LIST.match(v))
                            / max(sum(len(r.get("bibliography") or {}) for r in rows), 1)),
        "concl": sum(1 for r in untrunc if RE_CONCLUSION.search(r["fulltext"][-30000:])),
        "concl_of": len(untrunc),
    }


# (label, key, what a healthy value looks like, is it fatal)
CHECKS = [
    ("bibliography pasted into text", "reflist", "0", True),
    ("reference list inside the body", "reflist_body", "0", True),
    ("table flattened into prose", "table_prose", "0", True),
    ("LaTeXML page footer in text", "footer", "0", True),
    ("TikZ/pgf drawing commands", "pgf", "0", True),
    ("raw markup at the start", "markup_open", "0", True),
    ("abstract tag leaked", "abstract_tag", "0", True),
    ("thebibliography/bibitem in text", "bibenv", "0", True),
    ("duplicated text (>15%)", "dup", "0", True),
    ("numeric soup (>15%)", "numsoup", "0", True),
    ("stored longer than the cap", "over_cap", "0", True),
    ("references that are author lists (%)", "authorlist", "0", False),
    ("empty bibliography", "refs_zero", "near 0", False),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, help="domain id, comma list, or 'all'")
    ap.add_argument("--split", required=True, choices=["train", "test", "both"])
    a = ap.parse_args()

    known = [d["id"] for d in yaml.safe_load(open(CONFIG))["domains"]]
    domains = known if a.domain == "all" else [d.strip() for d in a.domain.split(",")]
    splits = ["train", "test"] if a.split == "both" else [a.split]

    fatal = warn = 0
    for d in domains:
        for s in splits:
            path = OUT / f"{d}_{s}.jsonl"
            r = audit(path)
            print(f"\n{'=' * 66}\n{path.name}")
            if r is None:
                print("  no output yet")
                continue
            print(f"{'=' * 66}")
            print(f"  papers            {r['n']:,}")
            print(f"  routes            {dict(r['routes'])}")
            print(f"  chars             median {r['chars_median']:,}   "
                  f"max {r['chars_max']:,}   cap {CAP:,}   truncated {r['truncated']:,} "
                  f"({100 * r['truncated'] / r['n']:.0f}%)")
            print(f"  references        median {r['refs_median']}")
            print(f"  reaches paper end {r['concl']}/{r['concl_of']} of untruncated "
                  f"papers ({100 * r['concl'] / max(r['concl_of'], 1):.0f}%)")
            print(f"\n  {'check':<34} {'count':>7}  {'want':>7}")
            for label, key, want, is_fatal in CHECKS:
                v = r[key]
                bad = v > 0 if want == "0" else v > 0.05 * r["n"]
                mark = "  <-- FAIL" if (bad and is_fatal) else ("  <-- check" if bad else "")
                print(f"  {label:<34} {v:>7,}  {want:>7}{mark}")
                if bad:
                    fatal += is_fatal
                    warn += not is_fatal

    print(f"\n{'=' * 66}")
    if fatal:
        print(f"{fatal} failing check(s). Do NOT spend money on Stage 3 with this output.")
        sys.exit(1)
    if warn:
        print(f"No fatal problems. {warn} non-fatal warning(s) above: papers with no "
              f"reference list cannot ground inspirations, so drop them before Stage 3 "
              f"rather than paying to decompose them.")
        return
    print("All checks clean.")


if __name__ == "__main__":
    main()
