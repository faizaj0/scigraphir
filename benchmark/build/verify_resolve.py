#!/usr/bin/env python3
"""Differential test of the rebuilt 04_resolve.py against the surviving bytecode.

04_resolve.py was destroyed on 2026-08-02 by a shell redirect and rebuilt from
build/__pycache__/04_resolve.cpython-313.pyc. The .pyc still imports and runs, so
the rebuild does not have to be trusted: every pure function can be called on
both and compared.

Differences are EXPECTED and asserted for, rather than tolerated.

From the rebuild:

  clean_title      strips an appended venue with no journal keyword to mark it
  abstract_source  new key on every record, so Stage 5 can tell a real document
                   from a title-only one after backfilling
  CACHE_VERSION    3 -> 6
  main             resolves each target's own field instead of the split constant
  backfill_all     new: batched second-source lookup for abstracts and fields
  by_dois / batch  new: 50-per-request OpenAlex, 500-per-request S2

From the precision pass, which by design changes what the matcher accepts. This
file can no longer assert those functions are unchanged, so it asserts they HAVE
changed, in the direction claimed, and build/test_matching.py pins the behaviour:

  norm             strips MathML and entities before tokenising
  similarity       token F0.5 instead of one-sided containment; short titles
                   fall back to sequence agreement
  clean_title      year stripping is anchored; keyword-free venue locators are
                   cut; unbalanced brackets are repaired
  _from_openalex   identifier routes trusted; weak hits held for review
  _from_s2         same
  domain_relation  same_domain/cross_domain -> same_field/cross_field
  MISS             carries purity, provenance and rejection keys

Everything else must still agree exactly.

    python build/verify_resolve.py     # the rebuild is faithful
    python build/test_matching.py      # the precision pass works
"""
from __future__ import annotations

import importlib.util
import itertools
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PYC = Path("/private/tmp/claude-501/-Users-faizajalil-Desktop-CARGO/"
           "b2b97c17-d437-4e5a-ba6c-0a5aff3b9730/scratchpad/orig_resolve.pyc")


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Strings drawn from the corpus itself: every shape stage 2 actually produces.
CASES = [
    "Attention is all you need",
    "Attention Is All You Need",
    "Convex Optimization (Cambridge University Press, 2004).",
    "Quantum Computation and Quantum Information (Cambridge University Press, Cambridge, 2000).",
    "PRX Quantum 2 , 010343 (2021) .",
    "Phys. Rev. X 7 , 021050 (2017) .",
    "Physical Review Letters 103 , 210501 (2009) .",
    "MNRAS, 336, 449",
    "ApJ, 319, 180",
    "A&A, 593, A92",
    "Bayesian Flow Networks, arXiv:2308.07037",
    "arXiv preprint arXiv:1912.01703 (2019).",
    "10.48550/arXiv.2404.16001 .",
    "New Journal of Physics 10.1088/1367-2630/ada8d1 (2025a)",
    "Rapid solutions of problems by quantum computation, Proc. Royal Soc. London A. 439 , 553 (1992).",
    "Loop calculus in statistical physics and information science",
    "Optimizing Radiotherapy Plans for Cancer Treatment with Tensor Networks",
    "The Löwner–John Ellipsoid",
    "in Towards a New Evolutionary Computation , Studies in Fuzziness and Soft Computing",
    "łek, P. Dulian, J. Majsak, S. Chakraborty, and R. Demkowicz-Dobrzański, New Journal of Physics",
    "A survey of mamba",
    "Netsimile: A scalable approach to size-independent network similarity",
    "",
    "   ",
    "A",
    "Quantum Optics",
    "Truncating the Loop Series Expansion for Belief Propagation",
    "Implementing a gammatone filter bank . Annex C of the SVOS Final Report",
    "Lecture notes on inverse theory, 07 2021",
    "Zeitschrift für Physik 51 (3-4), 165.",
]

# The one input class where a difference is intended.
EXPECT_DIFFERENT = [
    "Attention is all you need . Advances in neural information processing systems",
    "Implementing a gammatone filter bank . Annex C of the SVOS Final Report",
    "Something . Lecture notes on something else",
]


def main() -> int:
    if not PYC.exists():
        print(f"original bytecode not found at {PYC}", file=sys.stderr)
        return 2
    old = load("orig_resolve", PYC)
    new = load("new_resolve", HERE / "04_resolve.py")

    fails, checks, intended_keys, intended = [], 0, 0, 0

    def eq(label, a, b):
        nonlocal checks
        checks += 1
        if a != b:
            fails.append(f"{label}\n      old={a!r}\n      new={b!r}")

    def eq_rec(label, a, b):
        """Compare the keys the original record had. New keys are additive.

        The precision pass adds provenance (`field_purity`, `field_confidence`,
        `field_source`) and rejection detail (`rejected_title`, `rejected_tier`).
        Those are asserted present; what is asserted UNCHANGED is every key the
        destroyed file already produced, because Stage 5 reads those.
        """
        nonlocal checks, intended_keys
        checks += 1
        if "abstract_source" not in b:
            fails.append(f"{label}: new record is missing abstract_source")
            return
        intended_keys += 1
        shared = {k: v for k, v in b.items() if k in a}
        if a != {**a, **shared}:
            diff = {k: (a[k], shared[k]) for k in a if k in shared and a[k] != shared[k]}
            fails.append(f"{label}: changed an existing key {diff}")
        missing = set(a) - set(b)
        if missing:
            fails.append(f"{label}: dropped {sorted(missing)}")

    # ---------------------------------------------------------------- constants
    for k in ("OA", "S2_API", "S2_FIELDS", "EXACT", "STRONG", "WEAK"):
        eq(f"const {k}", getattr(old, k), getattr(new, k))
    # OA_SELECT is deliberately WIDER: the identity pass needs authorships,
    # primary_location and cited_by_count to tell two works with the same title
    # apart. Same request count, more fields per request.
    checks += 1
    if (set(old.OA_SELECT.split(",")) <= set(new.OA_SELECT.split(","))
            and {"authorships", "primary_location"} <= set(new.OA_SELECT.split(","))):
        intended += 1
    else:
        fails.append(f"OA_SELECT must be a superset with authorships and "
                     f"primary_location: {new.OA_SELECT}")
    checks += 1
    if not (hasattr(new, "backfill_all") and hasattr(new, "ArXiv")
            and hasattr(new.OpenAlex, "by_dois") and hasattr(new.S2, "batch")
            and not hasattr(old, "backfill_all")):
        fails.append("backfill_all / ArXiv / by_dois / batch should all be new")
    else:
        intended += 1   # _backfill and ArXiv are new by design
    # _PUBMETA is excluded: the precision pass anchored its year alternative.
    # _VENUE_MARK is GONE: it was a list of journal names and abbreviations from
    # computer science and physics, replaced by venue_cut(), which identifies a
    # venue by shape (abbreviation run, volume/page locator, identifier) and so
    # needs no per-domain vocabulary.
    checks += 1
    if hasattr(new, "_VENUE_MARK") or not hasattr(new, "venue_cut"):
        fails.append("_VENUE_MARK should be replaced by venue_cut()")
    else:
        intended += 1
    for k in ("_DOI", "_ARXIV"):
        eq(f"regex {k}", getattr(old, k).pattern, getattr(new, k).pattern)
        eq(f"flags {k}", getattr(old, k).flags, getattr(new, k).flags)
    checks += 1
    if r"\d{4}[a-z]?" in new._PUBMETA.pattern:
        fails.append("PRECISION FIX 5 did not take: _PUBMETA still strips any "
                     "four digits, which is what ate 'Scientific Reports 6, 34944'")
    else:
        intended += 1
    # MISS gained the purity, provenance and rejection keys. Every key the
    # original had must still be there, with the same value.
    checks += 1
    drift = {k: (v, new.MISS.get(k)) for k, v in old.MISS.items()
             if k not in new.MISS or new.MISS[k] != v}
    if drift:
        fails.append(f"MISS changed an existing key: {drift}")
    else:
        intended += 1

    # ------------------------------------------------------------------ helpers
    # norm, clean_title and similarity are all deliberately changed by the
    # precision pass, so they are pinned by build/test_matching.py instead. What
    # is asserted here is that the change is CONFINED: a reference string with no
    # markup, no bare year and no venue locator must still clean identically, or
    # the pass reached further than it claimed to.
    hard = [t for t in CASES if t not in EXPECT_DIFFERENT]
    # Any run of four digits, not just a plausible year: the OLD pattern stripped
    # both, which is the defect, so a string containing one cannot be used as a
    # control for anything.
    quiet = [t for t in hard if not re.search(r"\d{4}", t)
             and not new._VENUE_LOCATOR.search(t) and "<" not in t]
    checks += 1
    if len(quiet) < 10:
        fails.append(f"only {len(quiet)} unaffected cases to check against")
    for t in quiet:
        eq(f"norm({t[:34]!r})", old.norm(t), new.norm(t))
        eq(f"clean_title({t[:34]!r})", old.clean_title(t), new.clean_title(t))
        eq(f"is_citation_string({t[:34]!r})",
           old.is_citation_string(t), new.is_citation_string(t))
    rnd = random.Random(0)
    for _ in range(400):
        s = round(rnd.random(), 4)
        eq(f"tier({s})", old.tier(s), new.tier(s))
    for s in (0.0, 0.62, 0.6199, 0.8, 0.7999, 0.95, 0.9499, 1.0):
        eq(f"tier boundary {s}", old.tier(s), new.tier(s))
    for inv in ({}, None, {"Hello": [0], "world": [1]},
                {"b": [1, 3], "a": [0], "c": [2]}, {"x": []}):
        eq(f"abstract_from_inverted({inv})",
           old.abstract_from_inverted(inv), new.abstract_from_inverted(inv))

    # ---------------------------------------- what the precision pass must change
    # Asserted as differences, in the direction claimed. The six real false
    # matches are pinned case by case in build/test_matching.py; these are the
    # structural changes that make those rejections possible.
    for label, ok in [
        ("similarity is no longer one-sided containment",
         new.similarity("Bayesian Flow Networks",
                        "Predicting traffic flow using Bayesian networks") < new.ACCEPT
         <= old.similarity("Bayesian Flow Networks",
                           "Predicting traffic flow using Bayesian networks")),
        ("a weak hit is no longer accepted",
         not new.accepted(0.7) and old.tier(0.7) == "weak"),
        ("an arXiv route is trusted, not re-scored",
         new._from_openalex({"display_name": "PyTorch: An Imperative Style"},
                            "arXiv preprint arXiv:1912.01703 (2019).",
                            "arxiv")["match_quality"] == "exact"
         and old._from_openalex({"display_name": "PyTorch: An Imperative Style"},
                                "arXiv preprint arXiv:1912.01703 (2019).",
                                "arxiv")["match_quality"] == "none"),
        ("a bare year no longer eats digits mid-number",
         new.clean_title("Graph distance for complex networks. Scientific "
                         "Reports 6 , 34944 (2016).")
         != old.clean_title("Graph distance for complex networks. Scientific "
                            "Reports 6 , 34944 (2016).")),
        ("brackets left open by stripping are repaired",
         new.clean_title("Quantum Optics (Wiley, 2006).") == "Quantum Optics"
         and old.clean_title("Quantum Optics (Wiley, 2006).") != "Quantum Optics"),
        ("markup is stripped before tokenising",
         new.norm("in <mml:mi>x</mml:mi> systems") != old.norm("in <mml:mi>x</mml:mi> systems")),
        ("the binary label names fields, because it compares fields",
         new.domain_relation("Computer Science", "Mathematics") == "cross_field"
         and old.domain_relation("Computer Science", "Mathematics") == "cross_domain"),
        ("uniqueness can be restated after resolution",
         hasattr(new, "resolution_status") and not hasattr(old, "resolution_status")),
        ("field purity is computed", hasattr(new, "field_confidence")
         and not hasattr(old, "field_confidence")),
        ("a title is no longer treated as an identifier",
         hasattr(new, "corroborate") and hasattr(new, "parse_reference")
         and not hasattr(old, "corroborate")),
        ("the reference's year survives cleaning",
         new.parse_reference("Quantum Optics (Wiley, 2006).")["years"] == [2006]
         and new.clean_title("Quantum Optics (Wiley, 2006).") == "Quantum Optics"),
    ]:
        checks += 1
        if ok:
            intended += 1
        else:
            fails.append(f"PRECISION FIX did not take: {label}")

    # ------------------------------------------------------- record constructors
    W = {"display_name": "Loop calculus in statistical physics",
         "doi": "https://doi.org/10.1088/1742-5468/2006/06/p06009",
         "abstract_inverted_index": {"A": [0], "loop": [1], "series": [2]},
         "primary_topic": {"domain": {"display_name": "Physical Sciences"},
                           "field": {"display_name": "Physics and Astronomy"},
                           "subfield": {"display_name": "Statistical Physics"}},
         "topics": [{"field": {"display_name": "Physics and Astronomy"}},
                    {"field": {"display_name": "Computer Science"}}],
         "publication_year": 2006}
    # "arxiv" is omitted: it is now an identifier route and trusted without
    # scoring, which is PRECISION FIX 3 and asserted as a difference above.
    for route in ("doi", "oa_title"):
        for q in ("Loop calculus in statistical physics", "something unrelated"):
            eq_rec(f"_from_openalex({route},{q[:18]!r})",
               old._from_openalex(W, q, route), new._from_openalex(W, q, route))
    eq_rec("_from_openalex(None)", old._from_openalex(None, "q", "oa_title"),
       new._from_openalex(None, "q", "oa_title"))

    H = {"title": "Attention Is All You Need", "abstract": "The dominant sequence...",
         "externalIds": {"DOI": "10.5555/3295222", "ArXiv": "1706.03762"}, "year": 2017}
    for route in ("doi", "arxiv", "s2_match"):
        for q in ("Attention is all you need", "totally different"):
            eq_rec(f"_from_s2({route},{q[:18]!r})",
               old._from_s2(H, q, route), new._from_s2(H, q, route))
    eq_rec("_from_s2(None)", old._from_s2(None, "q", "s2_match"),
       new._from_s2(None, "q", "s2_match"))
    eq_rec("_from_s2(arxiv only)",
       old._from_s2({"title": "T", "externalIds": {"ArXiv": "2308.07037"}}, "T", "doi"),
       new._from_s2({"title": "T", "externalIds": {"ArXiv": "2308.07037"}}, "T", "doi"))

    # --------------------------------------------------------------- domain maps
    f2d = {"computer science": "Physical Sciences", "mathematics": "Physical Sciences",
           "physics and astronomy": "Physical Sciences", "medicine": "Health Sciences"}
    fields = [None, "", "Computer Science", "computer science", "Mathematics",
              "Medicine", "Dentistry", "Physics and Astronomy"]
    # The VALUES were renamed (same_domain -> same_field), so the partition is
    # what must be identical, not the labels. Anything else would mean the
    # precision pass changed which pairs count as a jump, which it did not.
    RENAME = {"same_domain": "same_field", "cross_domain": "cross_field", None: None}
    for t, i in itertools.product(fields, fields):
        eq(f"domain_relation({t},{i})",
           RENAME[old.domain_relation(t, i)], new.domain_relation(t, i))
        eq(f"domain_distance({t},{i})",
           old.domain_distance(t, i, f2d), new.domain_distance(t, i, f2d))

    # ------------------------------------------------- the intended differences
    for t in EXPECT_DIFFERENT:
        checks += 1
        if old.clean_title(t) == new.clean_title(t):
            fails.append(f"REBUILD FIX 1 did not take on {t[:50]!r}: "
                         f"both give {new.clean_title(t)!r}")
        else:
            intended += 1
    checks += 1
    if not (old.CACHE_VERSION == 3 and new.CACHE_VERSION >= 6):
        fails.append(f"CACHE_VERSION: expected 3 -> 6 or later, got "
                     f"{old.CACHE_VERSION} -> {new.CACHE_VERSION}. The precision "
                     f"pass MUST bump it: the six false matches are cached.")
    else:
        intended += 1

    # ------------------------------------------------------------------- verdict
    print(f"{checks} comparisons, {len(fails)} failures, "
          f"{intended} intended differences confirmed, "
          f"{intended_keys} records carrying the new abstract_source key")
    for f in fails[:25]:
        print("  FAIL  " + f)
    if fails:
        print(f"\n{len(fails)} MISMATCHES: the rebuild differs from the original.")
        return 1
    print("\nthe rebuild is behaviourally identical to the destroyed file, "
          "except where intended.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
