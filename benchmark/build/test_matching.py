#!/usr/bin/env python3
"""Regression test for the Stage 4 precision pass.

Every case below is a real string from data/_snap/run_B_resolved.jsonl, paired
with the paper the old matcher actually returned for it. The audit that prompted
this found six wrong gold documents across nine slots in five exportable papers,
all scored 0.97 ("exact") by a containment rule that never charged a candidate
for the words it added.

These are offline: the API responses are already known, so the only thing under
test is the scoring, cleaning and acceptance logic.

    python build/test_matching.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "resolve", Path(__file__).resolve().parent / "04_resolve.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


# ---------------------------------------------------------------- the six wrong
# (reference string, what the old matcher returned, why it is a different paper)
FALSE_MATCHES = [
    ("Bayesian Flow Networks",
     "Predicting traffic flow using Bayesian networks",
     "generative model vs a traffic-forecasting application"),
    ("A bound on the independent domination number of a tree",
     "A lower bound on the total outer-independent domination number of a tree",
     "independent domination vs total outer-independent domination"),
    ("Mathematical Foundations of Supervised Learning",
     "Mathematical Foundations of Graph-Based Bayesian Semi-Supervised Learning",
     "supervised vs graph-based semi-supervised"),
    ("Quantum Optics (Wiley, 2006).",
     "W. H. Louisell : Quantum Statistical Properties of Radiation, John Wiley, "
     "New York and London, 1973, 528ページ, 23.5×16cm, 10,170円 "
     "(Wiley Series in Pure and Applied Optics).",
     "Vogel and Welsch's book vs a review of a different Louisell book"),
    ("The Bayesian learning rule",
     "Training Binary Neural Networks using the Bayesian Learning Rule",
     "the JMLR article vs one application of it"),
    ("On maximal theories",
     "Completeness for the paraconsistent logic CG'3 based on maximal theories",
     "the 1991 JSL article vs an unrelated paraconsistent-logic paper"),
]

# Pairs that must still resolve. A precision fix that rejects everything is not a
# fix, so the same test carries the matches worth keeping.
TRUE_MATCHES = [
    ("Attention is all you need", "Attention Is All You Need"),
    ("Attention is all you need . Advances in neural information processing systems",
     "Attention Is All You Need"),
    ("Deep Residual Learning for Image Recognition.",
     "Deep Residual Learning for Image Recognition"),
    ("Quantum Computation and Quantum Information (Cambridge University Press, "
     "Cambridge, 2000).", "Quantum Computation and Quantum Information"),
    ("Loop calculus in statistical physics and information science",
     "Loop Calculus in Statistical Physics and Information Science"),
    ("Netsimile: A scalable approach to size-independent network similarity",
     "NetSimile: A Scalable Approach to Size-Independent Network Similarity"),
    ("Truncating the Loop Series Expansion for Belief Propagation",
     "Truncating the loop series expansion for belief propagation"),
    ("Bayesian Flow Networks, arXiv:2308.07037", "Bayesian Flow Networks"),
    # PRECISION FIX 11. One preposition apart, and the same paper. Scored 0.776
    # and was rejected until function words stopped counting as evidence.
    ("Mathematical Foundations of Supervised Learning",
     "Mathematical Foundations for Supervised Learning"),
    ("Deep Learning on Spatiotemporal Graphs",
     "Deep Learning for Spatiotemporal Graphs"),
]

# Reference strings the cleaner used to corrupt, and what it must leave.
CLEANING = [
    ("Graph distance for complex networks. Scientific Reports 6 , 34944 (2016).",
     "Graph distance for complex networks"),
    ("Quantum Optics (Wiley, 2006).", "Quantum Optics"),
    ("PRX Quantum 2 , 010343 (2021) .", "PRX Quantum 2 , 010343"),
    ("Attention is all you need", "Attention is all you need"),
    ("Quantum Computation and Quantum Information (Cambridge University Press, "
     "Cambridge, 2000).", "Quantum Computation and Quantum Information"),
]

# Identifier routes must be believed without re-scoring the reference string.
# arXiv:1912.01703 is the PyTorch paper; the reference string that carries it
# looks nothing like the title, which is why it was being thrown away.
ID_CASES = [
    ({"display_name": "PyTorch: An Imperative Style, High-Performance Deep "
                      "Learning Library", "doi": "https://doi.org/10.48550/arxiv.1912.01703"},
     "arXiv preprint arXiv:1912.01703 (2019).", "arxiv"),
    ({"display_name": "Attention Is All You Need", "doi": "https://doi.org/10.5555/3295222"},
     "some entirely unrelated reference string", "doi"),
]

# A field backed by one of three OpenAlex topics is a plurality of one.
PURITY = [
    ({"insp_field": "Computer Science",
      "insp_topic_fields": {"Computer Science": 3}}, "clear"),
    ({"insp_field": "Computer Science",
      "insp_topic_fields": {"Computer Science": 2, "Mathematics": 1}}, "clear"),
    ({"insp_field": "Computer Science",
      "insp_topic_fields": {"Computer Science": 1, "Mathematics": 1,
                            "Engineering": 1}}, "ambiguous"),
    ({"insp_field": "Computer Science", "insp_topic_fields": None}, None),
]


def main() -> int:
    fails: list[str] = []

    print("1. the six false matches must no longer be accepted")
    for ref, wrong, why in FALSE_MATCHES:
        s = R.similarity(R.clean_title(ref), wrong)
        ok = not R.accepted(s)
        print(f"   {'PASS' if ok else 'FAIL'}  {s:.3f} {R.tier(s):<6} "
              f"{ref[:38]:<38} -> {wrong[:36]}")
        if not ok:
            fails.append(f"still accepted at {s:.3f}: {ref!r} -> {wrong!r} ({why})")

    print("\n2. real matches must survive")
    for ref, right in TRUE_MATCHES:
        s = R.similarity(R.clean_title(ref), right)
        ok = R.accepted(s)
        print(f"   {'PASS' if ok else 'FAIL'}  {s:.3f} {R.tier(s):<6} "
              f"{ref[:38]:<38} -> {right[:36]}")
        if not ok:
            fails.append(f"lost a true match at {s:.3f}: {ref!r} -> {right!r}")

    print("\n3. cleaning must not corrupt the reference")
    for raw, want in CLEANING:
        got = R.clean_title(raw)
        ok = got == want
        print(f"   {'PASS' if ok else 'FAIL'}  {raw[:44]:<44} -> {got!r}")
        if not ok:
            fails.append(f"clean_title({raw!r}) = {got!r}, expected {want!r}")

    print("\n4. an identifier is trusted, never re-scored")
    for work, ref, route in ID_CASES:
        rec = R._from_openalex(work, ref, route)
        ok = rec["match_quality"] == "exact" and rec["similarity"] == 1.0
        print(f"   {'PASS' if ok else 'FAIL'}  route={route:<6} "
              f"{rec['match_quality']:<6} {str(rec['found_title'])[:44]}")
        if not ok:
            fails.append(f"identifier route {route} was re-scored: {rec['similarity']}")

    print("\n5. field purity below 2/3 is ambiguous")
    for rec, want in PURITY:
        got = R.field_confidence(rec)
        ok = got == want
        print(f"   {'PASS' if ok else 'FAIL'}  {str(rec['insp_topic_fields']):<52} "
              f"-> {got}")
        if not ok:
            fails.append(f"field_confidence({rec}) = {got}, expected {want}")

    print("\n6. a weak hit is held for review, not written as a gold")
    weak = R._from_openalex(
        {"display_name": "Predicting traffic flow using Bayesian networks"},
        "Bayesian Flow Networks", "oa_title")
    ok = (weak["match_quality"] == "none" and weak["found_title"] is None
          and weak["rejected_title"] == "Predicting traffic flow using Bayesian networks")
    print(f"   {'PASS' if ok else 'FAIL'}  match_quality={weak['match_quality']} "
          f"found_title={weak['found_title']} rejected={str(weak['rejected_title'])[:36]}")
    if not ok:
        fails.append(f"weak hit was not held for review: {weak}")

    print("\n7. the binary label says field, because it compares fields")
    rel = R.domain_relation("Computer Science", "Physics and Astronomy")
    same = R.domain_relation("Computer Science", "Computer Science")
    ok = rel == "cross_field" and same == "same_field"
    print(f"   {'PASS' if ok else 'FAIL'}  cs->physics={rel}  cs->cs={same}")
    if not ok:
        fails.append(f"domain_relation still names domains: {rel}, {same}")

    print("\n8. uniqueness is restated against what resolved")
    done = {"match_quality": "exact"}
    gone = {"match_quality": "none"}
    cases = [
        ({"uniqueness": {"uniqueness": "falsified", "M": [
            {"inspiration": [done, done]}, {"inspiration": [done]}]}}, "falsified"),
        ({"uniqueness": {"uniqueness": "falsified", "M": [
            {"inspiration": [done, gone]}, {"inspiration": [done]}]}},
         "resolution_incomplete"),
        ({"uniqueness": {"uniqueness": "not_falsified", "M": []}}, "not_falsified"),
    ]
    for rec, want in cases:
        got = R.resolution_status(rec)
        ok = got == want
        print(f"   {'PASS' if ok else 'FAIL'}  {want:<22} -> {got}")
        if not ok:
            fails.append(f"resolution_status = {got}, expected {want}")

    fails += check_identity()
    fails += check_smoke()

    print(f"\n{'=' * 70}")
    if fails:
        print(f"{len(fails)} FAILURES\n")
        for f in fails:
            print("  " + f)
        return 1
    print("all precision-pass checks green")
    return 0




# ============================================================ identity, not title
# IDENTITY CASES. Four wrong papers were accepted at similarity 1.0 because their
# titles matched exactly. A title is not an identifier: books have editions,
# conference papers become journal articles, and review indexes carry the
# reviewed work's title. Candidates below are the real OpenAlex records.
#
# NOTE what this replaces. The CLEANING table above asserts
#   "Quantum Optics (Wiley, 2006)." -> "Quantum Optics"
# which is correct as CLEANING but was doing real harm as a TEST: it encoded
# throwing away the publisher and year, the only two fields that separate the
# Wiley 2006 book from the Cambridge 1997 one. The cleaner still strips them;
# parse_reference now keeps a copy first.

def _w(title, year, doi, typ="article", authors=(), venue=None, cited=0):
    return {"display_name": title, "publication_year": year,
            "doi": f"https://doi.org/{doi}" if doi else None, "type": typ,
            "cited_by_count": cited,
            "authorships": [{"author": {"display_name": a}} for a in authors],
            "primary_location": {"source": {"display_name": venue}} if venue else None}


IDENTITY = [
    # (reference string, candidates, expected outcome, expected doi fragment)
    ("Quantum Optics (Wiley, 2006).",
     [_w("Quantum Optics", 1997, "10.1017/cbo9780511813993", "book",
         ("Marlan O. Scully", "M. Suhail Zubairy"), "Cambridge University Press", 9000),
      _w("Quantum Optics", 2006, "10.1002/3527608524", "book",
         ("Werner Vogel", "Dirk-Gunnar Welsch"), "Wiley-VCH", 900)],
     "accept", "10.1002/3527608524"),

    ("Modern computer algebra",
     [_w("Modern computer algebra", 2000, "10.5860/choice.37-5723", "review",
         (), "Choice Reviews Online", 40),
      _w("Modern Computer Algebra", 2013, "10.1017/cbo9781139856065", "book",
         ("Joachim von zur Gathen", "Jurgen Gerhard"), "Cambridge University Press", 3000)],
     "accept", "10.1017/cbo9781139856065"),

    ("Modal Model Theory",
     [_w("Modal model theory", 1973, "10.1007/bfb0066792", "book-chapter",
         ("Chen Chung Chang",), "Lecture Notes in Mathematics", 30),
      _w("Modal Model Theory", 2024, "10.1215/00294527-2024-0001", "article",
         ("Joel David Hamkins", "Wojciech Aleksander Woloszyn"),
         "Notre Dame Journal of Formal Logic", 12)],
     # Chang 1973 and Hamkins & Woloszyn 2024 are different papers with the same
     # title, and the reference says nothing that separates them. Asserting WHICH
     # one is correct would be asserting knowledge the resolver does not have.
     # What is asserted is that it resolves rather than vanishing, and that it
     # says so: basis "title_only", with the loser recorded under passed_over.
     "flagged", None),

    # The 2008 conference paper and the 2014 JMLR article share their authors,
    # so they are two records of one line of work rather than two different
    # papers. The published, more-cited one wins, which is the JMLR article the
    # reference meant.
    ("Natural evolution strategies",
     [_w("Natural Evolution Strategies", 2008, None, "article",
         ("Daan Wierstra", "Tom Schaul"), "IEEE Congress on Evolutionary Computation", 500),
      _w("Natural Evolution Strategies", 2014, "10.5555/2627435.2638566", "article",
         ("Daan Wierstra", "Tom Schaul"), "Journal of Machine Learning Research", 1200)],
     "accept", "10.5555/2627435.2638566"),

    # THE 33-POINT REGRESSION. OpenAlex holds the preprint and the published
    # version as separate works with identical titles. The first identity pass
    # called that "2 works share this title" and refused to choose, which took
    # resolution from 87.8% to 55.1% and put 104 perfect matches in the review
    # file. One work, two records; the published one wins.
    ("Relational knowledge distillation",
     [_w("Relational Knowledge Distillation", 2019, "10.1109/cvpr.2019.00409",
         "article", ("Wonpyo Park", "Dongju Kim"), "CVPR", 1200),
      _w("Relational Knowledge Distillation", 2019, "10.48550/arxiv.1904.05068",
         "preprint", ("Wonpyo Park", "Dongju Kim"), "arXiv", 300)],
     "accept", "10.1109/cvpr.2019.00409"),

    # Same shape, but the reference names nothing and the two records disagree
    # on everything. Still held.
    ("Quantum Optics",
     [_w("Quantum Optics", 1997, "10.1017/cbo9780511813993", "book",
         ("Marlan O. Scully",), "Cambridge University Press", 9000),
      _w("Quantum Optics", 2006, "10.1002/3527608524", "book",
         ("Werner Vogel",), "Wiley-VCH", 900)],
     # A bare title naming two real books. Resolved to the better-ranked one and
     # flagged `title_only`, because a missing gold is a silent hole and a noisy
     # one is visible. The Wiley/2006 version above still resolves correctly,
     # since there the reference DOES say which book it means.
     "flagged", None),

    # A control. One work, one title, nothing ambiguous: must still resolve.
    ("Attention is all you need",
     [_w("Attention Is All You Need", 2017, "10.5555/3295222", "article",
         ("Ashish Vaswani",), "NeurIPS", 100000)],
     "accept", "10.5555/3295222"),
]


class _FakeOA:
    """Stands in for OpenAlex.by_title so the four cases need no network."""

    def __init__(self, cands):
        self.cands = cands

    def by_title(self, title, n=8):
        return self.cands

    def by_doi(self, doi):
        return None


class _FakeS2:
    def match(self, title):
        return None

    def by_id(self, ident):
        return None


def check_identity() -> list[str]:
    fails = []
    print("\n9. a title several works share does not identify one of them")
    for ref, cands, want, doi_frag in IDENTITY:
        out = R._lookup(ref, _FakeOA(cands), _FakeS2())
        basis = (out.get("identity") or {}).get("basis")
        got = ("review" if out["match_quality"] == "none" else
               "flagged" if basis == "title_only" else "accept")
        ok = got == want
        if ok and want == "accept":
            ok = doi_frag in str(out.get("found_doi") or "")
        if ok and want == "flagged":
            ok = bool((out.get("identity") or {}).get("passed_over"))
        detail = (f"held: {str(out.get('rejected_reason'))[:48]}" if got == "review"
                  else f"-> {out.get('found_doi')}"
                       + (f"  (passed over "
                          f"{len((out.get('identity') or {}).get('passed_over') or [])})"
                          if got == "flagged" else ""))
        print(f"   {'PASS' if ok else 'FAIL'}  {want:<7} {ref[:32]:<32} {detail}")
        if not ok:
            fails.append(f"{ref!r}: wanted {want} {doi_frag}, got {got} "
                         f"{out.get('found_doi')} / {out.get('rejected_reason')}")
    return fails


def check_smoke() -> list[str]:
    """Call the things a real run calls, with no network.

    `python -m py_compile` does not catch a NameError inside a function body,
    and neither did the two suites: `matcher_fingerprint` referenced a regex
    that had been deleted, and the whole stage crashed on its first line for the
    user rather than for me. Anything reachable before the first HTTP request
    gets exercised here.
    """
    import tempfile
    fails = []
    print("\n10. the module runs end to end with no network")
    checks = [
        ("matcher_fingerprint", lambda: R.matcher_fingerprint()),
        ("Http + cache round trip", lambda: _http_roundtrip(tempfile)),
        ("report on an empty run", lambda: R.report([], R.L.DEFAULT_POLICY)),
        ("backfill with nothing to do", lambda: R.backfill_all({}, None, None, None)),
        ("resolution_status", lambda: R.resolution_status({})),
        ("all_inspirations", lambda: R.all_inspirations({})),
        # The exact shape that killed the full run 1,194 titles in: an OpenAlex
        # author whose display_name is whitespace. Truthy, but splits to [].
        ("surnames_of on blank author names",
         lambda: R.surnames_of({"authorships": [
             {"author": {"display_name": " "}},
             {"author": {"display_name": ""}},
             {"author": {"display_name": None}},
             {"author": {}}, {}, {"author": {"display_name": "Ada Lovelace"}}]})),
        ("corroborate against a blank-author work",
         lambda: R.corroborate(
             R.parse_reference("A. Smith, Some Title, J. Foo 1, 2 (2020)."),
             {"display_name": "Some Title", "publication_year": 2020,
              "authorships": [{"author": {"display_name": " "}}]})),
        ("domain_distance", lambda: R.domain_distance("a", "b", {})),
        ("abstract_from_inverted", lambda: R.abstract_from_inverted({"x": [0]})),
    ]
    for name, fn in checks:
        try:
            fn()
            print(f"   PASS  {name}")
        except Exception as exc:
            print(f"   FAIL  {name}: {exc.__class__.__name__}: {exc}")
            fails.append(f"{name} raised {exc.__class__.__name__}: {exc}")
    return fails


def _http_roundtrip(tempfile):
    """Construct Http, save, and reload: the path that crashed on the user."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "c.json"
        h = R.Http(p, 0.0)
        h.cache["k"] = {"found_title": "t"}
        h.save()
        again = R.Http(p, 0.0)
        assert again.cache.get("k"), "a cache written by this code must reload"


if __name__ == "__main__":
    sys.exit(main())
