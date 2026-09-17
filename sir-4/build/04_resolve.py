#!/usr/bin/env python3
"""Stage 4. Resolve each gold inspiration to a real paper, and label its domain.

Stage 3 gives every inspiration a `supposed_title` copied out of the target
paper's reference list. That string is not a document: it cannot be retrieved,
it has no abstract, and the gold documents of this benchmark ARE the resolved
papers. Until this stage runs there is no corpus and no retrieval result.

Fills the five schema fields Stage 3 leaves as None:

    found_title  found_abstract  found_doi  match_quality  similarity

and the domain labels:

    insp_domain / insp_field / insp_subfield   OpenAlex's three levels
    domain_relation   "same_field" | "cross_field"         binary, at field level
    domain_distance   "same_field" | "cross_field_same_domain" | "cross_domain"
    field_purity / field_confidence            is that field label load-bearing

OpenAlex is primary, not a fallback. It has the domain labels, it reconstructs
abstracts for papers Semantic Scholar has none for, the corpus is already
OpenAlex-native, and its polite pool is 100k requests a day against a Semantic
Scholar pool that returns 429 on the first burst. S2 is kept for what OpenAlex
misses, which measured 24 of 178 on the first real run.

Resolution order, most certain first:

    1. a DOI in the string        -> OpenAlex works/doi:...
    2. an arXiv id in the string  -> OpenAlex, then S2
    3. the cleaned title          -> OpenAlex title.search, best hit scored here
    4. the cleaned title          -> Semantic Scholar's title matcher

Every lookup is cached on disk under a schema version. Changing the matching
logic bumps the version and invalidates the cache, rather than serving verdicts
computed by code that no longer exists.

    export OPENALEX_MAILTO=you@ic.ac.uk      # polite pool
    export OPENALEX_API_KEY=...              # premium, optional
    export S2_API_KEY=...                    # fallback only, optional

    python build/04_resolve.py --input  data/_snap/run_B.jsonl \
                               --output data/_snap/run_B_resolved.jsonl
    python build/04_resolve.py --input ... --limit 10 --dry-run

PROVENANCE. The previous copy of this file was destroyed on 2026-08-02 by a
shell redirect that pointed a run log at the source path. It was rebuilt from
build/__pycache__/04_resolve.cpython-313.pyc, which was current, and the pure
functions are differentially tested against that bytecode by
build/verify_resolve.py. Three deliberate changes were made during the rebuild,
marked REBUILD FIX below.

PRECISION PASS (2026-08-02, marked PRECISION FIX below). An audit of the first
complete run found six wrong papers accepted as gold documents, e.g. the
reference "Bayesian Flow Networks" resolving to "Predicting traffic flow using
Bayesian networks". Every one came from the same place: a containment score that
rewarded the reference's words appearing in the candidate and never charged for
the candidate's extra words. Nine slots across five exportable papers carried a
wrong gold. The fixes are all deterministic and cost no extra API calls:

    1  token F1 instead of one-sided containment            similarity()
    2  short titles must agree almost exactly               similarity()
    3  a DOI or arXiv id is trusted, never re-scored        ID_ROUTES
    4  a weak hit is a review candidate, not a gold         accepted()
    5  year stripping no longer eats digits mid-number      _PUBMETA, _YEAR
    6  brackets left unbalanced by stripping are repaired   balance_brackets()
    7  full provenance reaches the output file              phase 3
    8  a field backed by 1 of 3 topics is "ambiguous"       field_confidence()
    9  M is restated after resolution, never silently       resolution_status()
   10  output is written atomically, never after an abort   main()
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels as L          # the same/cross definition, shared with Stage 5

ROOT = Path(__file__).resolve().parent.parent

CACHE_DEFAULT = ROOT / "data" / "_cache" / "resolve_titles.json"

# Bumped whenever matching, cleaning or scoring changes. A stale cache that
# silently serves old verdicts is worse than no cache: the first version of this
# file cached a batch of rate-limit failures as genuine "not found" results.
# 5 -> 6: the precision pass changes cleaning, scoring AND acceptance, so every
# verdict on disk was computed by rules that no longer exist. The six known false
# matches are all sitting in the v5 cache and would otherwise be served straight
# back out.
# 6 -> 7: similarity() now ignores function words, which changes scores and so
# changes verdicts. Cheap to redo: the 100-paper run was 312 distinct titles.
# 7 -> 8: the identity pass. Title equality is no longer a verdict, so every
# cached title-route match was decided by a rule that no longer exists, and the
# four known wrong identities are sitting in the v7 cache.
# 8 -> 9: the identity pass rejected duplicate RECORDS of one work as though
# they were different works, and a rerun without this bump served all 307 of
# those verdicts straight back out of the v8 cache. The cache stores VERDICTS:
# any change to matching logic must bump this, or the change does not run.
CACHE_VERSION = 9

OA = "https://api.openalex.org"
# IDENTITY FIX. `authorships` and `primary_location` are new, and they are the
# whole point: a title is not an identifier. Same request, wider select, so the
# call count does not change at all.
# TWO SELECTS, because a title search downloads every candidate and keeps one.
#
# `abstract_inverted_index` is the bulk of a work's payload and `topics` is most
# of the rest, and neither is needed to decide WHICH candidate is the right
# paper. Disambiguation needs the title, year, authors, type, venue and citation
# count; the heavy fields matter only for the winner. Searching lean and then
# fetching the chosen work by DOI, batched 50 at a time, costs about 2% more
# requests for roughly a tenth of the bytes.
OA_SELECT = ("id,doi,display_name,abstract_inverted_index,primary_topic,"
             "topics,publication_year,type,authorships,primary_location,"
             "cited_by_count")
OA_SELECT_LEAN = ("id,doi,display_name,publication_year,type,authorships,"
                  "primary_location,cited_by_count")
# Named S2_API, not S2: `class S2` below would rebind a constant called S2,
# and the f-strings in its methods would then interpolate the class object.
S2_API = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS = "title,abstract,externalIds,year,venue"

EXACT, STRONG, WEAK = (0.95, 0.8, 0.62)

# PRECISION FIX 4. Where a match stops being a gold document and becomes a thing
# for a human to look at. A weak hit used to be written into the benchmark as a
# gold, which is how "On maximal theories" acquired a paraconsistent-logic paper.
# Below this the candidate is kept on the record, under `rejected_title`, and the
# slot is reported unresolved.
ACCEPT = STRONG

# PRECISION FIX 2. At or below this many distinct words a title carries too
# little information for word overlap to identify anything: every word of
# "Quantum Optics" appears in a dozen unrelated papers. Short references are held
# to sequence agreement alone.
SHORT_TITLE = 4

# An identifier is not evidence about the title, it IS the paper. Re-scoring one
# against a reference string can only lose: arXiv:1912.01703 names the PyTorch
# paper exactly and was being thrown away because the reference string it came
# from did not look enough like "PyTorch: An Imperative Style...".
ID_ROUTES = frozenset({"doi", "arxiv", "arxiv_s2"})

# PRECISION FIX 8. OpenAlex gives a work three topics, each with a field. When
# only one of the three carries the field we label the paper with, that label is
# a plurality of one and should not be counted in a same/cross split.
PURITY_MIN = 2 / 3

# PRECISION FIX 5. A YEAR, not any four digits. The old pattern was a bare
# `\d{4}`, applied up to three times, which chewed through real numbers from the
# right: "Graph distance ... 34944 (2016)" lost "(2016)", then "4944", and ended
# as "Graph distance ... 3". Anchored on both sides so it can only take a whole
# number, and restricted to a plausible publication year.
_YEAR = r"(?<!\d)(?:1[6-9]|20)\d{2}[a-z]?(?!\d)"

# Publisher and place-of-publication furniture at the end of a reference string.
_PUBMETA = re.compile(rf"""\s*[\(\[]?\s*(?:
          [A-Z][A-Za-z.\s&]*?(?:Press|Publishers?|Publishing|Springer|Elsevier|Wiley|
          Academic|Books?|Verlag)[^)\]]*
        | (?:Cambridge|Oxford|Berlin|Heidelberg|London|New\s+York|Singapore|Dordrecht)
          \s*,?\s*{_YEAR}
        | {_YEAR}
        )\s*[\)\]]?\s*$""", re.X)
# Where a venue starts, when title and venue are run together in one string.
# GENERALISED. This was a list of journal names and abbreviations: Proc., J.,
# Phys., Nature, IEEE, ACM. That list is a computer-science and physics list. It
# does not contain J. Biol. Chem., Ann. Math., Acta Mater. or Astrophys. J., and
# extending it per domain is how a benchmark builder ends up with five different
# cleaners that disagree.
#
# A journal citation has a SHAPE that its name does not affect:
#   an abbreviation run   "Phys. Rev. Lett."   "J. Biol. Chem."   "Ann. Math."
#   a locator             "113, 140401"        "45, 2210"         volume, page
# Both are recognised without knowing a single venue name, so a materials science
# or biology split needs no new rules.
_ABBREV = re.compile(r"\b[A-Z][A-Za-z]{0,5}\.")
_LOCATOR = re.compile(r"\b\d{1,4}\s*[,:]\s*[A-Za-z]?\d")
# An identifier in the tail is as good a venue marker as an abbreviation run:
# "arXiv:2308.07037", "doi:10.1234/x", "hal-01234". Shape, not name.
_IDENT_TAIL = re.compile(r"\b(?:[A-Za-z]{2,8}:\s*)?\d{4}\.\d{4,5}\b|10\.\d{4,9}/")
_VENUE_NUMS = re.compile(r"\d{4}|\b\d{1,4}\s*[,:]\s*\d")
_DOI = re.compile(r"\b(10\.\d{4,9}/[^\s,;\)\]]+)", re.I)
_ARXIV = re.compile(r"(?:arxiv[:\s]*|abs/)(\d{4}\.\d{4,5})(?:v\d+)?", re.I)

# REBUILD FIX 1. A venue appended after the title with no journal keyword to
# mark it. Measured: "Attention is all you need . Advances in neural information
# processing systems" failed to resolve, because _VENUE_MARK looks for "J." or
# "Proc." and this has neither. One of the most findable papers in the corpus.
_VENUE_TAIL = re.compile(
    r"\s*\.\s+(?:Advances\s+in|Annex\b|In\s+Proceedings|Lecture\s+[Nn]otes|"
    r"Studies\s+in|Technical\s+[Rr]eport)\b.*$", re.I)

# PRECISION FIX 5. A venue tail with no keyword to announce it, identified by
# shape instead: a capitalised name followed by a volume/page locator.
# "Graph distance for complex networks. Scientific Reports 6 , 34944 (2016)."
# survived _VENUE_MARK because "Scientific Reports" is not in its keyword list,
# and no keyword list ever covers every journal. The locator is the giveaway.
_VENUE_LOCATOR = re.compile(
    r"[,.]\s+[A-Z][A-Za-z.&]*(?:\s+[A-Z][A-Za-z.&]*){0,4}\s+\d{1,4}\s*[,:]\s*\d")


# PRECISION FIX 1. OpenAlex serves some titles with the MathML still in them:
#
#   Phenomenon of a stronger trapping behavior in <mml:math xmlns:mml=
#   "http://www.w3.org/1998/Math/MathML"> <mml:mi mathvariant="normal">Λ</mml:mi>
#   </mml:math> -type quantum systems with symmetry
#
# Stripped of punctuation that becomes fourteen junk tokens (mml, math, xmlns,
# http, www, w3, org, 1998, ...), which wrecks token precision against the
# perfectly ordinary reference string for the same paper. Removed as markup
# rather than scored as words.
_MARKUP = re.compile(r"<[^>]+>")
_ENTITY = re.compile(r"&(?:[a-z]+|#\d+);", re.I)


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s or "").lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = _ENTITY.sub(" ", _MARKUP.sub(" ", s))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", s)).strip()


# A leading author list, which `clean_title` used to keep as part of the title.
# "T. Baumgratz, M. Cramer, and M. B. Plenio, Quantifying coherence, Phys. Rev.
# Lett. 113, 140401 (2014)." cleaned to "T. Baumgratz, ... Quantifying
# coherence", which scores 0.54 against the real title and never cleared the bar,
# so a perfectly findable paper resolved to nothing. Anchored at the start and
# requiring initials, so it can only eat an author run, never a title.
_AUTHOR_PREFIX = re.compile(
    r"^(?:(?:and\s+)?(?:[A-Z]\.\s*){1,4}(?:[a-z]{2,4}\s+)?[A-Z][\w'’-]+"
    r"(?:\s+et\s+al\.?)?"
    r"(?:\s*,\s*|\s+and\s+|\s*&\s*|\s*$))+", re.U)


def strip_author_prefix(t: str) -> str:
    """Remove a leading "A. Author, B. Author, and C. Author," run.

    Only fires when what remains still looks like a title, so a reference that
    is nothing BUT an author list is left alone for `is_citation_string` to
    reject rather than being silently reduced to its last surname.
    """
    m = _AUTHOR_PREFIX.match(t)
    if not m:
        return t
    rest = t[m.end():].strip(" ,;")
    return rest if len(rest.split()) >= 3 else t


def venue_cut(t: str) -> int | None:
    """Where a trailing venue begins, identified by shape rather than by name.

    A candidate boundary is any ", " or ". ". The remainder is a venue when it
    carries two or more abbreviation tokens, or a volume/page locator, or one
    abbreviation plus a number. Titles do not normally look like that; journal
    citations always do, in every field.
    """
    for m in re.finditer(r"[,.]\s+", t):
        head, tail = t[:m.start()], t[m.end():]
        # The head has to be able to BE a title. Without this the first boundary
        # in "S. Altschul et al., Basic local alignment search tool, J. Mol.
        # Biol. 215, 403 (1990)." is the period after the initial "S", and the
        # title comes out as "S".
        if len(head.split()) < 2:
            continue
        # The venue must START here, not merely appear somewhere later in the
        # string. Judged on a window, because otherwise every early boundary
        # inherits the locator belonging to the real venue further along.
        toks = tail.split()[:6]
        win = " ".join(toks)
        if _IDENT_TAIL.search(win) or _LOCATOR.search(win):
            return m.start()
        # CONSECUTIVE abbreviations, not merely several. An author list has the
        # same abbreviation density as a journal name and must not be mistaken
        # for one:
        #     "J. Mol. Biol. 215"        abbrev abbrev abbrev   -> a venue
        #     "J. Majsak, S. Chakraborty" abbrev name abbrev name -> authors
        # Counting without adjacency cut this reference list in half at its
        # second author.
        run = best = 0
        for tk in toks:
            run = run + 1 if re.fullmatch(r"[A-Z][A-Za-z]{0,5}\.", tk) else 0
            best = max(best, run)
        if best >= 2 or (best >= 1 and re.search(r"\d", win)):
            return m.start()
    return None


def balance_brackets(t: str) -> str:
    """Truncate at a bracket the strippers opened and left open.

    PRECISION FIX 6. `_PUBMETA` removes a trailing year including its closing
    paren, which turns "Quantum Optics (Wiley, 2006)" into
    "Quantum Optics (Wiley". The dangling "(Wiley" is not part of any title and
    drags the title search towards the publisher. Cutting at the unmatched
    opener leaves "Quantum Optics", which is what was cited.
    """
    for op, cl in (("(", ")"), ("[", "]")):
        depth, cut = 0, None
        for i, ch in enumerate(t):
            if ch == op:
                if depth == 0:
                    cut = i
                depth += 1
            elif ch == cl and depth:
                depth -= 1
                if depth == 0:
                    cut = None
        if depth and cut is not None:
            t = t[:cut]
    return t.strip(" .,;")


def clean_title(raw: str) -> str:
    """Strip the citation furniture a reference list carries.

    Entries in this corpus are one raw string with title and venue run together:
    "Quantum Computation and Quantum Information (Cambridge University Press,
    Cambridge, 2000)." Searched verbatim that finds nothing, because the second
    half is not part of any title.
    """
    t = str(raw or "").strip().strip('"“”')
    t = re.sub(r"\s+", " ", t)
    t = _VENUE_TAIL.sub("", t)                       # REBUILD FIX 1
    t = strip_author_prefix(t)                       # IDENTITY FIX
    for _ in range(3):
        t2 = _PUBMETA.sub("", t).strip(" .,;")
        if t2 == t:
            break
        t = t2
    cut = venue_cut(t)
    if cut is not None:
        t = t[:cut]
    if (m := _VENUE_LOCATOR.search(t)):               # PRECISION FIX 5
        t = t[:m.start()]
    return balance_brackets(t.strip(" .,;"))          # PRECISION FIX 6


# ===================================================================== identity
# IDENTITY FIX. A title is not an identifier, and `clean_title` was throwing away
# the only fields that could tell two works with the same title apart. Four wrong
# identities across 8 accepted slots, every one scored 1.0 on title equality:
#
#   "Quantum Optics (Wiley, 2006)"  -> Scully & Zubairy, CUP 1997
#                                      (should be Vogel & Welsch, Wiley 2006)
#   "Modern computer algebra"       -> 10.5860/choice.37-5723, a Choice REVIEW
#                                      of the book, not the book
#   "Modal Model Theory"            -> Chang's 1973 Springer chapter
#                                      (should be Hamkins & Wolo/szyn 2024)
#   "Natural evolution strategies"  -> the 2008 conference version
#                                      (should be the 2014 JMLR article)
#
# Only the first carries a year in the reference string, so a year check alone
# recovers one of four. What catches the rest is refusing to treat a title that
# several works share as though it identified one of them.

_REF_YEAR = re.compile(r"(?<!\d)((?:1[6-9]|20)\d{2})[a-z]?(?!\d)")
# A surname is only recognised where INITIALS precede it, which is how these
# strings actually encode authors: "T. Baumgratz", "M. O. Scully",
# "J. von zur Gathen".
#
# The first version matched any capitalised word outside the title span and got
# it badly wrong in both directions. Measured on five accepted-then-rejected
# references:
#
#   "T. Baumgratz, M. Cramer, and M. B. Plenio, Quantifying coherence,
#    Phys. Rev. Lett. 113, 140401 (2014)."   ->  surnames ['lett','phys','rev']
#   "A. Perelomov, ... (Springer, Berlin, 1986)."  ->  surnames ['berlin']
#   "J. Von Zur Gathen and J. Gerhard, Modern computer algebra"  ->  []
#
# The cause: `clean_title` frequently returns the author prefix AS PART of the
# title, so removing the title span removed the authors and left the venue. The
# journal abbreviation was then read as the author list and CONTRADICTED the
# real one, rejecting correct matches at similarity 1.0.
#
# Requiring initials fixes both directions at once. "Phys." and "Lett." are not
# single-letter initials, so a venue can no longer masquerade as an author, and
# scanning the whole string rather than the tail stops losing real ones.
_AUTHOR = re.compile(r"(?:\b[A-Z]\.\s*){1,4}((?:[A-Z][a-z]+[\s-]+){0,2}[A-Z][a-z]{2,})")
# Publisher and imprint names that pin a book to an edition.
# GENERALISED. This was a list of 16 publisher names, which is a list of the
# publishers that happened to appear in a computer science sample. The venue
# check now compares the reference's own non-title, non-author words against the
# venue string OpenAlex returns for the candidate, so it works for any publisher
# or journal in any field without being told the name in advance.
# OpenAlex types that are ABOUT a work rather than being it. "Modern computer
# algebra" resolved to a Choice review; a reference to a book never means its
# review, and the reference string gives no way to ask for one.
_META_TYPES = frozenset({"review", "paratext", "editorial", "erratum",
                         "letter", "peer-review", "grant", "retraction"})


def parse_reference(raw: str) -> dict:
    """The identity evidence in a reference string, before cleaning removes it.

    Cheap and deliberately conservative. Every field is optional, and a field
    that is absent is never treated as a disagreement: the checks below only
    fire when BOTH sides state something and they conflict.
    """
    s = str(raw or "")
    title = clean_title(s)
    # Only years outside the title, so "AlexNet 2012" in a title is not read as
    # a publication year and a year inside the title is not double counted.
    tail = s.replace(title, " ", 1) if title else s
    years = [int(y) for y in _REF_YEAR.findall(tail)]
    # Surnames from the part of the string that is NOT the title, same reason.
    # From the WHOLE string, not the tail: `clean_title` often keeps the author
    # prefix inside the title span, so the tail is usually the venue.
    names = {w.lower() for m in _AUTHOR.findall(s) for w in m.replace("-", " ").split()}
    names -= _STOP
    # The words the reference states that are NOT its title and NOT its authors.
    # Whatever venue or publisher it names is in here, without a list of venues.
    venue_words = ({w for w in re.findall(r"[A-Za-z]{3,}", tail)}
                   - {w.capitalize() for w in names} - set(names))
    return {
        "title": title,
        "years": sorted(set(years)),
        "surnames": sorted(names),
        "venue_words": sorted({w.lower() for w in venue_words}
                              - _STOP - {"press", "vol", "pp", "eds", "ed"}),
    }


def corroborate(ref: dict, w: dict) -> dict:
    """Does this candidate agree with the reference on anything besides its title?

    Returns a verdict and the reasons, both stored on the record so an accepted
    identity can be argued with afterwards rather than taken on trust.

        contradict  the two state different things. Never accepted.
        support     something beyond the title agrees. Breaks a tie.
        neutral     the reference says nothing more. Fine when the title is
                    unique, not enough when several works share it.
    """
    reasons: list[str] = []
    hard = False
    year = w.get("publication_year")
    if ref["years"] and year:
        # Two years, because a reference may cite an edition and OpenAlex may
        # hold the original. Nine years apart is a different book.
        gap = min(abs(year - y) for y in ref["years"])
        if gap <= 2:
            reasons.append(f"year~{year}")
        elif gap >= 5:
            hard = True
            reasons.append(f"YEAR {year} vs {ref['years']}")

    auth = surnames_of(w)
    if ref["surnames"] and auth:
        if auth & set(ref["surnames"]):
            reasons.append("author")
        else:
            hard = True
            reasons.append(f"AUTHORS {sorted(auth)[:3]} vs {ref['surnames'][:3]}")

    # Venue agreement by token overlap with whatever OpenAlex calls the source.
    # No publisher list: "Wiley-VCH" and "Cambridge University Press" are
    # compared to the reference's own words, so a chemistry or biology venue
    # works identically without being enumerated anywhere.
    venue = {w.lower() for w in re.findall(
        r"[A-Za-z]{3,}",
        ((w.get("primary_location") or {}).get("source") or {}).get("display_name") or "")}
    venue -= _STOP | {"journal", "review", "reviews", "letters", "proceedings",
                      "transactions", "press", "university", "international"}
    rv = set(ref["venue_words"])
    if rv and venue:
        if rv & venue:
            reasons.append("venue")
        else:
            # Both name a venue and they share no distinctive word. Soft, not
            # hard: a reference may name the publisher where OpenAlex names the
            # series, and neither is wrong.
            reasons.append(f"venue? {sorted(venue)[:2]} vs {sorted(rv)[:2]}")

    # NOT a contradiction. OpenAlex types survey and review ARTICLES as "review",
    # and a survey is a perfectly good inspiration. Treating the type as fatal
    # rejected real papers; it is only used below to demote an index stub when a
    # real work with the same title is also on offer.
    return {"verdict": "contradict" if hard else
                       ("support" if any(r[0].islower() for r in reasons) else "neutral"),
            "reasons": reasons}


def surnames_of(w: dict) -> set:
    """Last-name tokens of a work's authors. Never raises on odd metadata.

    The first version guarded on `if display_name` and then called
    `.split()[-1]`, which are not the same test: OpenAlex serves the occasional
    author whose display_name is a single space or a lone punctuation mark. That
    passes the truthiness guard, splits to an empty list, and IndexErrors on
    [-1]. It killed the full run 1,194 titles into the test split.
    """
    out = set()
    for a in (w.get("authorships") or []):
        parts = str((a.get("author") or {}).get("display_name") or "").split()
        if parts:
            out.add(parts[-1].lower())
    return out


# Kept so anything importing the old private name still works.
_surnames_of = surnames_of


def same_work(x: dict, y: dict) -> bool:
    """Two OpenAlex records of ONE paper, rather than two different papers.

    This is the distinction the first identity pass missed, and it cost 33
    points of resolution. OpenAlex routinely holds a preprint and the published
    version as separate works with byte-identical titles:

        Relational Knowledge Distillation   2019  10.1109/cvpr.2019.00409
        Relational Knowledge Distillation   2019  10.48550/arxiv.1904.05068

    Both score 1.0, neither contradicts a bare reference, and the old rule called
    that "2 works share this title" and refused to choose. It is one work with
    two records, and choosing between them is free.

    A shared author surname settles it. Failing that, and only when one side has
    no authors at all, publication year within two years does.
    """
    sx, sy = _surnames_of(x), _surnames_of(y)
    if sx and sy:
        return bool(sx & sy)
    # One side has no author list, so there is NO EVIDENCE these are different
    # works, only an absence. Treated as the same work, deliberately: a wrong
    # merge costs one edition of a book, while a wrong split throws the whole
    # reference away. Measured: "The Theory of Error-Correcting Codes" came back
    # as four records (1977, 1979, 1980, 1980) of one book and the year test
    # split them into different works, so the reference resolved to nothing.
    return True


def is_stub(w: dict) -> bool:
    """A record ABOUT a work rather than the work: a review index, an erratum.

    "Modern computer algebra" resolved to 10.5860/choice.37-5723, a Choice
    Reviews entry: no authors, type "review", carrying the reviewed book's
    title. Not rejected outright, because a survey article is also typed
    "review" and is a legitimate inspiration. Only demoted, and only when a
    record with authors shares the same title.
    """
    return not (w.get("authorships") or []) and \
        (w.get("type") or "").lower() in _META_TYPES


def cluster_works(ws: list) -> list:
    """Group records that are the same work. Single-link, because `same_work`
    is a similarity, not an equivalence: A may share an author with B and B with
    C without A and C overlapping, and all three are still one paper."""
    groups: list[list] = []
    for w in ws:
        hit = [g for g in groups if any(same_work(w, v) for v in g)]
        if not hit:
            groups.append([w])
            continue
        merged = [w] + [v for g in hit for v in g]
        groups = [g for g in groups if g not in hit] + [merged]
    return groups


def rank_key(w: dict):
    """Which record of one work to keep. Published over preprint, cited over not."""
    doi = str(w.get("doi") or "").lower()
    return (bool(doi) and "arxiv" not in doi,
            not is_stub(w),
            bool(w.get("authorships")),
            w.get("cited_by_count") or 0)


def is_citation_string(raw: str) -> bool:
    """True when the entry names a location in a journal rather than a title.

    "Physical Review B 104, 035118 (2021)" contains no title at all. Sent to a
    title matcher it returns a confident wrong answer, so it never reaches the
    network. Two words is enough to try: "Quantum Optics" is a real book, and the
    similarity score is what rejects a loose match on a short query.
    """
    t = clean_title(raw)
    if not t or len(norm(t).split()) < 2:
        return True
    digits = sum(c.isdigit() for c in t)
    return digits / max(len(t), 1) > 0.18


def similarity(a: str, b: str) -> float:
    """How much a reference string and a candidate title agree. `a` is the query.

    PRECISION FIX 1. This used to score containment against the SHORTER side:

        len(wa & wb) / min(len(wa), len(wb))

    which is one-sided. Every word of the reference appearing somewhere in the
    candidate scored 0.97, an exact match, no matter how many words the candidate
    added. The audit found six papers accepted that way, all of the same shape:

        "Bayesian Flow Networks"   ->  "Predicting traffic flow using Bayesian
                                        networks"                    0.97 -> 0.67
        "The Bayesian Learning Rule" -> "Training Binary Neural Networks using
                                         the Bayesian Learning Rule" 0.97 -> short

    Token F1 charges for both mistakes. Recall still forgives the trailing junk a
    reference carries, which is what containment was there for, but precision now
    charges for the words the candidate adds, and it is those added words that
    change which paper it is.

    PRECISION FIX 2. Below SHORT_TITLE distinct words, overlap decides nothing:
    "Quantum Optics" is contained in most of quantum optics. Those fall back to
    sequence agreement, which only a near-identical title can pass.

    The weighting is F0.5, not F1, because the two kinds of leftover word are not
    equally suspicious:

      extra words on the QUERY side    citation furniture the cleaner missed.
                                       Common, harmless, and forgiving them is
                                       the whole reason overlap beats edit
                                       distance here.
      extra words on the CANDIDATE     usually a DIFFERENT PAPER. "A bound on
                                       the independent domination number of a
                                       tree" is not "A lower bound on the TOTAL
                                       OUTER-independent domination number of a
                                       tree", and the three added words are the
                                       entire difference between them.

    Under F1 that pair scored 0.857 and was accepted. Weighting precision at
    twice recall puts it at 0.766, below ACCEPT, where it belongs.

    Sequence ratio is deliberately NOT max'd in for long titles. It is
    character-level, so words inserted in the MIDDLE of an otherwise matching
    title barely move it: the same pair scores 0.857 on sequence alone. That is
    exactly the case token precision exists to catch, so letting sequence
    override it would reinstate the bug one level down.
    """
    na, nb = norm(a), norm(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    wa, wb = content_words(na), content_words(nb)
    if min(len(wa), len(wb)) <= SHORT_TITLE:
        return SequenceMatcher(None, na, nb).ratio()
    inter = len(wa & wb)
    if not inter:
        return 0.0
    prec, rec = inter / len(wb), inter / len(wa)
    b2 = 0.25                                    # F-beta with beta = 0.5
    fbeta = (1 + b2) * prec * rec / (b2 * prec + rec)
    # Capped below 1.0: the same words in a different order is strong evidence,
    # but it is not the same evidence as an identical string.
    return fbeta * 0.97


# PRECISION FIX 11. Function words carry no evidence about which paper this is,
# but under plain token overlap they count as much as any other word. Measured on
# the 100-paper run:
#
#   "Mathematical Foundations OF Supervised Learning"
#   "Mathematical Foundations FOR Supervised Learning"      0.776, rejected
#
# The same paper, one preposition apart, and the mismatch was charged twice, once
# to precision and once to recall. Dropped from the token sets only. The sequence
# ratio still sees the whole string, so short titles are unaffected.
_STOP = frozenset("""a an the of for on in to and or with without via using by
    from at as is are be that this these those its their our we not no than then
    into over under between towards toward about""".split())


def content_words(n: str) -> set:
    """The tokens of a normalised title that carry identity. Never empty."""
    w = set(n.split())
    return (w - _STOP) or w


def tier(score: float) -> str:
    if score >= EXACT:
        return "exact"
    if score >= STRONG:
        return "strong"
    if score >= WEAK:
        return "weak"
    return "none"


def accepted(score: float) -> bool:
    """Whether a score is good enough to become a gold document.

    PRECISION FIX 4. Separate from `tier`, which still names all four bands, so
    a weak hit can be reported and reviewed rather than silently promoted. A
    benchmark's gold documents are its ground truth; a 0.7 guess does not belong
    in one.
    """
    return score >= ACCEPT


def field_purity(rec: dict) -> float | None:
    """What share of the work's OpenAlex topics agree with its primary field.

    OpenAlex assigns three topics per work and we label the paper with the field
    of the first. When the other two disagree, that field is a plurality of one.
    On the first run 119 of 453 occurrences were labelled that way, and 87 of the
    190 cross-field labels rested on them.
    """
    tf, f = rec.get("insp_topic_fields"), rec.get("insp_field")
    if not tf or not f:
        return None
    total = sum(tf.values())
    return round(tf.get(f, 0) / total, 3) if total else None


def field_confidence(rec: dict) -> str | None:
    """PRECISION FIX 8. "ambiguous" means: do not count this in a same/cross split."""
    p = field_purity(rec)
    if p is None:
        return None
    return "clear" if p >= PURITY_MIN - 1e-9 else "ambiguous"


def abstract_from_inverted(inv: dict | None) -> str | None:
    """OpenAlex stores abstracts as {word: [positions]} for licensing reasons."""
    if not inv:
        return None
    pos: dict[int, str] = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    if not pos:
        return None
    return " ".join(pos[i] for i in sorted(pos)) or None


# ========================================================================== http
def matcher_fingerprint() -> str:
    """A hash of every function whose output a cache entry records.

    CACHE_VERSION is a manual bump and it was forgotten exactly once, which cost
    a full rerun: the matching logic changed, the version did not, and 307 of 312
    lookups were served the old verdicts out of the v8 cache. The cache stores
    VERDICTS, not responses, so stale logic is indistinguishable from a fresh
    answer. This makes forgetting harmless: change any of these and the entries
    computed by the old code are dropped automatically.
    """
    import inspect
    src = "".join(inspect.getsource(f) for f in (
        norm, clean_title, strip_author_prefix, venue_cut, balance_brackets,
        content_words, similarity, tier, accepted, parse_reference, corroborate,
        same_work, is_stub, rank_key, cluster_works, is_citation_string,
        _reject, _from_openalex, _from_s2, _lookup))
    # Every regex in this module, by name, so a pattern can be added or removed
    # without this list needing to be maintained in step. The previous version
    # named them one by one and broke the moment _VENUE_MARK was deleted.
    src += repr(sorted((k, v.pattern, v.flags) for k, v in globals().items()
                       if isinstance(v, re.Pattern)))
    src += repr((EXACT, STRONG, WEAK, ACCEPT, SHORT_TITLE, PURITY_MIN,
                 sorted(ID_ROUTES), sorted(_META_TYPES), sorted(_STOP)))
    return hashlib.sha1(src.encode()).hexdigest()[:16]


class Http:
    """Shared rate-limited reader with one on-disk, version-stamped cache."""

    def __init__(self, cache_path: Path, delay: float, verbose: bool = False):
        self.delay = delay
        self.verbose = verbose
        self.cache_path = cache_path
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache: dict = {}
        if cache_path.exists():
            try:
                blob = json.loads(cache_path.read_text())
                fp = matcher_fingerprint()
                if blob.get("_v") == CACHE_VERSION and blob.get("_fp") == fp:
                    self.cache = blob.get("entries") or {}
                elif blob.get("_v") == CACHE_VERSION:
                    print(f"  cache is v{CACHE_VERSION} but the matching code "
                          f"changed since it was written: discarding "
                          f"{len(blob.get('entries') or {})} stale verdicts")
                else:
                    print(f"  cache is v{blob.get('_v')}, code is v{CACHE_VERSION}: "
                          f"discarding {len(blob.get('entries') or {})} stale entries")
            except (json.JSONDecodeError, AttributeError):
                print(f"  cache at {cache_path} unreadable, starting empty",
                      file=sys.stderr)
        self._lock = threading.Lock()
        self._last = 0.0
        self.calls = 0
        self.hits = 0
        self.rate_limited = 0

    def _wait(self) -> None:
        with self._lock:
            gap = time.monotonic() - self._last
            if gap < self.delay:
                time.sleep(self.delay - gap)
            self._last = time.monotonic()

    def get(self, url: str, headers: dict | None = None) -> dict | None:
        hdr = {"User-Agent": "sir4-stage4", **(headers or {})}
        for attempt in range(5):
            self._wait()
            try:
                self.calls += 1
                with urllib.request.urlopen(
                        urllib.request.Request(url, headers=hdr), timeout=30) as r:
                    return json.loads(r.read().decode())
            except urllib.error.HTTPError as e:
                if e.code in (400, 403, 404):
                    return None
                if e.code == 429:
                    # A quota that is actually spent says so in the body. A plain
                    # 429 is congestion, worth backing off for. Those are
                    # different failures and only one is worth giving up on.
                    body = ""
                    try:
                        body = e.read().decode()[:300]
                    except Exception:
                        pass
                    if "budget" in body.lower():
                        raise BudgetExhausted(body)
                    self.rate_limited += 1
                    time.sleep(1.5 * 2 ** attempt)
                    continue
                if e.code in (500, 502, 503, 504):
                    time.sleep(1.5 * 2 ** attempt)
                    continue
                if self.verbose:
                    print(f"    HTTP {e.code}", file=sys.stderr)
                return None
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                time.sleep(1.5 * 2 ** attempt)
                continue
        raise TransientError(url)

    def get_raw(self, url: str) -> bytes | None:
        """Same rate limiting, for a source that does not speak JSON."""
        for attempt in range(3):
            self._wait()
            try:
                self.calls += 1
                with urllib.request.urlopen(urllib.request.Request(
                        url, headers={"User-Agent": "sir4-stage4"}),
                        timeout=30) as r:
                    return r.read()
            except urllib.error.HTTPError as e:
                if e.code in (400, 403, 404):
                    return None
                time.sleep(1.5 * 2 ** attempt)
            except (urllib.error.URLError, TimeoutError):
                time.sleep(1.5 * 2 ** attempt)
        return None

    def post(self, url: str, payload: dict, headers: dict | None = None) -> list | None:
        hdr = {"User-Agent": "sir4-stage4", "Content-Type": "application/json",
               **(headers or {})}
        body = json.dumps(payload).encode()
        for attempt in range(4):
            self._wait()
            try:
                self.calls += 1
                with urllib.request.urlopen(urllib.request.Request(
                        url, data=body, headers=hdr), timeout=60) as r:
                    return json.loads(r.read().decode())
            except urllib.error.HTTPError as e:
                if e.code in (400, 403, 404):
                    return None
                if e.code == 429:
                    self.rate_limited += 1
                time.sleep(1.5 * 2 ** attempt)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                time.sleep(1.5 * 2 ** attempt)
        return None

    def save(self) -> None:
        tmp = self.cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"_v": CACHE_VERSION,
                                   "_fp": matcher_fingerprint(),
                                   "entries": self.cache}))
        tmp.replace(self.cache_path)


class TransientError(Exception):
    """Five attempts exhausted. Deliberately NOT cached, so a rerun retries it."""


class BudgetExhausted(Exception):
    """The API says the quota is spent. Backing off will not help."""


class OpenAlex:
    def __init__(self, http: Http):
        self.http = http
        self.mailto = os.environ.get("OPENALEX_MAILTO", "")
        self.key = os.environ.get("OPENALEX_API_KEY", "")

    def _params(self, extra: dict, lean: bool = False) -> str:
        p = {**extra, "select": OA_SELECT_LEAN if lean else OA_SELECT}
        if self.mailto:
            p["mailto"] = self.mailto
        if self.key:
            p["api_key"] = self.key
        return urllib.parse.urlencode(p)

    def by_doi(self, doi: str) -> dict | None:
        q = self._params({})
        return self.http.get(
            f"{OA}/works/doi:{urllib.parse.quote(doi, safe='/')}?{q}")

    # OpenAlex takes an OR of up to 50 values in one filter. The target fields
    # alone are 9,052 lookups on a full cs split; batched they are 181 requests.
    # Nothing else in this file changes the call count by two orders of magnitude.
    BATCH = 50

    def by_dois(self, dois: list) -> dict:
        """{normalised doi: work} for a list of DOIs, 50 per request."""
        out = {}
        clean = [str(d).strip().lower().replace("https://doi.org/", "")
                 for d in dois if d and not str(d).startswith("arXiv:")]
        for i in range(0, len(clean), self.BATCH):
            chunk = clean[i:i + self.BATCH]
            q = self._params({"filter": "doi:" + "|".join(chunk),
                              "per-page": self.BATCH})
            d = self.http.get(f"{OA}/works?{q}")
            for w in ((d or {}).get("results") or []):
                doi = (w.get("doi") or "").replace("https://doi.org/", "").lower()
                if doi:
                    out[doi] = w
        return out

    # 4, not 8. Measured on 4,107 resolved golds: 60.7% of searches had ONE
    # candidate tied on the exact title, 35.1% had two, and 99.4% had four or
    # fewer. The extra four were downloaded every time and used 0.6% of the time.
    TITLE_N = 4

    def by_title(self, title: str, n: int | None = None) -> list:
        q = self._params({"filter": f"title.search:{title}",
                          "per-page": n or self.TITLE_N}, lean=True)
        d = self.http.get(f"{OA}/works?{q}")
        return (d or {}).get("results") or []


class S2:
    """Fallback only, and paced independently of OpenAlex.

    A Semantic Scholar API key grants ONE REQUEST PER SECOND, cumulative across
    every endpoint. OpenAlex's polite pool allows ten. Both clients used to share
    one `Http` limiter, so a single delay had to serve both: 0.12s honours
    OpenAlex and violates S2 eightfold, while 1.0s honours S2 and makes OpenAlex
    eight times slower than it needs to be. Neither is acceptable at 23,000
    lookups.

    So S2 keeps its own lock and its own delay, the same pattern ArXiv already
    uses for its 3-second courtesy limit. The two services then run at their own
    correct rates concurrently: different titles are on different workers, so
    OpenAlex is never idle waiting for S2.
    """

    # Stated by Semantic Scholar on key issue. Cumulative across endpoints, so
    # `match`, `by_id` and `batch` all queue behind the same limiter.
    DELAY = 1.0

    def __init__(self, http: Http):
        self.http = http
        self.key = os.environ.get("S2_API_KEY", "")
        self._lock = threading.Lock()
        self._last = 0.0
        self.calls = 0

    def _hdr(self) -> dict:
        return {"x-api-key": self.key} if self.key else {}

    def _pace(self) -> None:
        """Held across the sleep, so workers queue rather than burst."""
        with self._lock:
            gap = time.monotonic() - self._last
            if gap < self.DELAY:
                time.sleep(self.DELAY - gap)
            self._last = time.monotonic()
            self.calls += 1

    def match(self, title: str) -> dict | None:
        self._pace()
        q = urllib.parse.urlencode({"query": title, "fields": S2_FIELDS})
        d = self.http.get(f"{S2_API}/paper/search/match?{q}", self._hdr())
        return ((d or {}).get("data") or [None])[0]

    # S2's batch endpoint takes up to 500 ids in one POST. The abstract backfill
    # is ~4,000 calls on a full split; batched it is 8 requests. It needs a key:
    # the anonymous pool refuses the endpoint outright.
    BATCH = 500

    def batch(self, ids: list) -> dict:
        """{id: paper} for DOI:... or arXiv:... ids, 500 per request."""
        out = {}
        for i in range(0, len(ids), self.BATCH):
            self._pace()
            chunk = ids[i:i + self.BATCH]
            d = self.http.post(
                f"{S2_API}/paper/batch?fields={S2_FIELDS}",
                {"ids": chunk}, self._hdr())
            if not isinstance(d, list):
                continue
            for ident, rec in zip(chunk, d):
                if rec:
                    out[ident] = rec
        return out

    def by_id(self, ident: str) -> dict | None:
        self._pace()
        q = urllib.parse.urlencode({"fields": S2_FIELDS})
        return self.http.get(
            f"{S2_API}/paper/{urllib.parse.quote(ident, safe=':')}?{q}", self._hdr())


class ArXiv:
    """Last resort for abstracts, and the slowest per request.

    Runs only after OpenAlex and S2 have both failed to supply an abstract.

    arXiv has no id-free batch endpoint, but its query parser takes boolean OR,
    so BATCH titles go in one request as `ti:"a" OR ti:"b" OR ...`. That is the
    difference between keeping this fallback and always passing --no-arxiv: on a
    full split, ~3,100 gaps at one request each is 2.6 hours, and at twelve per
    request it is about fifteen minutes.

    A batched reply is a flat list with no indication of which entry answers
    which term, so every title is scored against every entry and held to the
    same bar as any other match.
    """

    ATOM = {"a": "http://www.w3.org/2005/Atom",
            # Only for the totalResults counter, which is how a query that
            # parsed and found nothing is told apart from one that was refused.
            "os": "http://a9.com/-/spec/opensearch/1.1/"}
    # arXiv's terms ask for one request every three seconds. The shared Http
    # delay is 0.12s, tuned to OpenAlex's polite pool, which is 25x faster than
    # arXiv permits. Enforced here rather than by raising the global delay,
    # which would slow the other 250 lookups down to arXiv's pace.
    DELAY = 3.0
    # Twelve keeps the URL near 1kB and the reply small enough to score cheaply.
    # The parser gets unreliable well before the URL length limit bites.
    BATCH = 12

    # arXiv's parser refuses a quoted phrase containing quotes, colons or
    # brackets, and a refusal costs the whole batch. ti: matches on words
    # anyway, so the punctuation carries no search value to begin with.
    _QSAFE = re.compile(r"[^A-Za-z0-9 ]+")

    def __init__(self, http: Http):
        self.http = http
        self._last = 0.0
        self.calls = 0
        # Separated so the summary can say whether a low fill rate means arXiv
        # does not have these papers (empty) or that we asked badly (refused).
        # Those need opposite remedies: drop the step, or fix the query.
        self.empty = 0
        self.refused = 0
        self.stopped = False
        self._lock = threading.Lock()

    @classmethod
    def _qterm(cls, title: str) -> str:
        return " ".join(cls._QSAFE.sub(" ", str(title or "")).split())[:120]

    def _get(self, params: dict) -> str | None:
        # The lock is held across the sleep, so concurrent workers queue here
        # rather than all waking at once and firing a burst arXiv would refuse.
        with self._lock:
            gap = time.monotonic() - self._last
            if gap < self.DELAY:
                time.sleep(self.DELAY - gap)
            self._last = time.monotonic()
            self.calls += 1
        return self.http.get_raw(
            f"http://export.arxiv.org/api/query?{urllib.parse.urlencode(params)}")

    def _entries(self, raw: str | None) -> list[tuple[str, str]]:
        if not raw:
            return []
        try:
            root = ET.fromstring(raw)
        except ET.ParseError:
            return []
        out = []
        for e in root.findall("a:entry", self.ATOM):
            # arXiv hard-wraps both fields, so a raw summary arrives full of
            # newlines. Collapsing them here keeps the abstract usable as the
            # document text it is about to become.
            t = " ".join((e.findtext("a:title", default="",
                                     namespaces=self.ATOM) or "").split())
            s = " ".join((e.findtext("a:summary", default="",
                                     namespaces=self.ATOM) or "").split())
            if t and s:
                out.append((t, s))
        return out

    def answered(self, raw: str | None) -> bool:
        """True when arXiv understood the query, whatever it then found.

        An empty batch has two causes, and telling them apart is the entire
        cost of this step:

          arXiv parsed the OR query and holds none of the twelve titles.
          arXiv's parser REFUSED the query, so the twelve were never asked.

        Only the second is worth retrying one title at a time. Treating both as
        a refusal is what turned a 3s batch into 39s: on a full split most
        batches are legitimately empty, because these are the titles OpenAlex
        and S2 both already failed on, and many are not arXiv papers at all. At
        a 4% fill rate that made ~96% of batches pay 12 extra requests to ask
        the same index the same question and get the same answer, which is 13x
        the estimate printed at the top of the step.

        The reply carries an opensearch:totalResults counter whenever the query
        parsed, and reports a refusal as a lone entry titled "Error".
        """
        if not raw:
            return False                       # network or HTTP failure
        try:
            root = ET.fromstring(raw)
        except ET.ParseError:
            return False                       # not a feed at all
        first = (root.findtext("a:entry/a:title", default="",
                               namespaces=self.ATOM) or "").strip().lower()
        if first.startswith("error"):
            return False
        return root.find("os:totalResults", self.ATOM) is not None

    def abstract_for(self, title: str) -> str | None:
        """One title, one request. The per-batch fallback, and still correct."""
        q = self._qterm(title)
        if not q:
            return None
        hits = self._entries(self._get({"search_query": f'ti:"{q}"',
                                        "max_results": 1}))
        if not hits:
            return None
        found, summary = hits[0]
        # PRECISION FIX 4: an abstract attached to a weakly matching title is
        # the wrong abstract, and it lands on a document that was otherwise
        # correct, which makes it harder to spot than a bad title.
        return summary if accepted(similarity(title, found)) else None

    def abstracts_for(self, titles: list[str]) -> dict[str, str]:
        """{title: abstract} for whatever arXiv has, BATCH titles per request.

        Interruptible. This is by far the longest-running loop in the stage and
        the one most worth abandoning early, since its fill rate is a few per
        cent. Ctrl-C stops it and RETURNS what it has rather than unwinding, so
        the abstracts already paid for land on the records like any other.
        """
        out: dict[str, str] = {}
        # The slowest thing in the stage: 3s per request by arXiv's own courtesy
        # limit, and arXiv itself stalls often enough that the shared Http retry
        # (3 attempts, 30s timeout) dominates the tail. Shown, not guessed at.
        chunks = range(0, len(titles), self.BATCH)
        bar = tqdm(chunks, desc="  arxiv abstracts", unit="batch",
                   dynamic_ncols=True, leave=False, disable=len(titles) <= self.BATCH)
        for i in bar:
            chunk = [t for t in titles[i:i + self.BATCH] if self._qterm(t)]
            if not chunk:
                continue
            try:
                self._fill(chunk, out)
            except KeyboardInterrupt:
                self.stopped = True
                bar.close()
                print(f"  arxiv abstracts:   stopped early, keeping "
                      f"{len(out)} abstract(s) found so far", file=sys.stderr)
                return out
            bar.set_postfix_str(f"filled {len(out)}/{len(titles)}", refresh=False)
        bar.close()
        return out

    def _fill(self, chunk: list[str], out: dict[str, str]) -> None:
        """One request for one chunk of titles, writing straight into `out`."""
        raw = self._get({
            "search_query": " OR ".join(f'ti:"{self._qterm(t)}"' for t in chunk),
            "max_results": min(200, len(chunk) * 5)})
        hits = self._entries(raw)
        if not hits:
            if self.answered(raw):
                # arXiv parsed the query and holds none of these twelve. Twelve
                # single requests would ask the same index the same question,
                # so they cost 12 x DELAY and find exactly nothing.
                self.empty += 1
                return
            # The parser refused, so the titles were never actually asked. One
            # bad title must not cost the other eleven.
            self.refused += 1
            for t in chunk:
                a = self.abstract_for(t)
                if a:
                    out[t] = a
            return
        # A batched reply is a flat list with no indication of which entry
        # answers which term, so every title is scored against every entry.
        for t in chunk:
            best, score = None, 0.0
            for found, summary in hits:
                s = similarity(t, found)
                if s > score:
                    best, score = summary, s
            if best and accepted(score):
                out[t] = best


class Crossref:
    """Recovers references that carry NO TITLE, which is most of physics.

    THE PROBLEM THIS SOLVES.

    Physics reference lists omit paper titles by convention. What Stage 3 hands
    Stage 4 is therefore not a title at all:

        Phys. Rev. C, 101(6):064304, Jun 2020.
        Physics of Plasmas 2 , 2381 (1995)
        Prog. Theor. Phys. Suppl. 125, 97 (1996)

    Every route in this file searches by title, so all of them fail, and they
    fail expensively: on physics_train 16,821 such strings each cost an
    unbatchable OpenAlex title search, produced 8,603 rate-limit retries, and
    resolved nothing. 47.7% of golds came out unresolved against biology's 2.5%.

    Crossref indexes exactly what these strings DO carry. `query.bibliographic`
    takes the raw string, and the reply carries structured volume, page, issue
    and year, so the match can be checked on the LOCATOR rather than on prose.

    WHY THAT IS A STRONGER CHECK, NOT A WEAKER ONE.

    Journal plus volume plus first page is close to a primary key. Two different
    papers do not share one. So this route is verified more tightly than the
    title routes are, not less: `accepted()` tolerates paraphrase because titles
    get copied loosely, whereas a volume is right or it is wrong.

    A candidate is taken only when the volume matches exactly, the first page
    matches exactly (leading zeros normalised, ranges reduced to their start),
    the year is within one, and the abbreviated journal name expands to the
    container title. Anything short of that is dropped rather than guessed at.

    WHERE IT RUNS, AND WHY THAT MATTERS.

    In `backfill_all`, never in `_lookup`. `_lookup` is inside
    `matcher_fingerprint`, so touching it would invalidate every cached verdict
    in the project. This route is purely additive -- it only ever fires on a
    string that already resolved to nothing, and unresolved titles are never
    cached -- so no stored verdict changes meaning and the cache survives.

    For the same reason its patterns are CLASS attributes: the fingerprint hashes
    every re.Pattern in globals(), so a module-level regex here would discard the
    cache for no behavioural reason.

    Free. No key, polite pool, mailto in the query string. The DOIs it returns go
    through the existing batched `oa.by_dois`, which is where the OpenAlex topic
    fields the domain labels need come from.
    """

    API = "https://api.crossref.org/works"
    ROWS = 5                       # more than 3 has never changed a verdict
    YEAR_SLACK = 1                 # online-first and issue years differ by one

    _YEAR4 = re.compile(r"(?<!\d)((?:1[89]|20)\d{2})(?!\d)")
    # "101(6):064304" and "101 (6), 64304": volume, issue, page.
    _VOLISS = re.compile(r"(?<!\d)(\d{1,4})\s*\(\s*\d{1,3}[A-Za-z]?\s*\)\s*[:,]?\s*"
                         r"([A-Za-z]?\d{1,7})")
    # "120, 40002" and "46 , 1351": volume then page. Requires the separator so a
    # bare pair of numbers in prose cannot be read as a locator.
    _VOLPAGE = re.compile(r"(?<!\d)(\d{1,4})\s*[,:]\s*([A-Za-z]?\d{1,7})(?!\d)")
    _RANGE = re.compile(r"^([A-Za-z]?\d+)\s*[-–—]")

    def __init__(self, http: Http, mailto: str | None = None):
        self.http = http
        # The polite pool. Crossref asks for a contact and gives better service
        # in return; without it the shared pool is slower and rate-limited.
        self.mailto = mailto or os.environ.get("CROSSREF_MAILTO") or ""
        self.calls = 0
        self.matched = 0
        self.rejected = 0
        self._lock = threading.Lock()
        self._pace_lock = threading.Lock()
        self._last = 0.0
        if not self.mailto:
            print("  crossref:          no CROSSREF_MAILTO set, so requests go to "
                  "the SHARED pool,\n                     which is slower and rate "
                  "limited. Export it for the polite pool.", file=sys.stderr)

    # ------------------------------------------------------------------ parsing
    @classmethod
    def _page(cls, s) -> str:
        """First page, comparable across sources. '329-335' and '064304' -> '329', '64304'."""
        t = str(s or "").strip()
        m = cls._RANGE.match(t)
        if m:
            t = m.group(1)
        t = t.split(",")[0].strip()
        head = re.match(r"^([A-Za-z]?)0*(\d+)$", t)
        return (head.group(1) + head.group(2)).lower() if head else t.lower()

    @classmethod
    def locator(cls, s: str) -> tuple | None:
        """(volume, first_page, year, journal_head) from a citation string.

        Years are stripped before the volume search: "J. Comput. Phys., 2015,
        302, 329-335" otherwise reads 2015 as the volume and 302 as the page,
        which matches a real but wrong article.
        """
        raw = str(s or "")
        years = [int(y) for y in cls._YEAR4.findall(raw)]
        year = years[-1] if years else None
        t = cls._YEAR4.sub("  ", raw)
        m = cls._VOLISS.search(t) or cls._VOLPAGE.search(t)
        if not m:
            return None
        vol, page = m.group(1).lstrip("0") or "0", cls._page(m.group(2))
        if not page or page in ("0",):
            return None
        return vol, page, year, t[:m.start()].strip(" ,.;:")

    _STOPW = frozenset("of the and for in on a an".split())

    @staticmethod
    def _journal_ok(head: str, container: str) -> bool:
        """Does the cited venue correspond to this container title?

        Journals are shortened two incompatible ways and both are common here:

          TRUNCATION   "Phys. Rev. C"  -> "Physical Review C"
          INITIALISM   "ApJ"           -> "The Astrophysical Journal"
                       "MNRAS"         -> "Monthly Notices of the Royal
                                           Astronomical Society"

        Only the first was handled, and astronomy is almost entirely the second,
        so correct rank-1 hits with an exactly matching volume AND page were being
        thrown away on the venue test alone. Measured: 2 of 25 identified before
        this, and astronomy is the largest venue group in physics_train.

        Loose on purpose. It is a corroborating test, not the identifying one:
        volume plus first page plus year is already close to a primary key, so
        this only has to rule out the case where that key collides across two
        different journals.
        """
        if not container:
            return False
        # Drop the leading author list; what sits against the volume is the venue.
        h_all = re.sub(r"[^A-Za-z ]+", " ", str(head or "")).lower().split()
        c = re.sub(r"[^A-Za-z ]+", " ", str(container or "")).lower().split()
        if not h_all or not c:
            return False

        h = [w for w in h_all if len(w) > 1][-6:]
        if h:
            hit = sum(1 for w in h if any(f.startswith(w) or w.startswith(f) for f in c))
            if hit >= max(1, int(len(h) * 0.5)):
                return True

        # Initialism: the token adjacent to the volume, letter by letter, must be
        # a subsequence of the container's significant words, starting at its
        # first letter. "mnras" threads through "monthly notices royal
        # astronomical society"; "apj" threads through "astrophysical journal".
        flat = "".join(w for w in c if w not in Crossref._STOPW)

        # Candidates, in order of specificity. The second exists because
        # punctuation-separated initialisms shatter into single letters:
        # "A&A" normalises to ["a", "a"], which the first candidate skips.
        cands = []
        for w in reversed(h_all):
            if len(w) >= 2:
                cands.append(w)
                break
        run = []
        for w in reversed(h_all):
            if len(w) != 1:
                break
            run.insert(0, w)
        if len(run) >= 2:
            cands.append("".join(run))

        for w in cands:
            if not flat.startswith(w[0]):
                continue
            i = 0
            for ch in flat:
                if i < len(w) and ch == w[i]:
                    i += 1
            if i == len(w):
                return True
        return False

    # ------------------------------------------------------------------ network
    # Crossref's own pacing, INDEPENDENT of the shared Http limiter.
    #
    # That limiter serialises every request in the stage behind one lock at
    # 0.12s -- 8.3/sec across OpenAlex, S2, arXiv and Crossref TOGETHER -- because
    # it is tuned to OpenAlex's polite pool. Crossref's polite pool allows roughly
    # 50/sec, so routing through it throttled this route about 6x, and made it
    # compete with the very OpenAlex calls it exists to replace. Measured on
    # physics_train: 2.9 ref/sec, 45 minutes for 9,269 references.
    #
    # The mailto is what buys the polite pool. Without one, Crossref serves you
    # from the shared pool, which is both slower and rate limited, so raising the
    # worker count without setting CROSSREF_MAILTO buys very little.
    DELAY = 0.02

    def _fetch(self, url: str) -> bytes | None:
        """One Crossref request, paced and retried on its own terms."""
        with self._pace_lock:
            gap = time.monotonic() - self._last
            if gap < self.DELAY:
                time.sleep(self.DELAY - gap)
            self._last = time.monotonic()
        for attempt in range(3):
            try:
                ua = (f"sir4-stage4 (mailto:{self.mailto})" if self.mailto
                      else "sir4-stage4")
                req = urllib.request.Request(url, headers={"User-Agent": ua})
                with urllib.request.urlopen(req, timeout=30) as r:
                    return r.read()
            except urllib.error.HTTPError as e:
                # 400 is a query Crossref will never accept; 404 is no such
                # resource. Neither improves by asking again.
                if e.code in (400, 404):
                    return None
                time.sleep(0.5 * 2 ** attempt)
            except (urllib.error.URLError, TimeoutError):
                time.sleep(0.5 * 2 ** attempt)
        return None

    def _query(self, s: str) -> list:
        params = {"query.bibliographic": s[:400], "rows": str(self.ROWS),
                  "select": "DOI,title,container-title,volume,page,issued,type"}
        if self.mailto:
            params["mailto"] = self.mailto
        with self._lock:
            self.calls += 1
        raw = self._fetch(f"{self.API}?{urllib.parse.urlencode(params)}")
        if not raw:
            return []
        try:
            return (json.loads(raw).get("message") or {}).get("items") or []
        except (ValueError, AttributeError):
            return []

    def resolve(self, s: str) -> dict | None:
        """{doi, title, year, container} for a title-less citation, or None.

        None means "not confidently identified", which is the correct outcome for
        a gold set: an unresolved reference is visible, a wrong one is not.
        """
        loc = self.locator(s)
        if not loc:
            return None
        vol, page, year, head = loc
        for it in self._query(s):
            cv = str(it.get("volume") or "").lstrip("0") or "0"
            cp = self._page(it.get("page"))
            parts = (it.get("issued") or {}).get("date-parts") or [[]]
            cy = parts[0][0] if parts and parts[0] else None
            cont = (it.get("container-title") or [""])[0]
            if cv != vol or cp != page:
                continue
            if year and cy and abs(int(cy) - year) > self.YEAR_SLACK:
                continue
            if not self._journal_ok(head, cont):
                continue
            doi = str(it.get("DOI") or "").lower()
            if not doi:
                continue
            with self._lock:
                self.matched += 1
            return {"doi": doi, "title": (it.get("title") or [""])[0],
                    "year": cy, "container": cont}
        with self._lock:
            self.rejected += 1
        return None


# ======================================================================== records
MISS = {"found_title": None, "found_abstract": None, "found_doi": None,
        "match_quality": "none", "similarity": 0.0, "abstract_source": None,
        "insp_domain": None, "insp_field": None, "insp_subfield": None,
        "insp_topic_fields": None, "field_purity": None,
        "field_confidence": None, "openalex_id": None, "year": None,
        "rejected_title": None, "rejected_tier": None,
        "rejected_reason": None, "identity": None, "reference_parse": None}


def _reject(route: str, score: float, title: str) -> dict:
    """A candidate that did not clear ACCEPT. Kept, so it can be reviewed.

    Discarding it would leave no way to tell "nothing in either index matches
    this string" from "the best hit was a near miss", and those need different
    remedies: the first is a bibliography defect, the second is a threshold.
    """
    return {**MISS, "route": route, "similarity": round(score, 3),
            "rejected_title": title or None, "rejected_tier": tier(score)}


def _from_openalex(w: dict | None, query: str, route: str) -> dict:
    if not w:
        return {**MISS, "route": route}
    title = w.get("display_name") or ""
    # PRECISION FIX 3. An identifier route is not scored: the id names the paper.
    score = 1.0 if route in ID_ROUTES else similarity(query, title)
    if not accepted(score):                          # PRECISION FIX 4
        return _reject(route, score, title)
    pt = w.get("primary_topic") or {}
    topic_fields = dict(Counter(
        (t.get("field") or {}).get("display_name")
        for t in (w.get("topics") or []) if (t.get("field") or {}).get("display_name")))
    out = {
        "found_title": title,
        "found_abstract": abstract_from_inverted(w.get("abstract_inverted_index")),
        "found_doi": (w.get("doi") or "").replace("https://doi.org/", "") or None,
        "match_quality": tier(score),
        "similarity": round(score, 3),
        "route": route,
        # All three levels of OpenAlex's hierarchy, from one response. Domain is
        # what makes "which cross-domains are near and which are far" answerable:
        # Computer Science -> Mathematics stays inside Physical Sciences, while
        # Computer Science -> Medicine crosses into Health Sciences.
        "openalex_id": w.get("id"),
        "insp_domain": ((pt.get("domain") or {}).get("display_name")),
        "insp_field": ((pt.get("field") or {}).get("display_name")),
        "insp_subfield": ((pt.get("subfield") or {}).get("display_name")),
        "insp_topic_fields": topic_fields,
        "field_source": "openalex_primary_topic",
        # Which source supplied the abstract, or None. Stage 5 needs to tell a
        # real document from a title-only one, and after backfilling the answer
        # is no longer implied by the route.
        "abstract_source": "openalex" if w.get("abstract_inverted_index") else None,
        "year": w.get("publication_year"),
        "rejected_title": None, "rejected_tier": None,
    }
    # PRECISION FIX 8. Computed here, from the topics of this response, so the
    # output file can be audited without the cache that produced it.
    out["field_purity"] = field_purity(out)
    out["field_confidence"] = field_confidence(out)
    return out


def _from_s2(h: dict | None, query: str, route: str) -> dict:
    """S2 has no field taxonomy, so a paper resolved here carries no domain."""
    if not h:
        return {**MISS, "route": route}
    title = h.get("title") or ""
    score = 1.0 if route in ID_ROUTES else similarity(query, title)   # PRECISION FIX 3
    if not accepted(score):                                           # PRECISION FIX 4
        return _reject(route, score, title)
    ext = h.get("externalIds") or {}
    doi = ext.get("DOI")
    if not doi and ext.get("ArXiv"):
        doi = f"arXiv:{ext['ArXiv']}"
    return {**MISS,
            "found_title": title,
            "found_abstract": h.get("abstract"),
            "found_doi": doi,
            "match_quality": tier(score),
            "similarity": round(score, 3),
            "route": route,
            "abstract_source": "s2" if h.get("abstract") else None,
            "year": h.get("year")}


def _degrade(step: str, exc: Exception) -> None:
    """A backfill step that cannot run. Report it, do not take the stage down.

    Backfill is enrichment: it adds abstracts and field labels to records that
    are ALREADY resolved. Losing it costs coverage, not correctness, so an API
    running out of budget here must not discard the resolution work that
    preceded it. It did once: an uncaught BudgetExhausted from `by_dois` killed
    a 2h35m run at the final step, after every lookup had succeeded.
    """
    print(f"  !! {step} could not run: {str(exc)[:200]}\n"
          f"     Records stay resolved but keep whatever they already had. "
          f"Rerun once the budget is back;\n     the title cache means only "
          f"this step repeats.", file=sys.stderr)


# Concurrency for the Crossref backfill. THREE, and not more, because Crossref
# degrades under burst by answering rather than refusing: a throttled reply comes
# back parseable and empty, which this code cannot tell from "no such paper", so
# over-threading does not slow the route down -- it silently LOSES MATCHES.
#
# Measured on 48 real unresolved physics references:
#
#     workers   throughput   identified
#        1        1.62/s      21/48  44%
#        2        2.64/s      22/48  46%
#        4        2.92/s      19/48  40%
#        8        2.95/s      14/48  29%
#       12        3.49/s      12/48  25%
#       24        4.28/s       6/48  12%
#
# Throughput plateaus near 2.9/s whatever you do, so everything above ~4 workers
# is paid for entirely in lost golds. An earlier default of 12 was chosen from a
# mock harness that always returned 200, which could not show this.
CROSSREF_WORKERS = int(os.environ.get("CROSSREF_WORKERS", "3"))


def backfill_crossref(results: dict, queries: dict, cr: Crossref,
                      oa: OpenAlex) -> int:
    """Identify title-less citations through Crossref. Returns how many landed.

    Runs BEFORE the other backfills so anything it recovers is just another
    resolved record by the time they see it, and picks up its OpenAlex fields
    and abstract from the same batched call as everything else.

    Only touches records that resolved to nothing, so it can add coverage but
    never overwrite a verdict another route reached.
    """
    todo = [(k, queries.get(k) or "") for k, r in results.items()
            if r.get("match_quality") == "none"]
    todo = [(k, q) for k, q in todo if q and Crossref.locator(q)]
    if not todo:
        return 0
    print(f"  crossref:          {len(todo):,} unresolved reference(s) carry a "
          f"journal/volume/page locator")
    hits: dict = {}
    lock = threading.Lock()
    bar = tqdm(total=len(todo), desc="  crossref", unit="ref", dynamic_ncols=True,
               leave=False, disable=len(todo) < 20)

    def one(item):
        key, q = item
        try:
            got = cr.resolve(q)
        except (TransientError, BudgetExhausted):
            got = None
        with lock:
            if got:
                hits[key] = got
            bar.update(1)
            bar.set_postfix_str(f"identified {len(hits):,}", refresh=False)

    # CONCURRENT, because this loop is latency-bound and was running serially.
    #
    # A Crossref bibliographic query takes ~1.4s of round trip, of which the
    # shared Http limiter accounts for 0.12s. Run one at a time that is 1.5s per
    # reference: 53 minutes for matsci_train's 2,125. The limiter already caps
    # the aggregate rate, so threads only overlap the waiting and the ceiling
    # stays where it was -- about 8/sec, or roughly 4 minutes for the same work.
    #
    # Sized to the limiter rather than to the machine: more workers than the
    # rate allows just queue inside `_wait()`.
    try:
        with ThreadPoolExecutor(max_workers=CROSSREF_WORKERS) as pool:
            list(pool.map(one, todo))
    except KeyboardInterrupt:
        print(f"\n  crossref: stopped early, keeping {len(hits):,} identification(s)",
              file=sys.stderr)
    finally:
        bar.close()
    if not hits:
        print(f"  crossref:          identified 0 of {len(todo):,} "
              f"({cr.calls:,} queries)")
        return 0

    # One batched OpenAlex call per 50, exactly like every other bulk step. This
    # is what supplies the topic fields the same/cross policy reads; a Crossref
    # DOI alone would resolve the gold but leave it unlabelled.
    try:
        works = oa.by_dois(sorted({v["doi"] for v in hits.values()}))
    except (BudgetExhausted, TransientError) as exc:
        _degrade("crossref openalex enrichment", exc)
        works = {}
    n = 0
    for key, got in hits.items():
        w = works.get(got["doi"])
        rec = (_from_openalex(w, got["doi"], "crossref") if w
               else {**MISS, "route": "crossref",
                     "found_title": got["title"] or None,
                     "found_doi": got["doi"],
                     # EXACT is wrong here and STRONG is the honest label: the
                     # locator matched, which is a tighter test than the title
                     # routes pass, but no title was ever compared.
                     "match_quality": "strong", "similarity": 0.0})
        rec["route"] = "crossref"
        rec["crossref_locator"] = True
        results[key] = rec
        n += 1
    print(f"  crossref:          identified {n:,} of {len(todo):,} "
          f"({cr.calls:,} queries, {cr.rejected:,} rejected on locator mismatch)")
    return n


def backfill_all(results: dict, oa: OpenAlex, s2: S2, ax: "ArXiv | None",
                 verbose: bool = False, cr: "Crossref | None" = None,
                 queries: dict | None = None) -> None:
    """Fill missing abstracts and domain labels for a whole run, in batches.

    This used to run per title, inside the resolve loop: one extra S2 call and
    one extra OpenAlex call for each gap, serially. Both APIs take batches, so
    doing it once at the end turns ~7,000 requests on a full split into ~90.

    arXiv has no id-free batch endpoint, but its parser takes boolean OR, so it
    batches too. It still runs last: a three-second courtesy limit makes even a
    batched request dearer than an S2 or OpenAlex one, and it is worth spending
    only on the gaps the two cheap sources leave.
    """
    # 0. Crossref FIRST, so anything it identifies is an ordinary resolved record
    # by the time `live` is computed and picks up its abstract below.
    if cr is not None and queries:
        try:
            backfill_crossref(results, queries, cr, oa)
        except (BudgetExhausted, TransientError) as exc:
            _degrade("crossref backfill", exc)

    live = [r for r in results.values() if r.get("match_quality") != "none"]
    doi_of = lambda r: (str(r.get("found_doi") or "").lower()
                        if r.get("found_doi") and not str(r["found_doi"]).startswith("arXiv:")
                        else None)

    # 1. the heavy fields, from OpenAlex, batched 50 per request.
    #
    # This now serves TWO populations, because the title search went lean:
    #   - anything S2 resolved, which never had an OpenAlex field
    #   - anything the title search resolved, which no longer carries the
    #     abstract or the topic list
    # Both are one batched lookup per 50 records, against one full-payload
    # request per candidate before.
    need_f = [r for r in live
              if (not r.get("insp_field") or not r.get("found_abstract"))
              and doi_of(r)]
    if need_f:
        try:
            got = oa.by_dois([doi_of(r) for r in need_f])
        except (BudgetExhausted, TransientError) as exc:
            _degrade("openalex backfill (fields and abstracts)", exc)
            got = {}
        n = nab = 0
        for r in need_f:
            w = got.get(doi_of(r)) or {}
            if not r.get("found_abstract"):
                ab = abstract_from_inverted(w.get("abstract_inverted_index"))
                if ab:
                    r["found_abstract"] = ab
                    r["abstract_source"] = "openalex"
                    nab += 1
            pt = w.get("primary_topic") or {}
            if pt and not r.get("insp_field"):
                r["insp_domain"] = (pt.get("domain") or {}).get("display_name")
                r["insp_field"] = (pt.get("field") or {}).get("display_name")
                r["insp_subfield"] = (pt.get("subfield") or {}).get("display_name")
                # PRECISION FIX 8. A field arriving by backfill needs its purity
                # too, or the 2/3 test silently passes everything S2 resolved.
                r["insp_topic_fields"] = dict(Counter(
                    (t.get("field") or {}).get("display_name")
                    for t in (w.get("topics") or [])
                    if (t.get("field") or {}).get("display_name")))
                r["field_purity"] = field_purity(r)
                r["field_confidence"] = field_confidence(r)
                r["openalex_id"] = r.get("openalex_id") or w.get("id")
                r["field_source"] = "oa_doi_backfill"
                n += 1
        print(f"  openalex backfill: {n} fields, {nab} abstracts, "
              f"{len(need_f)} records in "
              f"{-(-len(need_f) // OpenAlex.BATCH)} request(s)")

    # 2. abstracts from S2, for what OpenAlex did not have.
    #    Runs SECOND: OpenAlex is the primary source and its pass is already
    #    fetching these records, so asking S2 first would query it for
    #    thousands of abstracts that arrive free one step later.
    need_ab = [r for r in live if not r.get("found_abstract") and doi_of(r)]
    if need_ab:
        try:
            got = s2.batch([f"DOI:{doi_of(r)}" for r in need_ab])
        except (BudgetExhausted, TransientError) as exc:
            _degrade("s2 abstract backfill", exc)
            got = {}
        n = 0
        for r in need_ab:
            rec = got.get(f"DOI:{doi_of(r)}")
            if rec and rec.get("abstract"):
                r["found_abstract"] = rec["abstract"]
                r["abstract_source"] = "s2_doi_backfill"
                n += 1
        print(f"  abstract backfill: S2 filled {n}/{len(need_ab)} "
              f"in {-(-len(need_ab) // S2.BATCH)} request(s)")

    # 3. arXiv, last and slowest per request, for what neither could supply
    if ax:
        still = [r for r in live if not r.get("found_abstract") and r.get("found_title")]
        if still:
            reqs = -(-len(still) // ArXiv.BATCH)
            print(f"  arxiv fallback:    {len(still)} titles in {reqs} request(s) at "
                  f"{ArXiv.DELAY}s each (~{reqs * ArXiv.DELAY / 60:.1f} min). "
                  f"--no-arxiv skips this")
            # One title can appear on two records, and the map is keyed by
            # title, so the lookup is deduplicated but the assignment is not.
            try:
                got = ax.abstracts_for(sorted({r["found_title"] for r in still}))
            except (BudgetExhausted, TransientError) as exc:
                _degrade("arxiv abstract backfill", exc)
                got = {}
            n = 0
            for r in still:
                a = got.get(r["found_title"])
                if a:
                    r["found_abstract"] = a
                    r["abstract_source"] = "arxiv_backfill"
                    n += 1
            print(f"  arxiv fallback:    filled {n}/{len(still)} "
                  f"in {ax.calls} request(s) "
                  f"({ax.empty} batch(es) empty, {ax.refused} refused)"
                  + ("  [STOPPED EARLY]" if ax.stopped else ""))
            if ax.stopped:
                raise KeyboardInterrupt          # the caller marks the run partial
            if n and n * 20 < len(still):
                print(f"     that is {n / len(still) * 100:.1f}%. These are the "
                      f"titles OpenAlex and S2 both missed, so most are not on\n"
                      f"     arXiv at all. --no-arxiv skips the step and costs "
                      f"you {n} abstract(s).")


def domain_relation(target_field: str | None, insp_field: str | None) -> str | None:
    """Binary, at OpenAlex's FIELD level.

    Field is where SIR-4's five splits sit. OpenAlex's own `domain` level would
    put cs, physics, matsci and maths all inside "Physical Sciences", collapsing
    four of the five into one bucket.

    PRECISION FIX. The values used to be "same_domain"/"cross_domain" while the
    comparison was, and still is, between FIELDS. A thesis sentence reading
    "cross_domain in 44.8% of golds" off this field would be claiming something
    the number does not support, and `domain_distance` right below reports that
    only 8.6% of them leave the OpenAlex domain. The names now say field.
    """
    if not target_field or not insp_field:
        return None
    return "same_field" if norm(target_field) == norm(insp_field) else "cross_field"


def domain_distance(target_field: str | None, insp_field: str | None,
                    field_to_domain: dict) -> str | None:
    """Three bands, so a cross-domain gold can be read as near or far.

    The binary label answers "did it leave the field". It cannot distinguish a
    computer science paper borrowing from mathematics, a neighbouring field
    inside the same OpenAlex domain, from one borrowing from medicine, which is
    not. Those are different claims about how far an idea travelled, and the
    first full run showed why it matters: 44.8% binary cross-domain, of which
    only 8.3% actually left Physical Sciences.

    `field_to_domain` is learned from the responses themselves rather than
    written down here: every resolved inspiration reports its own domain and
    field together, so the mapping accumulates for free and cannot drift out of
    step with whatever taxonomy version OpenAlex is serving.
    """
    if not target_field or not insp_field:
        return None
    if norm(target_field) == norm(insp_field):
        return "same_field"
    td = field_to_domain.get(norm(target_field))
    idm = field_to_domain.get(norm(insp_field))
    if not td or not idm:
        return "cross_field_unknown_domain"
    return "cross_field_same_domain" if td == idm else "cross_domain"


def resolution_status(rec: dict) -> str:
    """Restate the uniqueness verdict against what actually resolved.

    PRECISION FIX 9. Stage 3 decides uniqueness from decompositions it validated
    as TEXT. Resolution then fails on some of the papers those decompositions
    name, and a family that supported two complete alternatives before Stage 4
    may support one afterwards. Five samples in the first run still read
    `uniqueness = falsified` on the strength of alternatives that no longer have
    documents behind them.

    Deliberately NOT rewritten to "not_falsified". Failing to find a paper in
    OpenAlex is a fact about OpenAlex, not evidence that the decomposition is
    unique, and collapsing the two would turn a lookup failure into a claim.
    """
    u = rec.get("uniqueness") or {}
    if u.get("uniqueness") != "falsified":
        return u.get("uniqueness") or "unknown"
    complete = 0
    for m in (u.get("M") or []):
        insps = m.get("inspiration") or []
        if insps and all(x.get("match_quality") not in (None, "none") for x in insps):
            complete += 1
    return "falsified" if complete >= 2 else "resolution_incomplete"


def all_inspirations(rec: dict) -> list:
    """The primary set AND every alternative in M.

    M is the contribution: a family of valid decompositions for one hypothesis.
    Resolving only the primary set leaves every alternative as a bare title, so
    the family cannot be written in document ids and Stage 5 cannot emit it.
    Measured before this was fixed: 275 alternative inspirations, 0 resolved, 91
    titles appearing nowhere else.

    The same dict object is returned once even when it sits in several members
    of M, so updating it in place updates every reference. The title cache makes
    the shared ones free anyway; only the 91 unique titles cost a lookup.
    """
    out, seen = [], set()
    groups = [rec.get("inspiration") or []]
    groups += [m.get("inspiration") or []
               for m in ((rec.get("uniqueness") or {}).get("M") or [])]
    for g in groups:
        for x in g:
            if id(x) not in seen:
                seen.add(id(x))
                out.append(x)
    return out


# ======================================================================= resolving
def resolve_title(raw: str, oa: OpenAlex, s2: S2, cache: dict,
                  target_field: str | None = None,
                  ax: "ArXiv | None" = None,
                  prefetch: dict | None = None,
                  known_doi: str | None = None,
                  s2_first: bool = False) -> dict:
    """Resolve one reference-list string. Cached by normalised query.

    Importable, and deliberately independent of the target: the cache is shared
    across every paper in the run, so a title cited by two papers is fetched
    once. The domain label is applied after the cache lookup, since it depends
    on which paper is asking.
    """
    key = norm(raw)
    if not key:
        return {**MISS, "route": "empty", "domain_relation": None}

    if key in cache:
        cache_hit = True
        out = cache[key]
    else:
        cache_hit = False
        out = _lookup(raw, oa, s2, prefetch, known_doi, s2_first)
        cache[key] = out

    out = dict(out)
    out["cache_hit"] = cache_hit
    out["domain_relation"] = domain_relation(target_field, out.get("insp_field"))
    return out


def ref_doi(raw: str) -> tuple:
    """(doi, route) for whatever identifier a reference string carries.

    Shared by the batched prefetch and by `_lookup`, so the two cannot disagree
    about which DOI a reference names.
    """
    m = _DOI.search(raw)
    if m:
        return m.group(1).rstrip(".").lower(), "doi"
    a = _ARXIV.search(raw)
    if a:
        return f"10.48550/arxiv.{a.group(1)}".lower(), "arxiv"
    return None, None


def _lookup(raw: str, oa: OpenAlex, s2: S2, prefetch: dict | None = None,
            known_doi: str | None = None, s2_first: bool = False) -> dict:
    out: dict | None = None
    pre = prefetch or {}

    # 0. a DOI Stage 2b already matched to this reference. An identifier from
    #    the publisher's own deposited reference list, so it is trusted exactly
    #    like one found in the string, and it is served from the batch.
    if known_doi:
        w = pre[known_doi] if known_doi in pre else oa.by_doi(known_doi)
        out = _from_openalex(w, raw, "doi")
        if out["match_quality"] != "none":
            out["identity"] = {"basis": "stage2b_reference", "reasons": [],
                               "n_title_matches": None, "passed_over": []}
            return out

    # 1. an explicit DOI is the only identifier that cannot be a coincidence.
    #    Served from the batched prefetch when it is available, so this route
    #    costs no request of its own.
    doi, route = ref_doi(raw)
    if route == "doi":
        w = pre[doi] if doi in pre else oa.by_doi(doi)
        out = _from_openalex(w, raw, "doi")

    # 2. an arXiv id, through OpenAlex's DOI shim and then S2's own index
    if (not out or out["match_quality"] == "none") and (a := _ARXIV.search(raw)):
        doi = f"10.48550/arxiv.{a.group(1)}".lower()
        w = pre[doi] if doi in pre else oa.by_doi(doi)
        out = _from_openalex(w, raw, "arxiv")
        if out["match_quality"] == "none":
            out = _from_s2(s2.by_id(f"arXiv:{a.group(1)}"), raw, "arxiv_s2")

    if out and out["match_quality"] != "none":
        return out

    # 3. no identifier. If the string is a journal location rather than a title,
    #    a title matcher will answer confidently and wrongly, so stop here.
    if is_citation_string(raw):
        return {**MISS, "route": "citation_string"}

    ref = parse_reference(raw)
    title = ref["title"]

    # 3b. S2 BEFORE OpenAlex, when asked. Route order only; every candidate is
    #     still held to the same similarity bar, so this trades money for time
    #     and nothing else.
    #
    #     The title search is the one lookup NEITHER service can batch: OpenAlex
    #     bills $0.001 for each, S2 bills nothing but allows one per second. On
    #     physics_train's 16,821 title-route references that is $16.82 and 34
    #     minutes against $0 and 4.7 hours.
    #
    #     The domain labels survive because they never came from this call: a
    #     hit here carries a DOI, and `backfill_all` fetches OpenAlex topics 50
    #     DOIs at a time. So OpenAlex spend drops from one request per title to
    #     one per fifty, rather than to zero. Resolving through S2 with NO
    #     OpenAlex at all is what happens when the budget dies mid-run, and it
    #     is not the same thing: it leaves the same/cross stratum unlabelled.
    if s2_first:
        out = _from_s2(s2.match(title), title, "s2_match")
        if out["match_quality"] != "none":
            out["identity"] = {"basis": "s2_unverified", "reasons": [],
                               "n_title_matches": None}
            out["reference_parse"] = ref
            return out

    # 4. OpenAlex title search, scored and DISAMBIGUATED locally.
    #
    # IDENTITY FIX. This used to keep the single best title score and stop. That
    # is only sound if a title names one work, and for books, textbook editions,
    # and conference-then-journal pairs it does not:
    #
    #   "Quantum Optics"    Vogel & Welsch 2006 AND Scully & Zubairy 1997
    #   "Modal model theory" Hamkins & Woloszyn 2024 AND Chang 1973
    #
    # Both score 1.0, and which one `by_title` puts first is a property of
    # OpenAlex's relevance ranking, not of the reference. So: keep everything
    # tied at the top, then let the reference's own year, authors and publisher
    # choose. If they cannot, the reference does not identify a paper and it
    # goes to review rather than to a coin flip.
    cands = []
    for w in oa.by_title(title, n=8):
        s = similarity(title, w.get("display_name") or "")
        if accepted(s):
            cands.append((s, w, corroborate(ref, w)))
    if cands:
        # "Share this title" means the SAME TITLE, not a similar score. A 0.02
        # window pulled genuinely different papers into the ambiguity pool, and
        # one outlier there was enough to refuse the whole lookup. Ambiguity is
        # a property of identical titles; anything else is just a worse match.
        best = max(cands, key=lambda c: (c[0], rank_key(c[1])))
        btitle = norm(best[1].get("display_name"))
        tied = [c for c in cands if norm(c[1].get("display_name")) == btitle]

        # 4a. drop anything that disagrees with the reference about a fact
        ok = [c for c in tied if c[2]["verdict"] != "contradict"]
        if not ok:
            best = max(tied, key=lambda c: c[0])
            return {**MISS, "route": "oa_title", "similarity": round(best[0], 3),
                    "rejected_title": best[1].get("display_name"),
                    "rejected_tier": tier(best[0]),
                    "rejected_reason": "contradicted: "
                                       + "; ".join(best[2]["reasons"])[:120],
                    "reference_parse": ref}

        # 4b. an index stub loses to a real work with the same title
        real = [c for c in ok if not is_stub(c[1])]
        pool = real or ok

        chosen, why = None, None
        clusters = cluster_works([c[1] for c in pool])
        if len(pool) == 1:
            chosen, why = pool[0], "unique"
        elif len(clusters) == 1:
            # 4c. several RECORDS of one work: preprint and published version,
            # or two indexings. Choosing between them is free.
            #
            # Clustered, not all-pairs. `same_work` is not transitive, so
            # requiring every pair to agree let a single odd record veto an
            # otherwise obvious duplicate group.
            chosen, why = max(pool, key=lambda c: rank_key(c[1])), "duplicate_records"
        else:
            # 4d. genuinely different works. The reference must pick one.
            supported = [c for c in pool if c[2]["verdict"] == "support"]
            if supported:
                chosen = max(supported, key=lambda c: rank_key(c[1]))
                why = "corroborated"
            else:
                # Several works share this exact title and the reference names
                # no year, author or venue to separate them. RESOLVE ANYWAY,
                # to the best-ranked record, and record what was passed over.
                #
                # Refusing here was measured and it is the wrong trade: it lost
                # 8 references whose titles matched a real paper exactly, and a
                # missing gold is a silent hole in the benchmark while a noisy
                # one is at least visible and auditable. The alternatives are
                # kept on the record so `identity.basis == "title_only"` can be
                # excluded from a strict analysis later, which is a choice the
                # data still supports. Dropping them is not reversible.
                chosen = max(pool, key=lambda c: rank_key(c[1]))
                why = "title_only"

        out = _from_openalex(chosen[1], title, "oa_title")
        if out["match_quality"] != "none":
            out["identity"] = {
                "basis": why, "reasons": chosen[2]["reasons"],
                "n_title_matches": len(tied),
                # What was passed over, when the choice was not forced. Present
                # only for `title_only`, so a strict analysis can exclude those
                # rows or a human can check them, without a rerun.
                "passed_over": ([{"doi": (c[1].get("doi") or "").replace(
                                      "https://doi.org/", "") or None,
                                  "year": c[1].get("publication_year"),
                                  "cited_by": c[1].get("cited_by_count")}
                                 for c in pool if c is not chosen]
                                if why == "title_only" else [])}
            out["reference_parse"] = ref
            return out

    # 5. S2's title matcher, which finds things OpenAlex's search does not.
    #    Skipped when it already ran as step 3b; asking the same endpoint the
    #    same question twice costs a second of the 1 req/s budget for nothing.
    if s2_first:
        return {**MISS, "route": "oa_title", "reference_parse": ref}

    #    No corroboration is possible here: S2's match endpoint returns one
    #    answer with no alternatives, so a tie cannot even be detected. Recorded,
    #    so these are separable in an audit.
    out = _from_s2(s2.match(title), title, "s2_match")
    if out["match_quality"] != "none":
        out["identity"] = {"basis": "s2_unverified", "reasons": [],
                           "n_title_matches": None}
        out["reference_parse"] = ref
    return out


# ========================================================================== report
def report(rows: list, policy: str = L.DEFAULT_POLICY) -> None:
    n = len(rows)
    if not n:
        print("\nnothing resolved")
        return
    hit = sum(1 for r in rows if r["match_quality"] != "none")
    abst = sum(1 for r in rows if r.get("found_abstract"))
    qual = Counter(r["match_quality"] for r in rows)

    print(f"\n{'=' * 68}\nRESOLUTION  ({n} gold inspirations)\n")
    print(f"  resolved                  {hit:>5}  ({100 * hit / n:.1f}%)")
    for k in ("exact", "strong", "weak", "none"):
        if qual[k]:
            print(f"    {k:<22}{qual[k]:>5}  ({100 * qual[k] / n:.1f}%)")

    trans = [r for r in rows if r.get("transient")]
    if trans:
        print(f"\n  !! {len(trans)} lookups gave up on transient errors and were "
              f"NOT cached.\n     Rerun to retry only those.")

    print(f"\n  route taken:")
    for k, v in Counter(r["route"] for r in rows).most_common():
        print(f"    {str(k):<22}{v:>5}")

    print(f"\n  with an abstract          {abst:>5}  ({100 * abst / n:.1f}%)")
    print(f"  resolved, no abstract     {hit - abst:>5}   <- title-only documents")
    src = Counter(r.get("abstract_source") for r in rows if r.get("found_abstract"))
    if src:
        print(f"    abstract came from:")
        for k, v in src.most_common():
            print(f"      {str(k):<24}{v:>5}")
    ident = Counter((r.get("identity") or {}).get("basis") for r in rows
                    if r.get("match_quality") != "none")
    if ident:
        print(f"\n  how each identity was settled:")
        for k, v in ident.most_common():
            note = {"unique": "one candidate carried this title",
                    "duplicate_records": "several records of ONE work",
                    "corroborated": "the reference's year/author/venue chose",
                    "title_only": "several DIFFERENT works share the title, "
                                  "best-ranked taken",
                    "s2_unverified": "S2 returns no alternatives to compare",
                    None: "identifier route, not scored"}.get(k, "")
            print(f"    {str(k):<20}{v:>5}   {note}")
        if ident["title_only"]:
            print(f"    -> {ident['title_only']} rows are a best guess between "
                  f"same-titled works.\n       They carry identity.passed_over; "
                  f"filter on identity.basis to exclude them.")

    fs = Counter(r.get("field_source") for r in rows if r.get("field_source"))
    if fs:
        # Every OpenAlex-resolved record now records where its field came from,
        # so this is a breakdown, not a backfill count. It used to be printed as
        # "recovered by backfill" and read as though 268 of 312 needed rescuing.
        print(f"    field label came from:")
        for k, v in fs.most_common():
            print(f"      {str(k):<24}{v:>5}")

    # ------------------------------------------------------------ the domain split
    lab = Counter(r.get("domain_relation") for r in rows if r.get("domain_relation"))
    tot = sum(lab.values())
    print(f"\n{'=' * 68}\nDOMAIN  (policy: {policy!r})\n")
    if not tot:
        print("  no inspiration carries a field label yet")
    else:
        for k in ("same", "cross"):
            print(f"  {k:<26}{lab[k]:>5}  ({100 * lab[k] / tot:.1f}%)")
        print(f"  {'unlabelled':<26}{n - tot:>5}  "
              f"(unresolved, or neither side has a field)")

        # How each label was reached. The fallback rows come from a different,
        # coarser rule and must not be blended into the headline split.
        basis = Counter(r.get("label_basis") for r in rows if r.get("domain_relation"))
        if basis["primary_fallback"]:
            fb = [r for r in rows if r.get("label_basis") == "primary_fallback"]
            fbc = Counter(r["domain_relation"] for r in fb)
            clean = Counter(r["domain_relation"] for r in rows
                            if r.get("label_basis") == "topics")
            ct = sum(clean.values())
            print(f"\n  how the label was reached:")
            print(f"    {'topics':<24}{basis['topics']:>5}   full evidence, the "
                  f"{policy!r} policy")
            print(f"    {'primary_fallback':<24}{basis['primary_fallback']:>5}   "
                  f"one side had no topic list, primary fields compared")
            print(f"      of those: same {fbc['same']}, cross {fbc['cross']}")
            print(f"\n    headline over TOPIC-BASED rows only:  "
                  f"same {clean['same']}, cross {clean['cross']} "
                  f"({100 * clean['cross'] / max(ct, 1):.1f}% cross)")
            print(f"    the fallback rule is coarser: it called 40.8% of golds "
                  f"cross on the\n    pilot where {policy!r} called 12.4%. "
                  f"Filter label_basis == 'topics'\n    for a clean split, and "
                  f"report the fallback rows separately.")

        # The graded scale the policy cuts. Reported whatever the policy is, so
        # the effect of switching is visible without rerunning anything.
        band = Counter(r.get("domain_distance") for r in rows
                       if r.get("domain_distance"))
        bt = sum(band.values())
        if bt:
            print(f"\n  graded, by shared OpenAlex topic fields:")
            for k in L.BANDS:
                if band[k]:
                    print(f"    {k:<30}{band[k]:>5}  ({100 * band[k] / bt:.1f}%)")

        # What every OTHER policy would have said, from the same evidence. This
        # is the whole point of storing it: the choice stays reversible, and its
        # cost is visible here rather than discovered after the export.
        print(f"\n  the same data under every policy:")
        # The REAL stored evidence, not a reconstruction. An earlier version of
        # this table rebuilt a stand-in dict with same_openalex_domain=None,
        # which silently made `domain_jump` fall back to `disjoint` and report
        # an identical, wrong count.
        evs = [r["_evidence"] for r in rows
               if (r.get("_evidence") or {}).get("labelled")]
        for name in sorted(L.POLICIES):
            c = Counter(L.POLICIES[name](e) for e in evs)
            mark = "  <- in use" if name == policy else ""
            print(f"    {name:<14}same {c['same']:>5}   cross {c['cross']:>5}   "
                  f"({100 * c['cross'] / max(len(evs), 1):.1f}% cross){mark}")
        print(f"\n    change it later without rerunning:  "
              f"python3 build/relabel.py --benchmark <dir> --policy <name>")

        pairs = Counter(f"{r['_target_field']} -> {r['insp_field']}" for r in rows
                        if r.get("_target_field") and r.get("insp_field")
                        and norm(r["_target_field"]) != norm(r["insp_field"]))
        if pairs:
            print(f"\n  where golds come from when the two PRIMARY fields differ "
                  f"({sum(pairs.values())} of {n}).")
            print(f"  Not the same set as 'cross' above unless the policy is "
                  f"'primary':")
            for k, v in pairs.most_common(15):
                print(f"    {v:>4}  {k}")

    bad = [r for r in rows if r["match_quality"] == "none"]
    if bad:
        print(f"\n  unresolved ({len(bad)}), first 15:")
        for r in bad[:15]:
            why = r["route"] if not r.get("transient") else "TRANSIENT"
            print(f"    [{why:<16}] {r['query'][:72]}")


# ============================================================================ main
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--cache", type=Path, default=CACHE_DEFAULT)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--delay", type=float, default=0.12,
                    help="seconds between requests (default 0.12, OpenAlex polite pool)")
    ap.add_argument("--workers", type=int, default=8,
                    help="concurrent lookups (default 8). Http._wait still keeps "
                         "requests --delay apart, so this fills the gap left by "
                         "network latency rather than exceeding the rate limit")
    ap.add_argument("--no-arxiv", action="store_true",
                    help="skip the arXiv abstract fallback. It batches 12 titles "
                         "per request but is still rate limited to "
                         "one request every 3s, which is a minute on a sample and "
                         "hours on a full split")
    ap.add_argument("--no-crossref", action="store_true",
                    help="skip the Crossref locator route. It identifies "
                         "references that carry no title -- 'Phys. Rev. C, "
                         "101(6):064304, Jun 2020.' -- which is most of physics "
                         "and 70%% of what physics_train left unresolved. Free, "
                         "and only ever runs on references nothing else resolved")
    ap.add_argument("--crossref-mailto", default=None, metavar="EMAIL",
                    help="contact address for Crossref's polite pool (or set "
                         "CROSSREF_MAILTO). Without one the shared pool is "
                         "slower and more heavily rate limited")
    ap.add_argument("--s2-first", action="store_true",
                    help="ask Semantic Scholar for the TITLE search before "
                         "OpenAlex. Neither service can batch a title search: "
                         "OpenAlex bills $0.001 each, S2 bills nothing but "
                         "allows 1/second. On physics_train's 16,821 title-route "
                         "references that is $16.82 and 34min against $0 and "
                         "~4.7h. Domain labels are unaffected -- they come from "
                         "the batched by-DOI backfill, not from this call -- so "
                         "OpenAlex spend drops ~98%%, not to zero. An overnight "
                         "setting")
    ap.add_argument("--no-target-fields", action="store_true",
                    help="skip resolving each target's own field and fall back to "
                         "primary_field, which is a per-split constant")
    ap.add_argument("--label-policy", default=L.DEFAULT_POLICY,
                    choices=sorted(L.POLICIES),
                    help=f"how same/cross is decided (default {L.DEFAULT_POLICY!r}: "
                         f"cross means the two papers share no OpenAlex field). "
                         f"The EVIDENCE is stored either way, so this can be "
                         f"changed afterwards with build/relabel.py and no rerun")
    ap.add_argument("--dry-run", action="store_true", help="report, write nothing")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    if not a.dry_run and not a.output:
        ap.error("--output is required unless --dry-run")

    recs = [json.loads(l) for l in open(a.input) if l.strip()]
    if a.limit:
        recs = recs[:a.limit]
    todo = [(r, x) for r in recs for x in all_inspirations(r)
            if x.get("supposed_title")]
    print(f"{len(recs)} papers, {len(todo)} inspirations, "
          f"{len({norm(x['supposed_title']) for _, x in todo})} distinct titles")

    http = Http(a.cache, a.delay, a.verbose)
    oa, s2 = OpenAlex(http), S2(http)
    ax = None if a.no_arxiv else ArXiv(http)
    cr = None if a.no_crossref else Crossref(http, a.crossref_mailto)
    if not oa.mailto and not oa.key:
        print("  WARNING: no OPENALEX_MAILTO or OPENALEX_API_KEY. You are on the "
              "anonymous pool, which 429s immediately.", file=sys.stderr)
    print(f"  pacing: OpenAlex {1/a.delay:.1f} req/s | "
          f"S2 {1/S2.DELAY:.1f} req/s ({'key' if s2.key else 'ANONYMOUS'}) | "
          f"arXiv {1/ArXiv.DELAY:.2f} req/s   (independent limiters)")
    print(f"  cache {len(http.cache)} titles (v{CACHE_VERSION}) | "
          f"openalex {'key' if oa.key else 'mailto' if oa.mailto else 'anonymous'} | "
          f"s2 {'key' if s2.key else 'anonymous'}")

    # REBUILD FIX 2. The target's own field, resolved per paper rather than taken
    # from the split. Every record in cs_test_frozen.jsonl carries
    # primary_field="Computer Science", including papers that are plainly
    # quant-ph. Comparing a real per-paper label against that constant reported
    # 44.8% cross-domain, most of it quantum papers citing physics.
    target_field: dict = {}
    # The target's own TOPIC DISTRIBUTION, not just its top field. The label
    # policy compares topic sets, so this is half the evidence; storing only the
    # primary field is what made the old rule turn a three-way tie into a verdict.
    target_topics: dict = {}
    tdois = [] if a.no_target_fields else [r for r in recs if r.get("doi")]
    if tdois:
        fresh = [r["doi"] for r in tdois if f"__target__{norm(r['doi'])}" not in http.cache]
        print(f"\nresolving {len(tdois)} target fields "
              f"({len(fresh)} not cached, {-(-len(fresh) // OpenAlex.BATCH)} batched requests)")
        got = oa.by_dois(fresh) if fresh else {}
        for r in tdois:
            key = f"__target__{norm(r['doi'])}"
            if key not in http.cache:
                w = got.get(str(r["doi"]).strip().lower())
                http.cache[key] = _from_openalex(w, r["doi"], "doi") if w else {**MISS, "route": "doi"}
            ent = http.cache[key]
            # Stage 2 already stored this paper's topics when it was collected,
            # so a target OpenAlex cannot resolve by DOI is still labellable.
            # Measured: 6 of 71 targets, all recovered this way, coverage 100%.
            target_field[r["doi"]] = ent.get("insp_field") or r.get("primary_field")
            target_topics[r["doi"]] = (ent.get("insp_topic_fields")
                                       or r.get("topic_fields") or {})
        lab = Counter(v for v in target_field.values() if v)
        got_t = sum(1 for v in target_topics.values() if v)
        print(f"  {sum(lab.values())}/{len(tdois)} labelled: {dict(lab.most_common(6))}")
        print(f"  {got_t}/{len(tdois)} carry a topic distribution "
              f"(needed by the {L.DEFAULT_POLICY!r} label policy)")
        http.save()
    for r in recs:                      # papers with no DOI, and --no-target-fields
        target_field.setdefault(r.get("doi"), r.get("primary_field"))
        target_topics.setdefault(r.get("doi"), r.get("topic_fields") or {})

    # One pass per DISTINCT title. A paper cited by the primary set and by three
    # members of M occupies four slots; the cache already made the repeats free,
    # this makes the progress honest and the log legible.
    by_title: dict = {}
    for rec, x in todo:
        by_title.setdefault(norm(x["supposed_title"]), []).append((rec, x))
    print(f"  {len(todo)} slots over {len(by_title)} distinct titles "
          f"({len(todo) - len(by_title)} are repeats and cost nothing)")

    # ---- batched prefetch of every identifier the reference strings carry.
    #
    # The DOI and arXiv routes were the last per-item network calls in the
    # stage: one request each, inside the worker loop. Every other bulk step is
    # already batched (target fields 50/request, S2 abstracts 500, OpenAlex
    # fields 50, arXiv 12), and this closes the gap. The title route cannot be
    # batched at all, because OpenAlex has no bulk title.search, so it stays one
    # request per distinct title and dominates the count either way.
    prefetch: dict = {}
    want = {d for d, _ in (ref_doi(x["supposed_title"]) for _, x in todo) if d}

    # STAGE 2b SHORTCUT. If Stage 2b has run, each paper carries
    # `openalex_references`: its reference list as structured records with real
    # titles and DOIs, fetched in bulk. A reference matched against that list
    # needs NO title search at all, and a title search is the one request in
    # this stage that cannot be batched. On cs train that is 26,704 individual
    # searches; matched here they collapse into ~534 batched DOI lookups.
    #
    # Silently a no-op when 2b has not run, so nothing depends on it.
    known: dict = {}
    for rec in recs:
        refs = rec.get("openalex_references") or []
        if not refs:
            continue
        byt = {norm(r.get("title")): r for r in refs if r.get("title")}
        for x in all_inspirations(rec):
            t = x.get("supposed_title")
            if not t:
                continue
            key = norm(t)
            if key in known:
                continue
            ct = clean_title(t)
            hit = byt.get(norm(ct))
            if not hit:
                # Exact string equality alone left 5.2% of matchable titles on
                # the table (measured on 400 biology papers), and every miss
                # costs a full title search. The differences are trivial --
                # "Extraembryonic" against "Extra-embryonic", an author initial
                # the LLM kept -- and the file already owns a matcher tuned for
                # exactly that. Held to the SAME bar as any other match, so a
                # cheap route cannot buy a wrong paper.
                best, score = None, 0.0
                for r in refs:
                    s = similarity(ct, r.get("title") or "")
                    if s > score:
                        best, score = r, s
                if best and accepted(score):
                    hit = best
            if hit and hit.get("doi"):
                known[key] = str(hit["doi"]).lower()
    if known:
        want |= set(known.values())
        print(f"  stage 2b supplied a DOI for {len(known):,} of "
              f"{len(by_title):,} titles, so those skip the title search")
    if want:
        print(f"\nprefetching {len(want)} identifiers from the reference strings "
              f"({-(-len(want) // OpenAlex.BATCH)} batched requests)")
        try:
            prefetch = oa.by_dois(sorted(want))
            print(f"  OpenAlex had {len(prefetch)}/{len(want)}; the rest fall "
                  f"through to the title route")
        except (BudgetExhausted, TransientError) as exc:
            # This used to be the one OpenAlex call with no guard on it, so an
            # empty budget killed the whole stage HERE, before a single title was
            # looked up -- while the identical failure inside the worker loop was
            # handled and merely degraded. An optimisation must not be able to
            # fail harder than the path it optimises.
            #
            # Losing the prefetch is survivable: every DOI in it is retried by
            # the DOI route inside the loop, one request each instead of fifty
            # at a time. Slower and dearer, not wrong.
            _degrade("identifier prefetch", exc)
            print(f"  falling back to per-reference DOI lookups "
                  f"({len(want):,} of them). --s2-first avoids OpenAlex here.",
                  file=sys.stderr)
            prefetch = {}

    rows = []
    resolved: dict = {}
    tally = Counter()
    tlock = threading.Lock()
    budget_warned = [False]     # print the budget failure once, not 20,000 times

    # One line that updates in place, rather than a running log. Every other
    # stage in this pipeline uses tqdm, and a scrolling wall of matched titles
    # hides the two things worth watching: how far through it is, and whether
    # the miss rate is climbing.
    bar = tqdm(total=len(by_title), desc="resolve", unit="title",
               dynamic_ncols=True, disable=a.verbose)

    def work(item):
        """One distinct title. Runs on a worker; every shared write is atomic."""
        key, slots = item
        rec, x = slots[0]
        q = x["supposed_title"]
        tf = target_field.get(rec.get("doi")) or rec.get("primary_field")
        try:
            res = resolve_title(q, oa, s2, http.cache, tf, ax, prefetch,
                                known.get(key), a.s2_first)
        except TransientError:
            res = {**MISS, "route": "transient", "transient": True,
                   "domain_relation": None}
        except BudgetExhausted as e:
            # REBUILD FIX 3. OpenAlex being out of quota used to abort the whole
            # stage, even though S2 is a separate service that was working. One
            # bad hour cost every remaining paper.
            # The whole message, not the first 80 characters. OpenAlex states
            # the request's cost and the remaining balance in it, and the old
            # truncation cut it off at exactly "This request cost".
            if not budget_warned[0]:
                budget_warned[0] = True
                bar.write(f"\n  !! OPENALEX BUDGET EXHAUSTED. Every remaining "
                          f"lookup falls back to Semantic Scholar at 1 req/s,\n"
                          f"     which is slow AND loses the OpenAlex field "
                          f"labels. Stop, top up the API KEY\n"
                          f"     (credit is attached to the key, not the "
                          f"account), and rerun: nothing is cached\n"
                          f"     from this path, so the rerun redoes exactly "
                          f"these titles.\n     {str(e)[:400]}\n")
            try:
                ct = clean_title(q)
                res = _from_s2(s2.match(ct), ct, "s2_budget")
                res["domain_relation"] = None
            except Exception:
                res = {**MISS, "route": "transient", "transient": True,
                       "domain_relation": None}
        # Held, not applied. The batched backfill runs next and can still change
        # this record, so the slots are filled once, afterwards, in phase 3.
        resolved[key] = res
        with tlock:
            tally[res["match_quality"]] += 1
            tally["cached"] += bool(res.get("cache_hit"))
            n = sum(v for k, v in tally.items() if k != "cached")
            bar.set_postfix_str(
                f"hit {100 * (n - tally['none']) / max(n, 1):.0f}% "
                f"cache {100 * tally['cached'] / max(n, 1):.0f}%",
                refresh=False)
            bar.update(1)
        # -v keeps the old per-title log, for when a specific title is the
        # question rather than the run as a whole.
        if a.verbose:
            print(f"  [{sum(tally.values())}/{len(by_title)}] "
                  f"{res['match_quality']:<7} "
                  f"{str(res.get('domain_relation') or '-'):<13}"
                  f"{str(res.get('found_title') or q)[:52]}"
                  + (f"  x{len(slots)}" if len(slots) > 1 else ""))

    interrupted = False
    try:
        if a.workers > 1:
            with ThreadPoolExecutor(max_workers=a.workers) as pool:
                list(pool.map(work, list(by_title.items())))
        else:
            for item in by_title.items():
                work(item)
    except KeyboardInterrupt:
        # PRECISION FIX 10. The cache is still saved, so a rerun resumes cheaply,
        # but the run is now partial and must not be finalised as if it were not.
        interrupted = True
        bar.write("\ninterrupted, saving cache")
    finally:
        bar.close()
        http.save()

    # ---- phase 2: everything that needs a second source, in batches
    print(f"\nbackfilling")
    backfill_partial = False
    try:
        # The raw reference STRING, not the normalised key: Crossref needs the
        # journal, volume and page that normalisation is designed to throw away.
        backfill_all(resolved, oa, s2, ax, a.verbose, cr,
                     {k: slots[0][1]["supposed_title"]
                      for k, slots in by_title.items()})
    except Exception as exc:                      # belt and braces
        _degrade("backfill", exc)
    except KeyboardInterrupt:
        # A Ctrl-C HERE is not the same failure as a Ctrl-C in phase 1, and used
        # to be punished as if it were. Every title has already been looked up;
        # what is partial is the enrichment on top. The steps mutate records in
        # place as they go, so whatever OpenAlex and S2 filled is already on the
        # records and is kept. Killing the run instead threw all of it away and
        # made the slow arXiv tail impossible to abandon without paying twice.
        backfill_partial = True
        print("\ninterrupted during backfill. Every title still resolved; some "
              "abstracts are missing.", file=sys.stderr)
    # Routes whose result is a DEGRADED substitute, not the answer. Caching them
    # is worse than not resolving at all: the rerun you do after topping up the
    # budget serves the degraded verdict from cache and never retries it, so the
    # damage becomes permanent and invisible.
    #
    # `s2_budget` is the one that bites. It fires when OpenAlex is out of credit,
    # and S2 has no OpenAlex taxonomy, so the gold resolves but carries no
    # topic_fields -- which is precisely what the same/cross label policy reads.
    # A whole split can come out unlabelled from one bad hour of quota.
    #
    # `transient` is already excluded by the match_quality test, since it keeps
    # MISS's "none"; it is named here so the intent survives a change to that.
    DEGRADED = {"s2_budget", "transient"}
    poisoned = 0
    for key, res in resolved.items():
        if res.get("route") in DEGRADED:
            poisoned += 1
            continue
        if res.get("match_quality") != "none":
            http.cache[key] = {k: v for k, v in res.items()
                               if k not in ("cache_hit", "domain_relation")}
    http.save()
    if poisoned:
        print(f"  !! {poisoned:,} title(s) resolved through a DEGRADED route and were "
              f"NOT cached.\n     Top up the OpenAlex key and rerun; only these "
              f"repeat.", file=sys.stderr)

    # ---- phase 3: apply each result to every slot that asked for it
    for key, slots in by_title.items():
        res = resolved.get(key)
        if not res:
            continue
        for srec, sx in slots:
            stf = target_field.get(srec.get("doi")) or srec.get("primary_field")
            # PRECISION FIX 7. Everything the resolver computed, not a subset.
            # `insp_topic_fields` in particular: without it the purity behind a
            # cross-field label cannot be checked from the output file, and the
            # only copy lives in a mutable cache that the next run overwrites.
            sx.update({k: res.get(k) for k in
                       ("found_title", "found_abstract", "found_doi",
                        "match_quality", "similarity", "insp_domain",
                        "insp_field", "insp_subfield", "abstract_source",
                        "insp_topic_fields", "field_purity", "field_confidence",
                        "field_source", "openalex_id", "year",
                        "rejected_title", "rejected_tier", "rejected_reason",
                        "identity")})
            sx["resolution_route"] = res.get("route")
        rec0 = slots[0][0]
        rows.append({**res, "query": slots[0][1]["supposed_title"],
                     "_target_field": target_field.get(rec0.get("doi"))
                                      or rec0.get("primary_field"),
                     "_target_topics": target_topics.get(rec0.get("doi")) or {},
                     "_slots": len(slots)})

    # Second pass. The field -> domain map is only complete once every lookup has
    # been made, so it cannot be built inside the loop above without labelling
    # the early rows against a map that is still filling up.
    field_to_domain = {norm(r["insp_field"]): r["insp_domain"] for r in rows
                       if r.get("insp_field") and r.get("insp_domain")}
    for rec in recs:
        tf = target_field.get(rec.get("doi")) or rec.get("primary_field")
        ttop = target_topics.get(rec.get("doi")) or rec.get("topic_fields") or {}
        rec["target_field_resolved"] = target_field.get(rec.get("doi"))
        # Stored on the PAPER, not repeated on every inspiration: it is the same
        # for all of them, and relabel.py needs exactly one copy to recombine.
        rec["target_topic_fields"] = ttop
        rec["label_policy"] = a.label_policy
        for x in all_inspirations(rec):
            # The evidence, then the verdict. Keeping the evidence is what makes
            # `build/relabel.py --policy overlap2` a second's work later instead
            # of a rerun: every policy in labels.py is a pure function of this.
            ev = L.label_evidence(
                ttop, x.get("insp_topic_fields"), tf, x.get("insp_field"),
                field_to_domain.get(norm(tf)), x.get("insp_domain"))
            x["label_evidence"] = ev
            x.update(L.apply_policy(ev, a.label_policy))
            x["field_pair"] = (f"{tf} -> {x['insp_field']}"
                               if tf and x.get("insp_field") else None)
        # PRECISION FIX 9. Last, because it reads match_quality off every member
        # of M, which phase 3 has only just written.
        if rec.get("uniqueness"):
            rec["uniqueness"]["status_after_resolution"] = resolution_status(rec)
    for r in rows:
        rec0 = r.pop("_rec", None)
        ev = L.label_evidence(
            r.get("_target_topics") or {}, r.get("insp_topic_fields"),
            r.get("_target_field"), r.get("insp_field"),
            field_to_domain.get(norm(r.get("_target_field"))), r.get("insp_domain"))
        r.update(L.apply_policy(ev, a.label_policy))
        r["_evidence"] = ev

    if ax and ax.calls:
        print(f"  arxiv fallback used {ax.calls} times "
              f"({ax.calls * ArXiv.DELAY:.0f}s of its own rate limit)")
    print(f"\nnetwork calls {http.calls}, cache hits "
          f"{sum(1 for r in rows if r.get('cache_hit'))}, "
          f"rate-limit retries {http.rate_limited}")
    report(rows, a.label_policy)

    # PRECISION FIX 4. The near misses, so the threshold is auditable rather than
    # just restrictive. Anything here is a reference OpenAlex and S2 both had an
    # opinion about and neither was confident enough to be believed.
    review = sorted((r for r in rows if r.get("rejected_title")),
                    key=lambda r: -(r.get("similarity") or 0))
    if review:
        print(f"\n  {len(review)} near misses held for review "
              f"(best {review[0]['similarity']}):")
        for k, v in Counter(str(r.get("rejected_reason") or
                                "below threshold").split(":")[0]
                            for r in review).most_common():
            print(f"    {k:<28}{v:>5}")
        print("  top 8:")
        for r in review[:8]:
            print(f"    {r['similarity']:.2f} {str(r.get('rejected_tier')):<6} "
                  f"{r['query'][:40]:<40} -> {str(r['rejected_title'])[:44]}")

    if a.dry_run:
        return
    if interrupted:
        # PRECISION FIX 10. A partial file that looks complete is the one failure
        # mode nothing downstream can detect: Stage 5 would export it, and the
        # missing papers would read as unresolvable references rather than as
        # lookups that never ran.
        print(f"\nNOT writing {a.output}: the run was interrupted and this "
              f"result is partial.\nThe cache is saved, so rerunning the same "
              f"command resumes and costs only what is missing.", file=sys.stderr)
        sys.exit(1)

    # PRECISION FIX 10. Written whole, then renamed. A crash or a full disk
    # halfway through now leaves the previous output intact instead of a JSONL
    # truncated mid-line, which is how run_B_resolved.jsonl was lost once already.
    if backfill_partial:
        # Written, unlike an interrupted phase 1, because the difference is
        # detectable downstream: a record with no abstract is visibly missing an
        # abstract, whereas a record that was never looked up is indistinguishable
        # from a reference nothing could resolve. Said out loud anyway, because
        # "resolved" and "has an abstract" are not the same claim.
        n_ab = sum(1 for r in resolved.values() if r.get("found_abstract"))
        n_hit = sum(1 for r in resolved.values() if r.get("match_quality") != "none")
        print(f"\n!! backfill was interrupted: {n_ab}/{n_hit} resolved titles "
              f"carry an abstract.\n   Rerunning the same command fills the rest "
              f"from cache; only the backfill repeats.", file=sys.stderr)

    a.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.output.with_suffix(a.output.suffix + ".partial")
    with open(tmp, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    tmp.replace(a.output)
    print(f"\nwrote {len(recs)} papers -> {a.output}")

    if review:
        rv = a.output.with_suffix(".review.jsonl")
        with open(rv, "w") as f:
            for r in review:
                f.write(json.dumps({k: r.get(k) for k in (
                    "query", "rejected_title", "rejected_tier", "similarity",
                    "rejected_reason", "route", "_slots")}) + "\n")
        print(f"wrote {len(review)} near misses -> {rv}")


if __name__ == "__main__":
    main()
