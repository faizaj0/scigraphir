"""
Stage 2 -- full-text acquisition.

Inspirations are identified from the source paper's *citations*, so decomposition needs
the introduction and related work, not just the abstract. Every QUARTET domain is
restricted to sources that hand back structured text, so there is no PDF, no OCR and no
GPU stage anywhere:

  arXiv (cs, physics, matsci, maths)  ->  LaTeX source from the e-print tarball
  PubMed Central (biology)           ->  JATS XML with sections already labelled

Full text by default, plus the reference list. Both ResearchBench and TOMATO-Star
decompose from whole papers, and ResearchBench notes that inspirations appear in the
methodology and related work as well as the introduction. Use --front-matter-only to
truncate at the opening sections instead.

Usage:
  python build/02_fulltext.py --domain cs --split test --limit 20
"""
from __future__ import annotations

import argparse
import collections
import gzip
import html
import json
import os
import random
import re
import sys
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree as ET

import requests
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "domains.yaml"
IN = ROOT / "data" / "01_collected"
OUT = ROOT / "data" / "02_fulltext"
# The cap this stage actually used, written beside the data so Stage 3 and the auditor
# read one value rather than each guessing from the environment.
MANIFEST = OUT / "manifest.json"
# targets.per_domain from config/domains.yaml, filled in by main(). Used by --headroom so
# the fetch size comes from the agreed benchmark targets rather than a retyped number.
TARGETS: dict = {}
# targets.per_domain_override, for domains whose source pool cannot reach the common
# target. Materials science is the only one so far. Filled in by main().
OVERRIDES: dict = {}
# arXiv and Europe PMC both ask for a contact address in the User-Agent.
MAILTO = os.environ.get("QUARTET_MAILTO", "faizajalil0@gmail.com")
UA = {"User-Agent": f"scigraphir-sir4/0.1 (mailto:{MAILTO})"}

# Hard ceiling on stored text, and the budget --full raises to. Measured full-paper
# lengths: cs mean 26k, physics 40k, maths 54k, biology 70k with a 382k outlier. At
# 60k, --full silently truncated the average biology paper and most long maths and
# physics ones, so "FULL TEXT" was not true for a large share of the corpus. 150k holds
# essentially every real paper (~37k tokens, against a 1M context) while still capping
# the review-article tail.
def _max_chars() -> int:
    """The cap, from config/domains.yaml -- committed, shared, and identical everywhere.

    It used to be read from QUARTET_MAX_CHARS by this stage AND independently by Stage 3.
    An environment variable is per-terminal: the same corpus got fetched at 200,000 in one
    shell and 150,000 in another, and Stage 3 then cut what Stage 2 had stored. The env
    var still works as a deliberate one-off override, but it announces itself now instead
    of quietly changing the corpus.
    """
    try:
        cap = int(yaml.safe_load(open(CONFIG))["fulltext"]["max_chars"])
    except (OSError, KeyError, TypeError, ValueError):
        cap = 150_000
    env = os.environ.get("QUARTET_MAX_CHARS")
    if env and int(env) != cap:
        # IGNORED, not honoured. As an override it survived three separate attempts to
        # clear it -- an exported variable outlives the command that set it, so every
        # later run in that terminal silently fetched at a different length, and a
        # warning is no defence against a value that keeps winning. The corpus length is
        # a property of the benchmark, so it belongs in the committed config and nowhere
        # else. Change config/domains.yaml if you want a different cap.
        print(f"  !! ignoring QUARTET_MAX_CHARS={env}; the cap is {cap:,} from "
              f"config/domains.yaml. Edit that file to change it.")
    return cap


MAX_CHARS = _max_chars()
# Below this a fetch is treated as failed: an abstract-only deposit cannot support
# decomposition.
MIN_CHARS = 2_000
# Front-matter budget. Intro plus related work is where "motivated by" statements live;
# beyond that we are paying for tokens and risking answer leakage. Roughly 5k tokens.
FRONT_MATTER_CHARS = 20_000
MAX_SECTIONS = 3
# Whether to stop at the first Method/Results/Conclusion heading. Must be OFF in full
# mode. Left on, --full raised only the budget while the heading break still fired, so
# papers whose headings matched the regex were cut at the method and papers whose
# headings did not (surveys, unusual naming) kept everything up to MAX_CHARS. Measured:
# only 31% of "full text" rows contained a conclusion. That is worse than either pure
# option, because how much text a paper contributes then depends on whether its
# headings happened to match -- which correlates with paper type, not with content.
STOP_AT_METHOD = True
WORKERS = 4

# arXiv asks bulk clients to be gentle. 3s was my conservative choice and it was the
# entire bottleneck: at 3s, 100 papers take 5 minutes no matter how many workers run.
# 1s is still well inside polite use. Raise it with --rate if arXiv starts refusing.
ARXIV_INTERVAL = 1.0

# ar5iv is a separate host serving pre-rendered static HTML, so it does not compete
# with arxiv.org for the same limiter, and it is far lighter on arXiv's infrastructure
# than pulling e-print tarballs. Still arXiv Labs, so keep it moderate.
AR5IV = "https://ar5iv.labs.arxiv.org/html/"
AR5IV_INTERVAL = 0.5

# HTML markup emitted by LaTeXML, which renders arXiv's HTML papers.
# Top-level section OPENINGS only. A non-greedy `(.*?)</section>` ends at the first
# closing tag, which belongs to the first nested subsection, so everything after it
# was silently dropped from every section. We locate the openings and slice between
# them instead, which needs no balanced matching.
RE_HTML_SEC_OPEN = re.compile(r'<section[^>]*class="[^"]*ltx_section[^"]*"[^>]*>')
# Attribute order differs between the two hosts: arxiv.org/html writes class before id,
# ar5iv writes id before class. Requiring one order silently matched nothing on ar5iv.
RE_HTML_BIBITEM = re.compile(
    r'<li\b(?=[^>]*\bclass="[^"]*ltx_bibitem)(?=[^>]*\bid="([^"]*)")[^>]*>(.*?)</li>', re.S)
# Anchor on the opening tag. Matching from `class=` left `class="ltx_abstract">` in the
# text, because untag() only removes `<...>` and this fragment has no leading `<`. Every
# HTML paper therefore opened with markup, which is the first thing the extractor reads.
RE_HTML_ABSTRACT = re.compile(r'<div[^>]*class="[^"]*ltx_abstract[^"]*".*?</div>', re.S)
# Where the paper BODY ends. The last section was sliced to len(html), so it swallowed
# the bibliography, the appendices and LaTeXML's page footer: measured on CS, a third of
# HTML-route papers had their whole reference list pasted into the body, ending with
# "Generated on ... by LaTeXML". The bibliography is parsed separately and belongs in
# `bibliography`, not in the prose the extractor reads.
RE_HTML_BODY_END = re.compile(
    r'<[a-z]+[^>]*class="[^"]*ltx_(?:bibliography|page_footer)[^"]*"', re.I)
# LaTeXML cannot convert every picture. When it gives up it emits the raw LaTeX as TEXT
# inside an error span, so untag() -- which only removes <...> -- leaves it in place.
# Paper 10.1093/philmat/nkag001 came back as 150,000 characters of
# "pgfsys@stroke\pgfsys@invoke{ }..." repeated 4,243 times: the entire budget spent on
# drawing primitives, with no prose at all. This is the HTML-route twin of the pgfplots
# leak already handled on the LaTeX route by RE_COORDS / RE_PLOT_ENV.
RE_HTML_ERR = re.compile(r'<(span|div)[^>]*class="[^"]*ltx_ERROR[^"]*"[^>]*>.*?</\1>', re.S)
# MathML annotations. LaTeXML renders every formula as presentation MathML AND repeats
# it inside <annotation encoding="application/x-tex"> as the original LaTeX source (ar5iv
# adds an x-llamapun word rendering too). untag() only removes <...>, so each formula
# arrived two or three times over. One maths paper carried 2,108 annotation blocks,
# 173,542 characters. This is also where the surviving pgf commands came from: an inline
# diagram's LaTeX source lives in the annotation, not in an error span, so scrub_pgf was
# treating a symptom. Keep the presentation rendering, drop the duplicates.
RE_MATH_ANNOT = re.compile(
    r"<annotation(?:-xml)?\b[^>]*>.*?</annotation(?:-xml)?>", re.S)
# The identifier inside a rendered reference. LaTeXML links each entry's DOI or arXiv id,
# and untag() -- which keeps text and drops tags -- threw the href away with the tag.
# Measured across ar5iv: 27% of references carry a DOI and 14% an arXiv id.
#
# That loss is why the titleless-citation problem has no downstream fix. "PRX Quantum 2,
# 010343 (2021)" cannot be resolved from the string, and both Semantic Scholar and
# Crossref answer every query with their best guess rather than an error, so it does not
# fail at Stage 4 -- it silently returns the wrong paper. An identifier removes the
# guessing entirely.
RE_BIB_IDENT = re.compile(
    r'href="[^"]*?(?:(?:dx\.)?doi\.org/(10\.\d{4,9}/[^"?#\s]+)'
    r'|arxiv\.org/abs/([0-9]{4}\.[0-9]{4,5}|[a-z-]+/[0-9]{7}))"', re.I)
# Tables. The PMC route drops <table-wrap> and the LaTeX route drops RE_FLOAT, but the
# HTML route kept them, so 13-16% of arXiv papers carried a results grid flattened into
# prose -- "tt-GCRN [61] 3.8 -0.01 1m 5.6 0.0 6 7.7 0.0 9m MvRAM [55] ..." -- and another
# 17% carried tick-and-circle comparison tables as runs of unicode glyphs. Neither is
# readable once the row and column structure is gone. Strip the tabular data and keep the
# caption, which sits outside <table> in LaTeXML's markup and does carry meaning.
RE_HTML_TABLE = re.compile(r"<table\b[^>]*>.*?</table>", re.S)
# Residue from tables LaTeXML could NOT convert. It emits the tabular source as text, so
# there is no <table> element to strip and the grid arrives as prose:
#   "[HTML]ab9ac0 Object Detection Sq. Rel mIoU PA ... OmniDet ✓ ✗ ✗ ✗ 2.153 0.875 73.2"
# Two unambiguous signatures. A \cellcolor[HTML]{ab9ac0} argument never occurs in writing,
# and four or more tick marks in a row is a results grid -- a caption explaining that
# "✓ denotes the presence of existing work" uses them singly, so it survives.
RE_LTX_CELLCOLOR = re.compile(r"\[HTML\]\s*[0-9a-fA-F]{6}")
RE_GLYPH_RUN = re.compile(r"(?:[○●✓✗]\s*){4,}")
RE_PGF_MARK = re.compile(r"pgfsys@|lxSVG@|\\pgf")
_PGF_TOK = (r"\\?(?:pgfsys@[a-zA-Z@]*|pgf[a-zA-Z@]*|lxSVG@[a-zA-Z]*|hbox|vbox|hskip"
            r"|vskip|kern|hss|nullfont|color@|special|definecolor)")
# Two or more of those in a row, optionally separated by braces. Requiring a run means a
# stray "kern" or "special" in ordinary prose is left alone.
RE_PGF_RUN = re.compile(r"(?:" + _PGF_TOK + r"(?:\s*[{}\[\]()]+\s*|\s+)?){2,}")
RE_PGF_ONE = re.compile(r"\\?(?:pgfsys@|lxSVG@)[a-zA-Z@]*")
RE_ORPHAN_BRACE = re.compile(r"(?:\s|^)[{}\\]+(?=\s|$)")
# Plot data. A tarball paper measured here was 691,245 chars, its tail being pgfplots
# coordinates: "(1051.000000,0) (1053.000000,0) ...". Capped at 150k, that would have
# sent the model 150,000 characters of numbers. Strip coordinate runs and plot bodies.
RE_COORDS = re.compile(r"(?:\(\s*-?\d[\d.eE+-]*\s*,\s*-?\d[\d.eE+-]*\s*\)\s*){3,}")
RE_PLOT_ENV = re.compile(
    r"\\begin\{(axis|tikzpicture|pgfplots|filecontents\*?)\}.*?\\end\{\1\}", re.S)
RE_NUM_RUN = re.compile(r"(?:-?\d[\d.eE+-]*[\s,;]+){12,}")
RE_TAG = re.compile(r"<[^>]+>")
RE_HEAD = re.compile(r"<h[1-6][^>]*>(.*?)</h[1-6]>", re.S)

# Compiled once. strip_latex runs per paper over up to 60k chars, so recompiling these
# on every call was a measurable share of runtime.
RE_COMMENT = re.compile(r"(?m)^\s*%.*$")
# Match the whole environment, not "up to the next backslash". The old pattern
# stopped at the first command inside the float (usually \centering), so captions,
# includegraphics and labels all survived into the prompt.
RE_FLOAT = re.compile(
    r"\\begin\{(figure|table|tabular|algorithm|lstlisting|verbatim)\*?\}"
    r".*?\\end\{\1\*?\}", re.S)
# Where to stop reading. Everything from the method onward is the ANSWER: if the
# extractor sees Results and Conclusions it can write a background survey that quietly
# contains its own inspiration, which is exactly what the disjointness gate exists to
# catch. Cutting here is a correctness control, not just a token saving.
#
# The numbering prefix is the trap. `\d*` only handles Arabic, so "III Method",
# "IV Experiments" and "V Results" all sailed through uncut and 62% of CS rows ended up
# carrying conclusion text. Accept Arabic, Roman and single-letter numbering.
_STOP = r"(method|approach|experiment|result|evaluation|discussion|conclusion)"
RE_CUT = re.compile(r"\\section\*?\{\s*(?:[IVXLC]+|\d+|[A-Z])?[\.\)]?\s*" + _STOP, re.I)
RE_CUT_HTML = re.compile(r"^\s*(?:[IVXLC]+|\d+|[A-Z])?[\.\)]?\s*" + _STOP, re.I)
# The tarball route joins every .tex file, so in full mode an inline thebibliography or
# a \bibliography{...} line put the reference list inside the body text as well. Cut it
# in both modes: references reach Stage 3 through the `bibliography` field.
# Four bibliography environments occur in practice: the LaTeX default, amsrefs (maths),
# mciteplus (chemistry and materials science), and an external \bibliography{...}. Each
# one missed here means that paper's whole reference list stays in the body text.
RE_CUT_BIB = re.compile(
    r"\\begin\{(?:mcite)?thebibliography\}|\\bibliography\s*\{"
    r"|\\begin\{bibdiv\}|\\begin\{biblist\}"
    # mciteplus opens with a block of \providecommand definitions BEFORE \begin{...},
    # and writes them through \csname, so anchoring on "@ifundefined{" missed it. The
    # macro name itself is the reliable marker and appears at the very start of the block.
    r"|\\providecommand\*?\s*\{\\mcitethebibliography\}|endmcitethebibliography")
RE_NOISE = re.compile(r"\\(label|ref|eqref|usepackage|documentclass)\{[^}]*\}")
RE_CMD = re.compile(r"\\[a-zA-Z]+\*?")
RE_BRACE = re.compile(r"[{}$\\]")
RE_WS = re.compile(r"\s+")
# Unbounded body with a terminating lookahead, not a 400-character window. The window
# meant an entry longer than 400 chars had no match position where the lookahead held,
# so it was skipped entirely -- one maths paper had 31 \bibitem entries and parsed zero.
# \Z closes the last entry, which previously needed a following \bibitem to be seen at
# all. The body is truncated to 250 chars below, so removing the bound costs nothing.
RE_BIBITEM = re.compile(
    r"\\bibitem(?![A-Za-z])\s*(?:\[.*?\])?\s*\{([^}]+)\}"
    r"(.*?)(?=\\bibitem(?![A-Za-z])|\\end\{thebibliography\}|\Z)", re.S)
# REVTeX / apsrev, the American Physical Society style, and the default for a large share
# of physics and maths submissions. It writes
#     \bibitem [{\citenamefont {Smith}\ \emph {et~al.}(2020)}]{key}
#       \bibinfo {author} {...}, \bibinfo {title} {Real Title}, ...
# Three things broke on it: the space before "[", the braces inside the optional argument,
# and the \bibitemStop / \bibitemNoStop helper macros the style defines, which the old
# pattern matched as if they were entries. Papers using it parsed zero references while
# their reference list sat right there -- 101 citations in one, 29 in another.
RE_REVTEX = re.compile(r"\\bibinfo\s*\{|\\bibfield\s*\{")
# title for an article, booktitle for a chapter. Checked in that order.
RE_BIBINFO_TITLE = re.compile(r"\\bibinfo\s*\{title\}\s*\{")
RE_BIBINFO_BOOK = re.compile(r"\\bibinfo\s*\{booktitle\}\s*\{")
# APS house style frequently cites journal, volume, page and year with NO title at all:
# only 12 of 101 entries in one paper had one. But 75 carried a DOI, which identifies the
# work exactly, so Stage 4 resolves it without any string matching. Keep both -- the
# readable part for Stage 3, which has to judge whether the work is an inspiration, and
# the DOI for Stage 4, which only has to find it.
RE_BIB_DOI = re.compile(r"(?:doi\.org/|\\bibinfo\s*\{doi\}\s*\{)(10\.\d{4,9}/[^\s}\\,]+)")
RE_BIBINFO_ANY = re.compile(r"\\bibinfo\s*\{(journal|volume|pages|year|eprint)\}\s*\{")
# amsrefs, the AMS house format, used across mathematics:
#     \bib{Riordan2000}{article}{ author={Riordan, Oliver}, title={Spanning subgraphs...} }
# Neither the BibTeX @entry parser nor the \bibitem parser recognises it, so those papers
# came back with an empty bibliography AND the whole reference list dumped in the text.
RE_AMSREF = re.compile(r"\\bib\*?\{([^}]+)\}\s*\{[^}]*\}\s*\{")
RE_BIBNOISE = re.compile(r"\\[a-zA-Z]+\*?|[{}\\]")
RE_BIBENTRY = re.compile(r"@\w+\s*\{\s*([^,\s]+)\s*,", re.S)
RE_BIBTITLE = re.compile(r"\btitle\s*=\s*", re.I)
RE_PREAMBLE = re.compile(r"\\begin\{document\}")
# OpenAlex records arXiv landing pages in three shapes; the DOI form is 32% of them
# and the original regex missed all of it, which alone failed a third of the corpus.
RE_ARXIV = re.compile(
    r"(?:arxiv\.org/(?:abs|pdf)/|10\.48550/arxiv\.)([a-z-]*(?:/|\.)?[0-9]{4,7}\.?[0-9]{0,5})",
    re.I)
# OpenAlex writes PMC landing pages as .../pmc/articles/8371605 -- the bare numeric id,
# with no "PMC" prefix. Requiring the literal letters matched nothing and failed 100%
# of biology. Accept both forms and normalise to PMCnnnnnnn.
RE_PMC = re.compile(r"(?:pmc/articles/|/)(?:PMC)?(\d{5,9})\b", re.I)
# Reference labels in JATS mixed-citations: "1.", "[12]", "(3)".
RE_REF_LABEL = re.compile(r"[\[\(]?\d{1,3}[\.\]\)]?")
# Back matter that some publishers file inside <body> instead of <back>. JAMA writes the
# reference list as <sec sec-type="ref-list"> under body, so taking every top-level <sec>
# pasted the whole numbered reference list into the prose: measured on 997 biology papers,
# 36% carried it. Acknowledgements, conflict-of-interest statements and supplementary
# blocks arrive the same way. sec-type is authoritative when present; many deposits leave
# it empty, so the title is checked too.
PMC_DROP_SECTYPE = {
    "ref-list", "references", "associated-data", "supplementary-material", "appendix",
    "coi-statement", "conflict", "funding", "data-availability", "abbreviations",
    "acknowledgment", "acknowledgement"}
RE_PMC_DROP_TITLE = re.compile(
    r"^\s*(references?|bibliography|acknowledg|supplement|conflicts? of interest"
    r"|competing interests?|author contributions?|funding|data availability"
    r"|abbreviations|associated data|appendix)\b", re.I)


def front_matter(abstract: str, sections: list[tuple[str, str]],
                 budget: int | None = None, max_sections: int | None = None) -> str:
    """Abstract plus the paper's opening sections. One rule for arXiv and PMC alike.

    Keyword cutting ("stop at Methods") does not generalise, because papers name their
    sections however they like. A survey measured here ran
    Introduction / Overview / Knowledge Representation Learning / Knowledge Acquisition
    / Temporal Knowledge Graph / ... / Conclusion, so the first heading matching a stop
    word was section 8 and 145,000 characters were kept. 62% of CS rows ended up
    carrying conclusion text.

    Three bounds instead, whichever bites first:
      - a stop-word heading (still useful when it does fire)
      - MAX_SECTIONS, enough for Introduction plus Related Work plus one more
      - FRONT_MATTER_CHARS, so a single enormous section cannot blow the budget

    Bounding matters for correctness, not just cost: if the extractor reads Results and
    Conclusions it can write a background survey that quietly contains its own
    inspiration, which is exactly what the disjointness gate exists to catch. Applying
    the identical rule to both routes also removes a discipline-shaped asymmetry, since
    otherwise CS would arrive with whole papers and biology with abstracts.
    """
    budget = FRONT_MATTER_CHARS if budget is None else budget
    max_sections = MAX_SECTIONS if max_sections is None else max_sections
    # In FULL mode, assemble the whole paper and let the caller do the capping, so
    # `fulltext_chars_original` is the paper's real length instead of "the budget plus
    # whatever the last section added". The 2M ceiling is only a runaway guard.
    assembly = budget if budget < MAX_CHARS else 2_000_000
    parts: list[str] = []
    total = 0
    if abstract:
        parts.append(abstract)
        total += len(abstract)
    kept = 0
    for heading, body in sections:
        if STOP_AT_METHOD and heading and RE_CUT_HTML.search(heading):
            break
        if kept >= max_sections or total >= assembly:
            break
        parts.append(body)
        total += len(body)
        kept += 1
    # Deliberately NOT truncated here. Cutting to `budget` inside this function meant the
    # caller could never see how long the paper really was, so `fulltext_chars_original`
    # was recorded for the tarball route only and truncation was invisible on the HTML and
    # PMC routes -- which is 90% of the corpus, 10% of it sitting exactly at the cap.
    # The single cap now lives in fetch(), where the original length can be recorded.
    return RE_WS.sub(" ", " ".join(parts)).strip()


def scrub_pgf(text: str) -> str:
    """Remove raw TikZ/pgf drawing commands that survived as text content.

    Gated on a cheap marker search: the runs only appear in papers whose pictures
    LaTeXML failed to convert, so the expensive substitution runs on those alone.
    """
    if not RE_PGF_MARK.search(text):
        return text
    text = RE_PGF_ONE.sub(" ", RE_PGF_RUN.sub(" ", text))
    # The commands carried braces that the run pattern leaves behind as " } } } ".
    text = RE_ORPHAN_BRACE.sub(" ", text)
    return RE_WS.sub(" ", text).strip()


def scrub_table_residue(text: str) -> str:
    """Remove the traces of a table LaTeXML failed to convert. See RE_LTX_CELLCOLOR."""
    if "[HTML]" not in text and not RE_GLYPH_RUN.search(text):
        return text
    text = RE_GLYPH_RUN.sub(" ", RE_LTX_CELLCOLOR.sub(" ", text))
    return RE_WS.sub(" ", text).strip()


class RateLimiter:
    """Global token bucket. arXiv asks for one request per 3 seconds; that limit is
    per client, not per thread, so it must be shared. Sleeping inside each worker
    instead lets the wait overlap with download and parse time, which is where the
    speedup comes from: serial cost is sleep + work, here it is max(sleep, work/N)."""

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            sleep_for = max(0.0, self._next - now)
            self._next = max(now, self._next) + self.min_interval
        if sleep_for:
            time.sleep(sleep_for)

    def backlog(self) -> float:
        """Seconds until this limiter would release the next request. Used to choose
        between two hosts that serve the same thing, so one does not queue up hours of
        work while the other sits idle."""
        with self._lock:
            return max(0.0, self._next - time.monotonic())


# --------------------------------------------------------------------------- arXiv
def arxiv_id(url: str | None) -> str | None:
    if not url:
        return None
    m = RE_ARXIV.search(url)
    return m.group(1) if m else None


def untag(markup: str) -> str:
    """Strip tags, then decode entities.

    Order matters. Decoding first would turn "&lt;script&gt;" into real angle brackets
    that RE_TAG then eats as if they were markup. Stripping first means only text
    survives to be decoded.

    Without the decode, 71 of 93 HTML-route papers carried literal "&gt;", "&lt;" and
    "&amp;" into the prompt: "let F in R &gt; 0" instead of "> 0". Inequalities are
    load-bearing in a methods section, and the LaTeX route has no such problem, so the
    same paper read differently depending on which host served it.
    """
    return html.unescape(RE_WS.sub(" ", RE_TAG.sub(" ", markup))).strip()


def fetch_arxiv_html(aid: str, session: requests.Session,
                     base: str = "https://arxiv.org/html/") -> tuple[str, dict] | None:
    """LaTeXML-rendered HTML: 10-30x smaller than the e-print tarball and already
    parsed, so no LaTeX stripping and no tar extraction.

    Two hosts serve the same markup, so one parser covers both:
      arxiv.org/html/          submissions from December 2023 onward
      ar5iv.labs.arxiv.org/    everything older, rendered retrospectively

    That second host matters: 82% of the training window is 2020-2022, so without it
    86% of papers fell through to the tarball path, which is the whole bottleneck.
    """
    try:
        r = session.get(f"{base}{aid}", headers=UA, timeout=60)
        if r.status_code != 200 or len(r.content) < 5000:
            return None
    except requests.RequestException:
        return None
    # Drop the duplicate formula renderings once, up front, so every later pass -- bib
    # extraction, section slicing, untag -- works on markup that states each formula
    # exactly one time.
    html = RE_HTML_TABLE.sub(" ", RE_MATH_ANNOT.sub(" ", r.text))

    bib: dict[str, str] = {}
    for m in RE_HTML_BIBITEM.finditer(html):
        # strip_authors is essential here, not optional. LaTeXML emits
        # "[1] A. Author and B. Author. Real Title. Venue, Year." and the extractor
        # copies whatever it is given into supposed_title, which is then looked up in
        # Semantic Scholar. An author list never resolves and the gold is lost silently.
        text = strip_authors(untag(m.group(2)))
        if len(text) > 20:
            # Truncate FIRST, then append the identifier, so a long author list cannot
            # push the one part that resolves exactly past the 250-character cap.
            ident = RE_BIB_IDENT.search(m.group(2))
            if ident:
                text = text[:250] + (" doi:" + ident.group(1) if ident.group(1)
                                     else " arXiv:" + ident.group(2))
            bib[m.group(1)] = text[:300]
    if not bib:
        return None                     # references are mandatory; let the caller fall back

    am = RE_HTML_ABSTRACT.search(html)
    # Stop at the bibliography, which has already been parsed into `bib` above. Without
    # this the final slice below runs to the end of the document and pastes the whole
    # reference list, plus LaTeXML's footer, into the paper text.
    endm = RE_HTML_BODY_END.search(html)
    body_html = html[: endm.start()] if endm else html
    # Slice between top-level section openings: a non-greedy match to </section> ends
    # at the first nested subsection, dropping everything after it in every section.
    opens = [mm.start() for mm in RE_HTML_SEC_OPEN.finditer(body_html)]
    sections = []
    for start, end in zip(opens, opens[1:] + [len(body_html)]):
        block = RE_HTML_ERR.sub(" ", body_html[start:end])
        hm = RE_HEAD.search(block)
        sections.append((untag(hm.group(1)) if hm else "",
                         scrub_table_residue(scrub_pgf(untag(block)))))
    text = front_matter(scrub_pgf(untag(am.group(0))) if am else "", sections)
    return (text, bib) if len(text) >= MIN_CHARS else None


def fetch_arxiv(aid: str, session: requests.Session) -> tuple[str, str] | None:
    """Pull the e-print tarball; return (body_tex, refs_text).

    Body comes from .tex. References come from .bbl or .bib, which are SEPARATE files:
    modern submissions use BibTeX, so the reference list is almost never inline
    \\bibitem in the .tex. Reading only .tex found zero references for every paper.
    """
    # Streamed, and abandoned as soon as the magic bytes say PDF. A PDF-only submission
    # has no LaTeX source at all: ar5iv serves the abstract page instead of a rendering
    # and arxiv.org/html 404s, so this endpoint is the last thing tried and it hands
    # back the PDF itself. Measured on materials science, half the sampled papers were
    # PDF-only, at 0.8-4.3 MB each -- several GB downloaded across the domain purely to
    # discover there is nothing to parse. Reading the first chunk costs ~64 KB instead.
    try:
        with session.get(f"https://arxiv.org/e-print/{aid}", headers=UA,
                         timeout=120, stream=True) as r:
            if r.status_code != 200:
                return None
            chunks = r.iter_content(65_536)
            first = next(chunks, b"")
            if first.startswith(b"%PDF-"):
                return None
            content = first + b"".join(chunks)
        if not content:
            return None
    except requests.RequestException:
        return None

    body: list[str] = []
    refs: list[str] = []
    try:
        with tarfile.open(fileobj=BytesIO(content), mode="r:*") as tf:
            for member in tf.getmembers():
                if not member.isfile() or member.size > 5_000_000:
                    continue
                name = member.name.lower()
                if not name.endswith((".tex", ".bbl", ".bib")):
                    continue
                fh = tf.extractfile(member)
                if not fh:
                    continue
                text = fh.read().decode("utf-8", "ignore")
                (refs if name.endswith((".bbl", ".bib")) else body).append(text)
    except tarfile.TarError:
        # Single-file submissions arrive gzipped rather than tarred.
        try:
            body.append(gzip.decompress(content).decode("utf-8", "ignore"))
        except OSError:
            return None
    if not body:
        return None
    # Members arrive in tar order, which is arbitrary. strip_latex discards everything
    # before \begin{document}, so if the main file is not first, real content from the
    # other files is thrown away. Put the file holding \begin{document} in front.
    body.sort(key=lambda t: 0 if RE_PREAMBLE.search(t) else 1)
    joined = "\n".join(body)
    # Inline \bibitem still happens; keep the .tex as a reference source too.
    return joined, "\n".join(refs) + "\n" + joined


# Every LaTeX citation command, including the biblatex family and the optional
# pre/post-note arguments: \cite[see][p. 4]{key}. Keys are comma-separated.
RE_CITE = re.compile(
    r"\\(?:no)?(?:cite|citep|citet|citealt|citealp|citeauthor|citeyear|citeyearpar|"
    r"autocite|parencite|textcite|footcite|smartcite|Cite|Citep|Citet|Autocite|"
    r"Parencite|Textcite)\*?(?:\[[^\]]*\])*\{([^}]*)\}")


def cited_keys(tex: str) -> set[str]:
    """The bibliography keys the manuscript actually cites.

    A .bib file in an e-print tarball is NOT the paper's reference list. Two things
    live there that were never cited:

      - the template's sample bibliography, shipped inside the style package
        (IEEEexample.bib, ICML's example_paper.bib)
      - the authors' whole personal library. One paper here carried 2,086 entries
        spanning economics, psychology and philosophy behind a 41,000-character
        manuscript.

    Both are actively harmful, not merely noisy: Stage 3 grounds inspirations in this
    list, so an uncited entry becomes an inspiration the authors never had, and that
    becomes a fabricated gold document. A master library is topically diverse, so it
    manufactures CROSS-DOMAIN inspirations preferentially -- exactly the slice the
    benchmark exists to measure.

    Returning an empty set means "no \\cite found", and the caller then keeps
    everything rather than discarding a real bibliography: .bbl-only submissions and
    manuscripts whose citations are hidden behind macros must not be emptied out.
    """
    keys: set[str] = set()
    for m in RE_CITE.finditer(tex):
        keys.update(k.strip() for k in m.group(1).split(",") if k.strip())
    return keys


def strip_latex(tex: str) -> str:
    """Front matter only: preamble dropped, cut at the first Method/Experiment heading.

    Both cuts happen FIRST so every later substitution runs over the kept portion
    rather than the whole paper, typically a 5x reduction in work.
    """
    # Everything before \begin{document} is packages, macros and journal boilerplate.
    # Left in, it produced openings like "[manuscript,screen] jair cc 2025 positioning".
    start = RE_PREAMBLE.search(tex)
    if start:
        tex = tex[start.end():]
    # Only cut at Methods when front-matter mode is on. FRONT_MATTER_CHARS is raised to
    # MAX_CHARS in full-text mode (the default), so this test tracks the same switch the
    # HTML and PMC routes use. Cutting unconditionally meant tarball papers -- 18 of 95
    # in the last run -- stopped at Methods while HTML papers kept the whole text: a
    # route-shaped asymmetry in how much a paper was read, invisible in the output.
    if FRONT_MATTER_CHARS < MAX_CHARS:
        cut = RE_CUT.search(tex)
        if cut:
            tex = tex[: cut.start()]
    # Always drop the reference list from the body. It reaches Stage 3 through the
    # `bibliography` field; leaving it here duplicates it inside the prompt.
    bibcut = RE_CUT_BIB.search(tex)
    if bibcut:
        tex = tex[: bibcut.start()]
    tex = RE_COMMENT.sub("", tex)
    # Plot environments and raw coordinate data before anything else: they can be the
    # bulk of a submission and carry no meaning for decomposition.
    tex = RE_PLOT_ENV.sub(" ", tex)
    tex = RE_COORDS.sub(" ", tex)
    tex = RE_NUM_RUN.sub(" ", tex)
    tex = RE_FLOAT.sub(" ", tex)
    # Citation keys are the link back to the reference list; keep them visible.
    # \1, not \2. The broader RE_CITE added for cited_keys() has a single capture group
    # (the key list); the earlier two-group version it replaced put the keys in \2. The
    # mismatch was not a wrong result but a hard crash -- re compiles the replacement
    # template before looking at the text, so EVERY paper on the e-print route died with
    # "invalid group reference 2", which is 100% of PDF-free tarball submissions.
    tex = RE_CITE.sub(r" [CITE:\1] ", tex)
    tex = RE_NOISE.sub(" ", tex)
    tex = RE_CMD.sub(" ", tex)
    tex = RE_BRACE.sub(" ", tex)
    return scrub_pgf(RE_WS.sub(" ", tex).strip())


# A \bibitem body is "Authors. Title. Venue, Year." while a .bib entry gives the title
# directly. Handing the raw body to the extractor meant 12% of returned `supposed_title`
# values were author lists, which cannot be resolved against Semantic Scholar and would
# have lost those golds silently.
RE_AUTHOR_RUN = re.compile(
    r"^\s*(?:and\s+|&\s+)?"                                    # the final "and X. Y"
    # [\s-] not \s: East Asian names are routinely initialised as "D.-J. Lee" or
    # "S.-C. Zhang", and a hyphen there stopped the whole loop dead, leaving every
    # author from that point on sitting in front of the title.
    r"(?:[A-Z]\.(?:[\s-]?[A-Z]\.)*\s+[A-Z][A-Za-z'`-]+"        # J. K. Smith, D.-J. Lee
    r"|[A-Z][A-Za-z'`-]+,?\s+[A-Z]\.(?:[\s-]?[A-Z]\.)*"        # Smith, J. K.
    r")(?:\s*(?:,|and|&)\s*)?")
RE_LEAD_YEAR = re.compile(r"^\s*[\(\[]?(19|20)\d{2}[a-z]?[\)\].,]?\s*")
RE_ETAL = re.compile(r"^\s*et\.?\s?al\.?[,.]?\s*", re.I)
# Surname-only citations ("Norman and Shallice (1986) Attention to action") carry no
# initials, so the author-run pattern cannot see them. A parenthesised year near the
# start is the reliable marker: everything before it is attribution.
RE_YEAR_CUT = re.compile(r"^.{0,110}?[\(\[](19|20)\d{2}[a-z]?[\)\]][.,]?\s*")


# LaTeXML renders each reference with its label ("[1]", "[AB04]") and appends
# navigation cruft ("External Links: Link Cited by: S1, S4.1"). Both must go before the
# author stripper can see the start of the entry -- with "[1]" in front, the author
# pattern never matches and the entry looks clean when it is not.
RE_REF_TAG = re.compile(r"^\s*[\[\(][A-Za-z0-9+.,'\- ]{1,12}[\]\)]\s*")
RE_LATEXML_CRUFT = re.compile(
    r"\s*(External Links?:|Cited by:|MathSciNet|Digital Library).*$", re.I)
# DELIBERATELY NOT STRIPPED: full-name author runs ("Aharon Ben-Tal and Arkadi
# Nemirovski."). No regex can separate those from a title, because both are capitalised
# words joined by "and". A pattern that removed them also turned
#   "Distinguishing Separable and Entangled States"   -> "Entangled States"
#   "Theory Principles and Design of Magnetic Circuits" -> "Design of Magnetic Circuits"
# Corrupting titles is worse than leaving authors in front of them, so these entries are
# passed through whole and the Stage 3 prompt is told to take the title alone. A model
# can tell an author from a title; a regex cannot. The patterns below are kept only
# because they are unambiguous: no real title opens with "S. Tan," or "Smith, J. K.".


def strip_authors(entry: str) -> str:
    """Remove a leading author run and year from a reference line, leaving the title."""
    s = RE_LATEXML_CRUFT.sub("", entry.strip())
    s = RE_REF_TAG.sub("", s).strip()
    # Both author forms in one loop. Author lists mix them freely -- "Zofia Adamowicz,
    # Andres Cordon-Franco and F. Felix Lara-Martin" is two full names then an
    # initialled one -- so stripping each form in its own pass leaves the tail behind.
    # Each full-name strip is guarded on length: unguarded it eats titles that begin
    # with capitalised words.
    for _ in range(40):                       # reference lists can carry many authors
        new = RE_AUTHOR_RUN.sub("", s, count=1)
        new = RE_ETAL.sub("", new)
        new = RE_LEAD_YEAR.sub("", new)
        if new == s:
            break
        s = new
    cut = RE_YEAR_CUT.sub("", s, count=1)
    if len(cut) >= 12:
        s = cut
    s = re.sub(r"^\s*[.,;:]+\s*", "", s)
    # If stripping ate almost everything, the pattern misfired: keep the original.
    return s.strip() if len(s) >= 12 else entry.strip()


def _braced(text: str, i: int) -> str:
    """Read a brace- or quote-delimited BibTeX value starting at i, balancing nesting."""
    while i < len(text) and text[i] in " \t\n=":
        i += 1
    if i >= len(text):
        return ""
    if text[i] == '"':
        j = text.find('"', i + 1)
        return text[i + 1: j] if j > 0 else ""
    if text[i] != "{":
        j = i
        while j < len(text) and text[j] not in ",\n":
            j += 1
        return text[i:j]
    depth, j = 0, i
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1: j]
        j += 1
    return ""


def bibliography(refs: str) -> dict[str, str]:
    """Map citation key -> reference title.

    Handles both formats that occur in arXiv submissions:
      - BibTeX `.bib`  ->  @misc{key, title={...}, ...}     (the common case)
      - `.bbl` or inline `\\bibitem{key} ...`               (older submissions)
    """
    # Comments first. strip_latex() clears them from the body but this function never
    # did, so a .bbl with commented-out lines produced titles like
    #   "%Electronic properties of MXenes: a theoretical review, J. Mater. Chem. C 5..."
    # -- a real title wearing a comment marker, which will not resolve in Stage 4 -- and
    # dead entries like "% Tanabashi:2018oca" that are not references at all.
    refs = RE_COMMENT.sub("", refs)
    out: dict[str, str] = {}

    for m in RE_BIBENTRY.finditer(refs):
        key, start = m.group(1), m.end()
        # Stay inside this entry: stop at the next @entry.
        nxt = refs.find("@", start)
        block = refs[start: nxt if nxt > 0 else min(start + 4000, len(refs))]
        t = RE_BIBTITLE.search(block)
        if not t:
            continue
        title = RE_WS.sub(" ", _braced(block, t.end()).replace("{", "").replace("}", "")).strip()
        if len(title) > 8:
            out[key] = title[:250]

    for m in RE_AMSREF.finditer(refs):
        key = m.group(1)
        if key in out:
            continue
        # Stay inside this entry: the next \bib, or a bounded window.
        nxt = refs.find("\\bib", m.end())
        block = refs[m.end(): nxt if nxt > 0 else min(m.end() + 3000, len(refs))]
        t = RE_BIBTITLE.search(block)
        if not t:
            continue
        title = RE_WS.sub(" ", _braced(block, t.end()).replace("{", "").replace("}", "")).strip()
        if len(title) > 8:
            out[key] = title[:250]

    for m in RE_BIBITEM.finditer(refs):
        key = m.group(1)
        if key in out:
            continue
        raw = m.group(2)
        # REVTeX tags every field, so read the title rather than guessing where the
        # author list stops -- which is the whole reason strip_authors exists, and which
        # cannot work here because the markup interleaves names with commands.
        if RE_REVTEX.search(raw):
            title = ""
            for pat in (RE_BIBINFO_TITLE, RE_BIBINFO_BOOK):
                fm = pat.search(raw)
                if not fm:
                    continue
                cand = RE_WS.sub(" ", RE_BIBNOISE.sub(" ", _braced(raw, fm.end() - 1))).strip()
                if len(cand) > 8:
                    title = cand
                    break
            # No title field: build one from the tagged fields instead of falling through
            # to strip_authors, which on REVTeX markup produced entries like
            # "% author author G. ~ Thiering and author A." -- an author list dressed as a
            # title. Stage 4 looks these up in Semantic Scholar, where a mangled string
            # never resolves, so the gold would be lost with no error anywhere.
            if not title:
                bits = []
                for fm in RE_BIBINFO_ANY.finditer(raw):
                    v = RE_WS.sub(" ", RE_BIBNOISE.sub(" ", _braced(raw, fm.end() - 1))).strip()
                    if v:
                        bits.append(v)
                title = " ".join(bits[:5])
            dm = RE_BIB_DOI.search(raw)
            if dm:
                title = (title + " doi:" + dm.group(1)).strip()
            # Still nothing identifying: drop it. An absent reference is recoverable,
            # a wrong one is not.
            if len(title) > 8:
                out[key] = title[:250]
            continue
        body = RE_BIBNOISE.sub(" ", RE_WS.sub(" ", raw)).strip()
        body = strip_authors(body)
        if len(body) > 8:
            out[key] = body[:250]
    return out


# --------------------------------------------------------------------------- PMC
def pmc_id(url: str | None) -> str | None:
    if not url:
        return None
    m = RE_PMC.search(url)
    return f"PMC{m.group(1)}" if m else None


def fetch_pmc(pid: str, session: requests.Session) -> tuple[str, dict] | None:
    """JATS XML from Europe PMC; return (body_text, refs) to match the arXiv path.

    Two bugs lived here. First, this returned only text, so the caller left
    `bibliography` empty for every biology paper -- 5,491 of them -- and decomposition
    would have had no reference list to ground inspirations in.

    Second, references were read as `.//article-title`, which only exists in
    *structured* `<element-citation>` refs. Many publishers emit `<mixed-citation>`
    holding the whole citation as free text, so that lookup found nothing on half the
    corpus. We now take article-title when present and fall back to the ref's full
    text, which covers both encodings.
    """
    url = ("https://www.ebi.ac.uk/europepmc/webservices/rest/"
           f"{pid}/fullTextXML")
    try:
        r = session.get(url, headers=UA, timeout=90)
        if r.status_code != 200 or not r.content:
            return None
        root = ET.fromstring(r.content)
    except (requests.RequestException, ET.ParseError):
        return None

    abstract = RE_WS.sub(" ", " ".join(" ".join(a.itertext())
                                       for a in root.iter("abstract"))).strip()
    if abstract and not abstract.lower().startswith("abstract"):
        abstract = "Abstract. " + abstract

    # References FIRST, before any pruning. The reference list is not always in <back>:
    # JAMA and others file it as <sec sec-type="ref-list"> inside <body>, so the prune
    # below removes it from the tree. Reading refs afterwards therefore returned nothing
    # for exactly the 36% of papers the prune exists to fix -- trading a leak for an
    # empty bibliography, which is worse: those papers cannot ground inspirations at all.
    ref_nodes = list(root.iter("ref"))

    # TOP-LEVEL <sec> under <body> only, and nothing from <back>.
    #
    # root.iter("sec") also yields every nested subsection, and a parent's itertext()
    # already contains all of its children, so each subsection was emitted twice.
    # Measured on PMC12851941: 131,180 characters extracted against a true body length
    # of 100,702, i.e. 1.30x -- 30% of every biology prompt was a second copy of text
    # the model had just read, paid for at Stage 3. Nesting is common in biology
    # (Methods with six subsections), so this was systematic, not an outlier.
    #
    # It also reached into <back>, picking up acknowledgements and supplementary
    # material, and the f"{head}. " prefix repeated each heading that itertext()
    # already yields.
    body_el = root.find(".//body")
    sections: list[tuple[str, str]] = []
    if body_el is not None:
        # Tables are numeric dumps. The LaTeX route already discards floats via
        # RE_FLOAT, so dropping them here keeps the two routes reading the same notion
        # of "paper text" instead of giving biology a discipline-shaped advantage.
        drop = [(parent, child) for parent in body_el.iter()
                for child in list(parent)
                if child.tag in ("table-wrap", "table", "array", "ref-list")]
        for parent, child in drop:
            parent.remove(child)
        for sec in body_el.findall("sec"):
            t = sec.find("title")
            head = "".join(t.itertext()).strip() if t is not None else ""
            # Back matter filed inside <body>. See PMC_DROP_SECTYPE.
            if (sec.get("sec-type") or "").lower() in PMC_DROP_SECTYPE:
                continue
            if head and RE_PMC_DROP_TITLE.match(head):
                continue
            sections.append((head, RE_WS.sub(" ", " ".join(sec.itertext())).strip()))
        if not sections:
            # Some deposits have no <sec> at all, just paragraphs under <body>.
            sections = [("", RE_WS.sub(" ", " ".join(body_el.itertext())).strip())]
    text = front_matter(abstract, sections)

    refs: dict[str, str] = {}
    for i, ref in enumerate(ref_nodes):
        t = ref.find(".//article-title")
        if t is not None:
            cite = " ".join(t.itertext())
        else:
            # mixed-citation: the whole reference as one string. Drop the leading
            # label ("1.", "[12]") so the title is not buried behind a number.
            # Skip the pub-id values themselves: they are appended below in a tagged
            # form, and itertext() would otherwise leave the bare number in the middle
            # of the citation as well ("... 30207593 10.3322/caac.21492 pmid:30207593").
            skip = {"".join(p.itertext()).strip() for p in ref.iter("pub-id")}
            cite = " ".join(e for e in ref.itertext()
                            if e.strip() and e.strip() not in skip
                            and not RE_REF_LABEL.fullmatch(e.strip()))
        # A mixed-citation is the whole reference, authors first, exactly the shape
        # that made 12% of arXiv `supposed_title` values come back as author lists.
        # The same cleaner has to run here or biology inherits the same failure.
        cite = strip_authors(RE_WS.sub(" ", cite).strip())
        # JATS tags identifiers explicitly. Same reasoning as the arXiv route: an exact
        # identifier removes Stage 4's guessing, and both resolvers guess silently.
        for pid in ref.iter("pub-id"):
            kind = (pid.get("pub-id-type") or "").lower()
            val = "".join(pid.itertext()).strip()
            if kind == "doi" and val:
                cite = cite[:250] + " doi:" + val
                break
            if kind == "pmid" and val and "doi:" not in cite:
                cite = cite[:250] + " pmid:" + val
        if len(cite) > 20:
            refs[ref.get("id") or f"ref{i}"] = cite[:300]

    return (text, refs) if text else None


# --------------------------------------------------------------------------- main
def load_job(domain: str, split: str, a) -> tuple[list[dict], Path, bool]:
    """Sample and resume-filter one (domain, split). Returns (papers, dst, appending)."""
    src = IN / f"{domain}_{split}.jsonl"
    if not src.exists():
        print(f"[{domain}/{split}] missing {src.name}, skipped")
        return [], IN / "unused", False

    papers = [json.loads(l) for l in open(src) if l.strip()]
    # Stage 1 collects the WHOLE eligible pool on purpose (targets.collect_full_pool), so
    # its output is several times the benchmark target: 19,051 CS train papers against a
    # 6,000 target. Fetching all of it is 4.6x the work for no extra benchmark. --headroom
    # takes the size from config/domains.yaml so the number cannot drift per domain.
    n = a.sample
    if a.headroom:
        # Per-domain override first. Materials science is capped below the common target
        # because arXiv PDF-only deposits put ~40% of its papers out of reach; asking for
        # 6,000 there would just fetch the entire pool and still fall short.
        key = f"{split}_source_papers"
        target = (OVERRIDES.get(domain) or {}).get(key) or TARGETS.get(key)
        if target:
            n = int(target * a.headroom)
            over = " (per-domain override)" if (OVERRIDES.get(domain) or {}).get(key) else ""
            print(f"[{domain}/{split}] target {target:,}{over} x {a.headroom} headroom "
                  f"= {n:,} to fetch")
    if n:
        # Sort first so the sample does not depend on OpenAlex's cursor order, which
        # is not stable between runs.
        papers.sort(key=lambda p: p["doi"])
        papers = random.Random(a.seed).sample(papers, min(n, len(papers)))
    elif a.limit is not None:
        # `is not None`, not truthiness: --limit 0 is falsy, so it fell through to "no
        # limit at all" and started a full 13,000-paper run when a smoke test was meant.
        papers = papers[: a.limit]

    dst = OUT / f"{domain}_{split}.jsonl"
    done: set[str] = set()
    if dst.exists() and not a.no_resume:
        with open(dst) as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["doi"])
                except (json.JSONDecodeError, KeyError):
                    continue
    todo = [p for p in papers if p["doi"] not in done]
    print(f"[{domain}/{split}] {len(todo):,} to fetch"
          + (f" ({len(done):,} already done)" if done else "")
          + (f", sampled @ seed {a.seed}" if a.sample else ""))
    return todo, dst, bool(done)


def main() -> None:
    # Declared up front: argparse reads these as defaults below, and Python forbids a
    # global statement after the name has already been used in the same scope.
    global FRONT_MATTER_CHARS, MAX_SECTIONS, STOP_AT_METHOD
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True,
                    help="domain id, a comma-separated list (e.g. cs,biology), or 'all'. "
                         "Several domains in ONE process is the only real speedup: the "
                         "arXiv limiter stays global and correct, while PubMed Central "
                         "runs alongside on its own limiter. Separate processes per "
                         "domain do NOT go faster, they just multiply the request rate "
                         "arXiv sees for the same throughput.")
    ap.add_argument("--split", required=True, choices=["train", "test", "both"])
    ap.add_argument("--sample", type=int, help="random sample per (domain, split)")
    ap.add_argument("--headroom", type=float, metavar="X",
                    help="fetch X times the config target for each split (6,000 train / "
                         "1,000 test per domain) instead of the whole collected pool. "
                         "1.4 covers the ~9%% of papers with no machine-readable text "
                         "plus a margin for the Stage 3 quality gates. Overrides --sample.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, help="first N, unsampled; for quick smoke tests")
    ap.add_argument("--workers", type=int, default=WORKERS,
                    help=f"parallel fetches (default {WORKERS}); the arXiv rate limit is "
                         f"global, so this overlaps waiting with parsing rather than "
                         f"increasing request rate")
    ap.add_argument("--rate", type=float, default=ARXIV_INTERVAL,
                    help=f"seconds between arXiv requests, shared across workers "
                         f"(default {ARXIV_INTERVAL}); raise if arXiv starts refusing")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--front-matter-chars", type=int, default=FRONT_MATTER_CHARS,
                    help=f"character budget for kept text (default {FRONT_MATTER_CHARS:,})")
    ap.add_argument("--max-sections", type=int, default=MAX_SECTIONS,
                    help=f"sections kept after the abstract (default {MAX_SECTIONS})")
    ap.add_argument("--front-matter-only", action="store_true",
                    help=f"opposite of the default: keep only the abstract plus "
                         f"{MAX_SECTIONS} sections, capped at {FRONT_MATTER_CHARS:,} "
                         "chars. Cheaper at Stage 3, but deviates from ResearchBench "
                         "and TOMATO-Star and discards inspirations that appear in the "
                         "methodology and discussion.")
    a = ap.parse_args()

    # Running several processes does NOT go faster: the bottleneck is arXiv's rate
    # limit, not our concurrency, so N processes just make us N times ruder for the
    # same throughput. Running everything in ONE process is what actually helps,
    # because arXiv and PubMed Central then hold separate limiters and overlap.
    # Set the extraction bounds globally rather than threading them through every
    # fetcher; both routes read the same two knobs so they cannot drift apart.
    # FULL TEXT IS THE DEFAULT. Both source papers decompose from the whole paper, and
    # ResearchBench states that inspirations appear in the methodology and related work
    # as well as the introduction, so truncating at Methods discards real golds. The
    # leakage argument for truncating does not hold either: the abstract is always kept
    # and already states the contribution, so the extractor knows the answer regardless.
    # Disjointness is enforced by the Stage 3 gate, which is where it belongs.
    full = not a.front_matter_only
    FRONT_MATTER_CHARS = MAX_CHARS if full else a.front_matter_chars
    MAX_SECTIONS = 10_000 if full else a.max_sections
    STOP_AT_METHOD = not full
    print(f"[extract] keeping up to {FRONT_MATTER_CHARS:,} chars"
          + (" (FULL TEXT, the default)" if full
             else f", max {MAX_SECTIONS} sections after the abstract (front-matter only)"))

    # The mirror of drop_bad_rows.refuse_if_fetching(). That script rewrites each output
    # file and renames it into place; this one holds those files open in append mode for
    # the whole run. Whichever starts second, the result is the same: the fetcher ends up
    # writing into an unlinked inode and every paper after that point is silently lost.
    # Guarding only one direction left the race fully open, since the natural workflow is
    # drop-then-fetch and the drop takes minutes on a large corpus.
    try:
        import subprocess
        busy = subprocess.run(["pgrep", "-f", "drop_bad_rows.py"],
                              capture_output=True, text=True, timeout=10).stdout.split()
        if busy:
            sys.exit(f"REFUSING TO RUN: drop_bad_rows.py is running (pid "
                     f"{', '.join(busy)}). It rewrites the files this stage appends to, "
                     f"so anything fetched now would be discarded. Wait for it to "
                     f"finish, then start this again.")
    except (OSError, subprocess.SubprocessError):
        pass

    cfg = yaml.safe_load(open(CONFIG))
    global TARGETS, OVERRIDES
    _t = cfg.get("targets") or {}
    TARGETS = _t.get("per_domain") or {}
    OVERRIDES = _t.get("per_domain_override") or {}
    known = [d["id"] for d in cfg["domains"]]
    domains = known if a.domain == "all" else [d.strip() for d in a.domain.split(",")]
    if unknown := [d for d in domains if d not in known]:
        sys.exit(f"unknown domain(s): {unknown}. Known: {known}")
    splits = ["train", "test"] if a.split == "both" else [a.split]

    OUT.mkdir(parents=True, exist_ok=True)
    # Record the cap next to the data instead of leaving it in an environment variable.
    # QUARTET_MAX_CHARS is read by this stage AND by Stage 3, so a terminal that has it
    # exported and one that does not produce different limits for the same corpus: text
    # stored at 200,000 here would be silently cut to 150,000 in the prompt. That exact
    # drift has already cost one full rebuild. Stage 3 and the auditor read this file.
    # OSError as well as JSONDecodeError: biology runs against PubMed Central while the
    # arXiv domains run against arXiv, so the two are worth running as separate processes.
    # Both write this file, so a read can land on a partial write or on the instant
    # between exists() and read(). The cap now comes from config, so concurrent writers
    # always write the same bytes and a lost read is harmless.
    prev = {}
    try:
        prev = json.loads(MANIFEST.read_text())
    except (OSError, json.JSONDecodeError):
        prev = {}
    if prev and prev.get("max_chars") != FRONT_MATTER_CHARS:
        print(f"  !! this run keeps {FRONT_MATTER_CHARS:,} chars but existing output in "
              f"{OUT.name}/ was written at {prev['max_chars']:,}. Mixing the two gives a "
              f"corpus whose length depends on which terminal fetched it. Delete the old "
              f"output or match the setting.")
    MANIFEST.write_text(json.dumps(
        {"max_chars": FRONT_MATTER_CHARS, "mode": "full" if full else "front_matter"},
        indent=1))

    jobs: list[tuple[dict, Path]] = []
    handles: dict[Path, object] = {}
    for d in domains:
        for s in splits:
            todo, dst, appending = load_job(d, s, a)
            if not todo:
                continue
            handles[dst] = open(dst, "a" if appending else "w")
            jobs.extend((p, dst) for p in todo)
    if not jobs:
        print("nothing left to do")
        return

    # Interleave so the PubMed Central work is spread through the run rather than
    # queued behind every arXiv paper; otherwise its limiter sits idle for hours.
    jobs.sort(key=lambda j: (j[0]["doi"],))
    by_route: dict[str, list] = collections.defaultdict(list)
    for j in jobs:
        by_route[j[0]["fulltext_route"]].append(j)
    interleaved: list = []
    streams = list(by_route.values())
    for i in range(max(len(s) for s in streams)):
        for s in streams:
            if i < len(s):
                interleaved.append(s[i])
    jobs = interleaved

    local = threading.local()
    limiters = {"arxiv": RateLimiter(a.rate),
                "ar5iv": RateLimiter(AR5IV_INTERVAL),
                "pmc": RateLimiter(0.15)}

    def session() -> requests.Session:
        s = getattr(local, "s", None)
        if s is None:
            s = requests.Session()
            local.s = s
        return s

    write_lock = threading.Lock()

    def fetch(p: dict, dst: Path) -> tuple[Path, bool, bool, str]:
        # Catch here, not in the consumer: an exception escaping the worker loses `dst`,
        # so the failure could not be attributed to a file in the per-split summary.
        try:
            return _fetch(p, dst)
        except Exception as exc:                          # noqa: BLE001 - one paper only
            print(f"  !! {p.get('doi')}: {exc.__class__.__name__}: {exc}")
            return dst, False, False, ""

    def _fetch(p: dict, dst: Path) -> tuple[Path, bool, bool, str]:
        route = p["fulltext_route"]
        text, bib, via = None, {}, route
        # Why a paper dropped, so the loss is attributable rather than a bare count.
        # This matters for materials science, whose whole train pool is only 1.06x the
        # target: if the losses are PDF-only submissions there is no parser fix, and
        # the domain target has to move instead.
        reason = "no_id"
        if route == "arxiv":
            aid = arxiv_id(p.get("fulltext_url"))
            if aid:
                reason = "no_source"
                # arxiv.org/html only exists from December 2023, so for an OLDER paper
                # that call is a guaranteed 404 that still costs a rate-limit slot --
                # which doubled the request count for 82% of the training window. Those
                # papers must go to ar5iv first.
                #
                # For a RECENT paper both hosts serve the same LaTeXML markup, so send
                # it to whichever limiter is free soonest. Pinning every recent paper to
                # arxiv.org/html left the whole run waiting on the slower 1.0s limiter
                # while ar5iv's 0.5s limiter idled: on CS that is 16,302 of 32,109
                # papers queued on the slow host, about 4.5 hours against 3.0 balanced.
                ARX = ("arxiv_html", "https://arxiv.org/html/", "arxiv")
                AR5 = ("ar5iv", AR5IV, "ar5iv")
                recent = (p.get("publication_date") or "") >= "2023-12"
                if recent and limiters["arxiv"].backlog() < limiters["ar5iv"].backlog():
                    hosts = [ARX, AR5]
                else:
                    hosts = [AR5, ARX]
                for name, base, limiter in hosts:
                    limiters[limiter].wait()
                    got_html = fetch_arxiv_html(aid, session(), base)
                    if got_html:
                        text, bib = got_html
                        via = name
                        break
                if not text:
                    # Last resort: the e-print tarball. Up to 16 MB, plus tar
                    # extraction and eight regex passes over the LaTeX.
                    limiters["arxiv"].wait()
                    got = fetch_arxiv(aid, session())
                    if got:
                        body, refs = got
                        text, bib = strip_latex(body), bibliography(refs)
                        # Keep only what the manuscript cites. \nocite{*} pulls the
                        # whole file in deliberately, so honour it and skip the filter.
                        cited = cited_keys(body)
                        if cited and "*" not in cited:
                            kept = {k: v for k, v in bib.items() if k in cited}
                            # A .bbl-only submission yields keys the .tex never spells
                            # out; if the filter would empty the list, it is wrong.
                            if kept:
                                bib = kept
                        via = "arxiv_eprint"
        elif route == "pmc":
            limiters["pmc"].wait()
            pid = pmc_id(p.get("fulltext_url"))
            if pid:
                reason = "no_source"
                got_pmc = fetch_pmc(pid, session())
                if got_pmc:
                    text, bib = got_pmc
        if not text:
            return dst, False, False, reason
        if len(text) < MIN_CHARS:
            return dst, False, False, "too_short"
        # fulltext_chars must describe what is STORED, not what was fetched. Recording
        # the pre-truncation length made every length audit wrong for capped papers,
        # and hid truncation instead of surfacing it. Keep the original separately.
        # One cap for every route and both modes: FRONT_MATTER_CHARS is MAX_CHARS in
        # full mode and the user's smaller budget in front-matter mode.
        stored = text[:FRONT_MATTER_CHARS]
        row = {**p, "fulltext": stored, "bibliography": bib,
               "fulltext_chars": len(stored), "fulltext_via": via}
        if len(text) > len(stored):
            row["fulltext_chars_original"] = len(text)
        # Written HERE rather than returned. Returning the row kept every fetched paper
        # alive in the futures dict for the whole run: 49,347 physics papers at a 76k
        # mean is ~3.7 GB of text held in memory for no reason, and the run would have
        # died on the larger splits. The worker now returns four small values.
        line = json.dumps(row) + "\n"
        with write_lock:
            handles[dst].write(line)
            handles[dst].flush()
        return dst, True, bool(bib), via

    ok = fail = no_bib = 0
    per_job: dict[Path, list[int]] = collections.defaultdict(lambda: [0, 0, 0])  # ok, fail, no_bib
    via_counts: collections.Counter = collections.Counter()
    fail_reasons: collections.Counter = collections.Counter()
    # Submit in batches so the executor's pending-work queue stays bounded too, instead
    # of materialising a work item for all 162,030 papers before the first fetch starts.
    CHUNK = 2_000
    try:
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            bar = tqdm(total=len(jobs), desc="fulltext", unit="paper", dynamic_ncols=True)
            for i in range(0, len(jobs), CHUNK):
                batch = jobs[i: i + CHUNK]
                futures = [pool.submit(fetch, p, dst) for p, dst in batch]
                del batch
                for fut in as_completed(futures):
                    try:
                        dst, good, has_bib, via = fut.result()
                    except Exception as exc:
                        bar.write(f"  !! {exc.__class__.__name__}: {exc}")
                        fail += 1
                        bar.update(1)
                        continue
                    if not good:
                        fail += 1
                        per_job[dst][1] += 1
                        fail_reasons[via] += 1
                    else:
                        ok += 1
                        per_job[dst][0] += 1
                        no_bib += not has_bib
                        per_job[dst][2] += not has_bib
                        via_counts[via] += 1
                    # no_bib is the number that decides whether Stage 3 is worth paying
                    # for, so it belongs on the bar rather than only in the summary.
                    bar.update(1)
                    bar.set_postfix(ok=ok, fail=fail, no_refs=no_bib)
            bar.close()
    finally:
        for h in handles.values():
            h.close()

    print(f"\n{'file':<26} {'ok':>7} {'fail':>7} {'no refs':>8}")
    for dst, (o, f, nb) in sorted(per_job.items()):
        print(f"{dst.name:<26} {o:>7,} {f:>7,} {nb:>8,}")
    print(f"\nroutes used: {dict(via_counts)}")
    if fail_reasons:
        print(f"failures:    {dict(fail_reasons)}")
        print("             no_source = no machine-readable text exists (PDF-only arXiv "
              "deposit, or PMC has no full-text XML). Not a parser fault; those papers "
              "cannot enter the benchmark.")
    print(f"total: {ok:,}/{len(jobs):,} ({100*ok/max(len(jobs),1):.0f}%)")
    if no_bib:
        print(f"!! {no_bib}/{ok} have an EMPTY bibliography. Inspirations must be "
              f"grounded in the reference list, so decomposition on those papers will "
              f"either return nothing or invent citations. Fix the parser before spending.")


if __name__ == "__main__":
    main()
