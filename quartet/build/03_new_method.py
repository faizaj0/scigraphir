#!/usr/bin/env python3
"""QUARTET stage 3, PROPOSED method: fixed-target decomposition with a
uniqueness test.

This is a separate program from 03_decompose.py on purpose. That file is the
BASELINE (ResearchBench / MOOSE-Star as published) and nothing here can change
it. The two are compared by running both over the same papers, so any shared
code path would be a way for one arm to contaminate the other.

What IS shared, deliberately, is the checker. Both methods import the same
CHECKS prompt and the same DeepSeek client from 03_decompose.py, because the
whole argument rests on an alternative set I' being judged at exactly the bar
the original set I had to clear. A second copy of that prompt would let the two
drift, and it did drift once already.

    baseline                              proposed
    ------------------------------        ------------------------------------
    one call -> b, h, I together          call 1 -> b, h*, hypothesis parts
                                          call 2 -> VERIFY that target
                                                    freeze it
                                          call 3 -> I, for the frozen target
    four checks on I                      four checks on I  (same prompt)
    fail -> redo everything, h moves      fail -> re-search I only, h* fixed
    stop at the first passing set         then: is I the only set?
                                            controls, then hide each i_j and
                                            search for a replacement I'
    output: one gold set                  output: M(b,h*), every valid set

Why the target is extracted and verified alone, before any citation is seen:

  A single call writes b, h and I together, so the hypothesis is composed by a
  model that already has a set of citations in mind. Asking afterwards whether
  I is the only set explaining h is then close to circular.

  Extracting the target alone fails differently, and the verification exists for
  that. On a paper whose contribution is a shorter proof of a known theorem, the
  "hypothesis" comes back as the theorem, which already sits in the background.
  Nothing is needed to reach a result already stated, so every later question
  about necessity is vacuous. Check 4 catches exactly that.

The paper is read twice but billed once and a bit: EXTRACT_TARGET,
VERIFY_TARGET and FIND_INSPIRATIONS share a literal prompt PREFIX, so the
~15,000 tokens of paper text are charged at $0.435/M on the first call and
$0.003625/M thereafter.

Usage
-----
    export DEEPSEEK_API_KEY=...
    python build/03_new_method.py --domain cs --split test \
        --input  data/_snap/cs_test_frozen.jsonl \
        --output data/_snap/v2_new.jsonl --limit 80 --workers 12 -v

    # skip the uniqueness sweep (fixed target only, for isolating its effect)
    python build/03_new_method.py ... --no-uniqueness
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import textwrap
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent


def _load_baseline():
    """Import 03_decompose.py, whose name is not a legal module identifier.

    Taken from it: the DeepSeek client (retry, cost accounting, prompt caching),
    the CHECKS prompt, JSON salvage, the inspiration formatter, the delta
    normaliser and the leak score. Reimplementing the checker here would let the
    two methods be judged by different graders, which is the one thing the
    comparison cannot survive.
    """
    spec = importlib.util.spec_from_file_location(
        "_baseline", Path(__file__).resolve().parent / "03_decompose.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


B = _load_baseline()

MODEL_MAIN = os.environ.get("NEW_MODEL", B.MODEL_DECOMPOSE)
MODEL_CHECK = os.environ.get("NEW_MODEL_CHECK", B.MODEL_CHECK)
MAX_ROUNDS = int(os.environ.get("NEW_MAX_ROUNDS", str(B.MAX_ROUNDS)))
MAX_PROMPT_CHARS = B.MAX_PROMPT_CHARS
MAX_BIB = int(os.environ.get("NEW_MAX_BIB", "150"))
# The rebuild transcribes a decomposition over a set that `sub` has already
# settled. Its only judgement is the ordering. Measured at effort=low it still
# emitted ~8,700 output tokens for ~600 tokens of content, which was 20% of the
# whole run's cost. Set NEW_REBUILD_REASONING=on to put it back and compare.
REBUILD_REASONING = os.environ.get("NEW_REBUILD_REASONING", "off").lower()
# Alternatives are held to a looser standard than the primary set, on purpose.
# Discarding an alternative over a phrase costs a uniqueness result, which is
# the finding; dropping a paper costs one row. So the rebuild retries once on a
# prompt echo and then keeps the sequence. Set NEW_REBUILD_STRICT_ECHO=1 to make
# it reject instead, once the echo rate is known to be low enough to afford it.
REBUILD_STRICT_ECHO = os.environ.get("NEW_REBUILD_STRICT_ECHO", "").lower() in ("1", "true", "on")
OUT_DEFAULT = ROOT / "data" / "03_new_method"

log = B.log
as_json = B.as_json


# =========================================================================== prompts
# PAPER_BLOCK is the literal PREFIX of the three prompts below. DeepSeek caches
# on the prompt prefix, so the paper's tokens are billed at full rate once and
# then at 1/120th of it. Do not reorder this and do not let anything
# paper-specific appear before it.
PAPER_BLOCK = """THE PAPER
{text}

ITS REFERENCE LIST
{refs}

"""

EXTRACT_TARGET = PAPER_BLOCK + """Recover the TARGET of this paper: the question it asked, the state of the field it was dissatisfied with, and the claim it went on to make. Do NOT identify inspirations. That is a separate step, and naming citations now would let them shape the hypothesis you write.

FIRST, DECIDE WHETHER THIS PAPER IS ELIGIBLE AT ALL.

Only a paper advancing a novel claim of its own can be decomposed. A review, survey, tutorial or position paper reports what others found: it has no hypothesis, and its reference list is a bibliography of a field rather than a set of inspirations. Judge by what the paper DOES, not by what its title contains. "The Gaia-ESO Survey: mapping the radial abundance gradient" reports new observations and IS eligible. "A survey of professionals in formal methods" collects new questionnaire data and IS eligible. "Recent advances in X" that only summarises other people's results is NOT.

If it is not eligible, return exactly {{"eligible": false, "ineligible_reason": "<one sentence saying what the paper does instead>"}} and nothing else.

Otherwise set "eligible": true and produce three components.

1. research_question -- the specific question this paper set out to answer. One sentence, phrased as a question, as the authors would have posed it BEFORE they had the answer. It must not contain, name or hint at the solution.

2. background_survey -- the prior methods that already addressed this question, and their limitations, drawn from the introduction and related work.

   Describe prior approaches GENERICALLY, by what they do, never by what they are called: no system or model names, no acronyms, no author names, no cited titles. This text becomes a retrieval query, so anything that identifies one specific paper makes retrieval trivial and destroys the benchmark.

   State the gap as a DEFICIENCY OF THE PRIOR METHODS, what they fail to achieve. Never as the ABSENCE OF A SOLUTION, which names the answer. Write "no systematic method exists to identify these non-terminating patterns", not "although X has been studied in Y, it has not been applied to this problem".

3. fine_grained_hypothesis -- the paper's actual claim: the mechanism, and how it is applied. This is what the paper CONTRIBUTES, not what it assumes or relies on. If the contribution is a new proof of a known theorem, the hypothesis is the PROOF STRATEGY, not the theorem, which existed already and belongs in the background.

Return ONLY json:
{{"eligible": true,
  "research_question": "...",
  "background_survey": "...",
  "fine_grained_hypothesis": "..."}}
"""

VERIFY_TARGET = PAPER_BLOCK + """A proposed TARGET has been extracted from the paper above. Check it against the paper BEFORE it is frozen. Once frozen it cannot be revised, so anything wrong here propagates into every later step.

RESEARCH QUESTION: {rq}

BACKGROUND SURVEY: {bg}

HYPOTHESIS: {hyp}

Answer four questions. Be strict. A "true" here means this target is used unchanged for everything that follows.

1. faithful -- Is the research question actually the question THIS paper set out to answer? Not a broader question the field cares about, not something the paper merely touches. If the paper's real question is narrower or different, answer false and say what it should have been.

2. background_ok -- Does the background describe PRIOR WORK on this question and where it fell short? It must be about what came before, not about what this paper does. Answer false if it describes the paper's own contribution, or if it is a generic field summary with no stated limitation.

3. hypothesis_is_the_contribution -- Does the hypothesis state what THIS PAPER contributes? Where the contribution is a shorter proof, a faster algorithm, or a new measurement of a known quantity, the hypothesis must be the new proof strategy, the new algorithm, or the new measurement method. Answer false if it instead states the underlying result, which existed before the paper.

4. disjoint -- INFORMATION DISJOINTNESS between the background and the hypothesis.
The background must remain strictly independent, leaking no information present in the hypothesis.

Check TWO kinds of leak, and report either as a failure.

(a) IDEA LEAK. The background mentions, paraphrases or strongly hints at the hypothesis. Vocabulary distinctive to the hypothesis appearing in the background counts.

(b) NAME LEAK. The background names the method, system, model or formalism THIS PAPER introduces, as a named thing, or gives its acronym.

The test is identification, not word overlap. Some shared vocabulary is unavoidable and is NOT a leak: a paper about the event calculus must say "event calculus" in its background, and a paper about learning noise must say "noise". Naming the FIELD, the FORMALISM or the general class of prior method is fine and necessary.

It IS a leak when the background:
  - states the hypothesis, or a distinctive phrase from it
  - names the thing this paper introduces, as a named thing ("the hybrid temporal situation calculus", "FunSearch")
  - describes the mechanism the hypothesis proposes, even in entirely different words
  - ends with a list of open questions that restates the hypothesis. "Key questions remain: whether synthetic hard negatives improve over mined ones" hands over a hypothesis about synthesised hard negatives.

Read the background's LAST sentence with particular care. The gap statement lands there, and it is where the leak appears. A gap stated as a DEFICIENCY OF PRIOR METHODS is fine. A gap stated as an ABSENCE OF SOME PARTICULAR SOLUTION is a leak, because it names the answer.

Ask yourself: given only this background, could a reader state the hypothesis? If yes, it leaks. If the background merely places the work in the same subject area, it does not.

This check is the important one. The background becomes a retrieval query and the hypothesis is what the inspirations have to supply. A background that already carries the hypothesis makes retrieval trivially easy and makes every later question about which inspirations were necessary vacuous.

When a leak is found, quote the offending phrase in "leak" and say which part of the BACKGROUND to cut. Fix the background, never the hypothesis: the hypothesis is what the paper contributes, and weakening it to pass this check defeats the check.

A worked failure, so you can recognise the shape of it: a paper whose question is "can this theorem be proved more shortly?" and whose extracted hypothesis reads "this class of graphs is strongly perfect" fails checks 3 and 4. That theorem was already known and already sits in the background. The paper's contribution is the proof, not the statement.

Return ONLY json:
{{"faithful": true, "background_ok": true,
  "hypothesis_is_the_contribution": true,
  "disjoint": true, "leak": "",
  "problem": "<if any check is false, one sentence on what to fix>"}}
"""

# The delta contract, spliced VERBATIM into the primary decomposition and into
# the rebuild that produces every alternative. D_1 and D_-j are compared against
# one another, so any difference in how the two were instructed would surface as
# a difference between the decompositions and be read as a finding.
DELTA_SPEC = """THE DECOMPOSITION IS AN ORDERED SEQUENCE, NOT A PILE.

Return the inspirations in DEPENDENCY ORDER. The first must be understandable from the fixed background alone. Each later one may build on what the earlier ones have already supplied, and none may depend on one that comes after it. Where two are genuinely independent, put first the one that supplies more of the fixed hypothesis.

Each "delta" is what THIS inspiration adds once the fixed background and every EARLIER inspiration in your list are already in hand. Applied in order they must build the fixed hypothesis and stop there:

    background, then + delta 0, then + delta 1, ... ends at the fixed hypothesis.

So delta 0 is written against the background alone, and delta 2 against the background plus deltas 0 and 1. A delta that repeats what an earlier one already supplied is wrong, and so is one that reaches past the fixed hypothesis.

Each delta is published as part of the decomposition, and a reader sees it without seeing these instructions. Name the concept, limitation or method you mean, never a thing they cannot look up. Three phrases in particular have no referent for them:

  "the fixed background" or "the fixed hypothesis"   -- "fixed" is a word from these instructions
  "the first / previous / earlier inspiration"       -- an index into a list they may not be reading
  "delta 0", "delta 1"                               -- our internal numbering

Saying where you are in the sequence is fine, because the sequence is published in order. Pointing at an item by its number is not.

  BAD   "The previous inspiration provides attention, but the fixed background still lacks a fusion mechanism."
  GOOD  "Cross-modal attention permits exchange between modalities, but does not yet provide a shared representation for classification."

Give every inspiration a "delta" as an object with exactly these three keys:

  "delta": {{"motivation":  "<what is still missing at this point in the sequence, and why this direction closes it>",
             "mechanism":   "<why it can work>",
             "methodology": "<how it is combined with the background and the inspirations before it>"}}

All three must be present and filled in.

"""

FIND_INSPIRATIONS = PAPER_BLOCK + """Select the inspirations for a paper whose background and hypothesis are ALREADY FIXED. They were extracted and verified in a separate step. They are not yours to revise, reword, extend or improve.

FIXED RESEARCH QUESTION: {rq}

FIXED BACKGROUND SURVEY: {bg}

FIXED HYPOTHESIS: {hyp}

{rejected}
WHAT AN INSPIRATION IS. A cited paper is an inspiration when it supplies something that solves or alleviates a specific LIMITATION of the prior methods described in the fixed background. The prior methods define the problem; the inspiration supplies the missing piece; together they give the fixed hypothesis.

BACKGROUND AND INSPIRATION ARE DIFFERENT ROLES and no cited work can fill both. Prior work aimed at this same question, which the authors improve on, extend or compete with, is BACKGROUND. Knowledge brought IN to overcome one of its shortfalls is INSPIRATION. The test is role, not distance.

Rules:
  - Between 1 and 4 inspirations. Most papers have 2 or 3.
  - Every inspiration must be genuinely necessary: if the fixed hypothesis still follows without it, leave it out.
  - Inspirations must be mutually distinct, not restatements of one another.
  - Each must name a work from the reference list above, copied exactly into "supposed_title". Do NOT invent a reference. If you cannot ground it, omit it.
  - Describe the KNOWLEDGE, never the citation string. "[13]" or "Smith et al. 2019" is not an inspiration.
  - Do not restate or sharpen the hypothesis. If your inspirations lead somewhere else, they are the wrong inspirations.

""" + DELTA_SPEC + """Return ONLY json. It carries no background or hypothesis fields, because those are fixed:
{{"inspirations": [{{"insp": "<the knowledge this work supplies, one sentence>",
                     "insp_concise": "<at most 5 words>",
                     "supposed_title": "<exact title from the reference list>",
                     "relation": "<how it connects to the fixed hypothesis>",
                     "motivation": "<the gap in the fixed background it closes>",
                     "delta": {{"motivation": "...", "mechanism": "...",
                                "methodology": "..."}}}}]}}
"""

# Block order matters here too: the paper block is the same prefix every other
# call uses, so a paper's k substitution calls all read it from cache, and only
# the UNAVAILABLE block at the bottom changes between them. The excluded paper
# stays visible in the reference list and the exclusion is enforced on the reply,
# which is stricter than trusting the model to honour a shortened list, and keeps
# the prefix byte-identical.
SUBSTITUTE = PAPER_BLOCK + """A paper's background and hypothesis are FIXED and given below. They are not yours to revise. Your task is to find a DIFFERENT route to the same fixed hypothesis.

FIXED RESEARCH QUESTION: {rq}
FIXED BACKGROUND: {bg}
FIXED HYPOTHESIS: {hyp}

The hypothesis above is the exact endpoint. Any set you propose must supply what is needed to reach THAT hypothesis, not a related or improved one.

CURRENT INSPIRATIONS:
{insps}

AVAILABLE CITED PAPERS (this paper's own reference list):
{bib}

TASK. One of the current inspirations is named as UNAVAILABLE at the end of this prompt. Assume that paper does not exist. Using only the AVAILABLE list, find the smallest set of papers that, together with the fixed background and whichever current inspirations you keep, still supplies everything needed to derive the fixed hypothesis.

Rules:
  - You may KEEP any current inspiration except the unavailable one. List the ones you keep by index.
  - Every paper you ADD must come from the AVAILABLE list, quoted by its exact title.
  - You may NOT use the unavailable paper, or any restatement of it, even though it still appears in the AVAILABLE list above.
  - The replacement must COVER what the unavailable paper supplied. State which part it covers.
  - Do NOT pad. If two papers do the job, do not name three.
  - If nothing in the AVAILABLE list can supply what the unavailable paper supplied, say so and set "found" to false. A false answer is a real result, not a failure. Do not invent a weak substitute in order to return something.

UNAVAILABLE -- you may NOT use this paper, or any restatement of it:
  [{ex_idx}] {ex_title}
    it supplied: {ex_supplies}

This step CHOOSES the papers only. It does not write the decomposition: once the set is settled, the whole sequence is rebuilt from scratch in a separate step, because a kept inspiration's delta was written against a neighbour that has just been removed.

Return ONLY json:
{{"found": true,
  "keep": [0],
  "replacement": [{{"insp": "<the knowledge this paper supplies, one sentence>",
                    "insp_concise": "<at most 5 words>",
                    "supposed_title": "<exact title copied from AVAILABLE>",
                    "covers": "<what the unavailable paper supplied that this now supplies>"}}],
  "why_not": "<if found is false, what the unavailable paper supplied that nothing else can>"}}
"""

# Every alternative goes through here, so I' is a decomposition in its own right
# rather than I with one row swapped. Splicing kept records straight into I' left
# them narrated against a paper that is no longer in the set: for I = {A,B}
# becoming I' = {A,C}, the stored delta for A was the one written when B was
# doing part of the work, and under a SEQUENCE that is simply the wrong object.
REBUILD = PAPER_BLOCK + """A paper's background and hypothesis are FIXED and given below, along with a set of its cited works that has ALREADY been chosen. Decompose that set. Do not revise it.

FIXED RESEARCH QUESTION: {rq}
FIXED BACKGROUND: {bg}
FIXED HYPOTHESIS: {hyp}

THE CHOSEN SET, listed in no meaningful order:
{chosen}

Do not add a work, do not drop one, and do not swap one for another. Membership is settled and is not yours to revise. Write every delta from scratch: none of these works carries a description over from any earlier decomposition, and you should assume none exists.

If you believe this set cannot reach the fixed hypothesis, set "reachable" to false and still give your best ordering and deltas. Judging that is a later step's job, not yours.

""" + DELTA_SPEC + """Return ONLY json, with the inspirations in the order you chose:
{{"reachable": true,
  "inspirations": [{{"insp": "<the knowledge this work supplies, one sentence>",
                     "insp_concise": "<at most 5 words>",
                     "supposed_title": "<exact title, copied from THE CHOSEN SET>",
                     "relation": "<how it connects to the fixed hypothesis>",
                     "motivation": "<the gap it closes at this point in the sequence>",
                     "delta": {{"motivation": "...", "mechanism": "...",
                                "methodology": "..."}}}}]}}
"""


# =========================================================================== helpers
_PUBMETA = re.compile(
    r"\([^)]*\)"
    r"|\b(?:press|university|springer|elsevier|wiley|acad(?:emic)?|publish\w*|"
    r"edition|eds?|vol|volume|pp|no|issue|chap(?:ter)?|"
    r"cambridge|oxford|london|berlin|heidelberg|new\s+york|\d{4})\b", re.I)


def norm_t(s) -> str:
    if isinstance(s, list):
        s = " ".join(str(x) for x in s)
    return re.sub(r"[^a-z0-9 ]", " ",
                  unicodedata.normalize("NFKD", str(s or "").lower())).strip()


def title_core(s) -> set[str]:
    """Words of a title with publication metadata removed.

    Without the strip, the same work under two citation styles reads as two
    works: "Convex Optimization" against "Convex Optimization (Cambridge
    University Press, New York, 2004)".
    """
    return set(norm_t(_PUBMETA.sub(" ", str(s or ""))).split())


def same_t(a, b, thresh: float = 0.75) -> bool:
    """Do two strings name the same work?

    Containment alone is wrong for short titles: {networks} sits inside {graph,
    neural, networks} at ratio 1.0, so "Networks" would match "Graph neural
    networks" and a perfectly good replacement would be thrown away as a
    duplicate. Below three content words there is not enough signal for
    containment to mean anything, so equality of the stripped cores is required.
    """
    wa, wb = title_core(a), title_core(b)
    if not wa or not wb:
        return False
    if wa == wb:
        return True
    if min(len(wa), len(wb)) < 3:
        return False
    return len(wa & wb) / min(len(wa), len(wb)) >= thresh


def bib_titles(paper: dict) -> list[str]:
    out, seen = [], set()
    for e in (paper.get("bibliography") or {}).values():
        t = e.get("title") if isinstance(e, dict) else str(e)
        if not t:
            continue
        k = norm_t(t)[:70]
        if k and k not in seen:
            seen.add(k)
            out.append(str(t).strip())
    return out


# The three levels every delta must carry. A decomposition is a SEQUENCE, and a
# delta missing a level is not a short delta, it is a broken link: Motivation is
# what the chain still lacks at that point, Mechanism is why this closes it, and
# Methodology is how it joins to what came before. Drop one and the step no
# longer says how the hypothesis was reached.
_DELTA_LEVELS = ("motivation", "mechanism", "methodology")
# Matched anywhere in the block, not anchored to the start of a line. The earlier
# version required "- Motivation (WHY):" exactly, and rejected these, all of
# which carry all three levels with real content:
#
#     "Motivation: ...\nMechanism: ...\nMethodology: ..."        no dash, no (WHY)
#     "Motivation (WHY): ... Mechanism (HOW IT WORKS): ..."      all on one line
#
# That is a defect in the reader, not in the answer. Tightening the prompt to
# force the exact shape made it worse: decomposition fell to 18% and the gate
# fired 2.18 times per paper on content that was fine.
_DELTA_HEAD = re.compile(r"(motivation|mechanism|methodology)\s*(?:\([^)]*\))?\s*:", re.I)


def delta_gaps(block) -> list[str]:
    """Which of the three levels are missing or empty.

    The prompt now asks for an object, so the normal path is an exact key check
    rather than a guess at where one level ends and the next begins. The text
    path below is kept for replies that still arrive as a string: the model is
    not obliged to honour the shape, and a delta that is present but formatted
    unexpectedly must not be scored as missing. Reading it wrongly is what took
    decomposition from 84% to 18% earlier.
    """
    if isinstance(block, dict):
        # Prefix-matched, because the model returns "motivation", "Motivation"
        # and "Motivation (WHY)" interchangeably. Five characters separates all
        # three levels: motiv / mecha / metho.
        low = {str(k).lower(): v for k, v in block.items()}
        return [lv for lv in _DELTA_LEVELS
                if not str(next((v for k, v in low.items()
                                 if k.lstrip("- ").startswith(lv[:5])), "")).strip()]

    text = str(block or "")
    heads = list(_DELTA_HEAD.finditer(text))
    got: dict[str, str] = {}
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        got[h.group(1).lower()] = text[h.end():end].strip(" -\n\t")
    return [lv for lv in _DELTA_LEVELS if not got.get(lv)]


def delta_problems(insps: list[dict]) -> list[str]:
    """One complaint per malformed delta. Empty means every delta is whole."""
    out = []
    for i, x in enumerate(insps):
        # The RAW delta, not the rendered block. as_delta_block turns an object
        # into text, and checking the text would throw away the exact key test
        # the object makes possible and fall back to parsing prose.
        raw = x.get("delta")
        if not str(B.as_delta_block(raw, x.get("insp_concise", ""))).strip():
            out.append(f"inspiration {i} has no delta at all")
        elif (gaps := delta_gaps(raw)):
            out.append(f"inspiration {i} is missing: {', '.join(gaps)}")
    return out


# This prompt's own vocabulary, echoed back into the answer. Measured on the
# first full run at 20% of deltas for "the fixed background" and 8% for "the
# first inspiration". hypothesis_components is a published field: a reader sees
# the decomposition and never sees this prompt, so "fixed" has no referent for
# them and "the first inspiration" points at a list they were never shown.
_PROMPT_ECHO = re.compile(
    r"\bfixed\s+(?:background|hypothesis)\b"
    r"|\b(?:first|second|third|fourth|previous|earlier|preceding)\s+inspiration\b"
    r"|\bdelta\s+\d\b", re.I)


def delta_echoes(insps: list[dict]) -> list[str]:
    """Deltas that quote the prompt instead of describing the decomposition.

    Kept apart from delta_problems because the two carry different weight. A
    missing Mechanism breaks the chain and is fatal. A stray "the fixed
    background" is a wording defect in text that is otherwise correct, and
    discarding the record over it would cost far more than it fixes.
    """
    out = []
    for i, x in enumerate(insps):
        m = _PROMPT_ECHO.search(B.as_delta_block(x.get("delta"),
                                                 x.get("insp_concise", "")))
        if m:
            out.append(f'inspiration {i} says "{m.group(0)}"')
    return out


def off_pool(insps: list[dict], pool: list[str]) -> list[str]:
    """Titles that are not in this paper's own citation pool.

    An inspiration has to be a work THIS paper cited. A title the model produced
    from its own knowledge can name a paper that genuinely exists and is
    genuinely relevant, and it would still be wrong: the gold set would then
    contain a document the authors never drew on. Matched at 0.7 rather than
    exactly, because the model retypes titles and reference strings carry
    publisher and year metadata around the title itself.
    """
    return [t for t in (str(x.get("supposed_title", "") or "") for x in insps)
            if not t or not any(same_t(t, c, 0.7) for c in pool)]


def all_gates_pass(g: dict, n_insp: int) -> bool:
    """All four checks, at exactly the bar the original set I had to clear.

    Two things this must not do, both of which a naive version does.

    Accept a partial verdict: `necessity` is per-inspiration, so a reply judging
    only index 0 of a three-item set says nothing about items 1 and 2. Accepting
    it counts silence as approval and inflates every pass rate.

    Accept truthy strings: the model returns JSON, and "false" is a non-empty
    string, so bool("false") is True. A gate that reads a rejection as approval
    is worse than no gate.
    """
    if not isinstance(g, dict) or n_insp <= 0:
        return False
    nec = g.get("necessity")
    if not isinstance(nec, list) or len(nec) != n_insp:
        return False
    if {x.get("index") for x in nec if isinstance(x, dict)} != set(range(n_insp)):
        return False
    if not all(x.get("necessary") is True for x in nec):
        return False
    return (g.get("sufficient") is True and g.get("disjoint") is True
            and g.get("non_redundant") is True)


def fmt_seq(insps: list[dict]) -> str:
    """The baseline's inspiration listing, with each delta underneath it.

    The four checks stay B.CHECKS, unmodified, so their DEFINITIONS are byte
    identical across the two arms and the comparison survives. What changes is
    only what the {insps} slot renders. The baseline's formatter emits one line
    per citation, which was the whole object when a decomposition was a set of
    citations. Here it is a sequence of (i_j, delta_j) pairs, and a checker shown
    only the citation half is asked to rule on necessity and non-redundancy
    without seeing what each inspiration actually contributes.
    """
    out = ["(This decomposition is an ORDERED SEQUENCE: delta j is what "
           "inspiration j adds once the background and inspirations 0..j-1 are "
           "already in hand, so the deltas accumulate to the hypothesis.)", ""]
    for i, x in enumerate(insps):
        out.append(f"[{i}] {x.get('insp_concise', '')}: {x.get('insp', '')}"
                   f"  (from: {x.get('supposed_title', '')})")
        d = B.as_delta_block(x.get("delta"), x.get("insp_concise", ""))
        if d:
            out.append(textwrap.indent(d, "    "))
        out.append("")
    return "\n".join(out).rstrip()


def check_set(d: dict, insps: list[dict], llm, tag: str) -> dict:
    """The BASELINE's four-check prompt, unmodified, on a candidate set."""
    return as_json(llm(B.CHECKS.format(rq=d.get("research_question", ""),
                                       bg=d.get("background_survey", ""),
                                       hyp=d.get("fine_grained_hypothesis", ""),
                                       insps=fmt_seq(insps)),
                       MODEL_CHECK, thinking=False, tag=tag))


def tomato_block(insps: list[dict], bg: str) -> dict:
    """The TOMATO-Star record fields for ONE decomposition.

    Built for the primary set and for every alternative alike, so an alternative
    in M is a complete object -- inspirations, delta blocks, breakdown -- that
    Stage 4 can resolve and validate exactly as it resolves the primary. Storing
    alternatives as bare title lists made them a second-class record that no
    later stage could process.
    """
    deltas = [B.as_delta_block(x.get("delta"), x.get("insp_concise", ""))
              for x in insps]
    return {
        "inspiration": [{
            # Position in the sequence. The deltas are cumulative, so i_2's delta
            # is written against the background PLUS i_0 and i_1, and reordering
            # the list silently changes what each delta means.
            "order": ix,
            "insp": str(x.get("insp", "") or ""),
            "insp_concise": str(x.get("insp_concise", "") or ""),
            "supposed_title": str(x.get("supposed_title", "") or ""),
            # Filled by Stage 4 (Semantic Scholar), for alternatives too.
            "found_title": None, "found_abstract": None, "found_doi": None,
            "match_quality": None, "similarity": None,
            "relation": str(x.get("relation", "") or ""),
            "motivation": str(x.get("motivation", "") or ""),
            "delta": deltas[ix],
            "title_leak": round(B.title_leak(
                str(x.get("supposed_title", "") or ""), bg), 3),
        } for ix, x in enumerate(insps)],
        # Positionally aligned with "inspiration". Empty blocks are kept rather
        # than filtered: under a sequence a missing delta is a hole in the chain
        # and has to stay visible instead of closing up behind itself.
        "hypothesis_components": deltas,
        "detailed_breakdown": "\n\n".join(d for d in deltas if d),
    }


def rebuild_sequence(target: dict, body: str, refs_block: str,
                     chosen: list[dict], llm, tag: str) -> list[dict] | None:
    """Regenerate a COMPLETE ordered decomposition over an already-chosen set.

    Returns None unless the rebuild covers every chosen paper with a whole delta,
    after one retry. An earlier version filled a gap by appending the paper's
    record from D_1, which is the one thing this function exists to prevent: that
    delta was written when a paper now removed from the set was doing part of the
    work, so the alternative would claim to be a complete decomposition while
    carrying a step that describes a set nobody proposed. A rejected alternative
    costs one substitution; a silently patched one is a wrong gold record.

    Membership is enforced rather than trusted: the reply is matched back onto
    `chosen` by title and anything unrecognised is dropped, so the return value
    is always the chosen set reordered and re-narrated, never a different set
    that the later checks would then be judging by mistake.
    """
    listing = "\n".join(
        f"  - {x.get('supposed_title', '')}\n      supplies: {x.get('insp', '')}"
        for x in chosen)
    for attempt in range(2):
        r = as_json(llm(REBUILD.format(
            text=body, refs=refs_block,
            rq=target.get("research_question", ""),
            bg=target.get("background_survey", ""),
            hyp=target.get("fine_grained_hypothesis", ""),
            chosen=listing), MODEL_MAIN, thinking=True,
            tag=tag if attempt == 0 else f"{tag}-retry",
            reasoning=None if REBUILD_REASONING == "on" else "off"))

        ordered, used = [], set()
        for x in (r.get("inspirations") or []):
            if not isinstance(x, dict):
                continue
            t = x.get("supposed_title", "")
            hit = next((i for i, c in enumerate(chosen)
                        if i not in used and same_t(t, c.get("supposed_title", ""))),
                       None)
            if hit is None:
                continue
            used.add(hit)
            # The chosen title wins over the model's transcription of it, so a
            # reworded title cannot break the reference-list match later.
            ordered.append({**x,
                            "supposed_title": chosen[hit].get("supposed_title", "")})
        # Coverage and whole deltas are always required. A prompt echo buys a
        # second attempt, and by default the sequence is then kept: discarding a
        # structurally sound alternative over a phrase throws away a real
        # uniqueness result to fix a wording slip. REBUILD_STRICT_ECHO makes it
        # reject instead, for when the echo rate is low enough to afford it.
        whole = len(used) == len(chosen) and not delta_problems(ordered)
        if whole and not delta_echoes(ordered):
            return ordered
        if whole and attempt and not REBUILD_STRICT_ECHO:
            return ordered
    return None


# ================================================================ 1. lock the target
def extract_target(paper: dict, refs_block: str, llm,
                   rounds: int = MAX_ROUNDS) -> tuple[dict, dict]:
    """Recover b and h*, verify them against the paper, and only then freeze.

    Returns (target, log). target is empty if the paper is ineligible or if the
    target never verified, and the log says which.
    """
    body = paper["fulltext"][:MAX_PROMPT_CHARS]
    log_: dict = {"rounds": 0, "failures": []}
    feedback = ""
    for rnd in range(rounds):
        log_["rounds"] = rnd + 1
        prompt = EXTRACT_TARGET.format(text=body, refs=refs_block)
        if feedback:
            prompt += f"\nA previous attempt was rejected. Fix this and try again:\n{feedback}\n"
        d = as_json(llm(prompt, MODEL_MAIN, thinking=True, tag="extract-target"))

        if d.get("eligible") is False:
            return {}, {**log_, "ineligible": True,
                        "ineligible_reason": str(d.get("ineligible_reason", ""))[:300]}
        if not d.get("fine_grained_hypothesis"):
            log_["failures"].append("empty")
            feedback = "No hypothesis was returned. Produce all four components."
            continue

        # On MODEL_MAIN, not the cheap checker, and that is the cheaper choice.
        # DeepSeek caches per MODEL, so a flash call cannot reuse the ~15,000
        # paper tokens the pro extract call just cached, and pays $0.14/M for
        # them cold. The same call on pro reads them at $0.003625/M. Measured:
        # $0.0006 against $0.0023. thinking=False keeps the output short, which
        # is where pro is otherwise expensive. Better judge, quarter the price.
        v = as_json(llm(VERIFY_TARGET.format(
            text=body, refs=refs_block,
            rq=d.get("research_question", ""), bg=d.get("background_survey", ""),
            hyp=d.get("fine_grained_hypothesis", "")),
            MODEL_MAIN, thinking=False, tag="verify-target"))
        # Strict booleans, for the same reason as the four gates.
        failed = [k for k in ("faithful", "background_ok",
                              "hypothesis_is_the_contribution", "disjoint")
                  if v.get(k) is not True]
        if not failed:
            log_["verified"] = True
            return {"research_question": str(d.get("research_question", "") or ""),
                    "background_survey": str(d.get("background_survey", "") or ""),
                    "fine_grained_hypothesis": str(d.get("fine_grained_hypothesis", "") or ""),
                    }, log_

        log_["failures"].append(",".join(failed))
        feedback = f"The target failed these checks: {', '.join(failed)}. {v.get('problem', '')}"
        # A disjointness failure has two exits and only one of them is legal.
        # The retry reruns the whole extraction, so left to itself the model can
        # satisfy b _|_ h* by moving h* to something the background happens not
        # to mention. That passes the check and destroys the target. Name the
        # leaked phrase and say which side to cut.
        if "disjoint" in failed:
            leak = str(v.get("leak", "") or "")[:400]
            log_.setdefault("leaks", []).append(leak)
            feedback += (f"\nThe background leaks the hypothesis here: \"{leak}\"\n"
                         f"Cut that from the BACKGROUND SURVEY. Keep the hypothesis "
                         f"exactly as it is: it is what the paper contributes, and "
                         f"changing it to dodge this check invalidates the target.")
        if llm.verbose:
            log(f"    target rejected: {failed} -- {str(v.get('problem',''))[:110]}")

    return {}, {**log_, "verified": False}



# ============================================================ 3. the uniqueness test
def uniqueness(d: dict, paper: dict, insps: list[dict], body: str,
               refs_block: str, llm) -> dict:
    """Hide each inspiration in turn and search for another set explaining h*."""
    bib = bib_titles(paper)
    out: dict = {"n_bib": len(bib), "I": [x.get("supposed_title", "") for x in insps]}

    # -- the sweep
    avail = bib[:MAX_BIB]
    listing = "\n".join(f"  - {t}" for t in avail)
    sweep, valid = [], []
    for jx, ex in enumerate(insps):
        ex_title = ex.get("supposed_title", "")
        r = as_json(llm(SUBSTITUTE.format(
            text=body, refs=refs_block,
            rq=d.get("research_question", ""), bg=d.get("background_survey", ""),
            hyp=d.get("fine_grained_hypothesis", ""), insps=fmt_seq(insps),
            bib=listing, ex_idx=jx, ex_title=ex_title,
            ex_supplies=ex.get("insp", "")),
            MODEL_MAIN, thinking=True, tag=f"sub{jx}"))
        s = {"excluded": ex_title, "found": bool(r.get("found")),
             "why_not": str(r.get("why_not", ""))[:300]}

        # A kept index is dropped if it NAMES the hidden work: one paper's
        # inspiration list held the same title twice under different indices, so
        # keeping the other index put the hidden paper straight back into I'.
        keep = [i for i in (r.get("keep") or []) if isinstance(i, int)
                and 0 <= i < len(insps) and i != jx
                and not same_t(insps[i].get("supposed_title", ""), ex_title)]
        kept_titles = [insps[i].get("supposed_title", "") for i in keep]

        # The exclusion holds by construction, not by trusting the model. In
        # order: not the hidden paper; not a paper we are keeping (else the model
        # "replaces" one inspiration with a restatement of another, measured on
        # {Convex Optimization, Caratheodory}); not a repeat of another
        # replacement; and actually present in this reference list.
        repl, seen = [], []
        for x in (r.get("replacement") or []):
            if not isinstance(x, dict):
                continue
            t = x.get("supposed_title", "")
            if same_t(t, ex_title) or any(same_t(t, k) for k in kept_titles):
                continue
            if any(same_t(t, s2) for s2 in seen):
                continue
            if not any(same_t(t, c, 0.7) for c in avail):
                continue
            seen.append(t)
            repl.append(x)

        i_prime = [insps[i] for i in keep] + repl
        # I' must be a genuine ALTERNATIVE, not a pruned copy of I: dropping i_j
        # and adding nothing is the same explanation with a piece removed.
        orig = {norm_t(x.get("supposed_title", "")) for x in insps}
        newp = {norm_t(x.get("supposed_title", "")) for x in i_prime}
        if not s["found"] or not repl or not i_prime or newp <= orig:
            s.update(found=False, label="no_substitute_found")
            if repl and newp <= orig:
                s["why_not"] = "proposed set is a subset of the original, not an alternative"
        elif (rebuilt := rebuild_sequence(d, body, refs_block, i_prime, llm,
                                          f"rebuild{jx}")) is None:
            # The chosen set is decomposed from scratch BEFORE it is checked, and
            # a partial rebuild is discarded rather than topped up from D_1. An
            # alternative that cannot be decomposed completely is not evidence
            # that i_j is replaceable.
            s.update(found=False, label="rebuild_failed",
                     why_not="the alternative set could not be decomposed "
                             "completely: some paper came back with no delta, or "
                             "with one of the three levels missing")
        else:
            i_prime = rebuilt
            g = check_set(d, i_prime, llm, f"verify{jx}")
            s["gates"] = g
            s["verified"] = all_gates_pass(g, len(i_prime))
            s["I_prime"] = [x.get("supposed_title", "") for x in i_prime]
            s["replacement"] = repl
            s["label"] = "replaceable" if s["verified"] else "no_substitute_found"
            # A full TOMATO-Star record for the alternative, not just its titles,
            # so Stage 4 resolves it the same way it resolves the primary set.
            s["decomposition"] = tomato_block(i_prime, d.get("background_survey", ""))
            if s["verified"]:
                valid.append((s["I_prime"], i_prime))
        sweep.append(s)

    out["sweep"] = sweep
    # Deduplicate M: hiding i_1 and hiding i_2 can land on the same alternative,
    # and counting it twice inflates |M|, which is the headline number. Keyed on
    # the title SET, never on the delta sequence: two orderings of the same
    # papers are one explanation, and keying on order would count permutations.
    seen_sets, M = set(), []
    for titles, members in [(out["I"], insps)] + valid:
        k = frozenset(norm_t(t) for t in titles if t)
        if k and k not in seen_sets:
            seen_sets.add(k)
            # Each member of M is a complete decomposition, so every entry in the
            # family is the same kind of object as the primary one.
            M.append({"titles": titles,
                      **tomato_block(members, d.get("background_survey", ""))})
    out["M"] = M
    out["n_valid_sets"] = len(M)
    out["uniqueness"] = "falsified" if len(M) > 1 else "not_falsified"
    out["labels"] = {s["excluded"]: s["label"] for s in sweep if s.get("excluded")}
    # The budget the "no substitute found" verdicts are relative to. Stated,
    # because that claim is only ever as strong as the search that produced it.
    out["budget"] = {"model": MODEL_MAIN, "checker": MODEL_CHECK,
                     "effort": B.EFFORT,
                     "candidates_offered": min(len(bib), MAX_BIB),
                     "searches_per_inspiration": 1}
    return out


# =========================================================================== pipeline
def process(paper: dict, llm, do_uniq: bool) -> dict:
    refs = list(paper.get("bibliography", {}).values())
    refs_block = "\n".join(f"- {r}" for r in refs[:200]) or "(reference list unavailable)"
    # What a gold inspiration is allowed to be. Deliberately the WHOLE
    # bibliography, not the 200 entries the prompt shows: a title from further
    # down the list is still a work this paper cited, and rejecting it would
    # discard a valid inspiration for a reason that is about our truncation.
    pool = bib_titles(paper)
    body = paper["fulltext"][:MAX_PROMPT_CHARS]
    short = paper["doi"][-18:]
    gates = {"rounds": 0, "empty": 0, "checks_unparsed": 0,
             "off_pool": 0, "delta_incomplete": 0, "delta_prompt_echo": 0,
             "necessary_failed": 0, "sufficient_failed": 0,
             "disjoint_failed": 0, "redundant_failed": 0}

    # ---- 1. extract, verify and FREEZE the target, before any citation is seen
    if llm.verbose:
        log(f"  [{short}] extracting target ({len(paper['fulltext']):,} chars)")
    target, tlog = extract_target(paper, refs_block, llm)
    if tlog.get("ineligible"):
        return {**paper, "decomposed": False, "ineligible": True,
                "ineligible_reason": tlog.get("ineligible_reason", ""),
                "gates": gates, "target_log": tlog}
    if not target:
        # The target never satisfied all four checks. Freezing a hypothesis known
        # to be wrong about the paper would make everything after it meaningless.
        return {**paper, "decomposed": False, "target_unverified": True,
                "gates": gates, "target_log": tlog}

    # ---- 2. find inspirations for the frozen target; retries re-search only I
    feedback, prev, result = "", [], {}
    for rnd in range(MAX_ROUNDS):
        gates["rounds"] = rnd + 1
        rejected = (f"\nA previous selection was REJECTED for this reason:\n{feedback}\n"
                    f"\nPREVIOUS SELECTION (rejected):\n{fmt_seq(prev)}\n") if feedback else ""
        d = as_json(llm(FIND_INSPIRATIONS.format(
            text=body, refs=refs_block,
            rq=target["research_question"], bg=target["background_survey"],
            hyp=target["fine_grained_hypothesis"],
            rejected=rejected), MODEL_MAIN, thinking=True, tag="find-inspirations"))
        insps = d.get("inspirations") or []
        d = {**target, "inspirations": insps}

        if not insps:
            gates["empty"] += 1
            if gates["empty"] >= 2:
                return {**paper, "decomposed": False, "no_inspirations": True,
                        "gates": gates, "target_log": tlog, **target}
            feedback = ("No inspirations were grounded in the reference list. Look "
                        "again, and if the paper genuinely borrows nothing, say so.")
            continue
        prev = insps

        # Two structural gates before any checker call. Both are free, both are
        # decidable in code, and both catch things the LLM checker is not asked
        # about, so running them first also saves the checker call when they fire.
        if (off := off_pool(insps, pool)):
            gates["off_pool"] += 1
            feedback = ("These titles are not in this paper's reference list: "
                        + "; ".join(f'"{t}"' for t in off[:4])
                        + ". Every inspiration must be a work THIS paper cited, "
                          "copied exactly from the reference list above. A real "
                          "and relevant paper the authors did not cite is not an "
                          "inspiration. Replace them or drop them.")
            continue
        if (dprob := delta_problems(insps)):
            gates["delta_incomplete"] += 1
            # Keep the first rejected block. Without it this gate is a counter
            # with no witness: the malformed text is used to build the retry and
            # then dropped, so a regression shows up as a number going up and
            # nothing to look at. One block per paper, capped, costs nothing.
            bad = B.as_delta_block(insps[int(dprob[0].split()[1])].get("delta"),
                                   insps[int(dprob[0].split()[1])].get("insp_concise", ""))
            # 2,000, not 600: at 600 the sample was cut mid-sentence, so six of
            # seven blocks that looked like failures were only my own truncation.
            # A witness you have to distrust is worse than none.
            gates.setdefault("delta_reject", bad[:2000] or "<empty>")
            feedback = ("Some deltas are incomplete: " + "; ".join(dprob[:4])
                        + ". Every delta needs all three levels filled in with "
                          "real content: Motivation (WHY), Mechanism (HOW IT "
                          "WORKS) and Methodology (HOW IT'S INTEGRATED). This is "
                          "what came back for the first of them:\n" + bad[:600])
            continue
        # First round only. Enforcing this every round was tried and measured:
        # mean rounds went 1.54 -> 1.70 and the falsification rate fell from 82%
        # to 68%, because a paper that keeps echoing burns all of MAX_ROUNDS and
        # is dropped, and its alternatives are lost with it. The wording of a
        # published text field is not worth 14 points of the headline result.
        # The rule stays in the prompt, which is what took echoes from 60% of
        # papers to 24%; this gate only buys one call to clean up the residue.
        if rnd == 0 and (echo := delta_echoes(insps)):
            gates["delta_prompt_echo"] += 1
            feedback = ("These deltas quote the instructions instead of "
                        "describing the decomposition: " + "; ".join(echo[:4])
                        + ". A reader sees only the delta blocks, never these "
                          "instructions, so \"fixed\" has nothing to refer to and "
                          "\"the first inspiration\" points at a list they cannot "
                          "see. Say what the background contains, and name each "
                          "earlier inspiration by what it supplied.")
            continue

        c = check_set(d, insps, llm, "checks")
        if not c:
            gates["checks_unparsed"] += 1
            c = check_set(d, insps, llm, "checks-retry")
        if not c:
            # This used to accept the set with an UNVERIFIED flag. A record that
            # never passed the four checks is not a gold record: in the output it
            # is indistinguishable from one that did, so every rate computed over
            # the file silently becomes a rate over a mixture. Retry instead, and
            # if the rounds run out the paper is dropped, which is the honest
            # outcome for a decomposition nothing was able to verify.
            gates["checks_unparsed"] += 1
            feedback = ("The quality checks came back unreadable twice running. "
                        "Re-select the inspirations, keeping the set small and "
                        "each description short and concrete.")
            continue

        unnecessary = [x for x in (c.get("necessity") or []) if x.get("necessary") is not True]
        if not isinstance(c.get("necessity"), list) or len(c["necessity"]) != len(insps):
            gates["checks_unparsed"] += 1
            feedback = (f"The check returned a verdict for only "
                        f"{len(c.get('necessity') or [])} of {len(insps)} inspirations. "
                        f"Judge every one.")
            continue
        if unnecessary:
            gates["necessary_failed"] += 1
            feedback = ("These inspirations were judged unnecessary: "
                        + "; ".join(f"{u.get('index')}: {u.get('reason','')}" for u in unnecessary))
            continue
        if c.get("sufficient") is not True:
            gates["sufficient_failed"] += 1
            feedback = f"The set was insufficient. Missing: {c.get('missing','')}"
            continue
        if c.get("disjoint") is not True:
            # The background is FROZEN, so this cannot be fixed by rewording it.
            # A leak means the citation was misclassified: it is background, not
            # an inspiration, and the retry must drop it.
            gates["disjoint_failed"] += 1
            feedback = (f"The background identifies one of these cited works: "
                        f"{c.get('leak','')}. The background cannot be changed. That "
                        f"citation is background, not an inspiration, so drop it and "
                        f"choose a different one.")
            continue
        if c.get("non_redundant") is not True:
            gates["redundant_failed"] += 1
            feedback = (f"Inspirations {c.get('duplicate_pair') or []} express the same "
                        f"idea: {c.get('duplicate_reason','')}. Keep one and replace the other.")
            continue

        result = d
        break

    if not result:
        return {**paper, "decomposed": False, "gates": gates,
                "target_log": tlog, **target}

    # ---- 3. the uniqueness test. Reaching here means the set cleared all four
    # checks: there is no longer an UNVERIFIED path that could arrive with less.
    uniq: dict = {}
    if do_uniq:
        if llm.verbose:
            log(f"  [{short}] uniqueness: sweeping {len(result['inspirations'])} inspirations")
        uniq = uniqueness(result, paper, result["inspirations"],
                          body, refs_block, llm)

    return {
        "doi": paper["doi"], "source_id": paper["doi"].replace("/", "_"),
        "title": paper["title"], "abstract": paper["abstract"],
        "domain": paper["domain"], "split": paper["split"],
        "primary_field": paper["primary_field"],
        "publication_date": paper["publication_date"],
        "method": "new",
        "research_question": target["research_question"],
        "background_survey": target["background_survey"],
        "fine_grained_hypothesis": target["fine_grained_hypothesis"],
        "target_log": tlog,
        # Carried through for Stage 4, trimmed to the two fields it reads.
        #
        # This return builds a FRESH dict rather than extending `paper`, which
        # silently dropped the reference list on exactly the records that go on
        # to be resolved -- the dropped papers kept it, the kept ones did not. So
        # Stage 4's "stage 2b shortcut" could never fire, and every inspiration
        # took an individual OpenAlex title search, the one request in that stage
        # that cannot be batched. Measured on 400 biology papers: 92.4% of
        # inspirations are matchable against this list, so restoring it turns
        # ~16,600 single searches into ~1,600 requests.
        #
        # Trimmed to title and doi because that is all `_lookup` uses, and the
        # full records carry abstracts that would multiply the file size.
        "openalex_references": [
            {"title": r["title"], "doi": r["doi"]}
            for r in (paper.get("openalex_references") or [])
            if r.get("title") and r.get("doi")
        ],
        # The same builder every alternative in M goes through, so the primary
        # decomposition and its alternatives are indistinguishable in shape.
        **tomato_block(result["inspirations"], target["background_survey"]),
        "decomposed": True,
        "gates": gates,
        "uniqueness": uniq,
    }


# =============================================================================== main
def _status(llm, ok: int, done: int, failed: int, total: int, workers: int) -> str:
    """What the bar shows: spend so far, yield, and a projected total.

    THE PROJECTION HAS TO ACCOUNT FOR WORK IN FLIGHT. `llm.cost` is account-wide
    and includes every call made for papers that have NOT finished yet. Dividing
    it by `done` alone therefore charges the whole cohort's opening calls to the
    handful that have completed, and at 200 workers that read $42 against a $2
    run five papers in. The error shrinks as `done` grows and vanishes at the
    end, which is exactly when a projection stops being useful.

    So: while fewer papers have finished than are running, spend is attributed
    across everything actually touched and the estimate is shown as a RANGE.
    cost/touched is the floor (in-flight papers are only partly paid for),
    cost/done is the ceiling. Once a full cohort has completed, the two agree
    well enough to quote one number.
    """
    touched = done + min(workers, max(total - done, 0))
    hi = llm.cost / max(done, 1) * total
    lo = llm.cost / max(touched, 1) * total
    proj = (f"proj ${hi:.2f}" if done >= workers or done >= total
            else f"proj ${lo:.2f}-{hi:.0f}")
    return (f"${llm.cost:.2f} ({proj}) "
            f"| ok {100 * ok / max(done, 1):.0f}%"
            + (f" | {failed} errored" if failed else ""))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--input", type=Path)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--workers", type=int, default=48,
                    help="concurrent PAPERS (default 48). DeepSeek rate-limits by "
                         "concurrency, not requests per minute, and flash allows "
                         "2,500 simultaneous account-wide. Each paper is ~12 "
                         "SEQUENTIAL calls averaging 22s, so the stage is latency "
                         "bound: workers is the only lever on wall time. Under "
                         "load DeepSeek slows rather than failing, so raising this "
                         "cannot lose work, it just stops helping. Lower it if you "
                         "start seeing 429s")
    ap.add_argument("--no-uniqueness", action="store_true",
                    help="stop after the fixed-target decomposition, so the effect of "
                         "locking can be measured apart from the uniqueness test")
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    src = a.input or (ROOT / "data" / "02_fulltext" / f"{a.domain}_{a.split}.jsonl")
    out = a.output or (OUT_DEFAULT / f"{a.domain}_{a.split}.jsonl")
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key and not a.dry_run:
        sys.exit("DEEPSEEK_API_KEY is not set. Export it, or pass --dry-run.")

    papers = [json.loads(l) for l in open(src) if l.strip()]
    if a.limit:
        papers = papers[:a.limit]

    done = set()
    if out.exists():
        done = {json.loads(l).get("doi") for l in open(out) if l.strip()}
        papers = [p for p in papers if p.get("doi") not in done]
        if done:
            print(f"resuming: {len(done)} already done, {len(papers)} to go")
    print(f"{len(papers)} papers from {src.name}"
          f"{'' if a.no_uniqueness else '  (with uniqueness test)'}")

    llm = B.DeepSeek(key, dry_run=a.dry_run, verbose=a.verbose)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(out, "a") as fh, ThreadPoolExecutor(max_workers=a.workers) as pool:
        futs = {pool.submit(process, p, llm, not a.no_uniqueness): p
                for p in papers}
        # Live accounting on the bar, because this is the stage that spends
        # money and takes an hour: the two things worth watching are what it has
        # cost so far and what share of papers are surviving the gates. Both are
        # invisible until the final report otherwise.
        bar = tqdm(total=len(futs), desc="decompose", unit="paper",
                   dynamic_ncols=True, smoothing=0.05)
        done = ok = failed = 0
        for f in as_completed(futs):
            done += 1
            try:
                row = f.result()
            except Exception as exc:
                failed += 1
                bar.write(f"  !! {futs[f].get('title','?')[:60]}: "
                          f"{exc.__class__.__name__}: {exc}")
                bar.set_postfix_str(_status(llm, ok, done, failed, len(futs), a.workers), refresh=False)
                bar.update(1)
                continue
            ok += bool(row.get("decomposed"))
            rows.append(row)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            bar.set_postfix_str(_status(llm, ok, done, failed, len(futs), a.workers), refresh=False)
            bar.update(1)
        bar.close()

    ok = [r for r in rows if r.get("decomposed")]
    print(f"\n{len(ok)}/{len(rows)} decomposed")
    for k, lab in (("ineligible", "review or survey"),
                   ("target_unverified", "target failed its four checks"),
                   ("no_inspirations", "no groundable inspirations")):
        n = sum(1 for r in rows if r.get(k))
        if n:
            print(f"  dropped, {lab}: {n}")
    sw = [r for r in ok if "M" in (r.get("uniqueness") or {})]
    if sw:
        fal = sum(1 for r in sw if r["uniqueness"]["uniqueness"] == "falsified")
        print(f"\n  uniqueness falsified: {fal}/{len(sw)} ({100*fal/len(sw):.0f}%)")
        print(f"  mean |M|            : "
              f"{sum(r['uniqueness']['n_valid_sets'] for r in sw)/len(sw):.2f}")
    print(f"\n{llm.calls} calls   ${llm.cost:.4f}"
          + (f"   ${llm.cost/max(len(rows),1):.4f}/paper" if rows else ""))
    tot_in = llm.tok_in + llm.tok_cached
    if tot_in:
        print(f"  input {tot_in:,} tokens, {llm.tok_cached:,} cached "
              f"({100*llm.tok_cached/tot_in:.0f}%)   output {llm.tok_out:,}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
