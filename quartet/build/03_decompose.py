"""
Stage 3 -- inspiration decomposition.

ResearchBench's control flow (Liu et al., 2026, Figure 1), TOMATO-Star's output schema
(Yang & Bing, 2026, Sec. 4). Neither released construction code, so both are
reimplemented from the papers; Stage 8 validates the reimplementation by re-running it
over papers already in TOMATO-Star and scoring gold recovery.

Control flow (ResearchBench's loop, TOMATO-Star's four gates):

    Research Paper --> DECOMPOSE (v4-pro, thinking) --> b, h, [i_1 ... i_n] + deltas
                            ^                                    |
                            |                          CHECKS (v4-flash)
                            |                            1 Information Necessity
      (any gate fails, find again, max 3 rounds)          2 Information Sufficiency
                            |                            3 Information Disjointness
                            |                            4 Non-Redundancy
                            +--------- all pass -------------> End

Two calls per paper. The four gates take identical inputs and none consumes another's
verdict, so issuing them separately bought three extra round trips and nothing else.
Delta blocks come back from the decomposition call, which already holds everything
needed to write them.

Gates 3 and 4 are the ones that protect the benchmark rather than the extraction.
Disjointness matters because the extractor's pretraining postdates the test window, so
a memorising model could write a background survey quietly containing its own answer,
and that survey becomes the retrieval query. Non-redundancy matters because redundant
inspirations become near-duplicate gold documents, which inflates recall.

Models: v4-pro at medium effort for decomposition, v4-flash at low effort for the gates.
The gates need SOME reasoning: with thinking disabled they passed 90/90 papers on round
1 while an independent audit found 3 backgrounds naming their own cited paper.
`deepseek-chat` and `deepseek-reasoner` were retired on 2026-07-24; they are not options.

Usage:
  export DEEPSEEK_API_KEY=...
  python build/03_decompose.py --domain cs --split test --limit 20
  python build/03_decompose.py --domain cs --split test --limit 3 --dry-run   # prompts only
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "02_fulltext"
OUT = ROOT / "data" / "03_decomposed"

API = "https://api.deepseek.com/chat/completions"
# Flash everywhere. It was v4-pro here, on the argument that the judgement is
# genuinely hard, and that is still true -- but the two limits that bind this
# pipeline both favour flash by a factor of about five:
#
#   concurrency  2,500 simultaneous requests against pro's 500
#   price        $0.14/M in and $0.28/M out, against $0.435 and $0.87
#
# Both arms read this constant (03_new_method.py takes B.MODEL_DECOMPOSE), so
# changing it moves them together and the comparison stays on one model.
#
# To put a single run back on pro without editing anything:
#     NEW_MODEL=deepseek-v4-pro python build/03_new_method.py ...
#     python build/03_decompose.py --model deepseek-v4-pro ...
MODEL_DECOMPOSE = "deepseek-v4-flash"   # thinking still on: see llm() call sites
MODEL_CHECK = "deepseek-v4-flash"       # gates: thinking ON at low effort (see below)
MAX_ROUNDS = 3                          # ResearchBench's "find again" loop bound

# DeepSeek rate-limits by CONCURRENCY, not requests per minute: 500 simultaneous for
# v4-pro and 2,500 for v4-flash, account-wide. There is no RPM or TPM cap; under load
# responses slow rather than fail. On flash the ceiling is 2,500, so the binding
# constraint is now how many papers you have, not the account.
WORKERS = 32

# Must not sit below Stage 2's MAX_CHARS, or --full is a lie: Stage 2 stores up to
# 150k chars and a smaller cap here silently discards the rest. The old hardcoded 48k
# truncated the average biology paper (70k) and most long maths (54k) and physics (40k)
# ones with no warning. 150k is ~37k tokens against a 1M context, so the binding
# constraint is cost, not capacity, and truncations are now counted and reported.
# Read from the manifest Stage 2 wrote, NOT from the environment. Both stages used to
# read QUARTET_MAX_CHARS directly, so a terminal with it exported and one without gave
# different limits for the same corpus, and text stored at 200,000 was cut to 150,000
# here with no warning. The manifest is what Stage 2 actually did.
def _stage2_cap() -> int:
    root = Path(__file__).resolve().parent.parent
    try:
        return int(json.loads((root / "data" / "02_fulltext" / "manifest.json")
                              .read_text())["max_chars"])
    except (OSError, ValueError, KeyError):
        pass
    # Config, never the environment. QUARTET_MAX_CHARS is deliberately not consulted:
    # it is per-terminal, so it made the prompt length depend on which shell ran which
    # stage. Stage 2 ignores it too.
    try:
        import yaml
        return int(yaml.safe_load(
            open(root / "config" / "domains.yaml"))["fulltext"]["max_chars"])
    except (OSError, KeyError, TypeError, ValueError, ImportError):
        return 150_000


MAX_PROMPT_CHARS = _stage2_cap()
truncated_papers = 0

# DeepSeek defaults reasoning_effort to "high", which is slower still. We use "medium":
# measured at 113-230s and 4.9k-8.5k output tokens per decomposition, against 95-167s
# at "low". Low bought roughly 20% speed, not enough to risk extraction quality on the
# one call that actually reads the paper. Override with DEEPSEEK_EFFORT if needed.
EFFORT = os.environ.get("DEEPSEEK_EFFORT", "medium")
# The gate call is cheap (~900 token input) but must actually deliberate.
EFFORT_CHECK = os.environ.get("DEEPSEEK_EFFORT_CHECK", "low")

# Ceilings, so one rambling response cannot stall a worker.
#
# CAREFUL: reasoning tokens count toward max_tokens. A cap of 6,000 truncated a
# decomposition mid-JSON (out was exactly 6,000), which made the response unparseable
# and forced a full 167s retry -- the cap cost more than it saved. The ceiling must
# leave room for reasoning AND the ~1.5k tokens of actual content on top.
MAX_OUT_DECOMPOSE = int(os.environ.get("DEEPSEEK_MAX_OUT", "16000"))
# The gate reasons before answering, and reasoning counts toward max_tokens.
# 4,000 truncated it, which parsed to {} and was then read as a real verdict.
MAX_OUT_CHECK = int(os.environ.get("DEEPSEEK_MAX_OUT_CHECK", "8000"))

# A hung request should surface in minutes, not tie up a worker for the better part of
# an hour. 600s x 5 retries was 50 minutes of silence per paper.
TIMEOUT = int(os.environ.get("DEEPSEEK_TIMEOUT", "180"))

JSON_BLOCK = re.compile(r"\{.*\}", re.S)


# =========================================================================== prompts
DECOMPOSE = """You are analysing a scientific paper to recover how its central hypothesis was formed.

The premise: a research hypothesis h is formed by combining a research background b with one or more inspirations i drawn from other work. Your job is to recover b and i from the paper itself.

WHAT AN INSPIRATION IS. A paper serves as an inspiration when it supplies something that solves or alleviates a specific LIMITATION of the previous methods for this research question. The previous methods define the problem; the inspiration supplies the missing piece; combining the two gives the hypothesis.

A worked example:
  - Research question: how can the parameters of a multi-layer logistic regression be improved automatically from data?
  - Previous methods: can run the network forward, but cannot update its parameters to learn.
  - Inspiration: the chain rule, from mathematics.
  - Hypothesis: backpropagation.

BACKGROUND AND INSPIRATION ARE DIFFERENT ROLES, and no cited work can fill both.

  BACKGROUND is prior work aimed at THIS SAME QUESTION, and how it fell short. It is the state of the field the authors were dissatisfied with. A paper the authors improve on, extend, or compete with belongs here.

  INSPIRATION is the knowledge they brought IN to overcome one of those shortfalls. It is not part of the problem statement; it is what they added to it.

The test is about ROLE, not distance: does this citation state part of the problem, or supply part of the solution?
  - A paper proposing a better optimiser for the same task, which the authors improve on -> BACKGROUND
  - A technique from an adjacent subfield the authors adopt to fix a stated limitation -> INSPIRATION
  - The framework the authors extend -> BACKGROUND
  - A mechanism borrowed from a different field -> INSPIRATION

If a work belongs in the background, do NOT also list it as an inspiration. Choose one.

PAPER TEXT
{text}

REFERENCE TITLES (the inspirations MUST come from this list)
{refs}

FIRST, DECIDE WHETHER THIS PAPER IS ELIGIBLE AT ALL.

The decomposition only makes sense for a paper that ADVANCES a novel claim of its own. A review, survey, tutorial, primer or position paper reports what other people found; it has no hypothesis to decompose, and its reference list is a bibliography of a field rather than a set of inspirations. Forced to decompose one, you would invent a hypothesis and attach it to whichever references looked plausible, which is worse than returning nothing.

Judge by what the paper DOES, not by what its title contains. The word "survey" or "review" in a title proves nothing either way:
  - "The Gaia-ESO Survey: mapping the radial abundance gradient" reports new observations. ELIGIBLE.
  - "A survey of professionals in formal methods" collects new questionnaire data. ELIGIBLE.
  - "Recent advances in X" that only summarises other people's results. NOT eligible.
  - A paper whose contribution is a taxonomy or comparison of existing methods, with no new claim. NOT eligible.

Ask: did the authors produce a new result, method, theory, measurement or dataset that did not exist before this paper? If yes, continue. If the paper's contribution is to organise and describe work already published elsewhere, return exactly:

  {{"eligible": false, "ineligible_reason": "<one sentence saying what the paper does instead>"}}

and nothing else. Do not produce a decomposition in that case.

If the paper IS eligible, set "eligible": true and produce these components.

1. research_question: the specific question this paper set out to answer. One sentence, phrased as a question. State it as the authors would have posed it BEFORE they had the answer. It must not contain, name, or hint at the solution.

2. background_survey: a summary of the prior methods that already addressed this question, and their limitations. Drawn from the introduction and related work.

CRITICAL, and the most common way this goes wrong: the background survey must not mention, paraphrase, or gesture at any inspiration you list below, nor at the hypothesis. This includes naming the cited work in ANY form:
  - not its title
  - not its system, model or method name (e.g. writing "FunSearch" when the inspiration cites the paper titled "Mathematical discoveries from program search with large language models")
  - not its acronym, and not its authors
Describe prior approaches GENERICALLY, by what they do, never by what they are called. Write "evolutionary search driven by a language model" rather than naming a system.

A survey can also give an inspiration away WITHOUT naming it, by naming the technique and the field it comes from. This is just as damaging and is harder to notice. Two constructions do it, and both are BANNED:

  BANNED 1 -- the transfer announcement: "Although <technique> has been studied in <other field>, it has not been applied to <this problem>." This states the solution and calls it a gap. Example of the mistake: a survey ending "Although Zeno detection and regularization have been studied for timed and hybrid automata, these solutions have not been adapted to Event Calculus" while listing inspirations titled "Efficient Detection of Zeno Runs in Timed Automata" and "On the regularization of Zeno hybrid automata". Neither title is quoted, yet both are handed over.
    Write instead: "No systematic method exists to identify or prevent these non-terminating patterns in goal-directed reasoning."

  BANNED 2 -- the open-questions list: "Key questions remain: whether <idea A> helps, and whether <idea B> outperforms ...". Such a list is the inspirations restated as questions. Example of the mistake: "whether synthetic hard negatives can improve over mined ones, and whether merging domain experts can outperform a single model" alongside inspirations on synthesised hard negatives and on weight averaging.
    Write instead: "Which adaptation choices actually matter for this setting remains unquantified."

The rule behind both: state the gap as a DEFICIENCY OF THE PRIOR METHODS -- what they fail to achieve. Never as an ABSENCE OF THE SOLUTION -- which technique has not yet been tried here. The first describes the problem. The second describes the answer.

Before you finish, read the survey once more against your inspiration list. If a reader could identify any inspiration from the survey alone, rewrite it. Check the final sentence hardest: that is where the gap statement lands, and where the leak nearly always is.

3. fine_grained_hypothesis: the paper's central hypothesis, as a specific technical claim.

4. inspirations: a list, following the role distinction stated at the top. An inspiration is knowledge the authors brought in to overcome a limitation of the previous methods. It may come from any distance, near or far. Signalled by phrases like "motivated by", "inspired by", "we adapt", "following", "borrowing from". Each must correspond to a real entry in the reference list above.

Do not list a paper as an inspiration if you mentioned it, or the line of work it belongs to, in the background survey. Those are the same knowledge counted twice, and the decomposition is then incoherent: the background would already contain what the inspiration is meant to supply.

For each inspiration give:
  - insp: the transferable idea, one or two sentences. Describe the IDEA, not the paper.
  - fixes: the specific limitation of the previous methods that this inspiration overcomes, in one sentence.
  - insp_concise: a label of about four words.
  - supposed_title: the TITLE ONLY of the cited paper, taken from the reference list. Reference entries sometimes arrive with the authors, year or venue attached; strip all of that and give the title alone. Never return an author list, a year, or a venue name. If the entry reads "T. Karras, S. Laine, and T. Aila (2020) Analyzing and improving the image quality of StyleGAN", return "Analyzing and improving the image quality of StyleGAN". This title is looked up in an external database, so anything other than a clean title fails to resolve and the inspiration is lost.
  - relation: how that cited paper relates to this inspiration.
  - motivation: the Problem/Gap this inspiration resolves, in the form "Problem/Gap: ... "
  - delta: the increment this one inspiration contributes to the hypothesis, in exactly this form:

Inspiration: <insp_concise>
- Motivation (WHY): <why this direction was chosen>
- Mechanism (HOW IT WORKS): <why it can work>
- Methodology (HOW IT'S INTEGRATED): <how it is implemented>

Rules:
  - Between 1 and 4 inspirations. Most papers have 2 or 3.
  - Every inspiration must be genuinely necessary. If the hypothesis survives without it, leave it out.
  - Inspirations must be mutually distinct, not restatements of each other.
  - Do NOT invent a reference. If you cannot ground an inspiration in the list, omit it.

Return ONLY a JSON object with keys: eligible, research_question, background_survey, fine_grained_hypothesis, inspirations. If eligible is false, return only eligible and ineligible_reason.
"""

# --- the ablation ----------------------------------------------------------------
# The main DECOMPOSE prompt is deliberately distance-neutral: it selects on ROLE, never
# on how far an inspiration travelled. That is not fastidiousness. The cross-domain rate
# is the quantity this benchmark exists to measure, so a prompt that preferred distant
# inspirations would manufacture its own result.
#
# This second pass is the exception, and it is kept strictly separate for that reason.
#
# HYPOTHESIS: the standard prompt has a SAME-DOMAIN BIAS -- not because it is told to
# prefer near inspirations, but because topically adjacent citations read as obviously
# relevant while distant ones look like background noise. If so, cited work from far
# fields gets overlooked even when it is what enabled the hypothesis.
#
# If true, every benchmark in this lineage (MOOSE-Chem, ResearchBench, TOMATO-Star)
# systematically under-counts cross-domain inspirations, and their reported
# cross-domain rates are floors rather than estimates. That is a finding, and the
# second pass below is the fix.
#
# Run with --probe-cross to execute both passes and report the difference.
PROBE_CROSS = """You previously identified the inspirations for this paper. You are now looking for ones you may have missed.

RESEARCH QUESTION: {rq}
HYPOTHESIS: {hyp}

ALREADY FOUND:
{found}

REFERENCE TITLES
{refs}

Researchers frequently import an idea from a field far from their own: a method from statistics, a mechanism from biology, a formalism from physics. These borrowings are easy to overlook, because the cited paper looks unrelated to the topic on its surface while supplying the key move.

Re-read the reference list looking ONLY for such cases. For each, the cited paper should:
  - come from a visibly different research area than this paper, and
  - supply an idea genuinely load-bearing for the hypothesis, not merely background or a tool.

Apply the same standard as before. Do not lower the bar to find something: returning an empty list is a valid and common answer. Do not repeat anything already found.

For each additional inspiration give these fields. The definitions are the same ones the first pass used; listing the field names alone made 35% of returned items paste a raw reference string into `insp` instead of stating an idea:
  - insp: the transferable idea, one or two sentences. Describe the IDEA, not the paper. NEVER a citation, a BibTeX key, a "[12]" marker, an author list or a URL.
  - insp_concise: a label of about four words.
  - supposed_title: the TITLE ONLY of the cited paper, taken from the reference list. Strip any authors, year, venue or DOI. It must name the SAME work that `insp` describes.
  - relation: how that cited paper relates to this inspiration. State what the paper does, from the reference list and your own knowledge of it. If you cannot say without guessing, omit the inspiration rather than writing "likely" or "presumably".
  - motivation: the Problem/Gap this inspiration resolves, in the form "Problem/Gap: ... "
  - why_missed: one sentence on why a topic-focused reading would skip it.

Return ONLY JSON: {{"additional": [...]}}
"""

# --- the CONTROL arm for the probe above -------------------------------------------
# PROBE_CROSS on its own proves nothing. It is instructed to look for distant work, so
# whatever it returns is the prompt obeying its instruction, not evidence that the
# standard pass missed anything. The interpretable quantity is the DIFFERENCE between a
# second look that mentions distance and a second look that does not.
#
# Everything below is word-for-word PROBE_CROSS except the paragraph naming distance,
# which is replaced by a distance-neutral one of comparable length and force. Same
# inputs, same standard, same "empty is a valid answer" release valve, same output
# fields. If the two arms return similar counts, the standard pass was not biased and
# the probe was simply following orders; if the distance-targeted arm returns
# substantially more, the first pass has a blind spot worth reporting.
PROBE_NEUTRAL = """You previously identified the inspirations for this paper. You are now looking for ones you may have missed.

RESEARCH QUESTION: {rq}
HYPOTHESIS: {hyp}

ALREADY FOUND:
{found}

REFERENCE TITLES
{refs}

A first reading of a paper frequently overlooks an inspiration. The borrowed idea may be stated once in passing, folded into a methods paragraph, or credited so briefly that its role in the hypothesis is easy to miss on a single pass.

Re-read the reference list looking ONLY for such cases. For each, the cited paper should:
  - supply an idea genuinely load-bearing for the hypothesis, not merely background or a tool.

Apply the same standard as before. Do not lower the bar to find something: returning an empty list is a valid and common answer. Do not repeat anything already found.

For each additional inspiration give these fields. The definitions are the same ones the first pass used; listing the field names alone made returned items paste a raw reference string into `insp` instead of stating an idea:
  - insp: the transferable idea, one or two sentences. Describe the IDEA, not the paper. NEVER a citation, a BibTeX key, a "[12]" marker, an author list or a URL.
  - insp_concise: a label of about four words.
  - supposed_title: the TITLE ONLY of the cited paper, taken from the reference list. Strip any authors, year, venue or DOI. It must name the SAME work that `insp` describes.
  - relation: how that cited paper relates to this inspiration. State what the paper does, from the reference list and your own knowledge of it. If you cannot say without guessing, omit the inspiration rather than writing "likely" or "presumably".
  - motivation: the Problem/Gap this inspiration resolves, in the form "Problem/Gap: ... "
  - why_missed: one sentence on why a first reading would skip it.

Return ONLY JSON: {{"additional": [...]}}
"""


# All three gates take identical inputs and none needs another's verdict, so running
# them as separate calls bought nothing but three round trips per paper. Merged, a
# paper costs 2 calls instead of 5.
CHECKS = """Apply four independent quality checks to this decomposition. Answer all four.

RESEARCH QUESTION: {rq}
BACKGROUND SURVEY: {bg}
HYPOTHESIS: {hyp}
INSPIRATIONS:
{insps}

CHECK 1 -- INFORMATION NECESSITY (per inspiration).
Each inspiration must supply essential, complementary knowledge required to derive the hypothesis from the background. An inspiration is necessary when removing it makes the hypothesis underivable from what remains. It is NOT necessary when it merely elaborates, or is already implied by the background.

CHECK 2 -- INFORMATION SUFFICIENCY (overall).
The background and inspirations together must logically entail the hypothesis. A competent researcher given only these could reach it, with no further outside knowledge. If something load-bearing is missing, say what.

CHECK 3 -- INFORMATION DISJOINTNESS (overall).
The background must remain strictly independent, leaking no information present in either the inspirations or the hypothesis.

Check TWO kinds of leak, and report either as a failure.

(a) IDEA LEAK. The background mentions, paraphrases or strongly hints at an inspiration's idea, or at the hypothesis. Vocabulary distinctive to an inspiration appearing in the background counts.

(b) NAME LEAK. The background lets a reader IDENTIFY a specific cited work. Each inspiration above shows its source in "(from: ...)".

The test is identification, not word overlap. Some shared vocabulary is unavoidable and is NOT a leak: a paper about the event calculus must say "event calculus" in its background, and a paper about learning noise must say "noise". Naming the FIELD, the FORMALISM or the general class of prior method is fine and necessary.

It IS a leak when the background:
  - contains the cited paper's title, or a distinctive phrase from it
  - names the system, model, method or formalism THAT PAPER INTRODUCED, as a named thing ("the hybrid temporal situation calculus", "FunSearch")
  - gives its acronym or authors
  - names an inspiration's TECHNIQUE together with the FIELD it comes from, even without naming the work. "Although Zeno detection and regularization have been studied for timed and hybrid automata, these solutions have not been adapted here" identifies inspirations titled "Efficient Detection of Zeno Runs in Timed Automata" and "On the regularization of Zeno hybrid automata". Technique plus home field is an address; a reader can look the paper up from it.
  - ends with a list of open questions that restates the inspirations. "Key questions remain: whether synthetic hard negatives improve over mined ones, and whether merging domain experts beats a single model" hands over inspirations on synthesised hard negatives and on weight averaging.
Use your own knowledge: a background naming "FunSearch" leaks an inspiration citing "Mathematical discoveries from program search with large language models", because they are the same work, even though no words are shared.

Read the background's LAST sentence with particular care. The gap statement lands there, and it is where both of the constructions above appear. A gap stated as a DEFICIENCY OF PRIOR METHODS is fine. A gap stated as an ABSENCE OF SOME PARTICULAR SOLUTION is a leak, because it names the answer.

Ask yourself: given only this background, could a reader pick that one paper out of a large literature? If yes, it leaks. If the background merely places the work in the same subject area, it does not.

A name leak usually means the work was MISCLASSIFIED rather than merely mentioned. If a citation appears in the background as prior work on this question, it is background, not an inspiration: an inspiration must come from outside the question's own literature. Say so in "leak" so the retry drops it from the inspiration list rather than just rewording the survey.

This check is the important one: the background becomes a retrieval query and the inspirations become the documents to retrieve. A leak makes retrieval trivially easy and invalidates the benchmark. When a leak is found, quote the offending phrase in "leak".

CHECK 4 -- NON-REDUNDANCY (pairwise).
The inspirations must be mutually distinct and non-repetitive. Two inspirations are redundant when they express the same underlying idea in different words, or when one subsumes the other, even if each is individually necessary. Judge the IDEAS, not the source papers: two different cited papers can carry the same inspiration.
This is separate from necessity: a pair can each contribute something and still substantially overlap. It matters here because redundant inspirations become near-duplicate gold documents, which inflates retrieval scores.

Return ONLY JSON:
{{"necessity": [{{"index": 0, "necessary": true, "reason": "..."}}, ...],
  "sufficient": true, "missing": "",
  "disjoint": true, "leak": "",
  "non_redundant": true, "duplicate_pair": [], "duplicate_reason": ""}}
"""


# =========================================================================== client
# DeepSeek per-million-token prices, for turning measured usage into a real cost
# rather than my estimate. Peak hours (09:00-12:00 and 14:00-18:00 Beijing) cost 2x.
PRICES = {
    "deepseek-v4-pro":   {"in": 0.435, "cached": 0.003625, "out": 0.87},
    "deepseek-v4-flash": {"in": 0.14,  "cached": 0.0028,   "out": 0.28},
}


def log(msg: str) -> None:
    """tqdm-safe: writing with print would shred the progress bar."""
    tqdm.write(msg)


class PermanentRefusal(RuntimeError):
    """The API rejected the request itself, so resending it cannot help.

    Deliberately NOT a requests.RequestException: the whole point is that it
    escapes the retry loop, which catches that base class. The caller drops the
    paper, which is the same outcome as before but reached in one call instead
    of five.
    """


class DeepSeek:
    def __init__(self, key: str | None, dry_run: bool = False, verbose: bool = False):
        self.key, self.dry, self.verbose = key, dry_run, verbose
        # One Session per thread: requests.Session is not thread-safe, and sharing one
        # across workers silently corrupts connection state under load.
        self._local = threading.local()
        self._lock = threading.Lock()
        self.calls = 0
        self.tok_in = self.tok_cached = self.tok_out = 0
        # Reasoning is billed as output but is not content, so it is the only part
        # of the bill `reasoning_effort` actually moves. Totalled here so the
        # question "would effort=low be cheaper" is answerable from a finished run
        # instead of estimated: the answer is at most this many tokens' worth.
        self.tok_reasoning = 0
        self.truncated = 0
        self.cost = 0.0
        self.slowest = 0.0

    @property
    def session(self) -> requests.Session:
        s = getattr(self._local, "session", None)
        if s is None:
            s = requests.Session()
            self._local.session = s
        return s

    def _record(self, model: str, usage: dict, elapsed: float, tag: str) -> None:
        p = PRICES.get(model, PRICES["deepseek-v4-flash"])
        cached = usage.get("prompt_cache_hit_tokens", 0)
        fresh = usage.get("prompt_tokens", 0) - cached
        out = usage.get("completion_tokens", 0)
        reasoning = usage.get("reasoning_tokens", 0)
        cost = (fresh * p["in"] + cached * p["cached"] + out * p["out"]) / 1e6
        with self._lock:
            self.tok_in += fresh
            self.tok_cached += cached
            self.tok_out += out
            self.tok_reasoning += reasoning
            self.cost += cost
            self.slowest = max(self.slowest, elapsed)
        if self.verbose:
            log(f"    {tag:<12} {model:<18} {elapsed:6.1f}s  "
                f"in {fresh:,}(+{cached:,} cached)  out {out:,}"
                + (f" (reasoning {reasoning:,})" if reasoning else "")
                + f"  ${cost:.4f}")

    def __call__(self, prompt: str, model: str, thinking: bool = False,
                 tag: str = "call", *, reasoning: str | None = None) -> str:
        """reasoning="off" genuinely disables reasoning, which no other setting does.

        thinking=False does NOT turn reasoning off. It selects EFFORT_CHECK and a
        lower ceiling, and when DEEPSEEK_EFFORT is already "low" the two branches
        are identical apart from max_tokens. Measured: a call emitting 8,704
        output tokens at effort=low, for ~600 tokens of content.

        Only callers whose output is transcription rather than judgement should
        pass it. This method's default is None, so nothing in this file changes
        behaviour; it exists so the other arm does not have to reimplement the
        client to get at one field of the request body.
        """
        with self._lock:
            self.calls += 1
        if self.dry:
            print(f"\n{'=' * 70}\n[dry-run] {model} thinking={thinking}\n{'=' * 70}")
            print(prompt[:2500])
            return "{}"
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
        }
        # Thinking mode ignores temperature/top_p/penalties; sending them is a silent
        # no-op, so they are omitted rather than implying they did something.
        body["thinking"] = {"type": "enabled"}
        if reasoning == "off":
            body["thinking"] = {"type": "disabled"}
            # The ceiling stays at the HIGH value on purpose. Nothing here is paid
            # for tokens that are not generated, and if the API ever ignored the
            # disable flag, an 8,000 cap would truncate a call that used to fit.
            body["max_tokens"] = MAX_OUT_DECOMPOSE
        elif thinking:
            # Reasoning tokens are billed as output AND dominate latency: measured at
            # effort=medium, decomposition emitted 4,877-8,517 output tokens and took
            # 113-230s. Effort is the single biggest lever on both.
            body["reasoning_effort"] = EFFORT
            body["max_tokens"] = MAX_OUT_DECOMPOSE
        else:
            # The gates need SOME reasoning. Disabling thinking outright made them
            # rubber-stamp: 90/90 papers passed all four checks on round 1, while an
            # independent audit found 3 backgrounds naming their own cited paper. A
            # gate that never fires is worse than no gate, because it looks like
            # quality control. "low" is enough to judge four yes/no questions and the
            # input is only ~900 tokens, so the cost is negligible.
            body["reasoning_effort"] = EFFORT_CHECK
            body["max_tokens"] = MAX_OUT_CHECK
        for attempt in range(5):
            t0 = time.time()
            try:
                r = self.session.post(
                    API, json=body, timeout=TIMEOUT,
                    headers={"Authorization": f"Bearer {self.key}",
                             "Content-Type": "application/json"},
                )
                if r.status_code in (429, 500, 502, 503):
                    log(f"    HTTP {r.status_code} on {tag}, backing off "
                        f"{2 ** attempt}s (attempt {attempt + 1}/5)")
                    time.sleep(2 ** attempt)
                    continue
                if r.status_code != 200:
                    log(f"    HTTP {r.status_code} on {tag}: {r.text[:300]}")
                    # A 4xx that is not a 429 is the server saying the REQUEST is
                    # wrong, not that it is busy. Retrying sends the identical
                    # body and gets the identical refusal. DeepSeek's
                    # "Content Exists Risk" filter is the one that fires in
                    # practice, on clinical and genetics abstracts, and each hit
                    # used to burn 5 calls and 15s of backoff before failing the
                    # paper anyway: raise_for_status throws HTTPError, which is a
                    # RequestException, which the retry loop below catches.
                    if 400 <= r.status_code < 500 and r.status_code != 429:
                        raise PermanentRefusal(
                            f"HTTP {r.status_code} on {tag}: {r.text[:200]}")
                    r.raise_for_status()
                payload = r.json()
                self._record(model, payload.get("usage") or {}, time.time() - t0, tag)
                choice = payload["choices"][0]
                # Truncation is silent otherwise: the JSON simply fails to parse and
                # the caller retries the whole expensive call without knowing why.
                if choice.get("finish_reason") == "length":
                    with self._lock:
                        self.truncated += 1
                    log(f"    !! {tag} hit the max_tokens ceiling and was TRUNCATED. "
                        f"Raise DEEPSEEK_MAX_OUT or lower DEEPSEEK_EFFORT.")
                return choice["message"]["content"]
            except requests.RequestException as exc:
                log(f"    {tag} failed after {time.time()-t0:.0f}s: "
                    f"{exc.__class__.__name__} (attempt {attempt + 1}/5)")
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
        return "{}"


def as_json(text: str) -> dict:
    """Models occasionally wrap JSON in prose or a fence despite json_object mode."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = JSON_BLOCK.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {}


_LEAK_STOP = {"the", "a", "an", "of", "in", "on", "for", "and", "to", "with", "by",
              "from", "at", "as", "is", "are", "via", "using", "this", "that", "based",
              "approach", "method", "methods", "towards", "toward", "novel", "new"}
_LEAK_WORD = re.compile(r"[a-z0-9]+")


def title_leak(title: str, background: str) -> float:
    """Fraction of a cited title's content words already present in the background.

    A DETERMINISTIC backstop for the disjointness gate, recorded per inspiration and
    never acted on here. Stage 5 decides whether to exclude a pair; recording it keeps
    the decision auditable and keeps this stage from silently shrinking the corpus.

    Deliberately NOT fed back into the decomposition prompt. The background must be
    free to name its own subject -- a paper on the event calculus has to say "event
    calculus" -- so pushing the extractor away from shared vocabulary would distort the
    text rather than fix the leak. Strictness belongs at the filtering step, where it
    cannot bias what gets extracted.
    """
    def toks(s):
        return {w for w in _LEAK_WORD.findall((s or "").lower())
                if w not in _LEAK_STOP and len(w) > 3}
    t = toks(title)
    return len(t & toks(background)) / len(t) if t else 0.0


def as_delta_block(delta, label: str = "") -> str:
    """Normalise a delta into TOMATO-Star's text block.

    The prompt asks for a formatted string, but the model sometimes obliges the JSON
    mode instead and returns {"Motivation": ..., "Mechanism": ..., "Methodology": ...}.
    Both are valid readings of the instruction, so accept either rather than losing a
    paper that has already been paid for.
    """
    if isinstance(delta, str):
        return delta.strip()
    if isinstance(delta, dict):
        # Keys vary: "Motivation", "motivation", "Motivation (WHY)". Match on prefix.
        def pick(*names):
            for k, v in delta.items():
                kl = str(k).lower()
                if any(kl.startswith(n) for n in names):
                    return str(v).strip()
            return ""
        return "\n".join([
            f"Inspiration: {label}",
            f"- Motivation (WHY): {pick('motivation', 'why')}",
            f"- Mechanism (HOW IT WORKS): {pick('mechanism')}",
            f"- Methodology (HOW IT'S INTEGRATED): {pick('methodology', 'method')}",
        ])
    if isinstance(delta, list):
        return "\n".join(as_delta_block(d, label) for d in delta)
    return ""


# An `insp` that is really a citation. The probe prompts originally listed field names
# without definitions, so the model pasted the reference string it was scanning:
# "Farchi2019", "[13]", "J. Hintikka, Knowledge and belief... https://ezp.lib...".
# The prompts now define the field, but a structural guard survives prompt drift and
# costs nothing.
RE_IS_CITATION = re.compile(
    r"^\s*\[\d+\]"                                   # "[13]"
    r"|^\s*[A-Z][a-z]+\d{4}\s*$"                     # "Farchi2019"
    r"|^\s*[A-Z]\.\s?[A-Z]?\.?\s?[A-Z][a-z]+[^.]{0,60}?(?:,\s|\sand\s)"  # "J. Hintikka, ..."
    # "Newton, M. A. and A. E. Raftery (1994)." -- Surname, Initials, then anything.
    # An earlier version required "(" or "," right after the initials and so missed the
    # "and <second author>" form, which is how most two-author citations are written.
    r"|^\s*[A-Z][a-z]+,\s+[A-Z]\."
    r"|https?://|arXiv preprint|\bser\.\s|\bpp\.\s\d+")


def clean_probe_items(items: list[dict]) -> tuple[list[dict], int]:
    """Drop probe results whose `insp` is a citation rather than an idea.

    Returns (kept, n_dropped). Dropping beats repairing: an item whose idea field was
    never written cannot be recovered from the citation string, and a malformed entry
    that reaches Stage 4 becomes a gold document for an inspiration nobody stated.
    """
    kept, dropped = [], 0
    for i in items:
        idea = i.get("insp")
        if isinstance(idea, list):
            idea = " ".join(str(x) for x in idea)
        idea = str(idea or "")
        if not idea.strip() or RE_IS_CITATION.search(idea):
            dropped += 1
            continue
        kept.append(i)
    return kept, dropped


def fmt(insps: list[dict]) -> str:
    return "\n".join(
        f"[{i}] {x.get('insp_concise', '')}: {x.get('insp', '')}"
        f"  (from: {x.get('supposed_title', '')})"
        for i, x in enumerate(insps)
    )


# =========================================================================== pipeline
def insp_key(x: dict) -> str:
    """Identity of an inspiration across independent draws: its cited title.

    Normalised, because the same reference comes back with different capitalisation
    and trailing punctuation between runs. Two draws naming the same paper must
    collapse to one gold, or the closure double-counts and the draw frequency is wrong.
    """
    t = x.get("supposed_title")
    if isinstance(t, list):
        t = " ".join(str(y) for y in t)
    return re.sub(r"[^a-z0-9 ]", "", str(t or "").lower()).strip()[:60]


def closure(draws: list[dict]) -> dict:
    """Merge k independent decompositions into the inspiration closure I*.

    THE THEORY. MOOSE-Chem A.2 assumes a unique minimal inspiration set. Measured on
    two identical runs, the sets overlap at Jaccard 0.53, so the assumption is false:
    a paper admits MANY subsets that each pass all four gates, and a single run returns
    one of them. A gold set built from one run therefore omits roughly 28% of the
    citations that are valid inspirations in some admissible decomposition, and a
    retriever that finds one of those is scored wrong. The bias is in the evaluation,
    not the retriever.

    I* is the union of the admissible sets. It is unique even though no individual set
    is, which is the object the uniqueness assumption was reaching for.

    WHY UNION-THEN-GATE WOULD BE WRONG. Non-redundancy is a property of a SET: two
    inspirations can each be admissible in their own draw and be redundant with each
    other in the union. So admissibility is tested per draw -- which the gate loop
    already does -- and only the survivors are unioned. There is no second gating pass.

    The background and hypothesis are taken from the FIRST draw so the query stays a
    single coherent text; only the inspiration set is pooled. Each gold records how
    many draws found it, which is the confidence label the benchmark exposes.
    """
    base = draws[0]
    merged: dict[str, dict] = {}
    for d in draws:
        for x in d.get("inspirations", []):
            k = insp_key(x)
            if not k:
                continue
            if k in merged:
                merged[k]["_draws"] += 1
            else:
                merged[k] = {**x, "_draws": 1}
    # Most-corroborated first, so a truncated gold list keeps the surest items.
    ordered = sorted(merged.values(), key=lambda x: -x["_draws"])
    return {**base, "inspirations": ordered}


def decompose_paper(paper: dict, llm: DeepSeek, probe_cross: bool = False,
                    model: str = MODEL_DECOMPOSE) -> dict:
    """ResearchBench's loop: decompose, check necessary, check sufficient, retry.

    This is the BASELINE method: ResearchBench and MOOSE-Star as published. A
    failed gate re-runs DECOMPOSE in full, so the research question, background
    and hypothesis are regenerated along with the inspirations. The proposed
    method lives in 03_new_method.py and is a separate program, so neither can
    quietly change the other.
    """
    refs = list(paper.get("bibliography", {}).values())
    refs_block = "\n".join(f"- {r}" for r in refs[:200]) or "(reference list unavailable)"

    gates = {"rounds": 0, "empty": 0, "checks_unparsed": 0, "unverified": 0,
             "necessary_failed": 0, "sufficient_failed": 0,
             "disjoint_failed": 0, "redundant_failed": 0}
    feedback = ""
    result: dict = {}

    short = paper["doi"][-18:]

    for rnd in range(MAX_ROUNDS):
        gates["rounds"] = rnd + 1
        global truncated_papers
        body = paper["fulltext"]
        if len(body) > MAX_PROMPT_CHARS and rnd == 0:
            truncated_papers += 1

        prompt = DECOMPOSE.format(text=body[:MAX_PROMPT_CHARS], refs=refs_block)
        if feedback:
            prompt += f"\nA previous attempt failed for this reason. Fix it:\n{feedback}\n"
        tag = "decompose"
        if llm.verbose:
            log(f"  [{short}] round {rnd + 1}: {tag} "
                f"({len(paper['fulltext']):,} chars, {len(refs)} refs)")

        d = as_json(llm(prompt, model, thinking=True, tag=tag))

        # Reviews, surveys and tutorials have no hypothesis to decompose, and their
        # reference lists are bibliographies of a field rather than sets of
        # inspirations. Decomposing one manufactures a hypothesis and grounds it in
        # arbitrary citations, which becomes a gold document for a link that was never
        # drawn. The model decides this, not a keyword list: "survey" means a sky
        # survey in astronomy and a questionnaire study in software engineering, both
        # of which are primary research, so title matching drops real papers in some
        # domains while missing untitled reviews in others. Measured on biology, 24% of
        # abstracts announce a review while OpenAlex types every one of them "article".
        if d.get("eligible") is False:
            return {**paper, "decomposed": False, "ineligible": True,
                    "ineligible_reason": str(d.get("ineligible_reason", ""))[:300],
                    "gates": gates}

        insps = d.get("inspirations") or []
        if not insps:
            # Retry once: the first miss is often a formatting failure. A second empty
            # result means the paper genuinely has no groundable inspirations, and a
            # third v4-pro call would just pay again for the same answer. Papers of
            # this kind cannot yield a (query, gold) row, so drop them.
            gates["empty"] += 1
            if gates["empty"] >= 2:
                return {**paper, "decomposed": False, "no_inspirations": True, "gates": gates}
            feedback = ("No inspirations were grounded in the reference list. Look again, "
                        "and if the paper genuinely borrows nothing, return an empty list.")
            continue


        # --- All three gates in one call ---------------------------------------
        if llm.verbose:
            log(f"  [{short}] found {len(insps)} inspirations, running gates")
        checks_prompt = CHECKS.format(rq=d.get("research_question", ""),
                                      bg=d.get("background_survey", ""),
                                      hyp=d.get("fine_grained_hypothesis", ""),
                                      insps=fmt(insps))
        c = as_json(llm(checks_prompt, MODEL_CHECK, tag="checks"))
        if not c:
            # Empty means the response was truncated or unparseable, NOT that the
            # decomposition is bad. Treating it as a verdict costs a full
            # re-decomposition (~160s) for a transport problem. Retry the cheap call.
            gates["checks_unparsed"] += 1
            c = as_json(llm(checks_prompt, MODEL_CHECK, tag="checks-retry"))
        if not c:
            # Still nothing. Pass the paper rather than burn another decomposition,
            # and mark it so the audit can exclude unverified papers.
            log(f"  [{short}] checks unparseable twice; accepting UNVERIFIED")
            gates["unverified"] = 1
            result = d
            break
        # The necessity verdict must cover EVERY inspiration before it can be
        # read. It is a per-inspiration list, and the old test only asked whether
        # anything came back marked false. A reply judging index 0 of a
        # three-item set contains no false entries, so it passed, with two
        # inspirations never assessed: three students sit the exam, one hands in
        # a paper, the teacher finds no failing grades and passes all three.
        nec = c.get("necessity")
        if not isinstance(nec, list) or len(nec) != len(insps) or \
                {x.get("index") for x in nec if isinstance(x, dict)} != set(range(len(insps))):
            gates["checks_unparsed"] += 1
            if llm.verbose:
                log(f"  [{short}] GATE INCOMPLETE: {len(nec or [])} verdicts "
                    f"for {len(insps)} inspirations")
            feedback = (f"The necessity check returned a verdict for only "
                        f"{len(nec or [])} of {len(insps)} inspirations. Judge every "
                        f"one, and give each its own index from 0 to {len(insps) - 1}.")
            continue
        # `is not True` rather than a truthy test: the model returns JSON, and
        # bool("false") is True, so a truthy test reads a rejection as approval.
        unnecessary = [x for x in nec if x.get("necessary") is not True]
        # Checked in severity order so the retry gets the most actionable reason first.
        if unnecessary:
            gates["necessary_failed"] += 1
            if llm.verbose:
                log(f"  [{short}] GATE FAIL necessity")
            feedback = ("These inspirations were judged unnecessary or redundant: "
                        + "; ".join(f"{u.get('index')}: {u.get('reason', '')}" for u in unnecessary))
            continue
        if c.get("sufficient") is not True:
            gates["sufficient_failed"] += 1
            if llm.verbose:
                log(f"  [{short}] GATE FAIL sufficiency")
            feedback = f"The set was insufficient. Missing: {c.get('missing', '')}"
            continue
        if c.get("disjoint") is not True:
            gates["disjoint_failed"] += 1
            if llm.verbose:
                log(f"  [{short}] GATE FAIL disjointness -- background leaked the answer")
            feedback = (f"The background leaked the answer: {c.get('leak', '')}. "
                        f"Rewrite it stating only the problem and prior work.")
            continue
        if c.get("non_redundant") is not True:
            gates["redundant_failed"] += 1
            if llm.verbose:
                log(f"  [{short}] GATE FAIL non-redundancy")
            pair = c.get("duplicate_pair") or []
            feedback = (f"Inspirations {pair} express the same idea: "
                        f"{c.get('duplicate_reason', '')}. Keep one, or replace the other "
                        f"with a genuinely distinct inspiration from the reference list.")
            continue

        result = d
        break

    if not result:
        return {**paper, "decomposed": False, "gates": gates}

    # --- Ablation: two second passes, identical except for the distance framing ----
    n_pass1 = len(result["inspirations"])
    additional: list[dict] = []
    additional_neutral: list[dict] = []
    if probe_cross:
        args = dict(rq=result.get("research_question", ""),
                    hyp=result.get("fine_grained_hypothesis", ""),
                    found=fmt(result["inspirations"]), refs=refs_block)
        # Both arms run on every paper, so the comparison is within-paper and cannot be
        # confounded by which papers happened to land in which arm.
        p = as_json(llm(PROBE_CROSS.format(**args), model, thinking=True, tag="probe"))
        additional, gates["probe_malformed"] = clean_probe_items(p.get("additional") or [])
        q = as_json(llm(PROBE_NEUTRAL.format(**args), model, thinking=True, tag="probe_neutral"))
        additional_neutral, gates["probe_neutral_malformed"] = \
            clean_probe_items(q.get("additional") or [])
        # Kept separate from pass-1 inspirations, and from each other, so the three can
        # be compared. Merging them before measurement destroys the thing being measured.

        # Gate the probe results against the SAME bar pass 1 had to clear. Without this
        # the arms are not comparable: pass 1 is gated and the probes were raw, so part
        # of any gap between them is just the ungated arm keeping what a gate would have
        # rejected. Each extra is judged alongside the pass-1 set it is being added to,
        # because necessity and redundancy are properties of the SET, not of an item.
        for label, extras in (("probe", additional), ("probe_neutral", additional_neutral)):
            if not extras:
                continue
            combined = result["inspirations"] + extras
            base = len(result["inspirations"])
            cc = as_json(llm(CHECKS.format(rq=result.get("research_question", ""),
                                           bg=result.get("background_survey", ""),
                                           hyp=result.get("fine_grained_hypothesis", ""),
                                           insps=fmt(combined)),
                             MODEL_CHECK, tag=f"{label}-checks"))
            rejected = {n.get("index") for n in (cc.get("necessity") or [])
                        if not n.get("necessary", True)}
            rejected |= {i for i in (cc.get("duplicate_pair") or []) if i >= base}
            for n, x in enumerate(extras):
                x["survived_gates"] = (base + n) not in rejected
            gates[f"{label}_gate_rejected"] = sum(1 for x in extras
                                                  if not x["survived_gates"])

    # Delta blocks come back from the decomposition call itself. Asking for them
    # separately cost a whole round trip for information the model already had.
    deltas = [b for b in (as_delta_block(x.get("delta"), x.get("insp_concise", ""))
                          for x in result["inspirations"]) if b]

    return {
        "doi": paper["doi"],
        "source_id": paper["doi"].replace("/", "_"),
        "title": paper["title"],
        "abstract": paper["abstract"],
        "domain": paper["domain"],
        "split": paper["split"],
        "primary_field": paper["primary_field"],
        "publication_date": paper["publication_date"],
        "research_question": str(result.get("research_question", "") or ""),
        "background_survey": str(result.get("background_survey", "") or ""),
        "fine_grained_hypothesis": str(result.get("fine_grained_hypothesis", "") or ""),
        # TOMATO-Star's schema. found_title / found_abstract / found_doi /
        # match_quality / similarity are filled by Stage 4 (Semantic Scholar).
        # Every field is coerced to str: the model occasionally returns a nested object
        # where the schema asks for text, and one such case must not cost a paper.
        "inspiration": [{
            "insp": str(x.get("insp", "") or ""),
            "fixes": str(x.get("fixes", "") or ""),
            # Deterministic leak score: share of this cited title's content words
            # already in the background. Recorded, not enforced -- Stage 5 filters.
            "title_leak": round(title_leak(str(x.get("supposed_title", "") or ""),
                                           result.get("background_survey", "")), 3),
            "insp_concise": str(x.get("insp_concise", "") or ""),
            "supposed_title": str(x.get("supposed_title", "") or ""),
            "found_title": None, "found_abstract": None, "found_doi": None,
            "match_quality": None, "similarity": None,
            "relation": str(x.get("relation", "") or ""),
            "motivation": str(x.get("motivation", "") or ""),
        } for x in result["inspirations"]],
        "hypothesis_components": deltas,
        "detailed_breakdown": "\n\n".join(deltas),
        "decomposed": True,
        "gates": gates,
        # Ablation output. Not merged into `inspiration`: these are candidates the
        # standard pass missed, and Stage 4 must resolve and domain-label them before
        # any claim is made about whether they are genuinely cross-domain.
        "n_pass1": n_pass1,
        "probe_additional": additional,
        "probe_neutral_additional": additional_neutral,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--dry-run", action="store_true", help="print prompts, call nothing")
    ap.add_argument("--probe-cross", action="store_true",
                    help="second pass targeting distant-field references; tests whether "
                         "the standard prompt has a same-domain bias")
    ap.add_argument("--samples", type=int, default=1, metavar="K",
                    help="independent decompositions per paper; the gold set becomes the "
                         "union of what survived the gates in each (the inspiration "
                         "closure). K=1 reproduces ResearchBench/MOOSE-Star. Use K>=2 for "
                         "the test set: two identical runs agree at Jaccard 0.53, so a "
                         "single draw omits ~28%% of the citations that are valid "
                         "inspirations in some admissible decomposition")
    ap.add_argument("--workers", type=int, default=WORKERS,
                    help=f"parallel papers (default {WORKERS}); lower this on 429s")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="log every API call: model, seconds, tokens, cost")
    ap.add_argument("--model", default=MODEL_DECOMPOSE,
                    help=f"decomposition model (default {MODEL_DECOMPOSE}). "
                         f"deepseek-v4-flash is 284B against v4-pro's 1.6T, so several "
                         f"times faster and 3x cheaper; run both on the same papers "
                         f"before deciding whether the quality holds")
    # Stage 2 rewrites data/02_fulltext/<domain>_<split>.jsonl in place, so a fetch
    # running in another terminal truncates the file this stage is reading: a probe run
    # started against 77 papers and got 5. Point --input at a frozen snapshot to make a
    # test immune to that, and --output somewhere separate so the test does not clobber
    # the real decompositions either.
    ap.add_argument("--input", type=Path,
                    help="read papers from this jsonl instead of data/02_fulltext/")
    ap.add_argument("--output", type=Path,
                    help="write decompositions here instead of data/03_decomposed/")
    ap.add_argument("--no-resume", action="store_true",
                    help="overwrite output instead of skipping already-decomposed DOIs")
    a = ap.parse_args()

    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key and not a.dry_run:
        sys.exit("DEEPSEEK_API_KEY is not set. Export it, or pass --dry-run to inspect prompts.")

    src = a.input or IN / f"{a.domain}_{a.split}.jsonl"
    if not src.exists():
        sys.exit(f"missing {src} -- run 02_fulltext.py first")
    papers = [json.loads(l) for l in open(src) if l.strip()]
    if a.limit:
        papers = papers[: a.limit]

    # Inspirations must be grounded in the reference list. A paper without one cannot
    # produce a valid decomposition, so paying to try is waste.
    no_refs = [p for p in papers if not p.get("bibliography")]
    if no_refs:
        print(f"[skip] {len(no_refs)} paper(s) have no reference list; not decomposing them")
        papers = [p for p in papers if p.get("bibliography")]

    llm = DeepSeek(key, a.dry_run, a.verbose)
    dst = a.output or OUT / f"{a.domain}_{a.split}.jsonl"
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[io] reading {src}  ({len(papers)} papers)\n[io] writing {dst}")
    # Printed at the START, not just in the closing cost block, because these are
    # the settings you decide on BEFORE spending. They were previously visible
    # only as `reasoning_effort=` on the last line of a finished run, and
    # max_tokens was not shown at all, so a run at the wrong effort could not be
    # spotted until it had already been paid for. Also makes the log self-describing:
    # comparing two runs' falsification rates is only meaningful if these match.
    print(f"[cfg] model {MODEL_DECOMPOSE} (gates {MODEL_CHECK})\n"
          f"[cfg] reasoning_effort {EFFORT} (gates {EFFORT_CHECK})   "
          f"max_tokens {MAX_OUT_DECOMPOSE:,} (gates {MAX_OUT_CHECK:,})\n"
          f"[cfg] timeout {TIMEOUT}s   prompt cap {MAX_PROMPT_CHARS:,} chars   "
          f"gate rounds {MAX_ROUNDS}")

    # Resume: this stage costs money, so a crash at paper 5,000 must not throw away
    # 5,000 papers' worth of API spend. Append, and skip DOIs already written.
    done: set[str] = set()
    if dst.exists() and not a.no_resume:
        with open(dst) as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["doi"])
                except (json.JSONDecodeError, KeyError):
                    continue
        if done:
            print(f"[resume] {len(done):,} papers already decomposed, skipping them")
    todo = [p for p in papers if p["doi"] not in done]
    if not todo:
        print("nothing left to do")
        return

    ok = dropped_empty = dropped_gates = dropped_review = 0
    ineligible_reasons: list[tuple[str, str, str]] = []
    gate_totals = {"necessary_failed": 0, "sufficient_failed": 0,
             "disjoint_failed": 0, "redundant_failed": 0,
             "checks_unparsed": 0, "unverified": 0}
    n_insp = n_extra = papers_with_extra = 0
    n_extra_neutral = papers_with_extra_neutral = 0
    n_raw = n_raw_neutral = 0
    probe_drops: dict[str, int] = {"probe_malformed": 0, "probe_neutral_malformed": 0,
                                   "probe_gate_rejected": 0,
                                   "probe_neutral_gate_rejected": 0}
    write_lock = threading.Lock()
    workers = 1 if a.dry_run else a.workers

    with open(dst, "a" if done else "w") as fh, \
            ThreadPoolExecutor(max_workers=workers) as pool:
        # K independent draws per paper, submitted flat so every draw of every paper
        # competes for the same worker pool. Submitting paper-by-paper would serialise
        # the draws behind each other for no reason: they share no state.
        n_draws = max(1, a.samples)
        futures = {pool.submit(decompose_paper, p, llm, a.probe_cross, a.model): p
                   for p in todo for _ in range(n_draws)}
        pending: dict[str, list[dict]] = {}
        bar = tqdm(as_completed(futures), total=len(todo) * n_draws,
                   desc=f"{a.domain}/{a.split}",
                   unit="draw" if n_draws > 1 else "paper", dynamic_ncols=True)
        for fut in bar:
            try:
                r = fut.result()
            except Exception as exc:                      # one bad paper must not kill the run
                bar.write(f"  !! {futures[fut]['doi']}: {exc.__class__.__name__}: {exc}")
                continue
            if n_draws > 1:
                # Hold each paper's draws until all K land, then merge. A draw that
                # failed (review, no inspirations, gates) still counts toward K, so a
                # paper cannot wait forever for a draw that will never succeed.
                doi = futures[fut]["doi"]
                pending.setdefault(doi, []).append(r)
                if len(pending[doi]) < n_draws:
                    continue
                got = [x for x in pending.pop(doi) if x.get("decomposed")]
                if not got:
                    r = r                                  # all draws failed; keep the last
                else:
                    merged = closure([{"inspirations": x["inspiration"]} for x in got])
                    r = {**got[0], "inspiration": merged["inspirations"],
                         "n_draws": n_draws, "n_draws_decomposed": len(got)}
            for k in gate_totals:
                gate_totals[k] += r.get("gates", {}).get(k, 0)
            if r.get("decomposed"):
                ok += 1
                n_insp += len(r["inspiration"])
                # Headline counts are GATED: an extra that a gate rejected is not a
                # finding, and counting it would let the ungated arm win on volume.
                surv = [x for x in (r.get("probe_additional") or [])
                        if x.get("survived_gates")]
                surv_n = [x for x in (r.get("probe_neutral_additional") or [])
                          if x.get("survived_gates")]
                extra = len(surv)
                n_extra_neutral += len(surv_n)
                papers_with_extra_neutral += bool(surv_n)
                n_raw += len(r.get("probe_additional") or [])
                n_raw_neutral += len(r.get("probe_neutral_additional") or [])
                for k in ("probe_malformed", "probe_neutral_malformed",
                          "probe_gate_rejected", "probe_neutral_gate_rejected"):
                    probe_drops[k] += (r.get("gates") or {}).get(k, 0)
                n_extra += extra
                papers_with_extra += bool(extra)
                with write_lock:
                    fh.write(json.dumps(r) + "\n")
                    fh.flush()                            # crash-safe: never buffer results
            elif r.get("ineligible"):
                dropped_review += 1                       # review/survey: no hypothesis
                ineligible_reasons.append((r["doi"], r["title"][:70],
                                           r.get("ineligible_reason", "")))
            elif r.get("no_inspirations"):
                dropped_empty += 1                        # nothing groundable in the refs
            else:
                dropped_gates += 1                        # survived extraction, failed a gate
            post = {"ok": ok, "insp": n_insp, "calls": llm.calls}
            if a.probe_cross:
                post["probe"] = f"+{n_extra}"
            bar.set_postfix(post)
        bar.close()
    papers = todo

    print(f"\n[{a.domain}/{a.split}] {ok}/{len(papers)} papers decomposed "
          f"({100*ok/max(len(papers),1):.0f}%)")
    print(f"  inspirations per decomposed paper: {n_insp/max(ok,1):.2f}"
          f"   (TOMATO-Star test: 1.89)")
    print(f"  dropped, no groundable inspirations: {dropped_empty}")
    print(f"  dropped, failed a quality gate after {MAX_ROUNDS} rounds: {dropped_gates}")
    # Printed with reasons rather than as a bare count: this filter removes source
    # papers, so it needs to be auditable for a false positive that discards real
    # research (a sky survey, a questionnaire study) rather than a literature review.
    print(f"  dropped, not primary research (review/survey/tutorial): {dropped_review}")
    for doi, title, why in ineligible_reasons[:10]:
        print(f"      {title}\n        -> {why[:110]}")
    if no_refs:
        print(f"  skipped before any spend, no reference list: {len(no_refs)}")
    print(f"  gate retries: {gate_totals}")
    if truncated_papers:
        print(f"  !! {truncated_papers} papers exceeded {MAX_PROMPT_CHARS:,} chars and "
              f"were truncated. Raise QUARTET_MAX_CHARS if that share is material.")
    # Measured cost, not my estimate. Use it to size the full build.
    n_done = max(ok + dropped_empty + dropped_gates, 1)
    print(f"\n  MEASURED COST")
    print(f"    tokens        {llm.tok_in:,} in  (+{llm.tok_cached:,} cached)  "
          f"{llm.tok_out:,} out")
    print(f"    total         ${llm.cost:.4f}   per paper ${llm.cost/n_done:.4f}")
    print(f"    slowest call  {llm.slowest:.1f}s   reasoning_effort={EFFORT}")
    # The effort question, answered from the invoice rather than estimated. Output
    # is billed at 2x input, and reasoning is the share of it that carries no
    # content, so this line is the ceiling on what lowering effort could ever save.
    if llm.tok_reasoning:
        p_out = PRICES.get(MODEL_DECOMPOSE, PRICES["deepseek-v4-flash"])["out"]
        r_cost = llm.tok_reasoning * p_out / 1e6
        print(f"    reasoning     {llm.tok_reasoning:,} tokens "
              f"({llm.tok_reasoning / max(llm.tok_out, 1) * 100:.0f}% of output)  "
              f"= ${r_cost:.4f}, {r_cost / max(llm.cost, 1e-9) * 100:.0f}% of the bill")
        print(f"                  that is the CEILING on what a lower "
              f"reasoning_effort could save, not the saving")
    if llm.truncated:
        # Distinct from the prompt-truncation warning above: that one loses input,
        # this one loses the reply and forces a retry, so it is billed twice.
        print(f"    !! truncated  {llm.truncated} call(s) hit max_tokens "
              f"({llm.truncated / max(llm.calls, 1) * 100:.2f}% of {llm.calls:,}). "
              f"Each was paid for AND retried.")
    print(f"    extrapolated  7,000 papers = ${llm.cost/n_done*7000:,.0f}   "
          f"35,000 = ${llm.cost/n_done*35000:,.0f}")
    if a.probe_cross:
        print(f"\n  === ABLATION: does the standard prompt miss distant-field inspirations? ===")
        print(f"  standard pass              {n_insp:,} inspirations (gated)")
        print(f"\n  {'arm':<26s}{'returned':>9s}{'malformed':>11s}{'gate-rej':>10s}"
              f"{'SURVIVED':>10s}")
        print(f"  {'B. neutral look-again':<26s}{n_raw_neutral:>9,}"
              f"{probe_drops['probe_neutral_malformed']:>11,}"
              f"{probe_drops['probe_neutral_gate_rejected']:>10,}"
              f"{n_extra_neutral:>10,}")
        print(f"  {'A. distance-targeted':<26s}{n_raw:>9,}"
              f"{probe_drops['probe_malformed']:>11,}"
              f"{probe_drops['probe_gate_rejected']:>10,}{n_extra:>10,}")
        print(f"\n  surviving extras as a share of the standard pass: "
              f"B {100*n_extra_neutral/max(n_insp,1):.0f}%   "
              f"A {100*n_extra/max(n_insp,1):.0f}%")
        print(f"  papers gaining at least one: B {papers_with_extra_neutral}/{ok}   "
              f"A {papers_with_extra}/{ok}")
        # The comparison, not either arm alone, is the finding. A second look always
        # turns something up, so the distance-targeted count is uninterpretable on its
        # own: it has to be read against a second look that never mentions distance.
        delta = n_extra - n_extra_neutral
        print(f"\n  A - B = {delta:+,}")
        if delta <= 0:
            print("  -> No evidence of a distance blind spot. The targeted probe found no"
                  " more\n     than a neutral second look, so its yield is re-reading, not"
                  " distance.")
        else:
            print(f"  -> The targeted arm found {delta} more than the neutral arm. That is"
                  f" the\n     candidate blind spot, and it is only a candidate: Stage 4"
                  f" must resolve and\n     domain-label both sets before any of it can be"
                  f" called cross-domain.")
        print(f"  NOT a result either way until Stage 4 labels the domains.")
    print(f"  API calls: {llm.calls}   -> {dst}")


if __name__ == "__main__":
    main()
