"""
Verify a decomposition run before spending on the full build.

Three questions, in order of how badly a failure would hurt:

  1. GROUNDING. Is every `supposed_title` actually in that paper's reference list?
     A fabricated citation cannot be resolved to a real document in Stage 4, so it
     silently becomes a missing gold. This is the failure that quietly ruins the corpus.

  2. LEAKAGE. Does the background survey give away its own inspirations? The background
     becomes the retrieval query and the inspirations become the documents to retrieve.
     If they share distinctive vocabulary, retrieval is trivially easy and the benchmark
     measures nothing. The disjointness gate is supposed to catch this; here we check
     whether it actually did.

  3. SHAPE. Inspirations per paper, delta formatting, question phrased before the answer.

Usage:
  python build/inspect_output.py --domain cs --split test
  python build/inspect_output.py --domain cs --split test --show 2
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import re
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEC = ROOT / "data" / "03_decomposed"
FULL = ROOT / "data" / "02_fulltext"

STOP = {"the", "a", "an", "of", "in", "on", "for", "and", "to", "with", "by", "from",
        "at", "as", "is", "are", "via", "using", "this", "that", "we", "our", "it",
        "be", "can", "which", "these", "their", "has", "have", "such", "not", "but"}
WORD = re.compile(r"[a-z0-9]+")
# Mid-sentence capitalised tokens and CamelCase: how system names appear.
PROPER = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-Z][A-Za-z0-9]*(?:-[A-Z][A-Za-z0-9]*)*\b", re.M)
COMMON = {"the", "this", "these", "however", "thus", "prior", "we", "our", "as", "in",
          "for", "such", "while", "although", "moreover", "furthermore", "llm", "llms",
          "ai", "it", "they", "there", "alternative", "additionally", "recent", "by",
          "a", "an", "when", "if", "since", "because", "first", "second", "finally"}

DELTA_SHAPE = re.compile(
    r"Inspiration:.*?- Motivation \(WHY\):.*?- Mechanism \(HOW IT WORKS\):"
    r".*?- Methodology \(HOW IT'S INTEGRATED\):", re.S)


def toks(text: str) -> set[str]:
    return {w for w in WORD.findall((text or "").lower()) if w not in STOP and len(w) > 3}


def build_idf(docs: list[str]) -> dict[str, float]:
    """Inverse document frequency over this run's backgrounds.

    Raw token containment made the name-leak check useless: "Large language diffusion
    models" scored 3/4 against a background saying "large language models" and was
    reported as a leak, when the one word that identifies the paper -- diffusion -- was
    absent. Words that appear in most backgrounds of a domain carry no identifying
    information, so they must not count towards identification.
    """
    df = collections.Counter()
    for d in docs:
        df.update(toks(d))
    n = max(len(docs), 1)
    return {w: math.log(n / c) for w, c in df.items()}


def weighted_containment(title: str, bg_tokens: set[str], idf: dict[str, float]) -> float:
    """Share of a title's IDENTIFYING mass that the background already contains.

    A token unseen in the run's backgrounds gets the maximum weight: it is maximally
    distinctive, so its absence from the background is exactly what we want to reward.
    """
    t = toks(title)
    if not t:
        return 0.0
    default = max(idf.values(), default=1.0)
    total = sum(idf.get(w, default) for w in t)
    if total <= 0:
        return 0.0
    return sum(idf.get(w, default) for w in t & bg_tokens) / total


# The two constructions that give an inspiration away without naming it. Surfaced for
# review rather than judged: only reading the inspiration list can settle whether the
# technique named is one of them.
RE_TRANSFER = re.compile(
    r"\b(although|while|whereas)\b[^.]{0,200}?\b(has|have)\s+been\s+(studied|explored|"
    r"used|applied|developed|proposed|investigated)\b[^.]{0,200}?\b(not|never|yet)\b",
    re.I | re.S)
RE_OPEN_Q = re.compile(
    r"\b(key |open |several |important )?questions?\s+remain[^.]{0,40}?\bwhether\b"
    r"|\bit remains (unclear|unknown|an open question) whether\b"
    r"|\bwhether\b[^.]{0,120}\band whether\b", re.I | re.S)


def best_match(title: str, refs: list[str]) -> float:
    """Highest similarity between a claimed title and any real reference.

    Uses token CONTAINMENT, not symmetric similarity. A reference entry is a full
    bibliography line -- authors, venue, year, pages -- so a short title buried in it
    scores badly on any symmetric measure. "Double Thompson Sampling for Dueling
    Bandits" scored 0.42 and was flagged as fabricated when it is a real NeurIPS paper;
    the mismatch was length, not content.

    Containment asks the right question: are the title's words present in the reference?
    """
    t = toks(title)
    if not t:
        return 0.0
    best = 0.0
    for r in refs:
        rt = toks(r)
        if not rt:
            continue
        contained = len(t & rt) / len(t)
        # Keep a sequence score as a floor so a title matching a short reference
        # entry word-for-word still scores well.
        seq = SequenceMatcher(None, " ".join(sorted(t)), " ".join(sorted(rt))).ratio()
        best = max(best, contained, seq)
        if best > 0.95:
            break
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--show", type=int, default=1, help="papers to print in full")
    ap.add_argument("--ground-threshold", type=float, default=0.55)
    a = ap.parse_args()

    dec = [json.loads(l) for l in open(DEC / f"{a.domain}_{a.split}.jsonl") if l.strip()]
    bibs = {}
    fp = FULL / f"{a.domain}_{a.split}.jsonl"
    if fp.exists():
        for line in open(fp):
            r = json.loads(line)
            bibs[r["doi"]] = list((r.get("bibliography") or {}).values())

    print(f"{len(dec)} decomposed papers\n")

    # ---- 1. grounding -------------------------------------------------------------
    scores, ungrounded = [], []
    for p in dec:
        refs = bibs.get(p["doi"], [])
        if not refs:
            continue
        for i in p["inspiration"]:
            s = best_match(i["supposed_title"], refs)
            scores.append(s)
            if s < a.ground_threshold:
                ungrounded.append((p["doi"], i["supposed_title"], s))
    if scores:
        ok = sum(1 for s in scores if s >= a.ground_threshold)
        print(f"1. GROUNDING  {ok}/{len(scores)} inspirations match a real reference "
              f"({100*ok/len(scores):.0f}%)")
        print(f"   mean similarity {sum(scores)/len(scores):.2f}")
        for doi, t, s in ungrounded[:5]:
            print(f"   !! {s:.2f}  {t[:78]}")
        if not ungrounded:
            print("   no fabricated citations detected")
    else:
        print("1. GROUNDING  cannot check: no bibliographies loaded")

    # ---- 2a. NAME leak: does the background name a cited paper? --------------------
    # The damaging case. A background naming "FunSearch" leaks an inspiration citing
    # "Mathematical discoveries from program search with large language models" -- same
    # work, no shared words, so word-overlap cannot see it. Two signals: direct title
    # overlap, and proper nouns in the background, which are the way system names get
    # in. Proper nouns are flagged for review, not judged: only the gate can decide
    # whether "FunSearch" is one of the cited papers.
    print()
    idf = build_idf([p["background_survey"] for p in dec])
    title_hits, propernouns, constructions = [], [], []
    for p in dec:
        bg = p["background_survey"]
        bgt = toks(bg)
        for i in p["inspiration"]:
            w = weighted_containment(i["supposed_title"], bgt, idf)
            if w > 0.6:
                title_hits.append((w, p["doi"], i["supposed_title"]))
        for m in PROPER.finditer(bg):
            w = m.group(0)
            if w.lower() not in COMMON:
                propernouns.append((p["doi"], w))
        # The gap statement lands in the last sentence, and that is where a survey
        # gives an inspiration away by naming its technique and home field rather
        # than by naming the work.
        for name, rx in (("transfer-announcement", RE_TRANSFER), ("open-questions", RE_OPEN_Q)):
            m = rx.search(bg)
            if m:
                constructions.append((name, p["doi"], m.group(0)[:150]))
    title_hits.sort(reverse=True)
    print(f"2a. NAME LEAK  {len(title_hits)} background(s) contain a cited paper's title")
    print("    (weighted by inverse document frequency: common words like 'models' do "
          "not count towards identification, distinctive ones do)")
    for w, doi, t in title_hits[:5]:
        print(f"    !! {w:.2f}  {t[:70]}")

    uniq = sorted({w for _, w in propernouns})
    print(f"    proper nouns in backgrounds ({len(uniq)}) -- review: any that name a "
          f"cited work is a leak the gate must catch")
    if uniq:
        print(f"      {', '.join(uniq[:18])}")

    print()
    print(f"2b. LEAKY CONSTRUCTIONS  {len(constructions)} background(s) state the gap as "
          f"an absence of a solution rather than a deficiency of prior methods")
    for name, doi, frag in constructions[:6]:
        print(f"    !! [{name}] {' '.join(frag.split())[:110]}")
    if not constructions:
        print("    none found")

    # ---- 2c. idea leak ------------------------------------------------------------
    print()
    leaks = []
    for p in dec:
        bg = toks(p["background_survey"])
        if not bg:
            continue
        for i in p["inspiration"]:
            it = toks(i["insp"])
            if not it:
                continue
            overlap = len(bg & it) / len(it)     # share of inspiration words in the bg
            leaks.append((overlap, p["doi"], i["insp_concise"], sorted(bg & it)[:8]))
    leaks.sort(reverse=True)
    if leaks:
        mean = sum(x[0] for x in leaks) / len(leaks)
        bad = [x for x in leaks if x[0] > 0.5]
        print(f"2c. IDEA LEAK  mean {100*mean:.0f}% of an inspiration's distinctive words "
              f"already appear in its background survey")
        print(f"   {len(bad)}/{len(leaks)} inspirations above 50% overlap")
        print("   worst cases:")
        for ov, doi, label, shared in leaks[:4]:
            print(f"     {100*ov:3.0f}%  {label[:34]:36s} shared: {', '.join(shared)}")
        print("   Lower is better. High overlap means the query already contains its "
              "answer.")

    # ---- 3. shape -----------------------------------------------------------------
    print()
    n = [len(p["inspiration"]) for p in dec]
    good_delta = sum(1 for p in dec if DELTA_SHAPE.search(p.get("detailed_breakdown", "")))
    empty_q = sum(1 for p in dec if not p.get("research_question"))
    print(f"3. SHAPE      inspirations/paper {sum(n)/max(len(n),1):.2f} "
          f"(TOMATO-Star 1.89)   range {min(n)}-{max(n)}")
    print(f"   delta blocks correctly formatted: {good_delta}/{len(dec)}")
    print(f"   empty research_question: {empty_q}")
    print(f"   mean background survey {sum(len(p['background_survey']) for p in dec)//max(len(dec),1)} chars")

    # ---- full dump ----------------------------------------------------------------
    for p in dec[: a.show]:
        print("\n" + "=" * 78)
        print("TITLE     ", p["title"][:100])
        print("\nQUESTION  ", p["research_question"])
        print("\nBACKGROUND\n", p["background_survey"][:900])
        print("\nHYPOTHESIS\n", p["fine_grained_hypothesis"][:500])
        print("\nINSPIRATIONS")
        for i in p["inspiration"]:
            print(f"  - {i['insp_concise']}")
            print(f"      idea : {i['insp'][:150]}")
            print(f"      cites: {i['supposed_title'][:90]}")
        if p.get("probe_additional"):
            print("\nPROBE FOUND (distant-field pass)")
            for i in p["probe_additional"]:
                print(f"  - {i.get('insp_concise', '')}")
                print(f"      cites : {str(i.get('supposed_title', ''))[:90]}")
                print(f"      missed: {str(i.get('why_missed', ''))[:150]}")
        print("\nDELTA\n", p["detailed_breakdown"][:600])


if __name__ == "__main__":
    main()
