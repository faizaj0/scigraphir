#!/usr/bin/env python3
"""Stage 3b -- uniqueness falsification by substitution sweep.

Reads a finished stage-3 file, holds each paper's background b and hypothesis h*
FIXED, and asks whether the extracted inspiration set I is the only one that
works. For each i_j in I: hide it, search the paper's remaining citations for a
replacement set I', and put I' through the SAME four quality checks that I had
to pass. If I' survives, i_j is replaceable and I' joins the family M(b,h*).

    |M| = 1   uniqueness NOT falsified (under this search budget)
    |M| > 1   uniqueness falsified

Nothing here regenerates b or h*. That is the whole point: every candidate set
is judged against the same fixed hypothesis, so a set that "works" cannot work
by having quietly moved the target.

Two controls run first, and BOTH are recorded rather than used as filters.

  r_empty   how often the model produces h* from b alone, with no inspirations.
            A high value does NOT automatically mean contamination: it can also
            mean the hypothesis is an unsurprising next step from the
            background. Both readings weaken a necessity claim, and which one
            applies is a judgement for analysis time, not for a hard filter
            here. Papers are never dropped on this score.

  decoy     the four checks run on |I| random cited papers that are NOT in I.
            These SHOULD fail. If they pass, the checker is rubber-stamping on
            this paper and any substitute it later accepts is worthless.

Reviews and surveys are not handled here; the eligibility check inside stage 3's
DECOMPOSE prompt has already dropped them.

Usage
-----
    # the cheap gate: controls only, no sweep
    python build/03b_uniqueness.py \
        --decomposed data/_snap/probe_cs_out2.jsonl \
        --fulltext   data/_snap/probe_cs.jsonl \
        --out        data/_snap/uniq_cs_controls.jsonl \
        --stage controls

    # the full run
    python build/03b_uniqueness.py \
        --decomposed data/_snap/probe_cs_out2.jsonl \
        --fulltext   data/_snap/probe_cs.jsonl \
        --out        data/_snap/uniq_cs.jsonl \
        --stage sweep
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import re
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent


def _load_stage3():
    """Import 03_decompose.py, whose name is not a legal module identifier.

    Everything reusable lives there: the DeepSeek client with its retry and cost
    accounting, the CHECKS prompt, and the JSON salvage. Reimplementing any of it
    here would let the two drift, and the whole argument rests on I' being judged
    by exactly the same checker that I passed.
    """
    path = Path(__file__).resolve().parent / "03_decompose.py"
    spec = importlib.util.spec_from_file_location("_stage3", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S3 = _load_stage3()

MODEL_GEN = os.environ.get("UNIQ_MODEL_GEN", S3.MODEL_DECOMPOSE)    # search + guess
MODEL_CHK = os.environ.get("UNIQ_MODEL_CHK", S3.MODEL_CHECK)        # the four gates
MAX_BIB = int(os.environ.get("UNIQ_MAX_BIB", "150"))                # candidate pool cap


# =========================================================================== prompts
# NOT redefined here. The prompts and the set-comparison helpers live in
# 03_decompose.py and are imported, because two copies drifted once already:
# the standalone SUBSTITUTE grew a "must cover what the unavailable paper
# supplied" rule that the integrated one never had, so the two entry points
# were running measurably different experiments under one name.
CONTAMINATION = S3.CONTAMINATION
CONTAM_JUDGE  = S3.CONTAM_JUDGE
SUBSTITUTE    = S3.SUBSTITUTE

# =========================================================================== helpers
norm_title = S3._norm_t


same_title = S3._same_t


def bib_titles(paper: dict) -> list[str]:
    out, seen = [], set()
    for e in (paper.get("bibliography") or {}).values():
        t = e.get("title") if isinstance(e, dict) else str(e)
        if not t:
            continue
        k = norm_title(t)[:70]
        if k and k not in seen:
            seen.add(k)
            out.append(str(t).strip())
    return out


gates_pass = S3._all_gates_pass


def run_checks(rec: dict, insps: list[dict], llm, tag: str) -> dict:
    """The stage-3 checker, unmodified, applied to a candidate set."""
    raw = llm(S3.CHECKS.format(rq=rec.get("research_question", ""),
                               bg=rec.get("background_survey", ""),
                               hyp=rec.get("fine_grained_hypothesis", ""),
                               insps=S3.fmt(insps)),
              MODEL_CHK, thinking=False, tag=tag)
    return S3.as_json(raw)


# =========================================================================== controls
def contamination(rec: dict, llm, n: int) -> dict:
    """How often does b alone produce h*? Recorded, never used to drop a paper."""
    guesses = []
    for t in range(n):
        raw = llm(CONTAMINATION.format(rq=rec.get("research_question", ""),
                                       bg=rec.get("background_survey", "")),
                  MODEL_GEN, thinking=True, tag=f"contam{t}")
        guesses.append(str(S3.as_json(raw).get("hypothesis", ""))[:1500])

    listing = "\n\n".join(f"[{i}] {g}" for i, g in enumerate(guesses))
    j = S3.as_json(llm(CONTAM_JUDGE.format(n=n, ref=rec.get("fine_grained_hypothesis", ""),
                                           guesses=listing),
                       MODEL_CHK, thinking=False, tag="contam-judge"))
    m = [bool(x) for x in (j.get("matches") or [])][:n]
    return {"r_empty": (sum(m) / len(m)) if m else 0.0,
            "n_tries": len(m),
            "guesses": guesses,
            "match_reasons": (j.get("reasons") or [])[:n]}


def decoy(rec: dict, insps: list[dict], bib: list[str], llm, rng) -> dict:
    """The four checks on random cited papers. These SHOULD fail."""
    used = [x.get("supposed_title", "") for x in insps]
    pool = [t for t in bib if not any(same_title(t, u) for u in used)]
    if len(pool) < len(insps):
        return {"ran": False, "reason": "not enough unused references"}
    picked = rng.sample(pool, len(insps))
    fake = [{"insp": f"Knowledge reported in the cited work titled {t!r}.",
             "insp_concise": " ".join(t.split()[:5]),
             "supposed_title": t} for t in picked]
    g = run_checks(rec, fake, llm, "decoy")
    return {"ran": True, "titles": picked, "gates": g,
            "passed": gates_pass(g, len(fake))}


# =========================================================================== the sweep
def substitute_for(rec: dict, insps: list[dict], j: int, bib: list[str],
                   llm, verify: bool) -> dict:
    """Hide insps[j], search the rest of the bibliography, verify what comes back."""
    ex = insps[j]
    ex_title = ex.get("supposed_title", "")
    # The SAME list for every j, so the prompt prefix is byte-identical across a
    # paper's k calls and calls 2..k hit the cache. The excluded paper is still
    # in it; the exclusion is enforced on the reply below.
    avail = bib[:MAX_BIB]
    listing = "\n".join(f"  - {t}" for t in avail)

    raw = llm(SUBSTITUTE.format(rq=rec.get("research_question", ""),
                                bg=rec.get("background_survey", ""),
                                hyp=rec.get("fine_grained_hypothesis", ""),
                                insps=S3.fmt(insps),
                                ex_idx=j, ex_title=ex_title,
                                ex_supplies=ex.get("insp", ""),
                                bib=listing),
              MODEL_GEN, thinking=True, tag=f"sub{j}")
    d = S3.as_json(raw)

    out = {"excluded": ex_title, "excluded_index": j,
           "found": bool(d.get("found")), "why_not": str(d.get("why_not", ""))[:400]}
    if not out["found"]:
        out["label"] = "no_substitute_found"
        return out

    # Rebuild I' = kept originals + proposed replacements. Anything the model
    # "added" that is really the excluded paper under another name is dropped
    # here, not trusted: the exclusion has to hold by construction.
    keep_idx = [i for i in (d.get("keep") or []) if isinstance(i, int)
                and 0 <= i < len(insps) and i != j]
    repl = [r for r in (d.get("replacement") or []) if isinstance(r, dict)]
    repl = [r for r in repl if not same_title(r.get("supposed_title", ""), ex_title)]
    # and every added paper must actually be in the reference list
    grounded = [r for r in repl
                if any(same_title(r.get("supposed_title", ""), t, 0.7) for t in avail)]
    out["ungrounded_dropped"] = len(repl) - len(grounded)

    i_prime = [insps[i] for i in keep_idx] + grounded
    out["replacement"] = grounded
    out["I_prime"] = [x.get("supposed_title", "") for x in i_prime]

    if not grounded or not i_prime:
        out["found"] = False
        out["label"] = "no_substitute_found"
        out["why_not"] = out["why_not"] or "proposal was empty or ungrounded after checking"
        return out
    if not verify:
        out["label"] = "unverified"
        return out

    g = run_checks(rec, i_prime, llm, f"verify{j}")
    out["gates"] = g
    out["verified"] = gates_pass(g, len(i_prime))
    out["label"] = "replaceable" if out["verified"] else "no_substitute_found"
    if not out["verified"]:
        out["why_not"] = "a replacement was proposed but failed the four checks"
    return out


def process(rec: dict, paper: dict, llm, args, rng, prior: dict | None = None) -> dict:
    insps = rec.get("inspiration") or []
    bib = bib_titles(paper)
    res = {"source_id": rec.get("source_id"), "doi": rec.get("doi"),
           "title": rec.get("title"), "domain": rec.get("domain"),
           "n_insp": len(insps), "n_bib": len(bib),
           "I": [x.get("supposed_title", "") for x in insps],
           # the search budget this paper's verdict is relative to
           "budget": {"model_gen": MODEL_GEN, "model_chk": MODEL_CHK,
                      "effort": S3.EFFORT, "tries": args.tries,
                      "candidates_offered": min(len(bib), MAX_BIB)}}

    # The controls are a property of the paper, not of the sweep, so a previous
    # --stage controls run is reused rather than paid for twice. Contamination is
    # three pro calls with thinking on and is the single most expensive item here.
    if prior:
        res["contamination"] = prior.get("contamination", {})
        res["decoy"] = prior.get("decoy", {})
        res["controls_reused"] = True
    else:
        res["contamination"] = contamination(rec, llm, args.tries)
        res["decoy"] = decoy(rec, insps, bib, llm, rng)

    if args.stage == "controls":
        return res

    if res["decoy"].get("passed"):
        # The checker accepted random papers for this paper. Anything it accepts
        # below would be meaningless, so the sweep is skipped and said so.
        res["sweep"] = []
        res["uniqueness"] = "void_decoy_passed"
        return res

    sweep = [substitute_for(rec, insps, j, bib, llm, verify=not args.no_verify)
             for j in range(len(insps))]
    res["sweep"] = sweep

    seen_sets, m = set(), []
    for cand in [res["I"]] + [s["I_prime"] for s in sweep if s.get("verified")]:
        k = frozenset(norm_title(t) for t in cand if t)
        if k and k not in seen_sets:
            seen_sets.add(k)
            m.append(cand)
    res["M"] = m
    res["n_valid_sets"] = len(m)
    if args.no_verify:
        # Nothing was checked, so nothing can be concluded. Without this the
        # verdict reads "not_falsified" purely because no set carries
        # verified=True, which is an unverified run masquerading as a result.
        res["uniqueness"] = "unverified"
    else:
        res["uniqueness"] = "falsified" if len(m) > 1 else "not_falsified"
    res["labels"] = {s["excluded"]: s["label"] for s in sweep}
    return res


# =========================================================================== report
def report(rows: list[dict], stage: str) -> None:
    n = len(rows)
    if not n:
        print("no rows")
        return
    print(f"\n{'=' * 72}\nCONTROLS  ({n} papers)")
    r = [x["contamination"]["r_empty"] for x in rows if x.get("contamination")]
    if r:
        print(f"  r_empty (b alone reproduces h*)   mean {sum(r)/len(r):.2f}")
        for lo, hi, lbl in [(0, .01, "never"), (.01, .34, "1 of 3"),
                            (.34, .67, "2 of 3"), (.67, 1.01, "always")]:
            c = sum(1 for v in r if lo <= v < hi)
            print(f"      {lbl:<8} {c:>4}  ({100*c/len(r):.0f}%)")
    d = [x["decoy"] for x in rows if x.get("decoy", {}).get("ran")]
    if d:
        bad = sum(1 for x in d if x["passed"])
        print(f"  decoy sets that WRONGLY passed    {bad}/{len(d)}"
              f"  ({100*bad/len(d):.0f}%)   <- want this near 0")

    if stage == "controls":
        print("\ncontrols only; rerun with --stage sweep for the uniqueness test")
        return

    sw = [x for x in rows if x.get("sweep") is not None and "M" in x]
    if not sw:
        return
    fals = sum(1 for x in sw if x["uniqueness"] == "falsified")
    print(f"\n{'=' * 72}\nUNIQUENESS  ({len(sw)} papers swept)")
    print(f"  uniqueness FALSIFIED              {fals}/{len(sw)}"
          f"  ({100*fals/len(sw):.0f}%)")
    print(f"  mean |M(b,h*)|                    "
          f"{sum(x['n_valid_sets'] for x in sw)/len(sw):.2f}   (before: 1.00)")

    lab = {}
    for x in sw:
        for v in (x.get("labels") or {}).values():
            lab[v] = lab.get(v, 0) + 1
    tot = sum(lab.values()) or 1
    print("\n  per-inspiration verdict:")
    for k, v in sorted(lab.items(), key=lambda kv: -kv[1]):
        print(f"      {k:<22} {v:>4}  ({100*v/tot:.0f}%)")

    prop = sum(1 for x in sw for s in x["sweep"] if s.get("found"))
    ver = sum(1 for x in sw for s in x["sweep"] if s.get("verified"))
    print(f"\n  substitutes proposed              {prop}")
    print(f"  substitutes that passed 4 checks  {ver}"
          + (f"  ({100*ver/prop:.0f}% of proposals)" if prop else ""))

    # ---- the side-by-side. Retrieval cannot be compared until stage 4 resolves
    # these titles to corpus documents, so what changes here is the CONSTRUCTION:
    # how many gold sets exist, how many distinct gold documents they name, and
    # how many inspirations turn out to have a validated stand-in.
    before_docs = {norm_title(t) for x in sw for t in x["I"] if t}
    after_docs = {norm_title(t) for x in sw for s in x["M"] for t in s if t}
    n_insp = sum(x["n_insp"] for x in sw)
    repl = lab.get("replaceable", 0)
    grew = sum(1 for x in sw if x["n_valid_sets"] > 1)
    print(f"\n{'=' * 72}\nBEFORE vs AFTER  (construction, {len(sw)} papers)\n")
    print(f"  {'':<38}{'before':>10}{'after':>12}")
    print(f"  {'gold sets per paper':<38}{1.00:>10.2f}"
          f"{sum(x['n_valid_sets'] for x in sw)/len(sw):>12.2f}")
    print(f"  {'papers with more than one gold set':<38}{0:>10}"
          f"{grew:>12}  ({100*grew/len(sw):.0f}%)")
    print(f"  {'distinct gold documents':<38}{len(before_docs):>10}"
          f"{len(after_docs):>12}"
          + (f"  (+{100*(len(after_docs)-len(before_docs))/len(before_docs):.0f}%)"
             if before_docs else ""))
    print(f"  {'inspirations with a known substitute':<38}{0:>10}"
          f"{repl:>12}" + (f"  ({100*repl/n_insp:.0f}%)" if n_insp else ""))
    print(f"\n  A retriever that returns a substitute is scored WRONG before and")
    print(f"  RIGHT after. That is the difference the benchmark now records.")


# =========================================================================== main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decomposed", type=Path, required=True,
                    help="stage-3 output: the records carrying b, h* and I")
    ap.add_argument("--fulltext", type=Path, required=True,
                    help="stage-2 file, for each paper's bibliography")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stage", choices=["controls", "sweep"], default="controls",
                    help="controls: the cheap gate. sweep: the full uniqueness test")
    ap.add_argument("--controls", type=Path,
                    help="a finished --stage controls file; reuses its contamination "
                         "and decoy results instead of paying for them again")
    ap.add_argument("--tries", type=int, default=3,
                    help="contamination guesses per paper")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=17, help="decoy sampling")
    ap.add_argument("--no-verify", action="store_true",
                    help="propose substitutes but skip the four checks (cheap, unusable as a result)")
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report-only", action="store_true",
                    help="re-print the summary from an existing --out file")
    args = ap.parse_args()

    if args.report_only:
        rows = [json.loads(l) for l in open(args.out) if l.strip()]
        report(rows, args.stage)
        return

    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key and not args.dry_run:
        sys.exit("DEEPSEEK_API_KEY is not set")

    # join stage 3 back to stage 2 for the bibliography, which stage 3 does not carry
    src = {}
    for line in open(args.fulltext):
        if not line.strip():
            continue
        p = json.loads(line)
        for k in (p.get("doi"), norm_title(p.get("title"))[:70]):
            if k:
                src.setdefault(k, p)

    recs, unmatched = [], 0
    for line in open(args.decomposed):
        if not line.strip():
            continue
        r = json.loads(line)
        if not r.get("decomposed") or not (r.get("inspiration") or []):
            continue
        p = src.get(r.get("doi")) or src.get(norm_title(r.get("title"))[:70])
        if p is None:
            unmatched += 1
            continue
        recs.append((r, p))
    if args.limit:
        recs = recs[:args.limit]
    print(f"{len(recs)} papers with inspirations"
          + (f"  ({unmatched} could not be joined to a bibliography)" if unmatched else ""))

    done = set()
    if args.out.exists():
        done = {json.loads(l).get("source_id")
                for l in open(args.out) if l.strip()}
        recs = [(r, p) for r, p in recs if r.get("source_id") not in done]
        if done:
            print(f"resuming: {len(done)} already done, {len(recs)} to go")

    prior: dict[str, dict] = {}
    if args.controls:
        for line in open(args.controls):
            if not line.strip():
                continue
            c = json.loads(line)
            if c.get("source_id") and c.get("contamination") is not None:
                prior[c["source_id"]] = c
        hit = sum(1 for r, _ in recs if r.get("source_id") in prior)
        print(f"reusing controls for {hit}/{len(recs)} papers from {args.controls}")

    llm = S3.DeepSeek(key, dry_run=args.dry_run, verbose=args.verbose)
    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(args.out, "a") as fh, \
         ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(process, r, p, llm, args, random.Random(args.seed + i),
                            prior.get(r.get("source_id"))): r
                for i, (r, p) in enumerate(recs)}
        bar = tqdm(total=len(futs), desc=f"uniqueness/{args.stage}", unit="paper")
        for f in as_completed(futs):
            try:
                row = f.result()
            except Exception as exc:
                S3.log(f"  !! {futs[f].get('title','?')[:60]}: "
                       f"{exc.__class__.__name__}: {exc}")
                bar.update(1)
                continue
            rows.append(row)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            bar.update(1)
        bar.close()

    tot_in = llm.tok_in + llm.tok_cached
    print(f"\n{llm.calls} calls   ${llm.cost:.4f}"
          + (f"   ${llm.cost/len(rows):.4f}/paper" if rows else ""))
    print(f"  input {tot_in:,} tokens, {llm.tok_cached:,} of them cached "
          f"({100*llm.tok_cached/tot_in:.0f}%)   output {llm.tok_out:,}"
          if tot_in else "")
    # The search budget B. "No substitute found" is only as strong as this, so it
    # is printed and stored rather than left implicit.
    print(f"  budget B: models {MODEL_GEN} / {MODEL_CHK}, "
          f"effort {S3.EFFORT}/{S3.EFFORT_CHECK}, "
          f"{args.tries} contamination tries, {MAX_BIB} candidates offered")
    if done:
        rows = [json.loads(l) for l in open(args.out) if l.strip()]
    report(rows, args.stage)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
