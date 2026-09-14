#!/usr/bin/env python3
"""Compare the standard decomposition against the locked-target one.

Two arms, same papers, same prompt, same model. The ONLY difference is what a
gate failure re-runs:

    old   the whole DECOMPOSE prompt, so the research question, background AND
          hypothesis are regenerated. The target moves while the inspirations
          are being fixed.
    new   only the inspiration search, against a hypothesis frozen after round 0.

This script uses no API calls. Everything it reports is already in the two files.

    python build/compare_arms.py --old data/_snap/cmp_old.jsonl \
                                --new data/_snap/cmp_new.jsonl \
                                [--uniq data/_snap/cmp_new_uniq.jsonl]
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import unicodedata
from pathlib import Path

STOP = {"the", "a", "an", "of", "in", "on", "for", "and", "to", "with", "by", "from",
        "at", "as", "is", "are", "be", "that", "this", "it", "its", "which", "can",
        "will", "we", "our", "their", "these", "those", "than", "such", "more"}


def words(s) -> set[str]:
    s = unicodedata.normalize("NFKD", str(s or "").lower())
    return {w for w in re.findall(r"[a-z0-9]+", s) if w not in STOP and len(w) > 2}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 1.0


def title_key(t) -> str:
    if isinstance(t, list):
        t = " ".join(str(x) for x in t)
    return re.sub(r"[^a-z0-9 ]", "", str(t or "").lower()).strip()[:60]


def load(path: Path) -> dict[str, dict]:
    out = {}
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        k = r.get("doi") or title_key(r.get("title"))
        if k:
            out[k] = r
    return out


def arm_summary(name: str, recs: dict) -> dict:
    ok = [r for r in recs.values() if r.get("decomposed")]
    rounds = [r.get("gates", {}).get("rounds", 1) for r in ok]
    retried = sum(1 for x in rounds if x > 1)
    n_insp = [len(r.get("inspiration") or []) for r in ok]
    fails = {}
    for r in ok:
        for k, v in (r.get("gates") or {}).items():
            if k.endswith("_failed") and v:
                fails[k] = fails.get(k, 0) + v
    return {"name": name, "n": len(recs), "ok": len(ok),
            "rounds": st.mean(rounds) if rounds else 0,
            "retried": retried, "insp": st.mean(n_insp) if n_insp else 0,
            "fails": fails}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", type=Path, required=True)
    ap.add_argument("--new", type=Path, required=True)
    ap.add_argument("--uniq", type=Path, help="the 03b sweep output for the new arm")
    a = ap.parse_args()

    old, new = load(a.old), load(a.new)
    shared = sorted(set(old) & set(new))
    so, sn = arm_summary("old", old), arm_summary("new", new)

    print(f"\n{'=' * 70}\nARMS\n")
    print(f"  {'':<28}{'old':>10}{'new':>12}")
    print(f"  {'papers processed':<28}{so['n']:>10}{sn['n']:>12}")
    print(f"  {'decomposed':<28}{so['ok']:>10}{sn['ok']:>12}")
    print(f"  {'mean gate rounds':<28}{so['rounds']:>10.2f}{sn['rounds']:>12.2f}")
    print(f"  {'papers that needed a retry':<28}{so['retried']:>10}{sn['retried']:>12}")
    print(f"  {'mean inspirations':<28}{so['insp']:>10.2f}{sn['insp']:>12.2f}")

    print(f"\n  gate failures fired:")
    for k in sorted(set(so["fails"]) | set(sn["fails"])):
        print(f"    {k:<26}{so['fails'].get(k,0):>10}{sn['fails'].get(k,0):>12}")

    # ---------------------------------------------------------- did h move?
    both = [k for k in shared if old[k].get("decomposed") and new[k].get("decomposed")]
    if not both:
        print("\nno papers decomposed in both arms")
        return
    hyp_sim, ins_sim, bg_sim = [], [], []
    for k in both:
        o, n = old[k], new[k]
        hyp_sim.append(jaccard(words(o.get("fine_grained_hypothesis")),
                               words(n.get("fine_grained_hypothesis"))))
        bg_sim.append(jaccard(words(o.get("background_survey")),
                              words(n.get("background_survey"))))
        ins_sim.append(jaccard(
            {title_key(x.get("supposed_title")) for x in (o.get("inspiration") or [])},
            {title_key(x.get("supposed_title")) for x in (n.get("inspiration") or [])}))

    print(f"\n{'=' * 70}\nHOW FAR APART ARE THE TWO ARMS  ({len(both)} papers in both)\n")
    print(f"  hypothesis overlap   mean {st.mean(hyp_sim):.2f}   "
          f"identical {sum(1 for x in hyp_sim if x > 0.95)}/{len(both)}")
    print(f"  background overlap   mean {st.mean(bg_sim):.2f}")
    print(f"  inspiration overlap  mean {st.mean(ins_sim):.2f}   "
          f"identical {sum(1 for x in ins_sim if x > 0.99)}/{len(both)}   "
          f"disjoint {sum(1 for x in ins_sim if x == 0)}/{len(both)}")

    # The old arm regenerates the hypothesis on every retry, so its retried papers
    # are exactly where the two arms should diverge. If they do not, locking is
    # doing nothing and the whole distinction is cosmetic.
    ret = [i for i, k in enumerate(both) if old[k].get("gates", {}).get("rounds", 1) > 1]
    nor = [i for i in range(len(both)) if i not in set(ret)]
    if ret and nor:
        print(f"\n  split by whether the OLD arm had to retry:")
        print(f"    old retried    ({len(ret):>3} papers)  hypothesis overlap "
              f"{st.mean([hyp_sim[i] for i in ret]):.2f}")
        print(f"    old first try  ({len(nor):>3} papers)  hypothesis overlap "
              f"{st.mean([hyp_sim[i] for i in nor]):.2f}")
        print(f"\n  A gap here is the effect of locking. No gap means the retry was")
        print(f"  rewriting the hypothesis without changing what it says.")

    # ------------------------------------------------------------- uniqueness
    if not a.uniq or not a.uniq.exists():
        print(f"\n(pass --uniq to add the uniqueness result for the new arm)")
        return
    u = [json.loads(l) for l in open(a.uniq) if l.strip()]
    sw = [x for x in u if "M" in x]
    if not sw:
        print(f"\n{'=' * 70}\nUNIQUENESS: controls only, no sweep in {a.uniq.name}")
        return
    fals = sum(1 for x in sw if x["uniqueness"] == "falsified")
    print(f"\n{'=' * 70}\nUNIQUENESS  (new arm, {len(sw)} papers swept)\n")
    print(f"  {'':<34}{'old':>10}{'new':>12}")
    print(f"  {'gold sets per paper':<34}{1.00:>10.2f}"
          f"{sum(x['n_valid_sets'] for x in sw)/len(sw):>12.2f}")
    print(f"  {'papers with a second valid set':<34}{0:>10}{fals:>12}"
          f"  ({100*fals/len(sw):.0f}%)")
    before = {title_key(t) for x in sw for t in x["I"] if t}
    after = {title_key(t) for x in sw for s in x["M"] for t in s if t}
    print(f"  {'distinct gold documents':<34}{len(before):>10}{len(after):>12}")


if __name__ == "__main__":
    main()
