#!/usr/bin/env python3
"""Change the same/cross definition after the fact, without rerunning anything.

Stage 4 stores the EVIDENCE behind each label, not just the verdict: the
OpenAlex topic-field distribution of the query paper and of the gold, and which
fields they share. Every policy in build/labels.py is a pure function of that, so
switching between them is arithmetic over a file already on disk. No API calls,
no LLM calls, no refetch, seconds rather than hours.

    # what would each policy give? changes nothing
    python3 build/relabel.py --resolved data/_snap/v3_resolved.jsonl --compare

    # rewrite a resolved file under a different policy
    python3 build/relabel.py --resolved data/_snap/v3_resolved.jsonl \
                             --policy overlap2 --write

    # rewrite an exported benchmark in place (eval.json stratum + manifest)
    python3 build/relabel.py --benchmark data/benchmark/cs_test \
                             --policy overlap2 --write

WHY THIS EXISTS RATHER THAN A CONSTANT IN STAGE 4.

The choice between "cross means no shared field" and "cross means at most one
shared field" is a judgement, and on the pilot the two separate retrieval
difficulty equally well (9.1 against 9.3 nDCG@10 points). A judgement that close
should not be welded into a file that costs money to regenerate, and it should
not be re-litigated by rerunning the pipeline. It is one flag over stored facts.

Nothing here can change WHICH papers are gold, or their text, or the corpus. It
only re-partitions rows that already exist, so a relabelled benchmark stays
comparable to the run it came from.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels as L

ROOT = Path(__file__).resolve().parent.parent


def all_inspirations(rec: dict) -> list:
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


def evidence_of(x: dict, rec: dict | None = None) -> dict | None:
    """The stored evidence, or None if this row predates it.

    A file written before the evidence was stored can still be relabelled, as
    long as both topic distributions survived: the evidence is recomputed from
    them. What cannot be recovered is a row with no topic list at all, and those
    are reported rather than guessed at.
    """
    ev = x.get("label_evidence")
    if ev:
        return ev
    if rec is None:
        return None
    ttf = rec.get("target_topic_fields") or rec.get("topic_fields")
    itf = x.get("insp_topic_fields")
    if not (ttf and itf):
        return None
    return L.label_evidence(ttf, itf,
                            rec.get("target_field_resolved") or rec.get("primary_field"),
                            x.get("insp_field"))


def load(path: Path) -> list:
    return [json.loads(l) for l in open(path) if l.strip()]


def counts(recs: list, policy: str) -> Counter:
    c = Counter()
    for r in recs:
        for x in all_inspirations(r):
            if x.get("match_quality") in (None, "none"):
                continue
            ev = evidence_of(x, r)
            c[L.apply_policy(ev, policy)["domain_relation"] if ev else None] += 1
    return c


def show_compare(recs: list, current: str | None) -> None:
    print(f"\n{'policy':<16}{'same':>8}{'cross':>8}{'cross %':>10}{'no evidence':>14}")
    for name in sorted(L.POLICIES):
        c = counts(recs, name)
        n = c["same"] + c["cross"]
        mark = "   <- in use" if name == current else ""
        print(f"{name:<16}{c['same']:>8}{c['cross']:>8}"
              f"{100 * c['cross'] / max(n, 1):>9.1f}%{c[None]:>14}{mark}")
    band = Counter()
    for r in recs:
        for x in all_inspirations(r):
            if x.get("match_quality") in (None, "none"):
                continue
            ev = evidence_of(x, r)
            if ev:
                band[L.band_of(ev)] += 1
    if band:
        bt = sum(band.values())
        print(f"\nthe graded scale every policy cuts (policy-independent):")
        for k in L.BANDS:
            if band[k]:
                print(f"  {k:<22}{band[k]:>6}  ({100 * band[k] / bt:.1f}%)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--resolved", type=Path, help="a Stage 4 output .jsonl")
    src.add_argument("--benchmark", type=Path, help="a Stage 5 output directory")
    ap.add_argument("--policy", choices=sorted(L.POLICIES),
                    help=f"the new definition. Default just reports")
    ap.add_argument("--compare", action="store_true",
                    help="show every policy over this data and change nothing")
    ap.add_argument("--write", action="store_true",
                    help="apply it. Without this, nothing is modified")
    ap.add_argument("--no-backup", action="store_true")
    a = ap.parse_args()

    # ------------------------------------------------------------- resolved file
    if a.resolved:
        recs = load(a.resolved)
        cur = next((r.get("label_policy") for r in recs if r.get("label_policy")), None)
        print(f"{len(recs)} papers from {a.resolved.name}"
              + (f", currently labelled {cur!r}" if cur else
                 ", no policy recorded (written before policies existed)"))
        if a.compare or not a.policy:
            show_compare(recs, cur)
            if not a.policy:
                return
        n = miss = 0
        for r in recs:
            for x in all_inspirations(r):
                ev = evidence_of(x, r)
                if not ev:
                    miss += 1
                    continue
                x["label_evidence"] = ev
                x.update(L.apply_policy(ev, a.policy))
                n += 1
            r["label_policy"] = a.policy
        print(f"\nrelabelled {n} gold occurrences under {a.policy!r}"
              + (f", {miss} had no topic evidence and keep their old label" if miss else ""))
        if not a.write:
            print("(nothing written; add --write)")
            return
        if not a.no_backup:
            bak = a.resolved.with_suffix(a.resolved.suffix + ".prepolicy")
            if not bak.exists():
                shutil.copy2(a.resolved, bak)
                print(f"backup -> {bak}")
        tmp = a.resolved.with_suffix(a.resolved.suffix + ".partial")
        with open(tmp, "w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp.replace(a.resolved)
        print(f"wrote {a.resolved}")
        return

    # ------------------------------------------------------------ benchmark dir
    ev_path = a.benchmark / "eval.json"
    if not ev_path.exists():
        sys.exit(f"no eval.json in {a.benchmark}")
    rows = json.loads(ev_path.read_text())
    cur = next((r.get("quartet", {}).get("label_policy") for r in rows
                if r.get("quartet", {}).get("label_policy")), None)
    print(f"{len(rows)} eval rows from {ev_path}"
          + (f", currently {cur!r}" if cur else ""))

    have = [r for r in rows if r.get("quartet", {}).get("label_evidence")]
    if not have:
        sys.exit("these rows carry no label_evidence, so they cannot be relabelled "
                 "here.\nRelabel the Stage 4 output with --resolved and re-export.")
    if a.compare or not a.policy:
        print(f"\n{'policy':<16}{'same':>8}{'cross':>8}{'cross %':>10}")
        for name in sorted(L.POLICIES):
            c = Counter(L.apply_policy(r["quartet"]["label_evidence"], name)["domain_relation"]
                        for r in have)
            n = c["same"] + c["cross"]
            mark = "   <- in use" if name == cur else ""
            print(f"{name:<16}{c['same']:>8}{c['cross']:>8}"
                  f"{100 * c['cross'] / max(n, 1):>9.1f}%{mark}")
        if not a.policy:
            return

    changed = 0
    for r in rows:
        ev = r.get("quartet", {}).get("label_evidence")
        if not ev:
            continue
        new = L.apply_policy(ev, a.policy)
        if new["domain_relation"] and new["domain_relation"] != r.get("stratum"):
            changed += 1
        r["stratum"] = new["domain_relation"] or r.get("stratum")
        r["quartet"]["domain_distance"] = new["domain_distance"]
        r["quartet"]["label_policy"] = a.policy
    print(f"\n{changed} of {len(rows)} rows change stratum under {a.policy!r}")
    print("  ", dict(Counter(r["stratum"] for r in rows)))
    if not a.write:
        print("(nothing written; add --write)")
        return
    if not a.no_backup:
        bak = ev_path.with_suffix(".json.prepolicy")
        if not bak.exists():
            shutil.copy2(ev_path, bak)
            print(f"backup -> {bak}")
    ev_path.write_text(json.dumps(rows, ensure_ascii=False))
    man = a.benchmark / "manifest.json"
    if man.exists():
        m = json.loads(man.read_text())
        m["label_policy"] = a.policy
        m["stratum"] = dict(Counter(r["stratum"] for r in rows))
        man.write_text(json.dumps(m, indent=1))
    print(f"wrote {ev_path}"
          + (f" and {man.name}" if man.exists() else ""))


if __name__ == "__main__":
    main()
