"""Aggregate + report the downstream 4-arm composition comparison.

API-FREE.  Reads the matched-score and idea-arena jsonl files and prints:
  1. Matched-Score: mean total (0-12) + per-dimension, per arm, for Overall / Same /
     Cross, as a floor->ceiling ladder over whichever arms were run
     (none <= bm25 <= moose_chem <= oracle), plus paired differences with 95%
     bootstrap CIs -- headline MOOSE-Chem - BM25, and the ladder contrasts
     (vs the no-insp floor and the oracle ceiling).
  2. Idea Arena: round-robin over all run arms (official CoI judge, both orders).
     Per stratum we print a leaderboard with a multi-player Bradley-Terry ELO
     (now genuinely informative -- >2 contestants) and score-share, a head-to-head
     score-share matrix, and the MOOSE-Chem-vs-BM25 headline (direct-matchup
     score-share + 95% bootstrap CI + per-criterion win/tie/loss).
Also writes results/downstream/summary.json.

Run:
  cd TOMATO-Star
  python -m analysis.downstream.aggregate
"""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict

from analysis.downstream._common import OUT_DIR, index_by, read_jsonl
from analysis.downstream.idea_arena import CRITERIA, CRIT_DISPLAY

random.seed(42)


# --------------------------------------------------------------------------
# Matched-Score
# --------------------------------------------------------------------------
def _matched(arm: str, tag: str = "") -> dict:
    suffix = f"__{tag}" if tag else ""
    rows = [r for r in read_jsonl(OUT_DIR / f"matched_{arm}{suffix}.jsonl")
            if not r.get("failed") and r.get("total") is not None]
    return index_by(rows)


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _bootstrap_ci(diffs, n_boot=10000, alpha=0.05):
    if not diffs:
        return (float("nan"), float("nan"))
    n = len(diffs)
    means = []
    for _ in range(n_boot):
        s = sum(diffs[random.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot)]
    return (lo, hi)


# Arms in floor -> ceiling order, with display labels.  "gt" (the real paper
# hypothesis) is an ARENA-ONLY contestant -- it has no matched_gt.jsonl, so the
# Matched-Score ladder silently drops it; the arena leaderboard includes it.
ARM_ORDER = [("none", "No-insp"), ("random", "Random"), ("bm25", "BM25"), ("bge", "BGE"),
             ("qwen3", "Qwen3"), ("reasonir", "ReasonIR"),
             ("moose_chem", "MOOSE-Chem"), ("lattice", "LATTICE"),
             ("ours", "Ours-old"),
             ("scigraphir", "SciGraphIR"), ("oracle", "Oracle"),
             ("gt", "GT-real")]
ARM_DISPLAY = dict(ARM_ORDER)

# Paired contrasts (hi, lo) to report when BOTH arms are present.  The first is
# the headline; the rest characterise the floor/ceiling ladder.
DELTAS = [("scigraphir", "moose_chem"),  # headline: SciGraphIR vs the LLM retriever
          ("scigraphir", "qwen3"),       # vs its own dense backbone
          ("scigraphir", "reasonir"),    # vs the strongest off-the-shelf retriever
          ("scigraphir", "lattice"),     # vs the strongest LLM-search baseline (SIR-4)
          ("lattice", "none"),
          ("scigraphir", "none"),        # does it help over the floor
          ("moose_chem", "none"), ("qwen3", "none"), ("reasonir", "none"),
          ("bm25", "none"),
          ("random", "none"),            # does ANY document hurt, or only a wrong one
          ("scigraphir", "random"), ("qwen3", "random"),
          ("reasonir", "random"), ("moose_chem", "random"),
          ("ours", "moose_chem"), ("ours", "bge"),   # legacy June arm, if present
          ("oracle", "scigraphir"),      # headroom left
          ("oracle", "none")]            # max achievable lift (ceiling - floor)


def matched_report(tag: str = "") -> dict:
    data = {arm: _matched(arm, tag) for arm, _ in ARM_ORDER}
    data = {a: d for a, d in data.items() if d}          # keep only run arms
    present = [(a, disp) for a, disp in ARM_ORDER if a in data]
    if not present:
        raise FileNotFoundError                          # main() reports this

    # Paired set = queries scored under EVERY present arm.
    shared = None
    for a, _ in present:
        ks = set(data[a])
        shared = ks if shared is None else (shared & ks)
    shared = sorted(shared or [])

    print("=" * 72)
    print(f"MATCHED-SCORE (0-12, recall-style)   "
          f"arms: {', '.join(disp for _, disp in present)}")
    print(f"  paired queries (scored under all arms): {len(shared)}")
    print("=" * 72)

    first_arm = present[0][0]
    out = {"n_paired": len(shared),
           "arms_present": [a for a, _ in present], "strata": {}}

    for label, key in [("Overall", None), ("Same-domain", "same"),
                       ("Cross-domain", "cross")]:
        qs = [q for q in shared
              if key is None or data[first_arm][q]["stratum"] == key]
        if not qs:
            continue
        print(f"\n[{label}]  n={len(qs)}")
        print(f"  {'Arm':<12} total   ( M    Me    Mo )")

        arms_out = {}
        for arm, disp in present:
            tot = _mean([data[arm][q]["total"] for q in qs])
            dims = {d: _mean([data[arm][q]["scores"][d] for q in qs])
                    for d in ("motivation", "mechanism", "methodology")}
            print(f"  {disp:<12} {tot:5.2f}   "
                  f"({dims['motivation']:.2f}  {dims['mechanism']:.2f}  "
                  f"{dims['methodology']:.2f})")
            arms_out[arm] = {"total": tot, "dims": dims}

        print(f"  {'-' * 44}")
        deltas_out = {}
        for hi_arm, lo_arm in DELTAS:
            if hi_arm not in data or lo_arm not in data:
                continue
            diffs = [data[hi_arm][q]["total"] - data[lo_arm][q]["total"]
                     for q in qs]
            lci, hci = _bootstrap_ci(diffs)
            sig = "" if (lci <= 0 <= hci) else "  *"
            tag = f"{ARM_DISPLAY[hi_arm]}-{ARM_DISPLAY[lo_arm]}"
            print(f"  Δ({tag:<20}) {_mean(diffs):+5.2f}   "
                  f"95% CI [{lci:+.2f}, {hci:+.2f}]{sig}")
            deltas_out[f"{hi_arm}-{lo_arm}"] = {
                "delta": _mean(diffs), "ci95": [lci, hci]}

        out["strata"][label] = {
            "n": len(qs), "arms": arms_out, "deltas": deltas_out}
    return out


def hit_conditioned_report(tag: str = "") -> dict:
    """Matched score split by whether the fed inspiration was the gold.

    Pooled over every retrieval arm (oracle and none excluded: always / never
    gold), then per arm.  This is the causal reading: right inspiration in ->
    better hypothesis out, independent of which retriever found it."""
    comps = {}
    for arm, _ in ARM_ORDER:
        rows = read_jsonl(OUT_DIR / f"compositions_{arm}.jsonl")
        if rows:
            comps[arm] = index_by(rows)
    data = {arm: _matched(arm, tag) for arm, _ in ARM_ORDER}
    arms = [a for a, _ in ARM_ORDER if a in data and data[a] and a in comps
            and a not in ("none", "random", "oracle", "gt")]
    if not arms:
        return {}
    print("\n" + "=" * 72)
    print("HIT-CONDITIONED MATCHED-SCORE  (fed inspiration == gold  vs  not)")
    print("=" * 72)
    out = {}
    pooled = {True: [], False: []}
    for arm in arms:
        by = {True: [], False: []}
        for q, m in data[arm].items():
            c = comps[arm].get(q)
            if c is None:
                continue
            by[bool(c.get("is_gold"))].append(m["total"])
        for k in by:
            pooled[k].extend(by[k])
        hit, miss = _mean(by[True]), _mean(by[False])
        print(f"  {ARM_DISPLAY[arm]:<12} hit {hit:5.2f} (n={len(by[True]):3d})   "
              f"miss {miss:5.2f} (n={len(by[False]):3d})   Δ {hit - miss:+.2f}   "
              f"hit-rate {100 * len(by[True]) / max(1, len(by[True]) + len(by[False])):.1f}%")
        out[arm] = {"hit": hit, "n_hit": len(by[True]), "miss": miss,
                    "n_miss": len(by[False])}
    hit, miss = _mean(pooled[True]), _mean(pooled[False])
    diffs_note = "(unpaired; pooled over arms)"
    print(f"  {'POOLED':<12} hit {hit:5.2f} (n={len(pooled[True]):3d})   "
          f"miss {miss:5.2f} (n={len(pooled[False]):3d})   Δ {hit - miss:+.2f}  {diffs_note}")
    for ref, label in (("none", "closed-book floor"), ("random", "random-doc control")):
        if ref in data and data[ref]:
            v = _mean([m["total"] for m in data[ref].values()])
            print(f"  {ARM_DISPLAY[ref]:<12} {v:5.2f}   ({label}, for reference)")
            out[f"{ref}_floor"] = v
    out["pooled"] = {"hit": hit, "n_hit": len(pooled[True]), "miss": miss,
                     "n_miss": len(pooled[False])}
    return out


# --------------------------------------------------------------------------
# Idea Arena
# --------------------------------------------------------------------------
def _coi_points_pair(rec):
    """Apply CoI's change_winner_to_score to one round-robin record across BOTH
    orders.  orderA: idea0=arm0, idea1=arm1 ; orderB swapped.  Returns
    (arm0, arm1, {criterion: (p0, p1)}) with p0+p1 == 2 per criterion, where
    p0 are points to arm0 and p1 to arm1."""
    a0, a1 = rec["arm0"], rec["arm1"]
    res = {}
    for crit in CRITERIA:
        p0 = p1 = 0.0
        a = rec["orderA"][crit]          # 0->idea0(a0) win, 1->idea1(a1) win, 2->tie
        if a == "0":
            p0 += 1
        elif a == "1":
            p1 += 1
        else:
            p0 += 0.5; p1 += 0.5
        b = rec["orderB"][crit]          # 0->idea0(a1) win, 1->idea1(a0) win, 2->tie
        if b == "0":
            p1 += 1
        elif b == "1":
            p0 += 1
        else:
            p0 += 0.5; p1 += 0.5
        res[crit] = (p0, p1)
    return a0, a1, res


def _bradley_terry_elo(wins, games_between, arms, iters=1000):
    """Fractional Bradley-Terry strengths via MM iteration, returned as Elo
    ratings anchored to mean 1000.

    wins[arm]           = total points scored by arm (ties count as 0.5).
    games_between[(i,j)] = symmetric point-units played between i and j
                           (= 10 * #records, i.e. 5 criteria x 2 orders).
    Each criterion-order is one Bernoulli 'game'; ties are half-wins, so this is
    the standard BT MLE on aggregated sufficient statistics."""
    p = {a: 1.0 for a in arms}
    for _ in range(iters):
        new = {}
        for i in arms:
            denom = 0.0
            for j in arms:
                if j == i:
                    continue
                g = games_between.get((i, j), 0.0)
                if g:
                    denom += g / (p[i] + p[j])
            w = max(wins.get(i, 0.0), 1e-9)
            new[i] = w / denom if denom > 0 else p[i]
        logmean = sum(math.log(max(v, 1e-12)) for v in new.values()) / len(new)
        gm = math.exp(logmean)                 # geometric-mean normalise
        p = {a: new[a] / gm for a in arms}
    elo = {a: 400.0 * math.log10(max(p[a], 1e-12)) for a in arms}
    shift = 1000.0 - sum(elo.values()) / len(elo)
    return {a: elo[a] + shift for a in arms}


def _pair_share_ci(rs, target, other, n_boot=10000, alpha=0.05):
    """Bootstrap CI on `target`'s score-share in its DIRECT matchups vs `other`,
    plus per-criterion win/tie/loss (win = target out-points other across both
    orders for that criterion).  Returns (share, lo, hi, per_crit, n_records)
    where per_crit[crit] = (target_win, tie, other_win, target_win_rate)."""
    direct = [r for r in rs if {r["arm0"], r["arm1"]} == {target, other}]
    rec_share = []
    per_crit = {c: [0, 0, 0] for c in CRITERIA}   # [target_win, tie, other_win]
    for r in direct:
        a0, _a1, res = _coi_points_pair(r)
        tpts = 0.0
        for crit in CRITERIA:
            p0, p1 = res[crit]
            tp = p0 if a0 == target else p1
            op = p1 if a0 == target else p0
            tpts += tp
            if tp > op:
                per_crit[crit][0] += 1
            elif op > tp:
                per_crit[crit][2] += 1
            else:
                per_crit[crit][1] += 1
        rec_share.append(tpts / 10.0)            # 10 pts available per record
    n = len(rec_share)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"),
                {c: (0, 0, 0, float("nan")) for c in CRITERIA}, 0)
    share = sum(rec_share) / n
    means = []
    for _ in range(n_boot):
        s = sum(rec_share[random.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot)]
    per_crit_out = {c: (per_crit[c][0], per_crit[c][1], per_crit[c][2],
                        per_crit[c][0] / n) for c in CRITERIA}
    return (share, lo, hi, per_crit_out, n)


def arena_report(tag: str = "") -> dict:
    suffix = f"__{tag}" if tag else ""
    rows = [r for r in read_jsonl(OUT_DIR / f"idea_arena{suffix}.jsonl")
            if not r.get("failed") and r.get("orderA") and r.get("orderB")
            and r.get("arm0") and r.get("arm1")]
    if not rows:
        raise FileNotFoundError                          # main() reports this
    arms_present = [a for a, _ in ARM_ORDER
                    if any(r["arm0"] == a or r["arm1"] == a for r in rows)]

    print("\n" + "=" * 72)
    print(f"IDEA ARENA round-robin (official CoI judge, both orders)   "
          f"records: {len(rows)}")
    print(f"  arms: {', '.join(ARM_DISPLAY[a] for a in arms_present)}")
    print("  points = +1 win / +0.5 tie, summed over 5 criteria x 2 orders "
          "(10 pts/record)")
    print("=" * 72)

    out = {"n": len(rows), "arms_present": arms_present, "strata": {}}
    for label, key in [("Overall", None), ("Same-domain", "same"),
                       ("Cross-domain", "cross")]:
        rs = [r for r in rows if key is None or r["stratum"] == key]
        if not rs:
            continue

        points = defaultdict(float)         # arm -> total points scored
        avail = defaultdict(float)          # arm -> total points available
        h2h_pts = defaultdict(float)        # (i,j) -> points i scored vs j
        games_between = defaultdict(float)  # (i,j) -> symmetric point-units
        for r in rs:
            a0, a1, res = _coi_points_pair(r)
            s0 = sum(res[c][0] for c in CRITERIA)
            s1 = sum(res[c][1] for c in CRITERIA)
            tot = s0 + s1                   # == 10
            points[a0] += s0; points[a1] += s1
            avail[a0] += tot; avail[a1] += tot
            h2h_pts[(a0, a1)] += s0
            h2h_pts[(a1, a0)] += s1
            games_between[(a0, a1)] += tot
            games_between[(a1, a0)] += tot

        present = [a for a in arms_present if a in points]
        elo = _bradley_terry_elo(points, games_between, present)
        share = {a: (points[a] / avail[a] if avail[a] else float("nan"))
                 for a in present}
        order = sorted(present, key=lambda a: elo[a], reverse=True)

        print(f"\n[{label}]  n={len(rs)} records")
        print(f"  {'Arm':<12} {'Elo':>6}  {'score-share':>11}  {'points':>8}")
        for a in order:
            print(f"  {ARM_DISPLAY[a]:<12} {elo[a]:6.0f}  "
                  f"{100*share[a]:9.1f}%  {points[a]:8.1f}")

        # head-to-head score-share matrix (row arm's share of points vs col arm)
        print(f"\n  head-to-head score-share (row vs col):")
        print("  " + " " * 12 + "".join(f"{ARM_DISPLAY[a][:9]:>10}"
                                         for a in order))
        for i in order:
            cells = []
            for j in order:
                if i == j:
                    cells.append(f"{'--':>10}")
                else:
                    g = games_between.get((i, j), 0.0)
                    sh = h2h_pts[(i, j)] / g if g else float("nan")
                    cells.append(f"{100*sh:9.1f}%")
            print(f"  {ARM_DISPLAY[i]:<12}" + "".join(cells))

        # headline pairs: the target arm vs every opponent it met directly,
        # each with a bootstrap CI on score-share and per-criterion W/T/L.
        target = "scigraphir" if "scigraphir" in present else (
            "moose_chem" if "moose_chem" in present else present[0])
        head = {}
        for opp in present:
            if opp == target or not games_between.get((target, opp)):
                continue
            sh, lci, hci, per_crit, n_pair = _pair_share_ci(rs, target, opp)
            sig = "" if (lci <= 0.5 <= hci) else "  *"
            print(f"\n  {ARM_DISPLAY[target]} vs {ARM_DISPLAY[opp]}  "
                  f"(n={n_pair} direct matchups)")
            print(f"    {ARM_DISPLAY[target]} score-share {100*sh:.1f}%   "
                  f"95% CI [{100*lci:.1f}%, {100*hci:.1f}%]{sig}   "
                  f"(* = CI excludes 50%)")
            print(f"    {'Criterion':<16}   win  tie  loss   win-rate")
            for crit in CRITERIA:
                w, t, l, wr = per_crit[crit]
                print(f"    {CRIT_DISPLAY[crit]:<16}  {w:4d} {t:4d}  {l:4d}    "
                      f"{100*wr:5.1f}%")
            head[f"{target}_vs_{opp}"] = {
                "score_share": sh, "ci95": [lci, hci], "n_direct": n_pair,
                "per_criterion": {c: {"win": per_crit[c][0], "tie": per_crit[c][1],
                                      "loss": per_crit[c][2], "win_rate": per_crit[c][3]}
                                  for c in CRITERIA}}

        out["strata"][label] = {
            "n": len(rs),
            "leaderboard": [{"arm": a, "elo": elo[a],
                             "score_share": share[a], "points": points[a]}
                            for a in order],
            "h2h_score_share": {
                f"{i}_vs_{j}": (h2h_pts[(i, j)] / games_between[(i, j)]
                               if games_between.get((i, j)) else None)
                for i in present for j in present if i != j},
            "headline_pairs": head,
        }
    return out


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="",
                    help="read judge outputs namespaced by tag (e.g. gpt-4o) "
                         "and write summary__{tag}.json")
    args = ap.parse_args()
    suffix = f"__{args.tag}" if args.tag else ""

    summary = {}
    try:
        summary["matched_score"] = matched_report(args.tag)
        summary["hit_conditioned"] = hit_conditioned_report(args.tag)
    except FileNotFoundError:
        print("(matched_*.jsonl not found -- run score_matched first)")
    try:
        summary["idea_arena"] = arena_report(args.tag)
    except FileNotFoundError:
        print("(idea_arena.jsonl not found -- run idea_arena first)")

    path = OUT_DIR / f"summary{suffix}.json"
    path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
