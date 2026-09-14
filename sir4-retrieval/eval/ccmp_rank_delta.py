#!/usr/bin/env python3
"""Per-query rank change between a matched CCMP / no-CCMP pair.

Both inputs are prediction files in the usual shape:
  [{"id":..., "supporting_documents":[...], "stratum":..., "predictions":{"document":[[doc,score],...]}}, ...]

Usage:
  python3 sir4-retrieval/eval/ccmp_rank_delta.py CONTROL.json CCMP.json [--top 20]

Prints the aggregate first (so nothing per-query is mistaken for a dataset
result), then the queries whose gold moved most, which are the figure
candidates.  A gold outside the stored list is reported as ">N", never as a
number, so truncated files cannot silently inflate a delta.
"""
import argparse, json, statistics as st


def ranks(path):
    out = {}
    for r in json.load(open(path)):
        pr = r["predictions"]
        ranked = [d for d, _ in (pr["document"] if isinstance(pr, dict) else pr)]
        n = len(ranked)
        golds = r.get("supporting_documents") or []
        out[r["id"]] = {
            "stratum": r.get("stratum"),
            "n": n,
            "golds": {g: (ranked.index(g) + 1 if g in ranked else None) for g in golds},
        }
    return out


def recall_at(rk, k):
    return sum(1 for v in rk.values() if v is not None and v <= k) / max(len(rk), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("control"); ap.add_argument("ccmp")
    ap.add_argument("--top", type=int, default=20)
    a = ap.parse_args()
    A, B = ranks(a.control), ranks(a.ccmp)
    shared = sorted(set(A) & set(B))
    print(f"{len(shared)} queries in both files\n")

    print("aggregate (all shared queries, multi-gold recall):")
    print(f"{'':<10}{'R@1':>8}{'R@3':>8}{'R@5':>8}{'R@10':>8}")
    for nm, D in (("control", A), ("ccmp", B)):
        flat = {}
        for q in shared:
            for g, v in D[q]["golds"].items():
                flat[(q, g)] = v
        print(f"{nm:<10}" + "".join(f"{100*recall_at(flat,k):>8.1f}" for k in (1, 3, 5, 10)))

    rowsd = []
    for q in shared:
        for g in A[q]["golds"]:
            ra, rb = A[q]["golds"][g], B[q]["golds"].get(g)
            if ra is None and rb is None:
                continue
            sa = ra if ra is not None else 10**6
            sb = rb if rb is not None else 10**6
            rowsd.append((sa - sb, q, g, ra, rb, A[q]["stratum"], A[q]["n"], B[q]["n"]))
    rowsd.sort(reverse=True)

    def show(v, n):
        return str(v) if v is not None else f">{n}"

    print(f"\ntop {a.top} improvements (control rank -> ccmp rank):")
    for d, q, g, ra, rb, strat, na, nb in rowsd[: a.top]:
        print(f"  {show(ra,na):>6} -> {show(rb,nb):<6} [{strat}]  {q}  {g[:60]}")
    print(f"\ntop {a.top} regressions:")
    for d, q, g, ra, rb, strat, na, nb in rowsd[-a.top:]:
        print(f"  {show(ra,na):>6} -> {show(rb,nb):<6} [{strat}]  {q}  {g[:60]}")

    moved = [d for d, *_ in rowsd if d != 0]
    print(f"\n{len(moved)} of {len(rowsd)} gold pairs moved; median move {st.median(moved) if moved else 0}")
    print("NOTE: per-query ranks are anecdotes. Quote the aggregate above as the result.")


if __name__ == "__main__":
    main()
