"""Same/cross distribution for any split, computed from the PRECOMPUTED OpenAlex domains.

The labels already exist: domains/source_domains.jsonl + domains/inspiration_domains.jsonl
both contain train rows (107,059 source papers / 195,301 inspiration steps, already resolved).
This script just JOINS source-domain + gold-domain per (source_id, step_idx) and applies
Definition A via cargo.data.stratum_cross, i.e. it is identical to how test queries.json was
labelled. NO OpenAlex / OpenAI / API calls are made -- it is a pure local join.

Run:
  python -m cargo.train_strata                      # full train distribution
  python -m cargo.train_strata --split test         # sanity-check: should reproduce ~289 cross / 2843 same
  python -m cargo.train_strata --sample 7000        # what a random 7k-PAPER graph would contain (seed 42)
  python -m cargo.train_strata --write              # also dump per-query frame -> caches/train_strata.jsonl
                                                    # (use this frame to oversample cross into a small graph)
"""
from __future__ import annotations
import argparse, json, random
from collections import Counter
from tqdm import tqdm

from .config import SRC_DOMAINS, INSP_DOMAINS, CACHE
from .data import stratum_cross, stratum_A, stratum_gold, _load_overrides


def _gold_key(row: dict) -> str | None:
    """found_doi (lowercased) else found_title (lowercased) -- identical to cargo.data._norm_key."""
    doi = (row.get("found_doi") or "").strip().lower()
    if doi:
        return doi
    title = (row.get("found_title") or "").strip().lower()
    return title or None


def load_source_domains(split: str) -> dict[str, str | None]:
    src: dict[str, str | None] = {}
    with SRC_DOMAINS.open() as f:
        for line in tqdm(f, desc=f"read source_domains [{split}]"):
            r = json.loads(line)
            if r.get("split") == split:
                src[r["source_id"]] = r.get("openalex_domain")
    nn = sum(v is not None for v in src.values())
    print(f"[src] {len(src)} source papers ({split}); {nn} with a domain, {len(src) - nn} unresolved")
    return src


def _pct(c: int, tot: int) -> str:
    return f"{100 * c / tot:.2f}%" if tot else "0%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--sample", type=int, default=0, help="randomly keep N PAPERS (0 = all); graph is per-paper")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--write", action="store_true", help="dump per-query strata -> caches/<split>_strata.jsonl")
    args = ap.parse_args()

    src_dom = load_source_domains(args.split)

    # gold-domain overrides: test's file + this split's LLM-resolved file (from cargo.resolve_train)
    overrides = _load_overrides()  # caches/gold_domain_overrides.json (test golds; keyed by gold_key)
    tgo = CACHE / f"{args.split}_gold_domain_overrides.json"
    if tgo.exists():
        ext = json.loads(tgo.read_text())
        overrides = {**overrides, **ext}
        print(f"[data] +{len(ext)} {args.split} gold-domain overrides")
    # source-domain overrides: fill papers OpenAlex left domain-null
    tso = CACHE / f"{args.split}_source_domain_overrides.json"
    if tso.exists():
        so = json.loads(tso.read_text())
        filled = 0
        for sid, d in so.items():
            if src_dom.get(sid) is None:
                src_dom[sid] = d
                filled += 1
        print(f"[src] +{filled} {args.split} source-domain overrides applied")

    keep: set[str] | None = None
    if args.sample and args.sample < len(src_dom):
        rng = random.Random(args.seed)
        keep = set(rng.sample(sorted(src_dom), args.sample))
        print(f"[sample] restricting to {len(keep)} papers (seed {args.seed}) -> estimates a graph of this size")

    strat = Counter()        # PRIMARY (binary): same / cross
    stratA = Counter()       # 3-way: same / cross / method / None
    stratG = Counter()       # gold-only (Definition B)
    src_hist = Counter()     # source-paper domain (per query)
    gold_hist = Counter()    # gold-inspiration domain (per query / per occurrence)
    cross_field = Counter()  # OpenAlex field of the distant (cross) inspirations
    gold_unique: dict[str, str | None] = {}  # gold_key -> domain (answer/corpus side, deduped)
    papers_seen: set[str] = set()
    none_gold = n = 0
    records: list[dict] = []

    with INSP_DOMAINS.open() as f:
        for line in tqdm(f, desc=f"join inspiration_domains [{args.split}]"):
            r = json.loads(line)
            if r.get("split") != args.split:
                continue
            sid = r.get("source_id")
            if keep is not None and sid not in keep:
                continue
            n += 1
            papers_seen.add(sid)
            gkey = _gold_key(r)
            sd = src_dom.get(sid)
            gd = r.get("openalex_domain") or (overrides.get(gkey) if gkey else None)
            if gd is None:
                none_gold += 1
            st = stratum_cross(sd, gd)
            sa = stratum_A(sd, gd)
            sg = stratum_gold(gd)
            strat[st] += 1; stratA[sa] += 1; stratG[sg] += 1
            src_hist[sd] += 1; gold_hist[gd] += 1
            if gkey:
                gold_unique[gkey] = gd
            if st == "cross":
                cross_field[r.get("openalex_field")] += 1
            if args.write:
                records.append({"source_id": sid, "step_idx": r.get("step_idx"),
                                "gold_key": gkey, "src_domain": sd, "gold_domain": gd,
                                "gold_field": r.get("openalex_field"),
                                "stratum": st, "stratum_A": sa, "stratum_gold": sg})

    # ---------------- report ----------------
    print("\n" + "=" * 64)
    print(f"SPLIT = {args.split}   papers = {len(papers_seen)}   queries (paper,step) = {n}")
    print("=" * 64)

    print("\n[PRIMARY stratum -- Definition A, BIO->DIST]  (matches test queries.json)")
    for k in ("same", "cross"):
        print(f"  {k:5s}: {strat.get(k,0):>7,}  ({_pct(strat.get(k,0), n)})")

    print("\n[stratum_A -- 3-way]")
    for k in ("same", "cross", "method", None):
        if stratA.get(k): print(f"  {str(k):7s}: {stratA.get(k,0):>7,}  ({_pct(stratA.get(k,0), n)})")
    print("[stratum_gold -- Definition B, gold-only (upper bound on cross)]")
    for k in ("same", "cross", None):
        if stratG.get(k): print(f"  {str(k):5s}: {stratG.get(k,0):>7,}  ({_pct(stratG.get(k,0), n)})")

    print(f"\n[unresolved] {none_gold:,} queries have no gold domain "
          f"({_pct(none_gold, n)}) -> counted as 'same' (slight cross undercount)")

    print("\n[source-paper domain (per query)]")
    for d, c in src_hist.most_common():
        print(f"  {str(d):20s}: {c:>7,}")
    print("[gold-inspiration domain (per query / occurrence)]")
    for d, c in gold_hist.most_common():
        print(f"  {str(d):20s}: {c:>7,}")

    gu = Counter(gold_unique.values())
    print(f"\n[answer side: {len(gold_unique):,} UNIQUE inspiration papers (corpus)]")
    for d, c in gu.most_common():
        print(f"  {str(d):20s}: {c:>7,}  ({_pct(c, len(gold_unique))})")

    print(f"\n[what 'cross' contains -- OpenAlex field of the {strat.get('cross',0):,} distant inspirations]")
    for fld, c in cross_field.most_common(15):
        print(f"  {str(fld):28s}: {c:>6,}")

    print("\n[reference] test (Definition A): cross 289 / same 2,843 = 9.23% cross")
    print(f"[this run]  {args.split} (Definition A): cross {strat.get('cross',0):,} / "
          f"same {strat.get('same',0):,} = {_pct(strat.get('cross',0), n)} cross")

    if args.write:
        out = CACHE / f"{args.split}_strata.jsonl"
        with out.open("w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        print(f"\n[write] {len(records):,} per-query records -> {out}")
        print("        (filter stratum=='cross' to build a cross-oversampled training graph)")


if __name__ == "__main__":
    main()
