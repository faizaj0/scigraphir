"""
build_researchbench_pool.py — turn ResearchBench's inspiration-retrieval split into a
SciGraphIR-compatible corpus, and answer the two gating questions before any extraction is paid for.

  1. FEASIBILITY. How many *unique* documents are in the union of all per-query candidate
     sets? ResearchBench ships 75 candidates per query, but candidates are drawn from shared
     per-discipline pools and repeat heavily across queries, so the union -- not 1367 x 75 --
     is the affordance representation-extraction bill.

  2. CONTAMINATION. How many of those documents already appear in TOMATO-Star, which we train
     on? Run tomato_index.py first; this script consults its cache.

Contamination policy: we FLAG, we do not drop. Removing documents would change the candidate
sets and break comparability with ResearchBench's published Hit Ratio baselines, which is the
main reason to use this benchmark at all. Instead we report metrics twice -- over all queries,
and over the clean subset -- and let the gap speak. `--drop-contaminated` is available if a
strict corpus is wanted for a secondary table.

Emits, in TOMATO's raw layout so the existing hypothetical answers / handcrafted scorer / G-reasoner pipeline
runs unchanged:

  <out>/raw/documents.json   {doc_id: "Title. Abstract"}      union pool
  <out>/raw/test.json        [{id, question, supporting_documents, gold_strata, ...}]
  <out>/raw/candidates.json  {query_id: [doc_id, ...]}        Protocol A masking
  <out>/pool_report.json     feasibility + contamination + data-quality findings

Strata are per gold, not per query, matching the per-gold strict protocol used on TOMATO:
a query with one same-field and one cross-field gold contributes one row to each.  The labels
are produced by label_researchbench_domains.py from verified target and inspiration OpenAlex
records.  A query-level `domain_slice` is also exported: cross_involved when at least one gold
is verified cross-field, same_only only when every gold is verified same-field, and unlabelled
otherwise.

Usage:
  python researchbench/build_researchbench_pool.py --dry-run   # report only
  python researchbench/build_researchbench_pool.py             # write corpus
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

from norm import norm_title, norm_doi

BASE = (os.environ.get("SCIGRAPHIR_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
RB_DEFAULT = os.path.join(os.environ.get("EXTERNAL_REPOS") or os.path.join(BASE, "external"), "ResearchBench")
OUT_DEFAULT = f"{BASE}/retriever/data/researchbench"


def load_jsonl(path):
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


MIN_PREFIX_LEN = 30   # below this, titles prefix-match spuriously ("Functional Analysis")


def prefix_hits(keys, texts, min_len=MIN_PREFIX_LEN):
    """Secondary recall check: which of `keys` is a prefix of some normalised text?

    Only used for pool documents that L3 could not resolve to a DOI. L1/L2 store documents
    as 'Title. Abstract' with an unreliable separator, so their titles cannot be split off;
    prefix matching sidesteps that but over-fires on short generic titles, hence min_len.
    """
    keys = {k for k in keys if len(k) >= min_len}
    if not texts or not keys:
        return set()
    by_len = {n: set() for n in {len(k) for k in keys}}
    for t in texts:
        for n in by_len:
            if len(t) >= n:
                by_len[n].add(t[:n])
    return {k for k in keys if k in by_len[len(k)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rb", default=RB_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--index", default=None, help="tomato_index.json (default: <out>/tomato_index.json)")
    ap.add_argument("--labels", default=None,
                    help="corrected domain labels (default: <out>/domain_labels_v2.jsonl)")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    ap.add_argument("--drop-contaminated", action="store_true",
                    help="remove overlapping docs from the corpus (breaks Protocol A comparability)")
    a = ap.parse_args()
    index_path = a.index or f"{a.out}/tomato_index.json"

    retrieve_path = os.path.join(a.rb, "retrieve", "retrieve.jsonl")
    labels_path = a.labels or os.path.join(a.out, "domain_labels_v2.jsonl")
    if not os.path.exists(retrieve_path):
        sys.exit(f"missing {retrieve_path}")
    if not os.path.exists(labels_path):
        sys.exit(
            f"corrected field labels are missing: {labels_path}\n"
            "run label_researchbench_domains.py successfully before rebuilding the pool"
        )

    rows = list(load_jsonl(retrieve_path))
    print(f"[rb] {len(rows)} retrieval queries from {retrieve_path}")

    # ---- 1. union candidate pool -------------------------------------------------------
    pool = {}                       # title_key -> {"title", "abstract"}
    per_query_cands = {}            # sample_id -> [title_key]
    per_query_golds = {}            # sample_id -> [title_key]
    key_label = defaultdict(Counter)  # title_key -> {label: n}  (a doc is t1 here, t3 there)
    label_counter = Counter()
    tier_missing = Counter()
    occurrences = Counter()
    no_gold = []

    for r in rows:
        sid = r["sample_id"]
        gold_keys = {norm_title(t) for t in r.get("gold_titles", [])}
        seen_tiers = set()
        cands, golds = [], []
        for c in r.get("candidates", []):
            title = c.get("title") or ""
            key = norm_title(title)
            if not key:
                continue
            label = c.get("label", "?")
            label_counter[label] += 1
            seen_tiers.add(label)
            key_label[key][label] += 1
            if key not in pool:
                pool[key] = {"title": title, "abstract": c.get("abstract") or ""}
            elif not pool[key]["abstract"] and c.get("abstract"):
                pool[key]["abstract"] = c["abstract"]
            cands.append(key)
            occurrences[key] += 1
            if label == "gold" or key in gold_keys:
                golds.append(key)
        for tier in ("negative_t1", "negative_t2", "negative_t3"):
            if tier not in seen_tiers:
                tier_missing[tier] += 1
        per_query_cands[sid] = cands
        per_query_golds[sid] = sorted(set(golds))
        if not golds:
            no_gold.append(sid)

    n_pool = len(pool)
    reuse = Counter(occurrences.values())
    n_gold_rows = sum(len(v) for v in per_query_golds.values())

    print("\n=== FEASIBILITY ===")
    print(f"  candidate slots        {sum(len(v) for v in per_query_cands.values()):,}")
    print(f"  UNIQUE documents       {n_pool:,}   <-- affordance representation-extraction cost")
    print(f"  used by exactly 1 query {reuse.get(1, 0):,}"
          f"   ({100*reuse.get(1,0)/max(n_pool,1):.1f}%)")
    print(f"  max reuse              {max(occurrences.values(), default=0):,} queries")
    print(f"  gold rows (per-gold eval n) {n_gold_rows:,}"
          f"   mean {n_gold_rows/max(len(rows),1):.2f}/query")
    print(f"  labels                 {dict(label_counter)}")

    # ---- 2. data-quality findings for the write-up -------------------------------------
    doi_disc = defaultdict(set)
    for r in rows:
        doi_disc[norm_doi(r.get("doi"))].add(r.get("discipline"))
    multi_disc = {d: sorted(v) for d, v in doi_disc.items() if len(v) > 1}
    disc_pools = defaultdict(set)
    for r in rows:
        for c in r.get("candidates", []):
            disc_pools[r.get("discipline", "?")].add(norm_title(c.get("title")))
    sum_disc = sum(len(v) for v in disc_pools.values())

    print("\n=== DATA QUALITY (report these as benchmark limitations) ===")
    print(f"  source DOIs used under >1 discipline  {len(multi_disc):,} / {len(doi_disc):,}"
          f"  ({100*len(multi_disc)/max(len(doi_disc),1):.1f}%)")
    print(f"  queries missing a whole negative tier {dict(tier_missing)}")
    print(f"  per-discipline pools sum to {sum_disc:,} vs union {n_pool:,}"
          f"  -> {sum_disc - n_pool:,} cross-discipline pool overlap")
    print(f"  queries with no resolvable gold       {len(no_gold)}")
    print(f"  gold titles absent from candidates    "
          f"{sum(1 for r in rows for g in r.get('gold_titles', []) if norm_title(g) not in set(per_query_cands[r['sample_id']]))}")

    # ---- 3. contamination vs TOMATO-Star -----------------------------------------------
    contam = {"l1": set(), "l2": set(), "l3": set()}
    src_doi = {"l1": [], "l2": [], "l3": []}
    prefix_only = {"l1": set(), "l2": set()}
    if os.path.exists(index_path):
        idx = json.load(open(index_path))
        keys = set(pool)
        t2d = idx.get("l3_title2doi", {})
        # Primary path: exact normalised-title match against TOMATO-Star, then resolve the
        # matched paper's DOIs into the narrower layers. No fuzzy matching.
        contam["l3"] = keys & set(t2d)
        l1d, l2d = set(idx.get("l1_dois", [])), set(idx.get("l2_dois", []))
        for k in contam["l3"]:
            dois_k = set(t2d[k])
            if dois_k & l1d:
                contam["l1"].add(k)
            if dois_k & l2d:
                contam["l2"].add(k)
        # Secondary: pool docs L3 never saw could still sit in L1/L2 under a title we
        # cannot resolve. Prefix-match those only, with a length floor.
        unresolved = keys - contam["l3"]
        for lv, texts in (("l1", idx.get("l1_texts", [])), ("l2", idx.get("l2_texts", []))):
            prefix_only[lv] = prefix_hits(unresolved, texts)
            contam[lv] |= prefix_only[lv]
        rb_dois = {norm_doi(r.get("doi")) for r in rows} - {""}
        for lv in ("l1", "l2", "l3"):
            src_doi[lv] = sorted(rb_dois & set(idx.get(f"{lv}_dois", [])))
        print(f"\n[index] {index_path}")
        print(f"        exact title matches vs TOMATO-Star: {len(contam['l3']):,}"
              f"   + prefix-only additions L1 {len(prefix_only['l1'])}, L2 {len(prefix_only['l2'])}")
    else:
        print(f"\n[index] !! {index_path} not found -- run tomato_index.py first.")
        print("        Contamination is UNVERIFIED; do not claim zero-shot on this output.")

    gold_keys_all = {k for g in per_query_golds.values() for k in g}
    print("\n=== CONTAMINATION (ResearchBench pool vs TOMATO-Star) ===")
    print(f"  {'layer':<28} {'pool docs':>10} {'of which gold':>14} {'queries hit':>12} {'src DOIs':>9}")
    contam_report = {}
    for lv, desc in (("l1", "L1 SciAfford graph corpus"),
                     ("l2", "L2 training bundle"),
                     ("l3", "L3 full TOMATO-Star train")):
        hits = contam[lv]
        g = hits & gold_keys_all
        qs = sorted(sid for sid, gs in per_query_golds.items() if any(x in hits for x in gs))
        print(f"  {desc:<28} {len(hits):>10,} {len(g):>14,} {len(qs):>12,} {len(src_doi[lv]):>9,}")
        contam_report[lv] = {"desc": desc, "pool_docs": len(hits), "gold_docs": len(g),
                             "queries_with_leaked_gold": qs, "source_doi_overlap": src_doi[lv]}
    if contam["l3"] & gold_keys_all:
        print("  example leaked golds:")
        for k in list(contam["l3"] & gold_keys_all)[:5]:
            print(f"    - {pool[k]['title'][:76]}")

    # ---- 4. per-gold strata from verified paper-to-paper OpenAlex labels ----------------
    verdicts = {}
    if os.path.exists(labels_path):
        for row in load_jsonl(labels_path):
            verdicts[(row["source_paper_id"], norm_title(row["inspiration_title"]))] = row.get("verdict")
        print(f"\n[strata] {len(verdicts):,} verified gold field labels from {labels_path}")
    else:  # guarded above; retained as a defensive check for concurrent deletion
        sys.exit(f"corrected field labels disappeared during construction: {labels_path}")

    gold_strata = Counter()
    for sid, gs in per_query_golds.items():
        for k in gs:
            gold_strata[verdicts.get((sid, k), "unknown")] += 1
    labelled = sum(v for k, v in gold_strata.items() if k in ("same", "cross"))
    print(f"  per-gold strata: {dict(gold_strata)}")
    if labelled:
        print(f"  cross-field rate among labelled golds: "
              f"{100*gold_strata['cross']/labelled:.1f}%   (TOMATO test: 9.2%)")

    if a.dry_run:
        print("\n[dry-run] nothing written")
        return

    # ---- 5. emit TOMATO-format corpus ---------------------------------------------------
    raw = os.path.join(a.out, "raw")
    os.makedirs(raw, exist_ok=True)
    drop = contam["l3"] if a.drop_contaminated else set()
    keep = sorted(k for k in pool if k not in drop)
    key2id = {k: f"rb::{i:06d}" for i, k in enumerate(keep)}

    json.dump({key2id[k]: f"{pool[k]['title']}. {pool[k]['abstract']}" for k in keep},
              open(f"{raw}/documents.json", "w"))

    test, candidates = [], {}
    for r in rows:
        sid = r["sample_id"]
        golds = [k for k in per_query_golds[sid] if k in key2id]
        if not golds:
            continue
        q = (r.get("research_question") or "").strip()
        bg = (r.get("background_survey") or "").strip()
        strata = [verdicts.get((sid, k), "unknown") for k in golds]
        if "cross" in strata:
            domain_slice = "cross_involved"
        elif strata and all(s == "same" for s in strata):
            domain_slice = "same_only"
        else:
            domain_slice = "unlabelled"
        test.append({
            "id": sid,
            "question": f"{q}\n\n{bg}".strip(),
            "research_question": q,
            "answer": "",
            "supporting_documents": [key2id[k] for k in golds],
            "gold_strata": strata,
            "domain_slice": domain_slice,
            "gold_contaminated": [k in contam["l3"] for k in golds],
            "discipline": r.get("discipline", "?"),
            "doi": r.get("doi", ""),
        })
        candidates[sid] = [key2id[k] for k in per_query_cands[sid] if k in key2id]

    json.dump(test, open(f"{raw}/test.json", "w"))
    json.dump(candidates, open(f"{raw}/candidates.json", "w"))
    json.dump({
        "queries": len(test),
        "pool_unique": n_pool,
        "pool_written": len(keep),
        "dropped_contaminated": len(drop),
        "gold_rows": sum(len(t["supporting_documents"]) for t in test),
        "labels": dict(label_counter),
        "gold_strata": dict(gold_strata),
        "domain_slices": dict(Counter(t["domain_slice"] for t in test)),
        "domain_labels_path": os.path.realpath(labels_path),
        "contamination": contam_report,
        "data_quality": {
            "source_dois_multi_discipline": len(multi_disc),
            "source_dois_total": len(doi_disc),
            "queries_missing_tier": dict(tier_missing),
            "per_discipline_pool_sum": sum_disc,
            "queries_no_gold": no_gold,
        },
        "disciplines": dict(Counter(t["discipline"] for t in test)),
        "index_path": index_path if os.path.exists(index_path) else None,
    }, open(f"{a.out}/pool_report.json", "w"), indent=1)

    print(f"\n=== WROTE {a.out} ===")
    print(f"  raw/documents.json   {len(keep):,} docs"
          + (f"  (dropped {len(drop):,} contaminated)" if drop else "  (contaminated flagged, not dropped)"))
    print(f"  raw/test.json        {len(test):,} queries / "
          f"{sum(len(t['supporting_documents']) for t in test):,} gold rows")
    print(f"  raw/candidates.json  Protocol A masking sets")
    print(f"  pool_report.json")


if __name__ == "__main__":
    main()
