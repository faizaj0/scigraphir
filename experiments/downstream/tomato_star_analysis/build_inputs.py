"""Build per-query inputs for the BM25 vs MOOSE-Chem hypothesis-composition study.

API-FREE.  Writes results/downstream/inputs_500.jsonl: one record per stratified
query (250 same + 250 cross, the exact manifest set), carrying everything the
downstream steps need.

Record schema:
  query_id, source_id, step_idx, stratum            ("same" | "cross")
  research_question, background_survey
  prev_hypothesis                                   GT cumulative (steps 0..s-1), held
                                                    identical across both arms
  gt_delta                                          Matched-Score reference =
                                                    hypothesis_components[step_idx]
  gold_key                                          _norm_key of the gold inspiration
  bm25_top1   = {key, title, abstract, is_gold}
  moose_chem_top1 = {key, title, abstract, is_gold}
  gold_top1   = {key, title, abstract, is_gold=True} oracle/ceiling inspiration
                                                    (resolved from gold_key via meta)

Retrieval:
  BM25       -> recomputed via analysis.recall_baselines.rank_bm25 (canonical,
                deterministic; same code path as the recall table)
  MOOSE-Chem -> cached rankings (approach3-soft = the only complete 500-query run)
Top-1 = the highest-ranked corpus key that resolves to a (title, abstract); a rare
unresolvable key is skipped to the next-ranked one.

Run:
  cd TOMATO-Star
  python -m analysis.downstream.build_inputs
"""
from __future__ import annotations

import json

from analysis.config import TEST_JSONL
from analysis.recall_baselines import (
    _norm_key,
    get_stratified_queries,
    load_test_data,
    rank_bm25,
    stratum,
)
from analysis.downstream._common import INPUTS_PATH, MC_RANKINGS, PROJ

MANIFEST = PROJ / "results" / "stratified_250same_250cross_seed42.json"


def _corpus_meta_and_records():
    """key -> (title, abstract)  and  source_id -> test record."""
    meta: dict[str, tuple[str, str]] = {}
    recs: dict[str, dict] = {}
    with TEST_JSONL.open() as f:
        for line in f:
            r = json.loads(line)
            sid = r.get("source_id")
            if sid is not None:
                recs[sid] = r
            insps = r.get("inspiration", [])
            if isinstance(insps, str):
                insps = json.loads(insps)
            for ins in insps:
                key = _norm_key(ins)
                if not key:
                    continue
                title = (ins.get("found_title") or "").strip()
                abstract = (ins.get("found_abstract") or "").strip()
                meta.setdefault(key, (title, abstract))
    return meta, recs


def _top1(ranking: list[str], meta: dict[str, tuple[str, str]], gold_key: str):
    """First ranked key that resolves to non-empty (title, abstract)."""
    for k in ranking:
        kk = (k or "").strip().lower()
        if kk in meta and (meta[kk][0] or meta[kk][1]):
            title, abstract = meta[kk]
            return {"key": kk, "title": title, "abstract": abstract,
                    "is_gold": kk == gold_key}
    return None


def _prev_hypothesis(hc: list, step_idx: int) -> str:
    if step_idx <= 0:
        return "No previous hypothesis."
    prev = [hc[j] for j in range(step_idx)
            if j < len(hc) and isinstance(hc[j], str) and hc[j].strip()]
    return "\n\n".join(prev) if prev else "No previous hypothesis."


def main() -> None:
    print("Loading test data + domain labels ...")
    queries, corpus_text = load_test_data()
    sampled = get_stratified_queries(queries, 250, 250, seed=42)
    sampled_ids = [q["query_id"] for q in sampled]

    # Sanity: the sample must equal the saved manifest exactly.
    manifest_ids = set(json.loads(MANIFEST.read_text())["query_ids"])
    assert set(sampled_ids) == manifest_ids, (
        "Stratified sample does not match the saved manifest -- aborting.")
    print(f"  {len(sampled)} stratified queries (matches manifest)")

    print("Computing BM25 rankings ...")
    bm25 = rank_bm25(sampled, corpus_text)

    print("Loading MOOSE-Chem (approach3-soft) rankings ...")
    mc = json.loads(MC_RANKINGS.read_text())
    assert all(q["query_id"] in mc for q in sampled), "MOOSE-Chem cache missing queries"

    print("Indexing corpus title/abstract + test records ...")
    meta, recs = _corpus_meta_and_records()

    n_skipped = 0
    n_gold_empty = 0
    with INPUTS_PATH.open("w") as out:
        for q in sampled:
            qid = q["query_id"]
            sid, si = qid.rsplit("::", 1)
            si = int(si)
            rec = recs.get(sid)
            if rec is None:
                n_skipped += 1
                continue
            hc = rec.get("hypothesis_components", []) or []
            if not (si < len(hc) and isinstance(hc[si], str) and hc[si].strip()):
                n_skipped += 1
                continue
            gold_key = q["gold_key"]
            bm_top = _top1(bm25[qid], meta, gold_key)
            mc_top = _top1([k.strip().lower() for k in mc[qid]], meta, gold_key)
            if bm_top is None or mc_top is None:
                n_skipped += 1
                continue

            # Oracle/ceiling arm: the query's own gold inspiration, resolved to
            # (title, abstract) from the same corpus meta the other arms use.
            g_title, g_abstract = meta.get(gold_key, ("", ""))
            if not (g_title or g_abstract):
                n_gold_empty += 1
            gold_top = {"key": gold_key, "title": g_title,
                        "abstract": g_abstract, "is_gold": True}

            out.write(json.dumps({
                "query_id": qid,
                "source_id": sid,
                "step_idx": si,
                "stratum": stratum(q["src_domain"], q["gold_domain"]),
                "research_question": rec.get("research_question") or "",
                "background_survey": rec.get("background_survey") or "",
                "prev_hypothesis": _prev_hypothesis(hc, si),
                "gt_delta": hc[si].strip(),
                "gold_key": gold_key,
                "bm25_top1": bm_top,
                "moose_chem_top1": mc_top,
                "gold_top1": gold_top,
            }, ensure_ascii=False) + "\n")

    n = len(sampled) - n_skipped
    same = sum(1 for q in sampled
               if stratum(q["src_domain"], q["gold_domain"]) == "same")
    print(f"\nWrote {INPUTS_PATH}")
    print(f"  records: {n}  (skipped {n_skipped})")
    if n_gold_empty:
        print(f"  note: {n_gold_empty} gold inspiration(s) had empty title+abstract "
              f"(oracle arm degrades to no-insp for those)")
    print(f"  strata : ~{same} same / ~{len(sampled) - same} cross (before skips)")
    # Quick retrieval-accuracy sanity check (top-1 == gold).
    bm_hit = mc_hit = tot = 0
    with INPUTS_PATH.open() as f:
        for line in f:
            r = json.loads(line)
            tot += 1
            bm_hit += int(r["bm25_top1"]["is_gold"])
            mc_hit += int(r["moose_chem_top1"]["is_gold"])
    if tot:
        print(f"  top-1 == gold:  BM25 {bm_hit}/{tot} ({100*bm_hit/tot:.1f}%)  "
              f"MOOSE-Chem {mc_hit}/{tot} ({100*mc_hit/tot:.1f}%)")


if __name__ == "__main__":
    main()
