"""
run_moose_chem_sir4.py -- MOOSE-Chem screening baseline on one SIR-4 field.

Reuses TOMATO-Star's analysis/recall_moose_chem.py unchanged: its Screening wrapper, the
approach3-soft elimination schedule, the parallel one-round screening, and the tiered
survivorship ranking (screen_one_query). Only the data changes: the field's own test corpus
is the candidate pool and every query is screened against all of it, as on TOMATO.

Output: outputs/moose_chem/<field>/rankings.json  {query_id: [doi, ...]}  (full ranking,
best first), checkpointed every 10 queries and resumable. Score with score_rankings_sir4.py.

Cost. approach3-soft on N docs is sum_r ceil(N_r / W_r) calls per query with the survivor
trajectory N -> 2N/15 -> ... ; ~396 calls on 3,033 docs, so ~130 calls per 1,000 docs.
At ~2k input tokens per window call on gpt-4o-mini that is about $0.06 per 1,000 docs
per query: cs $0.26, biology $0.22, physics $0.20, matsci $0.08 per query.
--estimate prints the call count and cost for the chosen subset without calling the API.

Usage
-----
    export OPENAI_API_KEY=sk-...
    python3 llm_baselines/run_moose_chem_sir4.py --field matsci --estimate
    python3 llm_baselines/run_moose_chem_sir4.py --field matsci --limit 3          # smoke, ~$0.25
    python3 llm_baselines/run_moose_chem_sir4.py --field matsci                    # subset (default manifest)
    python3 llm_baselines/run_moose_chem_sir4.py --field cs --workers 8 --window-workers 32
    python3 llm_baselines/run_moose_chem_sir4.py --field cs --subset none          # every query
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from sir4_llm_data import OUT_ROOT, TOMATO_STAR_DIR, default_subset, load_corpus, load_queries, load_subset

sys.path.insert(0, TOMATO_STAR_DIR)
from analysis.recall_moose_chem import (  # noqa: E402  (also puts MOOSE-Chem on sys.path)
    Screening, _build_screening_args, _parallel_one_round, _write_corpus_json, screen_one_query,
)

HERE_DIR = os.path.dirname(os.path.abspath(__file__))

PRESETS = {
    "canonical":      "15:3,15:3,15:3,15:3",
    "approach3":      "15:1,2:1,10:5,10:5,5:2,10:5,5:1",
    "approach3-soft": "15:2,4:1,2:1,2:1,5:2,2:1,5:1",      # the TOMATO table row
}
USD_PER_CALL = 0.00045   # gpt-4o-mini, ~2.5k in / 200 out per 15-doc window (SIR-4 abstracts + query)


def parse_schedule(s: str) -> list[tuple[int, int]]:
    s = PRESETS.get(s, s)
    return [tuple(int(x) for x in st.split(":")) for st in s.split(",")]


def calls_per_query(n_docs: int, schedule: list[tuple[int, int]]) -> int:
    n, calls = n_docs, 0
    for w, k in schedule:
        windows = math.ceil(n / w)
        calls += windows
        n = min(n, windows * k)
    return calls


def screen_pool_query(screening, title_to_key, corpus, q, pool_ids, schedule, window_workers):
    """MOOSE-Chem's approach3-soft schedule run on a per-query candidate POOL instead of
    the whole corpus: the pool is the query's top-N from a dense retriever (best first).
    Same rounds, same prompts, same tiered survivorship ranking as screen_one_query; the
    tail after the survivors is the rest of the pool in dense order, then nothing (docs
    outside the pool were never seen by the method)."""
    cands = [[corpus[d][0], corpus[d][1]] for d in pool_ids if d in corpus]
    screening.dict_bkg2survey[q["research_question"]] = q["background_survey"]
    survivors_by_round, cur = [], cands
    for W, K in schedule:
        screen_results, cur = _parallel_one_round(screening, q["research_question"], q["background_survey"],
                                                  cur, window_workers=window_workers, window_size=W, keep_size=K)
        ordered = []
        for window in screen_results:
            for item in window:
                key = title_to_key.get(item[0])
                if key is not None and key not in ordered:
                    ordered.append(key)
        survivors_by_round.append(ordered)
    ranked, seen = [], set()
    for r in reversed(range(len(schedule))):
        for k in survivors_by_round[r]:
            if k not in seen:
                ranked.append(k); seen.add(k)
    for d in pool_ids:
        if d in corpus and d not in seen:
            ranked.append(d); seen.add(d)
    return ranked


def load_pool(path: str, n: int) -> dict[str, list[str]]:
    """{query_id: top-n doc ids, best first} from an eval/baselines_sir4.py predictions file."""
    recs = json.load(open(path))
    return {r["id"]: [d for d, _ in r["predictions"]["document"]][:n] for r in recs}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", required=True)
    ap.add_argument("--subset", default="default", help="manifest path, 'default', or 'none' for every query")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--schedule", default="approach3-soft")
    ap.add_argument("--workers", type=int, default=4, help="concurrent queries")
    ap.add_argument("--window-workers", type=int, default=32, help="concurrent window calls inside one query")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--estimate", action="store_true", help="print calls and cost, no API")
    ap.add_argument("--out", default=None)
    ap.add_argument("--pool", type=int, default=0,
                    help="screen only each query's top-N docs from --pool-pred (0 = whole corpus, the TOMATO protocol)")
    ap.add_argument("--pool-pred", default=None,
                    help="predictions file from eval/baselines_sir4.py; default data/pool_qwen3_<dataset>_test.json")
    a = ap.parse_args()

    schedule = parse_schedule(a.schedule)
    corpus = load_corpus(a.field)
    queries = load_queries(a.field)
    subset = None if a.subset == "none" else (default_subset(a.field) if a.subset == "default" else a.subset)
    queries = load_subset(subset, a.field, queries)
    if a.limit:
        queries = queries[: a.limit]
    ds = "mir" if a.field == "mir" else f"sir4_{a.field}"
    pool_pred = a.pool_pred or f"{os.path.dirname(HERE_DIR)}/data/pool_qwen3_{ds}_test.json"
    n_screen = min(a.pool, len(corpus)) if a.pool else len(corpus)
    cpq = calls_per_query(n_screen, schedule)
    print(f"{a.field}: {len(corpus)} docs, {len(queries)} queries, schedule {schedule}"
          + (f", POOL top-{a.pool} per query from {os.path.basename(pool_pred)}" if a.pool else ""))
    print(f"  {cpq} calls/query -> {cpq * len(queries):,} calls, ~${cpq * len(queries) * USD_PER_CALL:,.0f} on {a.model}")
    if a.estimate:
        return 0
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("set OPENAI_API_KEY")

    pools = None
    if a.pool:
        assert os.path.exists(pool_pred), f"no pool predictions at {pool_pred}; run make_pools.sh first"
        pools = load_pool(pool_pred, a.pool)
        missing = [q["query_id"] for q in queries if q["query_id"] not in pools]
        assert not missing, f"{len(missing)} queries have no pool entry, e.g. {missing[:3]}"
    out_dir = a.out or f"{OUT_ROOT}/moose_chem/{a.field}" + (f"_pool{a.pool}" if a.pool else "")
    os.makedirs(out_dir, exist_ok=True)
    corpus_json = f"{out_dir}/corpus_title_abstract.json"
    key_order = list(corpus)
    _write_corpus_json(corpus, key_order, __import__("pathlib").Path(corpus_json))
    title_to_key = {corpus[k][0]: k for k in key_order}
    assert len(title_to_key) == len(corpus)

    sa = _build_screening_args(api_key, a.model, corpus_json, len(schedule), schedule[0][0], schedule[0][1])
    screening = Screening(sa, custom_rq="__init__", custom_bs="")

    cache = f"{out_dir}/rankings.json"
    rankings: dict[str, list[str]] = json.load(open(cache)) if os.path.exists(cache) else {}
    pending = [q for q in queries if q["query_id"] not in rankings]
    print(f"  {len(rankings)} cached, {len(pending)} pending, workers {a.workers} x window-workers {a.window_workers}")
    meta = {"field": a.field, "model": a.model, "schedule": schedule, "subset": subset, "calls_per_query": cpq,
            "pool": a.pool, "pool_pred": pool_pred if a.pool else None}
    json.dump(meta, open(f"{out_dir}/run_meta.json", "w"), indent=1)

    lock, done, t0 = threading.Lock(), 0, time.time()

    def one(q):
        try:
            if pools is not None:
                r = screen_pool_query(screening, title_to_key, corpus, q, pools[q["query_id"]], schedule, a.window_workers)
                base = pools[q["query_id"]]
            else:
                r = screen_one_query(screening, title_to_key, q["research_question"], q["background_survey"],
                                     schedule, window_workers=a.window_workers)
                base = key_order
            # MOOSE-Chem swallows API errors inside its extraction retries; a query whose every
            # window failed comes back as the bare input order. Do not cache that as a result.
            if r[:20] == base[:20]:
                return q["query_id"], None, "no survivors recovered (API errors?); not cached, rerun to retry"
            return q["query_id"], r, None
        except Exception as e:  # noqa: BLE001
            return q["query_id"], None, repr(e)

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futs = [pool.submit(one, q) for q in pending]
        for f in as_completed(futs):
            qid, r, err = f.result()
            with lock:
                done += 1
                if err:
                    print(f"  FAIL {qid}: {err}", flush=True)
                else:
                    rankings[qid] = r
                if done % 10 == 0 or done == len(pending):
                    json.dump(rankings, open(cache, "w"))
                    el = time.time() - t0
                    print(f"  [{done}/{len(pending)}] {el/60:.1f} min, {done/el*60:.1f} q/min, "
                          f"eta {(len(pending)-done)/max(done,1)*el/60:.0f} min", flush=True)
    json.dump(rankings, open(cache, "w"))
    print(f"wrote {cache} ({len(rankings)} queries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
