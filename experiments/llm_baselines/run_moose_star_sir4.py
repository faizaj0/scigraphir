"""
run_moose_star_sir4.py -- MOOSE-Star hierarchical best-first search with gpt-4o-mini on one SIR-4 field.

Same protocol as TOMATO-Star's colab/run_stratified_gpt4omini.py (which made the TOMATO
MOOSE-Star gpt-4o-mini row): the published IRProbabilityExtractor with real top_logprobs
over the A-O labels, best-first search by geometric-mean probability, max 25 proposals,
format-error subtree pruning, OpenAI cap of 20 logprobs. Its patches are reproduced here
because that script parses its CLI at import time.

ONE CHANGE for SIR-4's 4 golds per query. The original search stops at the first proposal
that equals the single gold and reports its propose_rank. Here the search runs to
max_proposals regardless and records the proposal ORDER; the ranking scored by
score_rankings_sir4.py is that order, so recall counts every gold. On a single-gold query
this reduces to the original propose_rank exactly.

COST. Best-first search is deterministic at temperature 0, so the first k proposals do not
depend on the cap. The table needs k <= 5, so the default cap is 5 (~8 calls/query,
~$0.004/query) instead of TOMATO's 25 (~13 calls). Use --max-proposals 25 for R@10/R@25.

Output: outputs/moose_star/<field>/rankings.json {query_id: [doi x <=max_proposals]} plus
results_incremental.jsonl (one record per query with inference_calls, proposals); resumable.

Usage
-----
    python3 llm_baselines/build_moose_star_tree_sir4.py --field matsci      # once per field
    export OPENAI_API_KEY=sk-...
    python3 llm_baselines/run_moose_star_sir4.py --field matsci --limit 3   # smoke
    python3 llm_baselines/run_moose_star_sir4.py --field matsci --num-workers 16
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from sir4_llm_data import MOOSE_STAR_DIR, OUT_ROOT, default_subset, load_corpus, load_queries, load_subset

sys.path.insert(0, MOOSE_STAR_DIR)
from Inference.hierarchical_search_eval import HierarchicalSearchEvaluator  # noqa: E402
from Inference.ir_probability_extractor import (  # noqa: E402
    IRProbabilityExtractor, LABELS, SelectionResult, build_ir_prompt,
)

STATS = {"calls": 0, "fmt_err": 0, "api_err": 0, "long": 0, "tokens": 0}
SLOCK = threading.Lock()


def quiet_get_probs(self, research_question, background_survey, candidates, previous_hypothesis=None,
                    softmax_temperature=1.0, generation_temperature=0.0, max_tokens=8192, top_logprobs=20,
                    return_response=False):
    """run_stratified_gpt4omini.py PATCH 1: same call, OpenAI logprob cap, quiet."""
    valid = LABELS[: len(candidates)]
    prompt, _ = build_ir_prompt(research_question, background_survey, candidates, previous_hypothesis)
    with self._lock:
        client = self.clients[self._call_count % len(self.clients)]
        self._call_count += 1
    with SLOCK:
        STATS["calls"] += 1
    try:
        resp = client.chat.completions.create(model=self.model_name, messages=[{"role": "user", "content": prompt}],
                                              max_tokens=max_tokens, temperature=generation_temperature,
                                              logprobs=True, top_logprobs=min(top_logprobs, 20))
    except Exception as e:  # noqa: BLE001
        with SLOCK:
            STATS["api_err"] += 1
            if STATS["api_err"] <= 3:
                print(f"  [API ERROR] {str(e)[:120]}", flush=True)
        raw = {l: -1.0 for l in valid}
        probs = self._logprobs_to_probs(raw, softmax_temperature)
        return SelectionResult(probs, max(probs, key=probs.get), 0, len(candidates), raw)
    content = resp.choices[0].message.content
    ntok = resp.usage.completion_tokens if resp.usage else 0
    raw = self._extract_label_logprobs(content, resp.choices[0].logprobs, valid)
    with SLOCK:
        STATS["tokens"] += ntok
        STATS["long"] += ntok >= max_tokens - 50
        STATS["fmt_err"] += all(v == -1.0 for v in raw.values())
    probs = self._logprobs_to_probs(raw, softmax_temperature)
    sel = max(probs, key=probs.get)
    return SelectionResult(probs, sel, LABELS.index(sel), len(candidates), raw, content if return_response else None)


IRProbabilityExtractor.get_selection_probabilities = quiet_get_probs


def search_proposals(ev: HierarchicalSearchEvaluator, rq: str, bs: str, max_proposals: int) -> tuple[list[str], int]:
    """Best-first search (PATCH 3 semantics incl. format-error pruning) that does NOT stop at a
    gold: returns the ordered list of proposed leaf paper_ids and the number of IR calls."""
    frontier, expanded, proposals, calls = [], set(), [], 0
    root = ev.tree.root
    heapq.heappush(frontier, (0.0, 0, id(root), root, [root["node_id"]], 0.0))
    while frontier and len(proposals) < max_proposals:
        neg, depth, nid, cur, path, clp = heapq.heappop(frontier)
        if nid in expanded:
            continue
        expanded.add(nid)
        if cur.get("is_leaf"):
            proposals.append(cur["paper_id"])
            continue
        cands = ev.tree.get_candidates_at_node(cur)
        if len(cands) <= 1:
            if cands:
                ch = ev.tree.get_child_by_index(cur, 0)
                if ch and id(ch) not in expanded:
                    nd = depth + 1
                    heapq.heappush(frontier, (-(clp / nd), nd, id(ch), ch, path + [ch["node_id"]], clp))
            continue
        probs, _ = ev.infer_selection(research_question=rq, background_survey=bs,
                                      candidates=[{"title": c.title, "abstract": c.abstract} for c in cands],
                                      previous_hypothesis=None, node_id=cur["node_id"])
        calls += 1
        if len(set(round(p, 4) for p in probs.values())) == 1:      # format error -> prune subtree
            continue
        for label, p in probs.items():
            idx = LABELS.index(label)
            if idx < len(cands):
                ch = ev.tree.get_child_by_index(cur, idx)
                if ch and id(ch) not in expanded:
                    nlp, nd = clp + math.log(p + 1e-10), depth + 1
                    heapq.heappush(frontier, (-(nlp / nd), nd, id(ch), ch, path + [ch["node_id"]], nlp))
    return proposals, calls


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", required=True)
    ap.add_argument("--subset", default="default", help="manifest path, 'default', or 'none'")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--max-proposals", type=int, default=5, help="5 gives the same R@3/R@5/nDCG@5 as 25 (deterministic order) at ~40% fewer calls")
    ap.add_argument("--max-tokens", type=int, default=2048, help="outputs are a short reasoning + label; 2048 only trims runaways")
    ap.add_argument("--softmax-temperature", type=float, default=1.0)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("set OPENAI_API_KEY")
    import openai

    root = f"{OUT_ROOT}/moose_star/{a.field}"
    tree_dir = f"{root}/tree"
    assert os.path.exists(f"{tree_dir}/hierarchical_tree.json"), f"no tree at {tree_dir}; run build_moose_star_tree_sir4.py"
    ev = HierarchicalSearchEvaluator(tree_dir=tree_dir, sglang_urls=["http://placeholder.localhost/v1"], use_cache=True,
                                     softmax_temperature=a.softmax_temperature, max_tokens=a.max_tokens, top_logprobs=20)
    ev.extractor.clients = [openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])]
    ev.extractor.model_name = a.model

    corpus = load_corpus(a.field)
    title_to_doi = {t.lower().strip(): doi for doi, (t, _) in corpus.items()}
    pid_to_doi = {pid: title_to_doi.get(p["title"].lower().strip()) for pid, p in ev.tree.papers.items()}
    unmapped = sum(v is None for v in pid_to_doi.values())
    print(f"[tree] {len(ev.tree.papers)} papers, {unmapped} without a DOI mapping, corpus {len(corpus)}")

    subset = None if a.subset == "none" else (default_subset(a.field) if a.subset == "default" else a.subset)
    queries = load_subset(subset, a.field, load_queries(a.field))
    if a.limit:
        queries = queries[: a.limit]

    inc = f"{root}/results_incremental.jsonl"
    rank_path = f"{root}/rankings.json"
    rankings = json.load(open(rank_path)) if os.path.exists(rank_path) else {}
    done_ids = set(rankings)
    if os.path.exists(inc):
        for line in open(inc):
            if line.strip():
                r = json.loads(line)
                if r.get("error") or not r["ranking"]:
                    continue                      # failed earlier: leave it pending so this run retries it
                done_ids.add(r["query_id"])
                rankings.setdefault(r["query_id"], r["ranking"])
    pending = [q for q in queries if q["query_id"] not in done_ids]
    print(f"[run] {len(queries)} queries, {len(pending)} pending, workers {a.num_workers}, max_proposals {a.max_proposals}")
    fh, lock, t0, n = open(inc, "a"), threading.Lock(), time.time(), 0

    def one(q):
        t = time.time()
        try:
            props, calls = search_proposals(ev, q["research_question"], q["background_survey"], a.max_proposals)
            ranking = [pid_to_doi[p] for p in props if pid_to_doi.get(p)]
            gold_ranks = [ranking.index(g) + 1 for g in q["golds"] if g in ranking]
            rec = {"query_id": q["query_id"], "stratum": q["stratum"], "ranking": ranking, "inference_calls": calls,
                   "gold_ranks": gold_ranks, "search_time": time.time() - t}
            if not ranking:      # every expansion pruned (API/format errors): do not cache, rerun retries it
                rec["error"] = "no proposals (all expansions pruned)"
            return rec
        except Exception as e:  # noqa: BLE001
            return {"query_id": q["query_id"], "stratum": q["stratum"], "ranking": [], "inference_calls": 0,
                    "gold_ranks": [], "error": str(e)[:200], "search_time": time.time() - t}

    with ThreadPoolExecutor(max_workers=a.num_workers) as pool:
        for f in as_completed([pool.submit(one, q) for q in pending]):
            r = f.result()
            with lock:
                n += 1
                fh.write(json.dumps(r) + "\n"); fh.flush()
                if not r.get("error"):
                    rankings[r["query_id"]] = r["ranking"]
                hit = f"gold@{min(r['gold_ranks'])}" if r["gold_ranks"] else "miss"
                print(f"  [{n}/{len(pending)}] {r['query_id']} {hit} calls={r['inference_calls']} "
                      f"{r['search_time']:.0f}s" + (f" ERROR {r['error']}" if r.get("error") else ""), flush=True)
                if n % 10 == 0 or n == len(pending):
                    json.dump(rankings, open(rank_path, "w"))
    fh.close()
    json.dump(rankings, open(rank_path, "w"))
    el = time.time() - t0
    print(f"[done] {n} queries in {el/60:.1f} min; IR calls {STATS['calls']}, fmt_err {STATS['fmt_err']}, "
          f"api_err {STATS['api_err']}, avg output tokens {STATS['tokens']/max(STATS['calls'],1):.0f}")
    print(f"wrote {rank_path} ({len(rankings)} queries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
