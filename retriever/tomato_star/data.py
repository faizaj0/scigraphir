"""Build the full TOMATO-Star retrieval task: corpus + queries + golds + strata.

Definitions (identical to the existing TOMATO-Star/analysis/recall_baselines.py):
  query text  = research_question + " " + background_survey
  gold key    = found_doi (lowercased) else found_title (lowercased)
  corpus      = the unique gold inspirations across the test split ("title abstract")
  stratum     = 'same'  (biomedical source -> biomedical gold)
                'cross' (biomedical source -> distant gold: Physical/Social Sciences)
                None    (everything else; excluded from same/cross but still in 'all')
"""
from __future__ import annotations
import json
from pathlib import Path
from tqdm import tqdm

from .config import (TEST_JSONL, SRC_DOMAINS, INSP_DOMAINS, CACHE,
                     BIOMEDICAL, DISTANT)


def _norm_key(insp: dict) -> str | None:
    doi = (insp.get("found_doi") or "").strip().lower()
    if doi:
        return doi
    title = (insp.get("found_title") or "").strip().lower()
    return title or None


def load_domains(split: str = "test"):
    """Return (src_domain[source_id], insp_domain[(source_id, step_idx)])."""
    src_dom: dict[str, str | None] = {}
    n = 0
    with SRC_DOMAINS.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") == split:
                src_dom[r["source_id"]] = r.get("openalex_domain"); n += 1
    print(f"[domains] {n} source rows ({split})")
    insp_dom: dict[tuple[str, int], str | None] = {}
    m = 0
    with INSP_DOMAINS.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") == split:
                insp_dom[(r["source_id"], r["step_idx"])] = r.get("openalex_domain"); m += 1
    print(f"[domains] {m} inspiration rows ({split})")
    return src_dom, insp_dom


def stratum_source(src: str | None, gold: str | None) -> str | None:
    """Definition A (PRIMARY): cross = biomedical-vocabulary question -> distant-field gold.
    same  = BIO source -> BIO gold (BIO->BIO);  cross = BIO source -> distant gold (BIO->DIST).
    Method-heavy sources (DIST->*) are excluded (confounds: they share method vocab with gold).
    This is the definition that shows the real cross-domain gap (same 30.2 / cross 16.2 on BM25).
    """
    if src in BIOMEDICAL and gold in BIOMEDICAL:
        return "same"
    if src in BIOMEDICAL and gold in DISTANT:
        return "cross"
    return None


def _grp(d):
    return "BIO" if d in BIOMEDICAL else ("DIST" if d in DISTANT else None)


def stratum_cross(src: str | None, gold: str | None) -> str:
    """PRIMARY (binary, no None, pure cross):
      cross = biomedical-vocabulary question (BIO source) -> distant-field inspiration (DIST gold)
      same  = everything else
    Every query is same or cross. The cross bucket contains ONLY genuine hard cross-domain jumps
    (BIO->DIST). DIST->BIO (bio->bio) and DIST->DIST (method->method, no jump) both fall into same.
    BM25: cross ~16.2 nDCG@10 vs same ~30.6 -> the real cross-domain gap.
    """
    if _grp(src) == "BIO" and _grp(gold) == "DIST":
        return "cross"
    return "same"


def stratum_C(src: str | None, gold: str | None) -> str | None:
    """Definition C (alternative): cross = field jump (source-group != gold-group).
    Rejected as primary -- it mislabels DIST->BIO (bio->bio) as cross.
    """
    sg, gg = _grp(src), _grp(gold)
    if sg is None or gg is None:
        return None
    return "same" if sg == gg else "cross"


def stratum_A(src: str | None, gold: str | None) -> str | None:
    """Definition A (alternative, 3-way): same=BIO->BIO, cross=BIO->DIST, method=DIST->* source."""
    sg, gg = _grp(src), _grp(gold)
    if sg is None or gg is None:
        return None
    if sg == "BIO":
        return "same" if gg == "BIO" else "cross"
    return "method"


def stratum_gold(gold: str | None) -> str | None:
    """Definition B (ALTERNATIVE, NOT primary): gold-domain only. DO NOT use as the headline
    split -- it folds the easiest cell (DIST->DIST, method->method) into 'cross' and erases the gap.
    same  = gold biomedical (Life/Health);  cross = gold distant (Physical/Social).
    """
    if gold in BIOMEDICAL:
        return "same"
    if gold in DISTANT:
        return "cross"
    return None


def _load_overrides() -> dict:
    """Optional LLM-resolved gold domains for inspirations OpenAlex couldn't find."""
    p = CACHE / "gold_domain_overrides.json"
    if p.exists():
        o = json.loads(p.read_text())
        print(f"[data] gold-domain overrides applied: {len(o)}")
        return o
    return {}


def build(use_cache: bool = True):
    """Return (queries, corpus, golds).

    queries : list of dicts {query_id, source_id, step_idx, b_text, gold_key,
              src_domain, gold_domain, stratum}
    corpus  : dict insp_key -> "title abstract"
    golds   : dict query_id -> [gold_key]   (one per (paper, step) here)
    """
    qcache = CACHE / "queries.json"
    ccache = CACHE / "corpus.json"
    if use_cache and qcache.exists() and ccache.exists():
        queries = json.loads(qcache.read_text())
        corpus  = json.loads(ccache.read_text())
        golds   = {q["query_id"]: [q["gold_key"]] for q in queries}
        print(f"[data] cached: {len(queries)} queries | {len(corpus)} corpus docs")
        return queries, corpus, golds

    src_dom, insp_dom = load_domains("test")
    overrides = _load_overrides()
    queries: list[dict] = []
    corpus: dict[str, str] = {}

    n_papers = n_delta_papers = n_delta_queries = n_leak = 0
    with TEST_JSONL.open() as f:
        for line in tqdm(f, desc="build corpus+queries (test papers)"):
            paper = json.loads(line)
            sid = paper.get("source_id")
            if sid is None:
                continue
            n_papers += 1
            cold = ((paper.get("research_question") or "") + " " +
                    (paper.get("background_survey") or "")).strip()
            insps = paper.get("inspiration", [])
            if isinstance(insps, str):
                insps = json.loads(insps)
            # per-step hypothesis components (one per inspiration) -> delta context
            comps = paper.get("hypothesis_components", [])
            if isinstance(comps, str):
                try: comps = json.loads(comps)
                except Exception: comps = []
            aligned = isinstance(comps, list) and len(comps) == len(insps) and len(insps) > 0
            if aligned:
                n_delta_papers += 1
            for step_idx, insp in enumerate(insps):
                key = _norm_key(insp)
                if not key:
                    continue
                title    = (insp.get("found_title") or "").strip()
                abstract = (insp.get("found_abstract") or "").strip()
                corpus.setdefault(key, (title + " " + abstract).strip())
                sd = src_dom.get(sid)
                gd = insp_dom.get((sid, step_idx)) or overrides.get(key)  # B + override
                # ---- delta query = cold + hypothesis components from PRIOR steps only ----
                # component[step_idx] describes THIS step's gold -> excluded (no leakage).
                if aligned and step_idx > 0:
                    prior = " ".join(str(c) for c in comps[:step_idx]).strip()
                    delta = (cold + "  [HYPOTHESIS SO FAR]  " + prior).strip()
                    n_delta_queries += 1
                else:
                    delta = cold
                # leakage guard: this step's gold title must not appear in the delta context
                if title and title.lower() in delta.lower():
                    delta = cold
                    n_leak += 1
                queries.append({
                    "query_id": f"{sid}::{step_idx}",
                    "source_id": sid, "step_idx": step_idx,
                    "b_text": cold,           # cold: research_question + background_survey
                    "delta_text": delta,       # delta: cold + prior hypothesis components
                    "query_text": delta,       # PRIMARY text used for retrieval
                    "gold_key": key,
                    "src_domain": sd, "gold_domain": gd,
                    "stratum": stratum_cross(sd, gd),      # PRIMARY: cross = BIO->DIST only; same = everything else
                    "stratum_A": stratum_A(sd, gd),        # alt: 3-way same/cross/method
                    "stratum_gold": stratum_gold(gd),      # alt B: gold-domain only
                })

    golds = {q["query_id"]: [q["gold_key"]] for q in queries}
    # summary
    from collections import Counter
    strat = Counter(q["stratum"] for q in queries)
    print(f"[data] {n_papers} test papers -> {len(queries)} queries | "
          f"{len(corpus)} unique corpus docs")
    print(f"[data] strata: same={strat.get('same',0)} cross={strat.get('cross',0)} "
          f"none={strat.get(None,0)}")
    print(f"[data] DELTA: {n_delta_papers} papers aligned | {n_delta_queries} delta "
          f"queries (step>=1) | {n_leak} leakage-guarded->cold | query_text = delta")
    qcache.write_text(json.dumps(queries))
    ccache.write_text(json.dumps(corpus))
    print(f"[data] cached -> {qcache.name}, {ccache.name}")
    return queries, corpus, golds


if __name__ == "__main__":
    build(use_cache=False)
