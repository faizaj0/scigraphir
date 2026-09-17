"""Rebuild the DEV bundle to MIRROR the test construction exactly, so dev nDCG is comparable to test.

The old dev_bundle_big.json was per-PAPER multi-gold (mean 1.83 golds) -> nDCG inflated because a
cross paper's easy same-domain golds get scored too.  This rebuilds dev the SAME way data.py builds
the test set:
  - one query per (paper, step)         (NOT per paper)
  - SINGLE gold per query (that step's inspiration)
  - cold query text = research_question + background_survey
  - gold-only corpus (unique inspirations of the dev papers)
  - stratum = cross (BIO source -> DIST gold), per step  -- identical definition to test

Uses the SAME 1,500 dev papers already carved (read from the existing dev_bundle_big.json), so the
train/dev split is unchanged and disjoint.

Run:  python -m retriever.tomato_star.build_dev_like_test
Out:  outputs/caches/dev_bundle_big.json   (OVERWRITTEN, now per-step single-gold + 'stratum' field)
      outputs/caches/dev_strata.json        (index-aligned cross/same, for the notebook loader)

AFTER running, in Colab: re-upload BOTH files, DELETE the stale dev caches so they rebuild:
  emb_dev_d.npy emb_dev_q.npy emb_dev_p.npy  ee_dev.npy  triples_doc_dev.json triples_probe_dev.json
  probes_dev.json  cargo_graph_all.pkl
(train/test caches are untouched.)
"""
from __future__ import annotations
import json
from collections import Counter
from tqdm import tqdm
from .config import TRAIN_JSONL, CACHE
from .data import load_domains, stratum_cross, _norm_key


def main():
    # 1. the dev papers we already carved (keeps split fixed + disjoint from train)
    old = json.load(open(CACHE / "dev_bundle_big.json"))
    dev_sids = {e["source_id"] for e in old["examples"]}
    print(f"[dev-rebuild] {len(dev_sids)} dev papers (from existing bundle)")

    # 2. train-split domains (already on disk, no API)
    src_dom, insp_dom = load_domains("train")
    ov_path = CACHE / "gold_domain_overrides.json"
    overrides = json.load(open(ov_path)) if ov_path.exists() else {}

    # 3. per-(paper, step) single-gold queries, exactly like data.py builds test
    examples: list[dict] = []
    corpus: dict[str, str] = {}
    n_papers = 0
    with open(TRAIN_JSONL) as f:
        for line in tqdm(f, desc="scan train.jsonl for dev papers"):
            paper = json.loads(line)
            sid = paper.get("source_id")
            if sid not in dev_sids:
                continue
            n_papers += 1
            cold = ((paper.get("research_question") or "") + " " +
                    (paper.get("background_survey") or "")).strip()
            rq = (paper.get("research_question") or "").strip()
            bg = (paper.get("background_survey") or "").strip()
            insps = paper.get("inspiration", [])
            if isinstance(insps, str):
                insps = json.loads(insps)
            for step_idx, insp in enumerate(insps):
                key = _norm_key(insp)
                if not key:
                    continue
                title = (insp.get("found_title") or "").strip()
                abstract = (insp.get("found_abstract") or "").strip()
                corpus.setdefault(key, (title + " " + abstract).strip())
                sd = src_dom.get(sid)
                gd = insp_dom.get((sid, step_idx)) or overrides.get(key)
                examples.append({
                    "query": rq, "background": bg,           # load_bundle uses query+background = cold
                    "gold_ids": [key],                        # SINGLE gold per (paper, step)
                    "source_id": sid, "step_idx": step_idx,
                    "stratum": stratum_cross(sd, gd),         # cross = BIO->DIST (same as test)
                })

    documents = [{"id": k, "content": corpus[k]} for k in corpus]
    bundle = {"documents": documents, "examples": examples}
    json.dump(bundle, open(CACHE / "dev_bundle_big.json", "w"))

    # 4. index-aligned strata for the notebook loader
    strata = {str(i): e["stratum"] for i, e in enumerate(examples)}
    json.dump(strata, open(CACHE / "dev_strata.json", "w"))

    c = Counter(e["stratum"] for e in examples)
    ng = [len(e["gold_ids"]) for e in examples]
    print(f"[dev-rebuild] {n_papers} papers -> {len(examples)} queries (single-gold: "
          f"{sum(g==1 for g in ng)}/{len(ng)}) | {len(documents)} corpus docs")
    print(f"[dev-rebuild] strata: cross {c['cross']} / same {c['same']}")
    print(f"[dev-rebuild] wrote -> dev_bundle_big.json (per-step single-gold) + dev_strata.json")


if __name__ == "__main__":
    main()
