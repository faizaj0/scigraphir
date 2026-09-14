"""Carve a bigger train/dev split from the full TOMATO-Star train (107k papers) into bundle format
(same schema as tomato_train_bundle.json: {documents:[{id,content}], examples:[{query,background,gold_ids,source_id}]}).

Each example = one train paper; gold_ids = ALL its found inspirations (multi-gold, as the original bundle).

Run:  cd CARGO && python -m cargo.build_train --n_train 30000 --n_dev 3000
Out:  outputs/caches/train_bundle_big.json , dev_bundle_big.json
"""
from __future__ import annotations
import json, random, argparse
from tqdm import tqdm
from .config import TRAIN_JSONL, CACHE


def _key(insp: dict) -> str | None:
    return (insp.get("found_doi") or "").strip().lower() or (insp.get("found_title") or "").strip().lower() or None


def _read_papers():
    papers = []
    with open(TRAIN_JSONL) as f:
        for line in tqdm(f, desc="scan train.jsonl"):
            p = json.loads(line)
            rq = (p.get("research_question") or "").strip()
            if not rq:
                continue
            ins = p.get("inspiration", [])
            if isinstance(ins, str):
                ins = json.loads(ins)
            golds = []
            for it in ins:
                k = _key(it)
                if k:
                    content = ((it.get("found_title") or "") + " " + (it.get("found_abstract") or "")).strip()
                    golds.append((k, content))
            if golds:
                papers.append((p.get("source_id"), rq, (p.get("background_survey") or ""), golds))
    return papers


def _to_bundle(subset):
    corpus, examples = {}, []
    for sid, rq, bg, golds in subset:
        gids = []
        for k, content in golds:
            corpus.setdefault(k, content)
            gids.append(k)
        examples.append({"query": rq, "background": bg, "gold_ids": gids, "source_id": sid})
    documents = [{"id": k, "content": corpus[k]} for k in corpus]
    return {"documents": documents, "examples": examples}


def main(n_train: int, n_dev: int, seed: int = 1):
    papers = _read_papers()
    print(f"[build_train] {len(papers)} usable papers")
    rng = random.Random(seed); rng.shuffle(papers)
    dev = papers[:n_dev]; train = papers[n_dev:n_dev + n_train]
    tb = _to_bundle(train); db = _to_bundle(dev)
    (CACHE / "train_bundle_big.json").write_text(json.dumps(tb))
    (CACHE / "dev_bundle_big.json").write_text(json.dumps(db))
    print(f"[build_train] train: {len(tb['examples'])} papers / {len(tb['documents'])} docs")
    print(f"[build_train] dev:   {len(db['examples'])} papers / {len(db['documents'])} docs")
    print(f"[build_train] wrote -> {CACHE}/train_bundle_big.json, dev_bundle_big.json")
    print(f"[build_train] est. CARGO LLM cost on train: probes ~${len(tb['examples'])*0.0008:.0f} "
          f"+ triples ~${(len(tb['documents'])+len(tb['examples']))*0.0006:.0f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=30000)
    ap.add_argument("--n_dev", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    main(a.n_train, a.n_dev, a.seed)
