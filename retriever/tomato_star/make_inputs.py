"""
Stage 0b - write GFM-RAG raw inputs for tomato_test, tomato_train, tomato_mini.

documents.json : {doc_key: "title. abstract"}
{train,test}.json : [{id, question, answer, supporting_documents, stratum}]

Test  text : outputs/caches/{queries.json, corpus.json}.
Train text : data/train.jsonl  (research_question+background_survey -> question;
             each inspiration's found_title+found_abstract -> gold doc text, keyed by found_doi|found_title).
Question field = b_text (cold), matching the existing zero-shot run.
No API key needed. (The train.jsonl scan is ~1.9 GB / 107k papers, a few minutes.)
"""

import argparse
import json
import logging
import os
from collections import Counter

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("make_inputs")

QUESTION_FIELD = "b_text"  # cold query (research_question + background_survey)


def write_raw(out_root: str, name: str, documents: dict, examples: list) -> None:
    raw = os.path.join(out_root, name, "raw")
    os.makedirs(raw, exist_ok=True)
    qa = "test.json" if name.endswith("_test") else "train.json"
    json.dump(documents, open(os.path.join(raw, "documents.json"), "w"))
    json.dump(examples, open(os.path.join(raw, qa), "w"))
    log.info("[%s] wrote %d docs + %d examples -> %s", name, len(documents), len(examples), raw)
    if examples:
        s = examples[0]
        log.info("  sample example: id=%s stratum=%s gold=%s q=%r",
                 s["id"], s["stratum"], s["supporting_documents"], s["question"][:90])


def build_test(cache: str, out_root: str) -> None:
    log.info("=== tomato_test ===")
    queries = json.load(open(os.path.join(cache, "queries.json")))
    corpus = json.load(open(os.path.join(cache, "corpus.json")))
    examples, dropped = [], 0
    for q in tqdm(queries, desc="test queries"):
        g = q["gold_key"]
        if g not in corpus:
            dropped += 1
            continue
        examples.append({
            "id": q["query_id"], "question": q[QUESTION_FIELD], "answer": "",
            "supporting_documents": [g], "stratum": q["stratum"],
        })
    if dropped:
        log.warning("test: dropped %d queries (gold not in corpus)", dropped)
    log.info("test strata: %s", dict(Counter(e["stratum"] for e in examples)))
    write_raw(out_root, "tomato_test", dict(corpus), examples)


def build_train(train_jsonl: str, selection_path: str, out_root: str, mini_docs: int) -> None:
    log.info("=== tomato_train (+ tomato_mini) ===")
    sel = json.load(open(selection_path))
    chosen_golds = set(sel["gold_keys"])
    chosen_golds_lower = {g.lower(): g for g in chosen_golds}  # selection DOIs are lowercased; train.jsonl found_doi keeps original case
    qrows = sel["queries"]
    needed_sources = {q["source_id"] for q in qrows}
    log.info("need text for %d golds across %d source papers", len(chosen_golds), len(needed_sources))

    src2question: dict[str, str] = {}
    gold2text: dict[str, str] = {}
    with open(train_jsonl) as f:
        for line in tqdm(f, desc="scan train.jsonl"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            sid = r.get("source_id")
            if sid not in needed_sources:
                continue
            src2question[sid] = (r.get("research_question", "") + " " + r.get("background_survey", "")).strip()
            insp = r.get("inspiration")
            if isinstance(insp, str):
                try:
                    insp = json.loads(insp)
                except Exception:
                    insp = []
            for e in insp or []:
                raw_key = e.get("found_doi") or e.get("found_title")
                if not raw_key:
                    continue
                sel_key = chosen_golds_lower.get(raw_key.lower())  # case-insensitive match (fixes DOI casing)
                if sel_key is None:
                    continue
                title = e.get("found_title") or e.get("supposed_title") or ""
                abstract = e.get("found_abstract") or ""
                if abstract:
                    text = (title + ". " + abstract).strip()
                else:
                    text = (title or e.get("insp", "")).strip()
                if text:
                    gold2text[sel_key] = text
    log.info("resolved: %d/%d golds have text, %d/%d source questions",
             len(gold2text), len(chosen_golds), len(src2question), len(needed_sources))
    missing = chosen_golds - set(gold2text)
    if missing:
        log.warning("%d golds have no text -> their queries are dropped", len(missing))

    examples, used_golds = [], set()
    for q in tqdm(qrows, desc="train examples"):
        g, sid = q["gold_key"], q["source_id"]
        if g not in gold2text or sid not in src2question:
            continue
        examples.append({
            "id": f"{sid}::{q['step_idx']}", "question": src2question[sid], "answer": "",
            "supporting_documents": [g], "stratum": q["stratum"],
        })
        used_golds.add(g)
    documents = {g: gold2text[g] for g in used_golds}
    log.info("train strata: %s", dict(Counter(e["stratum"] for e in examples)))
    write_raw(out_root, "tomato_train", documents, examples)

    # tomato_mini: smallest validation slice (first mini_docs golds + their examples)
    mini_golds = set(sorted(used_golds)[:mini_docs])
    mini_ex = [e for e in examples if e["supporting_documents"][0] in mini_golds]
    write_raw(out_root, "tomato_mini", {g: gold2text[g] for g in mini_golds}, mini_ex)


def main() -> None:
    base = (os.environ.get("SCIGRAPHIR_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=f"{base}/outputs/caches")
    ap.add_argument("--train_jsonl", default=f"{base}/data/train.jsonl")
    ap.add_argument("--selection", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_selection.json"))
    ap.add_argument("--out", default=f"{base}/retriever/data")
    ap.add_argument("--mini_docs", type=int, default=150)
    args = ap.parse_args()

    build_test(args.cache, args.out)
    build_train(args.train_jsonl, args.selection, args.out, args.mini_docs)
    log.info("done. GFM-RAG raw inputs under %s", args.out)


if __name__ == "__main__":
    main()
