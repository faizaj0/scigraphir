"""
HyDE baseline (Gao et al. 2022, arXiv:2212.10496) on the TOMATO test set -> predictions in the
unified schema. Zero-shot, no training.

For each query: an LLM writes a hypothetical answer passage; we encode it with the same encoder as
the BGE baseline and retrieve real documents by cosine similarity. Standard HyDE pools the
hypothetical doc(s) with the query embedding:  emb(q) = mean( enc(hyde_doc_1..n), enc(query) ).

Generation is cached/resumable (run once with your key). Same corpus / cold query (b_text) /
single-gold as BM25 / BGE / Qwen3, so the numbers are directly comparable.

Deps: openai, sentence-transformers, numpy, tqdm.
Run:  OPENAI_API_KEY=...  python eval/hyde.py --workers 32
      python eval/score.py eval/predictions_hyde.json --name HyDE
"""
import argparse, json, os, time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from openai import OpenAI

HYDE_PROMPT = (
    "Write a short scientific passage (one paragraph, like a paper abstract) that directly answers "
    "the following research question. Be specific and use precise scientific terminology. Do not hedge "
    "or say you are unsure; write it as if it were taken from a relevant paper.\n\n"
    "Question: {q}\n\nPassage:"
)


def gen_one(client, q, model, n):
    for _ in range(4):
        try:
            r = client.chat.completions.create(
                model=model, temperature=(0.7 if n > 1 else 0.0), max_tokens=300, n=n,
                messages=[{"role": "user", "content": HYDE_PROMPT.format(q=q)}])
            return [c.message.content.strip() for c in r.choices if c.message.content]
        except Exception:
            time.sleep(2)
    return [""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(os.environ.get("CARGO_ROOT") or (os.environ.get("CARGO_ROOT") or os.path.expanduser("~/Desktop/CARGO")), "outputs/caches"))
    ap.add_argument("--field", default="b_text")
    ap.add_argument("--gen_model", default="gpt-4o-mini")
    ap.add_argument("--embed_model", default="BAAI/bge-large-en-v1.5")  # match the BGE baseline
    ap.add_argument("--n", type=int, default=1, help="hypothetical docs generated per query")
    ap.add_argument("--no_query", action="store_true", help="drop the query from the embedding average")
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "predictions_hyde.json"))
    a = ap.parse_args()

    corpus = json.load(open(f"{a.cache}/corpus.json"))
    queries = json.load(open(f"{a.cache}/queries.json"))
    doc_ids = list(corpus)
    print(f"corpus {len(doc_ids)} | {len(queries)} queries | field={a.field} | "
          f"gen={a.gen_model} | embed={a.embed_model} | n={a.n} | use_query={not a.no_query}")

    # ---- 1. generate hypothetical documents (LLM, cached/resumable) ----
    gen_cache = os.path.join(a.cache, f"hyde_{a.gen_model}_n{a.n}.jsonl")
    done = {}
    if os.path.exists(gen_cache):
        for line in open(gen_cache):
            r = json.loads(line); done[r["query_id"]] = r["docs"]
    todo = [q for q in queries if q["query_id"] not in done]
    print(f"[hyde] {len(done)} cached, {len(todo)} to generate -> {gen_cache}")
    if todo:
        client = OpenAI()
        with open(gen_cache, "a") as fout, ThreadPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(gen_one, client, q[a.field], a.gen_model, a.n): q["query_id"] for q in todo}
            for fu in tqdm(as_completed(futs), total=len(futs), desc="HyDE gen"):
                qid = futs[fu]; docs = fu.result()
                fout.write(json.dumps({"query_id": qid, "docs": docs}) + "\n"); fout.flush()
                done[qid] = docs

    # ---- 2. encode corpus + (hypothetical docs [+ query]) ----
    m = SentenceTransformer(a.embed_model)
    D = np.asarray(m.encode([corpus[d] for d in doc_ids], normalize_embeddings=True,
                            batch_size=64, show_progress_bar=True), dtype=np.float32)
    spans, flat = [], []                       # one batched encode for all per-query texts
    for q in queries:
        texts = [d for d in done.get(q["query_id"], []) if d]
        if not a.no_query:
            texts = texts + [q[a.field]]
        if not texts:
            texts = [q[a.field]]
        spans.append((len(flat), len(texts))); flat.extend(texts)
    E = np.asarray(m.encode(flat, normalize_embeddings=True, batch_size=64, show_progress_bar=True),
                   dtype=np.float32)
    Q = np.zeros((len(queries), D.shape[1]), np.float32)
    for i, (off, cnt) in enumerate(spans):
        v = E[off:off + cnt].mean(0); Q[i] = v / (np.linalg.norm(v) + 1e-9)

    # ---- 3. retrieve ----
    out = []
    for qi, q in enumerate(tqdm(queries, desc="HyDE score")):
        s = D @ Q[qi]
        k = min(a.topk, len(doc_ids))
        top = np.argpartition(-s, k - 1)[:k]; top = top[np.argsort(-s[top])]
        out.append({"id": q["query_id"], "stratum": q["stratum"],
                    "supporting_documents": [q["gold_key"]],
                    "predictions": {"document": [[doc_ids[i], float(s[i])] for i in top]}})
    json.dump(out, open(a.out, "w"))
    print("wrote", a.out, "->  python eval/score.py", a.out, "--name HyDE")


if __name__ == "__main__":
    main()
