"""Rank a corpus with ReasonIR-8B.  Standalone: no repo imports, runs on a CUDA Colab.

Same encoder settings as colab_reasoning_baselines.ipynb (right-padded last-token
pooling, max_len 512, the TOMATO inspiration instruction on the query side only).

Colab:
  !pip install -q transformers accelerate
  !python reasonir_rank_colab.py --input reasonir_input_tomato.json \
        --output predictions_reasonir_tomato.json
Then download predictions_reasonir_tomato.json to
  TOMATO-Star/outputs/caches/reasonir/  and run augment_arms.

Output format = SciGraphIR prediction list:
  [{id, supporting_documents, predictions:{document:[[doc_id, score], ...]}}]
so augment_arms.py (and eval/score_* tools) read it unchanged.
~10 min on an A100 for 3k docs + 500 queries.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

REASONIR_MODEL = "reasonir/ReasonIR-8B"
REASONIR_INSTR = ("Given a scientific research question, find papers whose ideas "
                  "could have inspired the hypothesis answering it.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--model", default=REASONIR_MODEL)
    a = ap.parse_args()

    bundle = json.load(open(a.input))
    queries, docs = bundle["queries"], bundle["docs"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(a.model, trust_remote_code=True,
                                      torch_dtype=torch.bfloat16).to(dev).eval()

    def encode(texts, instruction=""):
        embs = []
        for s in tqdm(range(0, len(texts), a.batch_size), desc="encode"):
            batch = texts[s:s + a.batch_size]
            if instruction:
                batch = [f"Instruct: {instruction}\nQuery: {t}" for t in batch]
            inp = tok(batch, padding=True, truncation=True, max_length=a.max_len,
                      return_tensors="pt").to(dev)
            with torch.no_grad():
                h = model(**inp).last_hidden_state
            last = inp["attention_mask"].sum(dim=1) - 1
            pooled = h[torch.arange(h.size(0)), last]
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
            embs.append(pooled.cpu().float().numpy())
        return np.vstack(embs)

    d_emb = encode([d["text"] for d in docs])
    q_emb = encode([q["text"] for q in queries], instruction=REASONIR_INSTR)
    sims = q_emb @ d_emb.T
    top = np.argsort(-sims, axis=1)[:, :a.topk]

    out = []
    for i, q in enumerate(queries):
        ranked = [[docs[j]["id"], float(sims[i, j])] for j in top[i]]
        out.append({"id": q["id"], "supporting_documents": q.get("gold", []),
                    "predictions": {"document": ranked}})
    json.dump(out, open(a.output, "w"))

    hit1 = sum(1 for r in out if r["predictions"]["document"][0][0]
               in set(r["supporting_documents"]))
    print(f"Wrote {a.output}: {len(out)} queries; top1==gold {hit1}/{len(out)}")


if __name__ == "__main__":
    main()
