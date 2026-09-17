"""
precompute_handcrafted_components.py — cache the handcrafted scorer's RAW ingredients (not the final score) so the
handcrafted scorer's weights w=[w0,w1,w2] and exponent beta can be LEARNED jointly inside FusionGraphReasoner.

The handcrafted scorer score is   S_op = w0 z(dense) + w1 z(S/dem^beta) + w2 z(M/dem^beta),  where
  dense = q . d            (query-doc similarity in the selected encoder space)
  S, M  = hypothetical answer-sum / hypothetical answer-max similarity   (HyDE bridge hypothetical answers vs docs)
  dem   = total_S - S      (leave-one-out specificity correction background matchability; total_S = S.sum over queries)
Caching dense, S, M and total_S lets the wrapper recompute S_op live with LEARNABLE w, beta (dem is
derived as total_S - S). All arrays are aligned to the graph's nodes.csv document order.

Saves data/<graph>/operator_components.npz {dense,S,M: float16 [Q x n_doc], total_S: float32 [n_doc],
query_ids:[...]}. Local, no API (reuses the op_emb caches). Mirrors precompute_operator_scores.py.
  python precompute/precompute_handcrafted_components.py --graph tomato_train_v16sc --split train
  python precompute/precompute_handcrafted_components.py --graph tomato_test_v16sc  --split test
"""
import argparse, csv, hashlib, json, os, sys
import numpy as np
from tqdm import tqdm

# SCIGRAPHIR_ROOT so this runs off-laptop (Colab unzips to /content/scigraphir). Unset,
# it resolves to exactly the path this replaced.
_ROOT = os.environ.get("SCIGRAPHIR_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
BASE = f"{_ROOT}/retriever"
sys.path.insert(0, BASE)
# One resolver for corpus + caches; default "tomato" reproduces every legacy path.
sys.path.insert(0, _ROOT)
from scigraphir_paths import add_dataset_arg, banner, corpus_dir, emb_dir, answers_path, set_dataset  # noqa: E402
csv.field_size_limit(10 ** 7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True, help="graph dir (for doc-node order), e.g. tomato_train_v16sc")
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--model", default="BAAI/bge-large-en-v1.5",
                    help="sentence-transformer used for dense and hypothetical answer similarities")
    add_dataset_arg(ap)
    a = ap.parse_args()
    set_dataset(a.dataset)
    print(banner())

    s1 = f"{BASE}/data/{a.graph}/processed/stage1"
    docnodes = [n["name"] for n in csv.DictReader(open(f"{s1}/nodes.csv")) if n["type"] == "document"]
    corpus = json.load(open(f"{corpus_dir(a.split)}/raw/documents.json"))
    doc_ids = list(corpus)
    d2i = {d: i for i, d in enumerate(doc_ids)}
    assert all(n in d2i for n in docnodes), "graph has document nodes not in documents.json"
    col = [d2i[name] for name in docnodes]              # corpus order -> nodes.csv doc order
    raw = json.load(open(f"{corpus_dir(a.split)}/raw/{a.split}.json"))
    qids = [q["id"] for q in raw]
    qkey = hashlib.md5("|".join(qids).encode()).hexdigest()[:8]

    answers = {json.loads(l)["id"]: json.loads(l).get("answers", json.loads(l).get("probes", []))
              for l in open(answers_path(a.split))}

    # ENCODE ON MISS, rather than assuming a previous run happened to use the
    # same query subset. The embedding cache is keyed by an md5 of the query
    # ids, and handcrafted_scorer.py only ever encodes its FIT and DEV samples
    # (e.g. 30 + 10 of 50, or 2,500 + 600 of 5,445). This script needs ALL of
    # them, so its hash never matched and the load died with a bare
    # FileNotFoundError naming a hash that nothing had produced.
    import importlib.util
    spec = importlib.util.spec_from_file_location("op", f"{BASE}/eval/handcrafted_scorer.py")
    op = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(op)                          # same encoder, instruction and cache key
    slug = op.model_slug(a.model)
    qi    = op.query_instruction(a.model)
    qslug = f"{slug}{op.qi_tag(qi)}"        # the instruction is part of what queries ARE
    dpath = f"{emb_dir()}/{a.split}_doc{slug}.npy"
    qpath = f"{emb_dir()}/{a.split}_query_{qkey}{qslug}.npy"
    ppath = f"{emb_dir()}/{a.split}_probe_{qkey}{slug}.npy"
    if not all(os.path.exists(p) for p in (dpath, qpath, ppath)):
        from sentence_transformers import SentenceTransformer
        print(f"[op-comp] {a.model} full-split embeddings absent for {a.split} "
              f"({len(qids)} queries); encoding once")
        model = SentenceTransformer(a.model)
        model.max_seq_length = 512
        op.cached_encode(model, [corpus[d] for d in doc_ids], f"{a.split}_doc{slug}")
        op.cached_encode(model, [q["question"] for q in raw],
                         f"{a.split}_query_{qkey}{qslug}", instruct=qi)
        flat = [pr for q in raw for pr in answers.get(q["id"], [])]
        op.cached_encode(model, flat, f"{a.split}_probe_{qkey}{slug}")

    de = np.load(dpath).astype(np.float32)
    qe = np.load(qpath).astype(np.float32)
    pe = np.load(ppath).astype(np.float32)
    own = []
    for q in raw:
        own += [q["id"]] * len(answers.get(q["id"], []))
    own = np.array(own)

    Q, D = len(raw), len(doc_ids)
    print(f"[op-comp] {Q} queries x {D} docs; computing raw components ...", flush=True)
    print(f"[op-comp] dense: one {Q}x{len(qe[0])} @ {len(de[0])}x{D} matmul ...", flush=True)
    dense = qe @ de.T                                    # [Q, D]
    S = np.zeros((Q, D), np.float32); M = np.zeros((Q, D), np.float32)
    # The hypothetical answer loop is the long pole: one matmul per query against the whole
    # corpus. It ran completely silently, so a 5,445-query train split looked
    # indistinguishable from a hang for several minutes.
    for i, q in enumerate(tqdm(raw, desc='[op-comp] answers', unit="q")):
        idx = np.where(own == q["id"])[0]
        if len(idx):
            H = np.clip(pe[idx] @ de.T, 0, None); S[i] = H.sum(0); M[i] = H.max(0)

    # reorder columns to nodes.csv document order (so they align with graph.nodes_by_type['document'])
    dense = dense[:, col]; S = S[:, col]; M = M[:, col]
    total_S = S.sum(0)                                   # [D] background matchability (for dem = total_S - S)

    # MODEL-SCOPED, because the components ARE the encoder's output. Writing every
    # encoder to one filename would let a Qwen3 run silently clobber the BGE
    # components that already-trained checkpoints were calibrated against -- and the
    # ResearchBench transfer is consuming exactly those right now. The slug is empty
    # for the default encoder, so existing paths are unchanged.
    out = f"{BASE}/data/{a.graph}/operator_components{slug}.npz"
    # Record the instruction, not just the encoder. `dense` is the only array the
    # instruction touches (hypothetical answers and documents are encoded without one), so a
    # components file built under a different instruction is a different baseline
    # wearing the same filename. Consumers can now check instead of assuming.
    np.savez_compressed(out,
                        dense=dense.astype(np.float16), S=S.astype(np.float16), M=M.astype(np.float16),
                        total_S=total_S.astype(np.float32), query_ids=np.array(qids),
                        encoder=np.array(a.model), query_instruct=np.array(qi))
    sz = os.path.getsize(out) / 1e6
    print(f"[op-comp] saved {out}  ({sz:.0f} MB)  dense/S/M={dense.shape} f16, total_S={total_S.shape} f32")
    print(f"[op-comp] env:  HANDCRAFTED_COMPONENTS{'_TEST' if a.split=='test' else ''}={out}")


if __name__ == "__main__":
    main()
