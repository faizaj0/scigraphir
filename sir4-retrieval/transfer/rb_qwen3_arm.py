# ---- Qwen3-Embedding dense arm on ResearchBench (Table 7.5) ------------------
# Self-contained on purpose. Routing through bge_sir4.py would take the
# instruction from the BUNDLED cargo_operator.py, and the bundle on Drive still
# carries the OLD tuned prompt -- so this row would silently use a different
# instruction from the Qwen3 row in Table 7.3. The prompt is pinned here instead.
#
# Cost: one encode of the ResearchBench corpus (~20k docs) plus the queries.
# Cached to Drive, so a re-run is free.
import os, json, glob, hashlib, shutil
import numpy as np

DRIVE   = globals().get("DRIVE", "/content/drive/MyDrive/cargo-gfmrag")
S4      = globals().get("S4")
DATASET = globals().get("DATASET", "researchbench")
QUERIES = globals().get("QUERIES")
SETS    = globals().get("SETS")
BGE_PRED= globals().get("BGE_PRED")
OUT_T   = globals().get("OUT_T", f"{DRIVE}/outputs/researchbench/transfer")
CARGO_ROOT = globals().get("CARGO_ROOT", "/content/cargo")
assert S4 and QUERIES, "run the notebook's setup cells first (S4 / QUERIES unset)"
os.makedirs(OUT_T, exist_ok=True)

# The Qwen3-Embedding model card's own example task, verbatim. Identical to the
# string used for the Qwen3 row in Table 7.3, so the two tables are comparable.
QWEN_QI = ("Instruct: Given a web search query, retrieve relevant passages that "
           "answer the query\nQuery:")
TOPK = 300

QWEN = "/content/qwen3"
if not os.path.isdir(QWEN):
    for cand in (f"{DRIVE}/qwen3-embedding-0.6b", f"{DRIVE}/models/qwen3"):
        if os.path.isdir(cand):
            shutil.copytree(cand, QWEN); print("restored Qwen3 from", cand); break
assert os.path.isdir(QWEN), f"{QWEN} missing; run the Qwen3 cell first"

CORPUS = f"{CARGO_ROOT}/kg-construction/data/{DATASET}_test/raw/documents.json"
assert os.path.exists(CORPUS), CORPUS
corpus  = json.load(open(CORPUS))
doc_ids = list(corpus)
queries = json.load(open(QUERIES))
print(f"corpus {len(doc_ids):,} docs | {len(queries):,} queries")

# Cache keyed on the corpus/query sets AND the instruction. Without the
# instruction in the key, changing the prompt reloads the previous prompt's
# vectors and the change does nothing.
qi_tag = "_i" + hashlib.md5(QWEN_QI.encode()).hexdigest()[:6]
ckey   = hashlib.md5(("|".join(doc_ids)).encode()).hexdigest()[:8]
qkey   = hashlib.md5(("|".join(q["id"] for q in queries)).encode()).hexdigest()[:8]
CACHE  = f"{DRIVE}/outputs/researchbench/cache"
os.makedirs(CACHE, exist_ok=True)
DV = f"{CACHE}/qwen3_doc_{ckey}.npy"                 # documents carry no instruction
QV = f"{CACHE}/qwen3_query_{qkey}{qi_tag}.npy"

if not (os.path.exists(DV) and os.path.exists(QV)):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(QWEN); m.max_seq_length = 512
    if not os.path.exists(DV):
        print("encoding documents ...")
        D = np.asarray(m.encode([corpus[d] for d in doc_ids], normalize_embeddings=True,
                                batch_size=32, show_progress_bar=True), np.float32)
        np.save(DV, D)
    if not os.path.exists(QV):
        print("encoding queries ...")
        Q = np.asarray(m.encode([QWEN_QI + q["question"] for q in queries],
                                normalize_embeddings=True, batch_size=32,
                                show_progress_bar=True), np.float32)
        np.save(QV, Q)
D, Q = np.load(DV).astype(np.float32), np.load(QV).astype(np.float32)
assert D.shape[0] == len(doc_ids) and Q.shape[0] == len(queries), (D.shape, Q.shape)
print("embeddings:", D.shape, Q.shape)

# Chunked, because the full [Q x D] matrix is large and only the top-k is needed.
pred = f"{OUT_T}/predictions_qwen3_{DATASET}_test.json"
recs, k = [], min(TOPK, len(doc_ids))
for i0 in range(0, len(queries), 256):
    Sm = Q[i0:i0 + 256] @ D.T
    for r, s in enumerate(Sm):
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.argsort(-s[top])]
        q = queries[i0 + r]
        recs.append({"id": q["id"], "question": q["question"],
                     "predictions": {"document": [[doc_ids[j], float(s[j])] for j in top]}})
json.dump(recs, open(pred, "w"))
print("wrote", pred)

# Byte-identical rankings to BGE would mean the encoder never actually changed --
# the failure mode a shared embedding cache produces, and one that looks like a
# real result in the table.
if BGE_PRED and os.path.exists(BGE_PRED):
    a = hashlib.md5(open(pred, "rb").read()).hexdigest()
    b = hashlib.md5(open(BGE_PRED, "rb").read()).hexdigest()
    assert a != b, "qwen3 predictions are byte-identical to BGE's -- wrong cache loaded"
    print("qwen3 rankings differ from BGE: ok")

# SAME flags as the other Table 7.5 rows, so the columns line up.
RB_COLS = ("mrr,ndcg@5,hits@1,hits@5,recall@1,recall@3,recall@5,"
           "recall@10,recall@15,recall@20,completeset@5,completeset@10")
args = (f"--pred {pred} --queries {QUERIES} --cols {RB_COLS} --gold-strata "
        f"--name 'Qwen3-Embedding' "
        f"--json-out {OUT_T}/qwen3_scores.json "
        f"--per-query-out {OUT_T}/qwen3_perquery.json")
if SETS and os.path.exists(SETS):        args += f" --sets {SETS}"
if BGE_PRED and os.path.exists(BGE_PRED): args += f" --bge {BGE_PRED}"
rc = sh(f"python3 -u eval/score_sir4.py {args}", S4)
assert rc == 0, f"scoring failed, exit {rc}"

j = json.load(open(f"{OUT_T}/qwen3_scores.json"))
print("\nTable 7.5 row (Qwen3-Embedding, supervision = None):")
for slc in ("same", "cross"):
    if slc in j:
        print(f"  {slc:6} n={j[slc]['n']:4}  MRR {j[slc]['mrr']:.4f}  "
              f"nDCG@5 {j[slc]['ndcg@5']:.4f}")
print(f"\nwrote {OUT_T}/qwen3_scores.json and qwen3_perquery.json")
print("re-run the per-discipline cell to pick Qwen3 up: it globs *_perquery.json.")
