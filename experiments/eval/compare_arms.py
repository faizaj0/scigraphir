# ---- All arms, all domains, one table ----------------------------------------
# BM25 / BGE / Qwen3 / handcrafted scorer / trained fusion, scored through the SAME
# score_sir4.py call so every column is comparable. Needs the corpora unpacked;
# paste it into the transfer-matrix notebook after the unpack cell, which unpacks
# all four domains at once.
#
# Three arms are RECONSTRUCTED here rather than re-run, because everything they
# need is already cached and re-running them would burn GPU to reproduce files
# that are deterministic:
#
#   qwen3     the components npz stores dense = Q @ D.T, and cached_encode
#             normalizes, so that array IS cosine similarity. Ranking it
#             reproduces `bge_sir4.py --model /content/qwen3` exactly.
#   handcrafted scorer  S_op recomputed from the same npz plus the fitted w and beta.
#             This is handcrafted_scorer.py's formula, not an approximation.
#   bm25      lexical, so it needs no model at all. Computed from documents.json.
#
# No GPU, no encoder, no API call anywhere in this cell.
import os, csv, json, glob, math, re, subprocess, sys
from collections import Counter
import numpy as np

DOMS       = globals().get("DOMS", ["cs", "biology", "physics", "matsci"])
OP_SLUG    = globals().get("OP_SLUG", "_content-qwen3")
SCIGRAPHIR_ROOT = globals().get("SCIGRAPHIR_ROOT", "/content/scigraphir")
DRIVE      = globals().get("DRIVE", "/content/drive/MyDrive/cargo-gfmrag")
KGDIR, S4  = f"{SCIGRAPHIR_ROOT}/retriever", f"{SCIGRAPHIR_ROOT}/experiments"
DATA_ROOT  = f"{KGDIR}/data"
OUT_C      = f"{DRIVE}/outputs/_arm_comparison"
os.makedirs(OUT_C, exist_ok=True)
csv.field_size_limit(10 ** 7)

TOPK = 300                        # same depth as the fusion's predict_top_k
ARMS = ["bm25", "bge", "qwen3", 'operator', "fusion"]

# THE QWEN3 INSTRUCTION: the model card's own example task, verbatim and untuned,
# matching how BGE is run with its stock instruction.
#
# DEFINED HERE, not read from the bundle. Reading it from handcrafted_scorer.py meant
# an out-of-date bundle silently reverted the swap and reported the old baseline
# under the new name -- the cell printed the old string and carried on. The
# bundle is now CHECKED against this instead, so a stale one is loud and the
# baseline is still correct.
QWEN_QI   = ("Instruct: Given a web search query, retrieve relevant passages "
             "that answer the query\nQuery:")
QWEN3_DIR = globals().get("OP_MODEL", "/content/qwen3")

def qi_tag(instruct):
    """Instruction fingerprint for the query cache key. Without it, a changed
    instruction reloads the previous one's embeddings and changes nothing."""
    import hashlib
    return "_i" + hashlib.md5(instruct.encode()).hexdigest()[:6]

# Does the shipped code agree? If not, the handcrafted scorer and fusion arms below were
# built with a different prompt from the qwen3 arm, and that must be visible.
BUNDLE_QI = None
try:
    import importlib.util as _ilu2
    _ospec = _ilu2.spec_from_file_location("_op", f"{KGDIR}/eval/handcrafted_scorer.py")
    _op = _ilu2.module_from_spec(_ospec); _ospec.loader.exec_module(_op)
    BUNDLE_QI = _op.QWEN_QI
except Exception as e:
    print(f"could not read the bundled handcrafted_scorer ({e})")

STALE_BUNDLE = BUNDLE_QI is not None and BUNDLE_QI.strip() != QWEN_QI.strip()
print(f"Qwen3 instruction: {QWEN_QI!r}")
if STALE_BUNDLE:
    print("\n!! STALE BUNDLE: the uploaded sir4_*_bundle.zip still carries the OLD\n"
          "   Qwen3 instruction:\n"
          f"     {BUNDLE_QI!r}\n"
          "   The qwen3 row below is correct (encoded here, with the new prompt).\n"
          '   The handcrafted scorer and fusion rows are NOT on that prompt. Rebuild and\n'
          "   reupload the bundles, then rebuild components, to make them agree.\n")

# WHICH MRR? The key `mrr` means opposite things in the two scorer versions -- the
# NAME flipped, not just the definition -- so a table built across them would mix
# two metrics silently and look fine:
#
#   bundled (<= 8 Aug 17:03)   mrr = mean RR over ALL golds,  mrr_best = best gold
#   current (>= 8 Aug 18:49)   mrr = best gold,               mgrr     = all golds
#
# That is why the fusion scores.json on Drive shows MRR 0.31 while a fresh
# best-gold run shows 0.65: same model, different metric under the same name.
# hypothetical answer the scorer that will actually run and report BOTH, named by what they
# compute rather than by whichever key happens to hold them.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("_sc", f"{S4}/eval/score_sir4.py")
_sc   = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_sc)
_KEYS = set(_sc.score_one([], set(), []))
if "mgrr" in _KEYS:
    MRR_BEST, MRR_ALL, SCORER = "mrr", "mgrr", "current"
else:
    MRR_BEST, MRR_ALL, SCORER = "mrr_best", "mrr", "bundled"
print(f"scorer: {SCORER}  ->  best-gold MRR is {MRR_BEST!r}, all-gold MRR is {MRR_ALL!r}")

CKEYS  = [MRR_BEST, MRR_ALL, "ndcg@5", "recall@3", "recall@5", "recall@10", "completeset@5"]
LABELS = ["MRR(best)", "MRR(allgold)", "nDCG@5", "R@3", "R@5", "R@10", "CGS@5"]
COLS   = ",".join(CKEYS)


def sets_path(d):
    tag = "cs_test_final" if d == "cs" else f"{d}_test_low"
    return f"{SCIGRAPHIR_ROOT}/sir-4/data/benchmark/{tag}/sets.json"


def find(*cands):
    for c in cands:
        hits = sorted(glob.glob(c), key=os.path.getmtime, reverse=True)
        if hits:
            return hits[0]
    return None


def stamp_of(*paths):
    return "|".join(f"{p}@{int(os.path.getmtime(p))}" for p in paths)


def fresh(dst, stamp):
    """Cache on the INPUTS' mtimes. A filename-only cache would keep serving the
    previous run's numbers after a retrain, with no warning."""
    s = dst + ".src"
    return os.path.exists(dst) and os.path.exists(s) and open(s).read() == stamp


def write_preds(dst, stamp, recs):
    json.dump(recs, open(dst, "w"))
    open(dst + ".src", "w").write(stamp)
    return dst


def corpus_of(d):
    return json.load(open(f"{DATA_ROOT}/sir4_{d}_test/raw/documents.json"))


def queries_of(d):
    return json.load(open(f"{DATA_ROOT}/sir4_{d}_test/raw/test.json"))


def rank_to_recs(scores, doc_ids, qids, qtext):
    """[Q, D] score matrix -> score_sir4's prediction records."""
    recs = []
    for i, qid in enumerate(qids):
        order = np.argsort(-scores[i])[:TOPK]
        recs.append({"id": qid, "question": qtext.get(qid, ""),
                     "predictions": {"document": [[doc_ids[j], float(scores[i, j])]
                                                  for j in order]}})
    return recs


# ---- arm builders ------------------------------------------------------------
def zr(X):                        # row-wise over documents, as in handcrafted_scorer
    return (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-6)


def load_components(d):
    """(npz, params_json_path) for a domain, or (None, reason)."""
    g   = f"sir4_{d}_test_v16sc"
    npz = f"{DATA_ROOT}/{g}/operator_components{OP_SLUG}.npz"
    pj  = find(f"{DRIVE}/outputs/sir4_{d}/cache/operator_params_sir4_{d}{OP_SLUG}.json",
               f"{KGDIR}/eval/operator_params_sir4_{d}{OP_SLUG}.json")
    if not os.path.exists(npz):
        return None, "components npz missing"
    if not pj:
        return None, 'handcrafted scorer params missing'
    return (npz, pj), None


def doc_nodes(d):
    """Component columns are ordered to the graph's document nodes, not corpus order."""
    g = f"{DATA_ROOT}/sir4_{d}_test_v16sc/processed/stage1/nodes.csv"
    return [n["name"] for n in csv.DictReader(open(g)) if n["type"] == "document"]


def build_qwen3(d):
    """Qwen3 dense with the current (model card) instruction.

    Only the QUERIES are re-encoded: passage_instruct is null, so the cached
    document embeddings are instruction-free and valid under any prompt. About a
    minute per domain."""
    emb = f"{SCIGRAPHIR_ROOT}/outputs/caches/op_emb/sir4_{d}"
    dv  = f"{emb}/test_doc_content-qwen3.npy"
    if not os.path.exists(dv):                       # restore the encoder cache
        src = f"{DRIVE}/outputs/sir4_{d}/cache/op_emb/test_doc_content-qwen3.npy"
        if os.path.exists(src):
            os.makedirs(emb, exist_ok=True); __import__("shutil").copy(src, dv)
    if not os.path.exists(dv):
        return None, "Qwen3 doc embeddings not cached"
    if not os.path.isdir(QWEN3_DIR):
        return None, f"{QWEN3_DIR} missing (run the Qwen3 cell to enable this arm)"

    cf  = f"{DATA_ROOT}/sir4_{d}_test/raw/documents.json"
    dst = f"{OUT_C}/predictions_qwen3_sir4_{d}.json"
    # The instruction is part of the cache key, so swapping it invalidates this.
    st  = stamp_of(dv, cf) + "|" + qi_tag(QWEN_QI)
    if fresh(dst, st):
        return dst, "cached"

    from sentence_transformers import SentenceTransformer
    qs = queries_of(d)
    m  = SentenceTransformer(QWEN3_DIR); m.max_seq_length = 512
    Q  = np.asarray(m.encode([QWEN_QI + q["question"] for q in qs],
                             normalize_embeddings=True, batch_size=32,
                             show_progress_bar=False), np.float32)
    D  = np.load(dv).astype(np.float32)
    doc_ids = list(json.load(open(cf)))               # op_emb is in CORPUS order
    if len(doc_ids) != D.shape[0]:
        return None, f"doc cache is {D.shape[0]} rows but corpus is {len(doc_ids)}"
    recs = rank_to_recs(Q @ D.T, doc_ids, [q["id"] for q in qs],
                        {q["id"]: q["question"] for q in qs})
    return write_preds(dst, st, recs), "encoded with the model-card instruction"


def build_handcrafted(d):
    got, why = load_components(d)
    if not got:
        return None, why
    npz, pj = got
    dst, st = f"{OUT_C}/predictions_operator_sir4_{d}.json", stamp_of(npz, pj)
    if fresh(dst, st):
        return dst, "cached"
    P = json.load(open(pj))
    w, beta = np.asarray(P["w"], np.float32), float(P["beta"])
    z = np.load(npz, allow_pickle=True)
    enc_c, enc_p = str(z["encoder"]), str(P.get("encoder", ""))
    # w and beta were fitted on one encoder's score distributions. Applying them
    # to another's is silent nonsense, so it is checked rather than assumed.
    assert ("qwen" in enc_c.lower()) == ("qwen" in enc_p.lower()), \
        f"{d}: components={enc_c} but params={enc_p}"

    dense = z["dense"].astype(np.float32)
    S     = z["S"].astype(np.float32)
    M     = z["M"].astype(np.float32)
    dem   = np.clip(z["total_S"].astype(np.float32)[None, :] - S, 1e-6, None)
    degb  = dem ** beta
    S_op  = w[0] * zr(dense) + w[1] * zr(S / degb) + w[2] * zr(M / degb)

    docs  = doc_nodes(d)
    assert len(docs) == S_op.shape[1], f"{d}: {len(docs)} doc nodes vs {S_op.shape[1]} cols"
    qtext = {q["id"]: q["question"] for q in queries_of(d)}
    recs  = rank_to_recs(S_op, docs, [str(x) for x in z["query_ids"]], qtext)
    return write_preds(dst, st, recs), f"rebuilt ({enc_c})"


_TOK = re.compile(r"[a-z0-9]+")


def build_bm25(d, k1=1.5, b=0.75):
    """Okapi BM25, self-contained. The lexical floor: it shows how much of each
    domain is solvable by word overlap alone, which is the honest thing to read a
    dense or trained retriever against."""
    cf = f"{DATA_ROOT}/sir4_{d}_test/raw/documents.json"
    qf = f"{DATA_ROOT}/sir4_{d}_test/raw/test.json"
    dst, st = f"{OUT_C}/predictions_bm25_sir4_{d}.json", stamp_of(cf, qf)
    if fresh(dst, st):
        return dst, "cached"

    corpus  = json.load(open(cf))
    doc_ids = list(corpus)
    toks    = [_TOK.findall(corpus[i].lower()) for i in doc_ids]
    N       = len(toks)
    dl      = np.array([len(t) for t in toks], np.float32)
    avgdl   = float(dl.mean()) or 1.0

    tf, df = [], Counter()
    for t in toks:
        c = Counter(t); tf.append(c); df.update(c.keys())
    idf = {t: math.log(1 + (N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    # Inverted index with the per-posting weight precomputed, so scoring a query
    # is a few array adds rather than a pass over the corpus.
    post = {}
    for j, c in enumerate(tf):
        for t, f in c.items():
            post.setdefault(t, []).append((j, f))
    for t, lst in post.items():
        idx = np.fromiter((j for j, _ in lst), np.int32, len(lst))
        f   = np.fromiter((f for _, f in lst), np.float32, len(lst))
        post[t] = (idx, (idf[t] * f * (k1 + 1) / (f + k1 * (1 - b + b * dl[idx] / avgdl))
                         ).astype(np.float32))

    recs = []
    for q in queries_of(d):
        s = np.zeros(N, np.float32)
        for t in set(_TOK.findall(q["question"].lower())):
            p = post.get(t)
            if p is not None:
                s[p[0]] += p[1]
        order = np.argsort(-s)[:TOPK]
        recs.append({"id": q["id"], "question": q["question"],
                     "predictions": {"document": [[doc_ids[j], float(s[j])] for j in order]}})
    return write_preds(dst, st, recs), f"computed ({N} docs, {len(recs)} queries)"


# ---- scoring -----------------------------------------------------------------
def score(pred, d, arm):
    """One scorer, one flag set, so the arms are directly comparable."""
    js    = f"{OUT_C}/{arm}_sir4_{d}.json"
    stamp = stamp_of(pred)
    if os.path.exists(js):
        old = json.load(open(js))
        if old.get("_src") == stamp:
            return old
        print(f"  {d}/{arm}: predictions changed since last score, re-scoring")
    bge = find(f"{DRIVE}/outputs/sir4_{d}/predictions_bge_test.json",
               f"{DRIVE}/outputs/sir4_{d}/cache/predictions_bge_sir4_{d}_test.json",
               f"{S4}/data/predictions_bge_sir4_{d}_test.json")
    cmd = [sys.executable, "-u", "eval/score_sir4.py", "--pred", pred,
           "--queries", f"{DATA_ROOT}/sir4_{d}_test/raw/test.json",
           "--cols", COLS, "--name", f"{arm} {d}", "--json-out", js]
    if os.path.exists(sets_path(d)):
        cmd += ["--sets", sets_path(d)]
    if bge:
        cmd += ["--bge", bge]
    r = subprocess.run(cmd, cwd=S4, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:]); return None
    res = json.load(open(js)); res["_src"] = stamp
    json.dump(res, open(js, "w"), indent=1)
    return res


# ---- collect -----------------------------------------------------------------
RES, notes = {}, []
for d in DOMS:
    if not os.path.exists(f"{DATA_ROOT}/sir4_{d}_test/raw/test.json"):
        notes.append(f"{d:9} corpus not unpacked, skipped"); continue
    preds = {}

    for arm, fn in (("bm25", build_bm25), ("qwen3", build_qwen3),
                    ('operator', build_handcrafted)):
        p, why = fn(d)
        notes.append(f"{d:9} {arm:9} {why if p else 'UNAVAILABLE, ' + why}")
        if p:
            preds[arm] = p

    p = find(f"{DRIVE}/outputs/sir4_{d}/predictions_bge_test.json",
             f"{DRIVE}/outputs/sir4_{d}/cache/predictions_bge_sir4_{d}_test.json")
    notes.append(f"{d:9} {'bge':9} " + (p if p else "UNAVAILABLE, no BGE predictions"))
    if p:
        preds["bge"] = p

    # Only a qwenop run, so a stale BGE-era fusion is never reported as current.
    p = find(f"{DRIVE}/outputs/sir4_{d}/*qwenop*/predictions_sir4_{d}_test_v16sc.json")
    notes.append(f"{d:9} {'fusion':9} " + (p if p else "UNAVAILABLE, no finished qwenop run"))
    if p:
        preds["fusion"] = p

    for arm, pred in preds.items():
        r = score(pred, d, arm)
        if r:
            RES[(d, arm)] = r

for n in notes:
    print(" ", n)

# ---- table -------------------------------------------------------------------
SLICES = ["all", "same", "similar", "cross", "dissimilar"]
lines  = []
def out(s=""):
    print(s); lines.append(s)

for sl in SLICES:
    if not any((d, a) in RES and sl in RES[(d, a)] for d in DOMS for a in ARMS):
        continue
    out(f"### {sl}")
    out("| domain | arm | n | " + " | ".join(LABELS) + " |")
    out("|---|---|--:|" + "--:|" * len(CKEYS))
    for d in DOMS:
        present = [a for a in ARMS if (d, a) in RES and sl in RES[(d, a)]]
        if not present:
            continue
        best = {c: max(present, key=lambda a: RES[(d, a)][sl].get(c, -1)) for c in CKEYS}
        for a in present:
            m = RES[(d, a)][sl]
            cells = [(f"**{m.get(c, 0):.4f}**" if best[c] == a else f"{m.get(c, 0):.4f}")
                     for c in CKEYS]
            out(f"| {d} | {a} | {m['n']} | " + " | ".join(cells) + " |")
    out()

out("**bold** = best arm for that domain and metric. Every arm went through one "
    "score_sir4.py invocation with identical flags, so the columns are directly "
    "comparable.")
out()
out("Reading notes:")
out(f"- Scored with the **{SCORER}** score_sir4.py. `MRR(best)` is the rank of the "
    f"first gold; `MRR(allgold)` averages the reciprocal rank of every gold. Do "
    f"NOT compare either against a number from the other scorer version: the key "
    f"`mrr` names a different one in each.")
out("- `bm25` is the lexical floor: how far word overlap alone gets you.")
out("- `bge` defines the similar/dissimilar split, so its own `dissimilar` row is "
    "near zero by construction. That is the definition, not a result.")
out('- `qwen3` and `handcrafted` share an encoder, a cache AND an instruction, so '
    "their gap isolates what the hypothetical answer and specificity correction terms add over plain dense "
    "retrieval.")
out(f"- Qwen3 instruction: `{QWEN_QI.strip()}` -- the model card's own example "
    f"task, untuned, matching how BGE is run with its stock instruction.")
out('- CAVEAT while components are stale: `handcrafted` and `fusion` read a '
    "components npz whose `dense` term was encoded with the PREVIOUS instruction. "
    "Until those are rebuilt and the fusion retrained, they are not on the same "
    "prompt as the `qwen3` row above.")
open(f"{OUT_C}/arm_comparison.md", "w").write("\n".join(lines))
print(f"\nwrote {OUT_C}/arm_comparison.md")


# ---- Table 7.3 shape ---------------------------------------------------------
# The thesis table pairs same/cross under each metric and adds a relative gap, so
# it is built here rather than reshaped by hand. MRR is BEST-GOLD, which is what
# the existing Table 7.3 uses (BM25 physics reads 0.712 there, and best-gold is
# the only one of the two definitions in that range).
T73     = [(MRR_BEST, "MRR"), ("ndcg@5", "nDCG@5"),
           ("recall@5", "Recall@5"), ("completeset@5", "CompleteSet@5")]
DISPLAY = {"bm25": "BM25", "bge": "BGE", "qwen3": "Qwen3", 'operator': 'Handcrafted semantic scorer',
           "fusion": "SciGraphIR"}
TITLE   = {"cs": "Computer Science", "biology": "Biology",
           "physics": "Physics", "matsci": "Materials Science"}

def cell(d, a, key, sl):
    r = RES.get((d, a), {}).get(sl)
    return None if not r else r.get(key)

def gap(d, a):
    """Relative same -> cross change in Recall@5. Closer to zero is a smaller
    cross-field penalty, which is the column the claim rests on."""
    s, c = cell(d, a, "recall@5", "same"), cell(d, a, "recall@5", "cross")
    return None if not s else (c - s) / s * 100.0

tl = []
def tout(s=""):
    print(s); tl.append(s)

tout("## Table 7.3 (in-domain, same vs cross)")
tout()
tout("| Method | " + " | ".join(f"{lab} Same | {lab} Cross" for _, lab in T73)
     + " | dR@5 Gap |")
tout("|---|" + "--:|" * (2 * len(T73) + 1))
for d in DOMS:
    present = [a for a in ARMS if (d, a) in RES]
    if not present:
        continue
    tout(f"| *{TITLE.get(d, d)}* |" + " |" * (2 * len(T73) + 1))
    best = {(k, sl): max(present, key=lambda a: cell(d, a, k, sl) or -1)
            for k, _ in T73 for sl in ("same", "cross")}
    for a in present:
        row = []
        for k, _ in T73:
            for sl in ("same", "cross"):
                v = cell(d, a, k, sl)
                row.append("--" if v is None else
                           (f"**{v:.3f}**" if best[(k, sl)] == a else f"{v:.3f}"))
        g = gap(d, a)
        tout(f"| {DISPLAY.get(a, a)} | " + " | ".join(row)
             + f" | {'--' if g is None else f'{g:+.1f}%'} |")
tout()

# LaTeX, because this goes into the thesis and hand-transcribing 40 numbers per
# domain is exactly where a transcription error would enter unnoticed.
tout("```latex")
for d in DOMS:
    present = [a for a in ARMS if (d, a) in RES]
    if not present:
        continue
    tout(r"\textit{" + TITLE.get(d, d) + r"} \\")
    for a in present:
        vals = [cell(d, a, k, sl) for k, _ in T73 for sl in ("same", "cross")]
        g = gap(d, a)
        tout(DISPLAY.get(a, a) + " & "
             + " & ".join("--" if v is None else f"{v:.3f}" for v in vals)
             + " & " + ("--" if g is None else f"${g:+.1f}\\%$") + r" \\")
    tout(r"\midrule")
tout("```")

open(f"{OUT_C}/table_7_3.md", "w").write("\n".join(tl))
print(f"wrote {OUT_C}/table_7_3.md")
