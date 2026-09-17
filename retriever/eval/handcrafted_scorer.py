"""
Handcrafted semantic scorer and shared encoder/cache helpers.

This comparison predates the learned multi-view semantic scorer. It combines
the direct query similarity with the sum and maximum of the positive
hypothetical-answer similarities:
    score = w0*z(dense) + w1*z(S/dem**beta) + w2*z(M/dem**beta)
Here dem is leave-one-out background matchability from the other queries in
the split. The weights and exponent are fitted on training data.

The thesis scorer is SortedMLPScorer with MatchabilityPredictor in
experiments/eval/semantic_scorer.py. It reuses this module's encoder settings
and cache layout; `handcrafted` remains the comparison/configuration identifier.

From the repository root, after generating hypothetical answers:
    python retriever/eval/handcrafted_scorer.py --dataset sir4_cs --train_fit 2500 --dev 600
"""
import argparse, hashlib, json, os, sys
import numpy as np
from tqdm import tqdm

# SCIGRAPHIR_ROOT so this runs off-laptop (Colab unzips to /content/scigraphir). Unset,
# it resolves to exactly the path this replaced.
_ROOT = os.environ.get("SCIGRAPHIR_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
BASE = f"{_ROOT}/retriever"
BGE_QI = "Represent this sentence for searching relevant passages: "
# The Qwen3-Embedding model card's own example task, verbatim. It replaced a
# hand-written task description that named cross-domain transfer explicitly.
# Reason: BGE is run with its stock model-card instruction, so a tuned Qwen3
# instruction made the two dense baselines non-comparable, and any margin over
# Qwen3 partly measured prompt engineering rather than method.
QWEN_QI = ("Instruct: Given a web search query, retrieve relevant passages that "
           "answer the query\nQuery:")
DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"
KS = [1, 5, 10, 25, 50, 100]

# Corpus + cache resolution lives in one module so the pipeline scripts cannot
# drift. Default dataset "tomato" reproduces every path this replaced. The
# embedding cache is the dangerous one: `{split}_doc.npy` carries no corpus
# name, so a stale TOMATO matrix would load into a SIR-4 run and pass the only
# check `cached_encode` makes (row count).
sys.path.insert(0, _ROOT)
from scigraphir_paths import (add_dataset_arg, banner, corpus_dir, emb_dir,  # noqa: E402
                         answers_path, set_dataset)


def model_slug(name):
    """Return a cache suffix so embeddings from different encoders cannot mix."""
    if name == DEFAULT_MODEL:
        return ""
    import re
    return "_" + re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def query_instruction(name):
    """Use the retrieval instruction intended for the selected encoder."""
    return QWEN_QI if "qwen" in name.lower() else BGE_QI


def qi_tag(instruct):
    """Instruction fingerprint, for the QUERY cache key only.

    WITHOUT THIS, CHANGING AN INSTRUCTION DOES NOTHING. The query cache is keyed
    on which query ids are in the split plus the model slug, and cached_encode
    validates row count and nothing else -- so a new instruction reloads the
    previous instruction's embeddings and reports them as the new result. Silent,
    and indistinguishable from "the instruction did not matter".

    Documents and hypothetical answers are encoded with no instruction at all, so only the
    query cache needs this.
    """
    return "_i" + hashlib.md5(instruct.encode()).hexdigest()[:6]


def load_split(split):
    """Return doc_ids, doc_texts, queries(list of dict id/question/gold/stratum), hypothetical answers{id:[...]}."""
    root = os.path.join(corpus_dir(split), "raw")
    corpus = json.load(open(os.path.join(root, "documents.json")))           # {doc_id: text}
    queries = json.load(open(os.path.join(root, f"{split}.json")))
    answers = {}
    for line in open(answers_path(split)):
        r = json.loads(line); answers[r["id"]] = r.get("answers", r.get("probes", []))
    return list(corpus), corpus, queries, answers


def cached_encode(model, texts, tag, instruct="", batch=64, chunk=1500):
    """Encode in chunks, releasing MPS unified-memory buffers between chunks so the encoder
    can't snowball into swap (the Apple-Silicon MPS accumulation bug). Cached to .npy."""
    import torch
    p = os.path.join(emb_dir(), f"{tag}.npy")   # emb_dir() makes the dir
    if os.path.exists(p):
        e = np.load(p)
        if e.shape[0] == len(texts):
            print(f"  [emb] loaded {tag}: {e.shape}")
            return e
        print(f"  [emb] {tag} stale ({e.shape[0]} != {len(texts)}), re-encoding")
    print(f"  [emb] encoding {tag}: {len(texts)} texts (chunk={chunk}) ...")
    parts = []
    for i in tqdm(range(0, len(texts), chunk), desc=f"  enc {tag}"):
        sub = [instruct + t for t in texts[i:i + chunk]]
        v = model.encode(sub, normalize_embeddings=True, batch_size=batch, show_progress_bar=False)
        parts.append(np.asarray(v, dtype=np.float32))
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()             # release GPU buffers so memory stays bounded
    e = np.concatenate(parts, 0)
    np.save(p, e)
    print(f"  [emb] saved {tag}: {e.shape}")
    return e


def signals(qe, de, pe, own, NQ):
    """dense/S/M/dem exactly as the handcrafted scorer notebook (section 5)."""
    dense = qe @ de.T
    ND = de.shape[0]
    S = np.zeros((NQ, ND), np.float32); M = np.zeros((NQ, ND), np.float32)
    for i in tqdm(range(NQ), desc="  signals", leave=False):
        idx = np.where(own == i)[0]
        if len(idx):
            H = np.clip(pe[idx] @ de.T, 0, None)          # [n_answers_i, ND], ReLU
            S[i] = H.sum(0); M[i] = H.max(0)
    dem = np.clip(S.sum(0, keepdims=True) - S, 1e-6, None)  # leave-one-out background matchability
    return dense.astype(np.float32), S, M, dem.astype(np.float32)


def build_signals(model, model_name, split, q_subset=None):
    """Encode (cached) and assemble signals for a split. q_subset = list of query dicts to use."""
    import hashlib
    doc_ids, corpus, queries, answers = load_split(split)
    if q_subset is not None:
        queries = q_subset
    qkey = hashlib.md5("|".join(q["id"] for q in queries).encode()).hexdigest()[:8]  # cache by WHICH queries
    slug = model_slug(model_name)
    docpos = {d: i for i, d in enumerate(doc_ids)}
    de = cached_encode(model, [corpus[d] for d in doc_ids], f"{split}_doc{slug}")
    qi = query_instruction(model_name)
    qe = cached_encode(model, [q["question"] for q in queries],
                       f"{split}_query_{qkey}{slug}{qi_tag(qi)}", instruct=qi)
    flat, own = [], []
    for i, q in enumerate(queries):
        for pr in answers.get(q["id"], []):
            flat.append(pr); own.append(i)
    pe = cached_encode(model, flat, f"{split}_probe_{qkey}{slug}")
    own = np.array(own)
    gold_idx = [[docpos[g] for g in (q.get("supporting_documents") or []) if g in docpos] for q in queries]
    strat = [q.get("stratum") for q in queries]
    sig = signals(qe, de, pe, own, len(queries))
    return sig, gold_idx, strat


def ndcg10(order, gold):
    g = set(gold)
    if not g:
        return 0.0
    dcg = sum(1 / np.log2(r + 2) for r, d in enumerate(order[:10]) if d in g)
    idcg = sum(1 / np.log2(r + 2) for r in range(min(len(g), 10)))
    return dcg / idcg if idcg else 0.0


def evaluate(score, gold_idx, strat, name):
    order = np.argsort(-score, 1)
    buck = {"all": [], "same": [], "cross": []}
    for i, gs in enumerate(gold_idx):
        if not gs:
            continue
        o = list(order[i][:200]); g = set(gs)
        row = ({k: len(set(o[:k]) & g) / len(g) for k in KS}, ndcg10(o, gs))
        buck["all"].append(row)
        if strat[i] in ("same", "cross"):
            buck[strat[i]].append(row)
    print(f"\n=== {name} ===")
    print(f"{'stratum':6} " + " ".join(f'R@{k:<4}' for k in KS) + " nDCG  n")
    out = {}
    for s in ("all", "cross", "same"):
        r = buck[s]
        if not r:
            continue
        mr = {k: 100 * np.mean([x[0][k] for x in r]) for k in KS}
        nd = 100 * np.mean([x[1] for x in r])
        out[s] = mr
        print(f"{s:6} " + " ".join(f'{mr[k]:5.1f}' for k in KS) + f" {nd:4.1f} {len(r)}")
    return out


def main():
    import torch, torch.nn as nn
    from sentence_transformers import SentenceTransformer

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--train_fit", type=int, default=2500, help="train queries used to fit the 4 params")
    ap.add_argument("--dev", type=int, default=600, help="held-out train queries for model selection")
    ap.add_argument("--epochs", type=int, default=401)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true", help="force CPU encoding (steady, no MPS swap blowup)")
    add_dataset_arg(ap)
    a = ap.parse_args()
    set_dataset(a.dataset)
    print(banner())

    dev_t = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    enc_dev = "cpu" if a.cpu else dev_t
    print(f"compute={dev_t} | encode={enc_dev} | model={a.model}")
    model = SentenceTransformer(a.model, device=enc_dev)
    model.max_seq_length = 512

    # ---- TRAIN: sample fit + dev queries (shared train corpus), build signals ----
    _, _, train_q, _ = load_split("train")
    rng = np.random.default_rng(a.seed); rng.shuffle(train_q)
    dev_q = train_q[:a.dev]
    fit_q = train_q[a.dev:a.dev + a.train_fit]
    print(f"\n[train] fit={len(fit_q)} dev={len(dev_q)} queries (corpus shared)")
    (fD, fS, fM, fDe), fgold, _ = build_signals(model, a.model, "train", q_subset=fit_q)
    (dD, dS, dM, dDe), dgold, _ = build_signals(model, a.model, "train", q_subset=dev_q)

    # ---- handcrafted scorer (4 params), InfoNCE on fit, select on dev nDCG@10 ----
    DEVt = dev_t
    T = lambda x: torch.tensor(x, device=DEVt)
    def zr(X): return (X - X.mean(1, keepdim=True)) / (X.std(1, keepdim=True) + 1e-6)
    class Operator(nn.Module):
        def __init__(s):
            super().__init__(); s.logbeta = nn.Parameter(torch.zeros(1))
            s.w = nn.Parameter(torch.tensor([1., 1., 0.2]))
        def forward(s, dense, ssum, smax, dem):
            degb = dem.clamp_min(1e-6) ** torch.exp(s.logbeta)
            return s.w[0] * zr(dense) + s.w[1] * zr(ssum / degb) + s.w[2] * zr(smax / degb)

    fDt, fSt, fMt, fDet = T(fD), T(fS), T(fM), T(fDe)
    dDt, dSt, dMt, dDet = T(dD), T(dS), T(dM), T(dDe)
    qi, gi = [], []
    for i, gs in enumerate(fgold):
        for g in gs:
            qi.append(i); gi.append(g)
    qi, gi = torch.tensor(qi, device=DEVt), torch.tensor(gi, device=DEVt)

    m = Operator().to(DEVt); opt = torch.optim.Adam(m.parameters(), lr=0.05)
    best, bsd, stale = -1, None, 0
    # Stop once the dev score stops improving. The 401 gradient steps are cheap
    # (four parameters), but each evaluation scores 600 dev queries against the
    # whole corpus and argsorts each one in Python -- that is the real cost, and
    # it runs 21 times. The fit typically plateaus by ~ep 40. Set OP_PATIENCE=0
    # to disable and run the full schedule.
    PATIENCE = int(os.environ.get("OP_PATIENCE", "3"))
    for ep in range(a.epochs):
        m.train(); opt.zero_grad()
        loss = -torch.log_softmax(m(fDt, fSt, fMt, fDet), 1)[qi, gi].mean()
        loss.backward(); opt.step()
        if ep % 20 == 0:
            with torch.no_grad():
                nd = np.mean([ndcg10(list((-m(dDt, dSt, dMt, dDet).cpu().numpy()[i]).argsort()), dgold[i])
                              for i in range(len(dgold)) if dgold[i]])
            tag = ""
            if nd > best:
                best = nd; bsd = {k: v.clone() for k, v in m.state_dict().items()}; tag = "*"
                stale = 0
            else:
                stale += 1
            print(f"ep{ep:3d} loss {loss.item():.3f} dev nDCG@10 {nd:.3f} "
                  f"beta {torch.exp(m.logbeta).item():.2f} w {m.w.detach().cpu().numpy().round(2)} {tag}")
            if PATIENCE and stale >= PATIENCE:
                print(f"[early stop] ep{ep}: no dev gain for {stale} evals "
                      f"({stale * 20} epochs); keeping best {best:.3f}")
                break
    m.load_state_dict(bsd)
    print(f"\nLEARNED beta={torch.exp(m.logbeta).item():.3f} w={m.w.detach().cpu().numpy().round(3)} best dev {best:.3f}")

    # SAVE the fitted parameters. They used to be printed and nothing else, while
    # FusionGraphReasoner warm-starts w and beta from hardcoded W_INIT/BETA_INIT
    # fitted on TOMATO. So a SIR-4 fusion run began from TOMATO's calibration and
    # this fit was thrown away. Written per-dataset so one corpus can never read
    # another's, same contract as scigraphir_paths.
    # ALSO SCOPED BY ENCODER. w and beta are fitted against one encoder's score
    # distributions, so a Qwen3 fit and a BGE fit are different calibrations of the
    # same four parameters. Sharing a filename would let a refit destroy the values
    # that already-trained checkpoints were warm-started from, with nothing to show
    # it happened. Empty slug for the default encoder keeps existing paths intact.
    sfx = ("" if a.dataset == "tomato" else f"_{a.dataset}") + model_slug(a.model)
    params = {"dataset": a.dataset, "encoder": a.model,
              "w": [round(float(x), 6) for x in m.w.detach().cpu().numpy()],
              "beta": round(float(torch.exp(m.logbeta).item()), 6),
              "dev_ndcg10": round(float(best), 6),
              "train_fit": a.train_fit, "dev": a.dev}
    pp = os.path.join(BASE, "eval", f"operator_params{sfx}.json")
    json.dump(params, open(pp, "w"), indent=1)
    print(f"[params] wrote {pp}")
    print(f"[params] fusion warm start -> W_INIT={tuple(params['w'])} BETA_INIT={params['beta']}")

    # ---- TEST: signals on test corpus, apply handcrafted scorer, score by stratum ----
    print("\n[test] building signals ...")
    (teD, teS, teM, teDe), tegold, testrat = build_signals(model, a.model, "test")
    with torch.no_grad():
        S_op = m(T(teD), T(teS), T(teM), T(teDe)).cpu().numpy()
    summary = {}
    dense_name = f"{a.model} (dense)"
    scorer_name = f"Handcrafted ({a.model}; hypothetical answers + specificity correction)"
    summary[dense_name] = evaluate(teD, tegold, testrat, dense_name)
    summary[scorer_name] = evaluate(S_op, tegold, testrat, scorer_name)
    out = os.path.join(BASE, "eval", f"operator_results{sfx}.json")
    json.dump(summary, open(out, "w"), indent=2)
    # save per-query handcrafted scorer rankings (for downstream handcrafted scorer-seeded PPR)
    te_doc_ids, _, te_q, _ = load_split("test")
    preds = []
    for i, q in enumerate(te_q):
        order = np.argsort(-S_op[i])[:100]
        preds.append({"id": q["id"], "stratum": q.get("stratum"),
                      "supporting_documents": q.get("supporting_documents", []),
                      "predictions": {"document": [[te_doc_ids[j], float(S_op[i, j])] for j in order]}})
    json.dump(preds, open(os.path.join(BASE, "eval", f"predictions_operator{sfx}.json"), "w"))
    print(f"wrote eval/predictions_operator{sfx}.json  encoder={a.model}")
    bo = summary[scorer_name]; bg = summary[dense_name]
    print(f"\nHEADLINE cross R@10:  handcrafted scorer {bo['cross'][10]:.1f}  vs  dense {bg['cross'][10]:.1f}  "
          f"(HyDE 25.3) | wrote {out}")


if __name__ == "__main__":
    main()
