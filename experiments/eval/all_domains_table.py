# ---- ONE CLEAN TABLE: every domain, every arm, correct MRR --------------------
# Rescores the predictions already on Drive. No retraining, no re-encoding, no GPU.
#
# WHY THIS CELL INLINES THE SCORER instead of calling eval/score_sir4.py: the
# bundles currently on Drive ship a version where `mrr` is the mean reciprocal
# rank over ALL golds. That is MGRR, a multi-gold diagnostic, not conventional
# MRR -- which is why the qwenop runs report 0.35 while the older runs report
# 0.75 for a model that is not worse. Inlining pins the definition to this cell,
# so the table cannot silently depend on which bundle a runtime happens to hold.
#
# Reports BOTH: `MRR` (rank of the first relevant doc, the standard) and `MGRR`
# (mean over all golds, what the earlier tables printed), so old numbers stay
# traceable rather than being quietly replaced.
import os, re, csv, json, glob, math, zipfile, shutil
from collections import defaultdict

DRIVE      = globals().get("DRIVE", "/content/drive/MyDrive/cargo-gfmrag")
SCIGRAPHIR_ROOT = globals().get("SCIGRAPHIR_ROOT", "/content/scigraphir")
DATA_ROOT  = globals().get("DATA_ROOT", f"{SCIGRAPHIR_ROOT}/retriever/data")
DOMAINS    = ["cs", "biology", "physics", "matsci"]
OUT        = f"{DRIVE}/outputs/_all_domains"
os.makedirs(OUT, exist_ok=True)
BIG = 10 ** 9

# --- scorer, byte-for-byte the semantics of the current eval/score_sir4.py -----
def ranked_docs(rec):
    p = rec.get("predictions", rec)
    docs = p.get("document", p) if isinstance(p, dict) else p
    return [d[0] if isinstance(d, (list, tuple)) else d for d in docs]

def best_gold_rank(ranked, golds):
    for i, d in enumerate(ranked, 1):
        if d in golds:
            return i
    return BIG

def score_one(ranked, golds, sets_):
    r   = best_gold_rank(ranked, golds)
    pos = {d: i + 1 for i, d in enumerate(ranked)}
    rks = [pos.get(g, BIG) for g in golds]
    out = {"mrr":  1.0 / r if r < BIG else 0.0,
           "mgrr": (sum(1.0 / x for x in rks if x < BIG) / len(golds)) if golds else 0.0}
    for k in (1, 3, 5, 10, 20, 100):
        topk = set(ranked[:k])
        out[f"recall@{k}"]      = len(topk & golds) / len(golds) if golds else 0.0
        out[f"hits@{k}"]        = float(r <= k)
        out[f"completeset@{k}"] = float(any(set(s) <= topk for s in sets_)) if sets_ else 0.0
    rel  = [float(d in golds) for d in ranked[:5]]
    dcg  = sum(v / math.log2(i + 2) for i, v in enumerate(rel))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(golds), 5)))
    out["ndcg@5"] = dcg / idcg if idcg else 0.0
    return out

# --- inputs, unpacking any bundle this runtime does not already have ----------
def sets_path(d):
    tag = "cs_test_final" if d == "cs" else f"{d}_test_low"
    return f"{SCIGRAPHIR_ROOT}/sir-4/data/benchmark/{tag}/sets.json"

def ensure(d):
    """Return (queries, sets) for a domain, or None if its bundle is unavailable."""
    q = f"{DATA_ROOT}/sir4_{d}_test/raw/test.json"
    if not os.path.exists(q):
        z = f"{DRIVE}/sir4_{d}_bundle.zip"
        if not os.path.exists(z):
            return None
        zipfile.ZipFile(z).extractall(SCIGRAPHIR_ROOT)
    if not os.path.exists(q):
        return None
    queries = {r["id"]: r for r in json.load(open(q))}
    sets_by_q = {}
    sp = sets_path(d)
    if os.path.exists(sp):
        raw = json.load(open(sp))
        for qid, v in raw.items():
            sets_by_q[qid] = [list(s) for s in (v.get("sets") if isinstance(v, dict) else v)]
    else:
        print(f"  ! {d}: {sp} missing, CompleteSet@k will read 0")
    return queries, sets_by_q

# --- arm discovery ------------------------------------------------------------
# Every arm is a predictions file. The run-directory name is the only record of
# which encoder the handcrafted scorer used, so it is also the arm label.
BANNED = ("smoke", "routed", "leverd", "lossv2", "nodistill")

def arms_for(d):
    out = []
    for src in (f"{DRIVE}/outputs/sir4_{d}/cache/predictions_bge_sir4_{d}_test.json",
                f"{DRIVE}/outputs/sir4_{d}/predictions_bge_test.json"):
        if os.path.exists(src):
            out.append(("BGE dense", src)); break
    for p in sorted(glob.glob(f"{DRIVE}/outputs/sir4_{d}/*/predictions_sir4_{d}_test_v16sc.json"),
                    key=os.path.getmtime):
        run = os.path.basename(os.path.dirname(p))
        if any(b in run.lower() for b in BANNED):
            continue
        out.append(("fusion Qwen3-op" if "qwenop" in run.lower() else "fusion BGE-op", p))
    # A domain retrained more than once keeps only its newest run per label, so the
    # table has one column per arm rather than one per attempt.
    keep = {}
    for label, p in out:
        keep[label] = p
    return keep

# --- score --------------------------------------------------------------------
RES = {}          # RES[domain][arm][slice] = {metric: value}
for d in DOMAINS:
    got = ensure(d)
    if not got:
        print(f"{d:9} no corpus and no bundle on Drive, skipped"); continue
    queries, sets_by_q = got
    arms = arms_for(d)
    if not arms:
        print(f"{d:9} no predictions on Drive yet"); continue

    # similar/dissimilar needs the dense baseline: a query is `dissimilar` when
    # plain BGE ranks its BEST gold below 100.
    bge_rank = {}
    if "BGE dense" in arms:
        for rec in json.load(open(arms["BGE dense"])):
            q = queries.get(rec["id"])
            if q:
                bge_rank[rec["id"]] = best_gold_rank(ranked_docs(rec),
                                                     set(q["supporting_documents"]))
    else:
        print(f"{d:9} no BGE baseline, similar/dissimilar slices skipped")

    RES[d] = {}
    for label, path in arms.items():
        preds = json.load(open(path))
        if isinstance(preds, dict):
            preds = list(preds.values())
        slices, seen = defaultdict(list), set()
        for rec in preds:
            qid = rec.get("id"); q = queries.get(qid)
            if q is None or qid in seen:
                continue
            seen.add(qid)
            m = score_one(ranked_docs(rec), set(q["supporting_documents"]),
                          sets_by_q.get(qid, []))
            names = ["all"] + ([q["stratum"]] if q.get("stratum") else [])
            if qid in bge_rank:
                names.append("dissimilar" if bge_rank[qid] > 100 else "similar")
            for n in names:
                slices[n].append(m)
        RES[d][label] = {s: {"n": len(rows),
                             **{k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}}
                         for s, rows in slices.items()}
        print(f"{d:9} {label:16} {len(seen):5} queries  ({os.path.basename(os.path.dirname(path))})")

assert RES, f"nothing scored: no predictions found under {DRIVE}/outputs/sir4_*"

# --- tables -------------------------------------------------------------------
ARMS    = ["BGE dense", "fusion BGE-op", "fusion Qwen3-op"]
METRICS = [("mrr", "MRR"), ("mgrr", "MGRR"), ("ndcg@5", "nDCG@5"),
           ("recall@3", "R@3"), ("recall@5", "R@5"), ("completeset@5", "CS@5")]
lines = []
def out(s=""):
    print(s); lines.append(s)

out("MRR  = reciprocal rank of the FIRST relevant document (standard).")
out("MGRR = mean reciprocal rank over ALL golds. This is what the run-time")
out("       scores.json files called `mrr`; it is a multi-gold diagnostic, not MRR.")
out()

for slc in ("all", "same", "cross", "similar", "dissimilar"):
    present = [d for d in DOMAINS if d in RES and any(slc in RES[d][a] for a in RES[d])]
    if not present:
        continue
    out(f"### {slc}")
    out("| domain | n | " + " | ".join(f"{lb} {a}" for a in ARMS for _, lb in METRICS) + " |")
    out("|---|--:|" + "--:|" * (len(ARMS) * len(METRICS)))
    for d in present:
        n = next((RES[d][a][slc]["n"] for a in ARMS if a in RES[d] and slc in RES[d][a]), 0)
        cells = []
        for a in ARMS:
            r = RES[d].get(a, {}).get(slc)
            cells += [(f"{r[k]:.4f}" if r else "-") for k, _ in METRICS]
        out(f"| {d} | {n} | " + " | ".join(cells) + " |")
    out()

# The comparison the retrain was for, on its own, because the wide table above is
# hard to read across three arms at once.
out('### Qwen3 handcrafted scorer minus BGE handcrafted scorer (fusion, `all` slice)')
out("| domain | n | " + " | ".join(lb for _, lb in METRICS) + " | wins |")
out("|---|--:|" + "--:|" * len(METRICS) + "--:|")
for d in DOMAINS:
    a, b = RES.get(d, {}).get("fusion Qwen3-op"), RES.get(d, {}).get("fusion BGE-op")
    if not (a and b and "all" in a and "all" in b):
        out(f"| {d} | - | " + " | ".join("-" for _ in METRICS) + " | not both trained |")
        continue
    ds = [a["all"][k] - b["all"][k] for k, _ in METRICS]
    out(f"| {d} | {a['all']['n']} | " + " | ".join(f"{x:+.4f}" for x in ds)
        + f" | {sum(1 for x in ds if x > 0)}/{len(ds)} |")
out()

json.dump(RES, open(f"{OUT}/all_domains_scores.json", "w"), indent=1)
open(f"{OUT}/all_domains_table.md", "w").write("\n".join(lines))
print(f"wrote {OUT}/all_domains_scores.json and all_domains_table.md")
