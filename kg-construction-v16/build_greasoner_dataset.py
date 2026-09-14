"""
Materialize the v16 soft+canon(0.85) graph as a GFM-RAG / G-Reasoner stage1 dataset for a split.

Reads   cache/frames_doc_{split}.jsonl, cache/frames_query_{split}.jsonl
Writes  <out>/{dataset}_{split}_v16sc/processed/stage1/{nodes.csv, edges.csv, relations.csv, {split}.json}

Construction (identical to the PPR champion config):
  1. base frame graph (build_graph.py logic)
  2. consolidation: per-type single-linkage clusters at cosine >= tau_canon (0.85), merge to canonical node
  3. soft edges: per-type mutual-8NN in [tau_edge, cap] with typed relations + polarity guard
  4. query start_nodes: each query phrase snapped to top-3 nearest canonical node >= tau_snap (0.60)

  python build_greasoner_dataset.py --split test
  python build_greasoner_dataset.py --split train   # needs train frames extracted first
"""
import argparse, csv, json, os, sys, re
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("CARGO_ROOT") or os.path.dirname(HERE)   # repo root
KG = os.path.join(ROOT, "kg-construction")
CACHE = f"{HERE}/cache"
# One resolver for corpus + caches; default "tomato" reproduces every legacy path.
sys.path.insert(0, ROOT)
from cargo_paths import (add_dataset_arg, banner, corpus_dir, frames_path,  # noqa: E402
                         graph_cache_dir, graph_name, is_legacy, set_dataset)
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
csv.field_size_limit(10 ** 8)

CTYPES = ("function", "method", "limitation", "task", "mechanism", "finding")
REL = {"function": "is_specific_case_of", "method": "is_variant_of", "task": "shares_purpose",
       "mechanism": "shares_mechanism", "limitation": "same_limitation", "finding": "corroborates"}
POL_T = {"limitation", "finding"}
POS = set("increase increases increased higher high more greater improve improves improved gain gains enhance enhanced overestimate overestimation excess faster larger rise rises up upward".split())
NEG = set("decrease decreases decreased lower low less fewer reduce reduces reduced loss lack lacks not no non poor fail fails failure slower smaller underestimate underestimation insufficient limited unable inability down downward drop drops deficit weak".split())


def norm(s): return " ".join(str(s).lower().split()).strip()
def strip(n): return n.split("] ", 1)[-1]
def n_task(s): return f"[task] {norm(s)}"
def n_dom(s): return f"[domain] {norm(s)}"
def n_fun(s): return f"[function] {norm(s)}"
def n_lim(s): return f"[limitation] {norm(s)}"
def n_mech(s): return f"[mechanism] {norm(s)}"
def n_meth(s): return f"[method] {norm(s)}"


def polarity(p):
    t = re.findall(r"[a-z]+", p.lower())
    s = sum(1 for x in t if x in POS) - sum(1 for x in t if x in NEG)
    return (s > 0) - (s < 0)


def build_base(docs):
    """v16 build_graph.py logic -> node_type, edges, doc_provs"""
    node_type, edges = {}, set()

    def add_node(name, typ):
        if name and name.split("] ")[-1]: node_type[name] = typ

    def add_edge(s, rel, t):
        if s in node_type and t in node_type: edges.add((s, rel, t))

    for did, fr in docs.items():
        add_node(did, "document")
        for tk in (fr.get("task") or []):
            add_node(n_task(tk), "task"); add_edge(did, "addresses", n_task(tk))
            for lim in (fr.get("task_limitations") or []):
                add_node(n_lim(lim), "limitation"); add_edge(n_task(tk), "limited_by", n_lim(lim))
        if fr.get("domain"):
            add_node(n_dom(fr["domain"]), "domain"); add_edge(did, "in_field", n_dom(fr["domain"]))
        contribs = []

        def wire(src, c):
            for dep in (c.get("builds_on") or []):
                if isinstance(dep, str):
                    dm = n_meth(dep); add_node(dm, "method"); add_edge(src, "builds_on", dm)
            for pm in (c.get("improves_on") or []):
                if isinstance(pm, dict) and pm.get("name"):
                    pmn = n_meth(pm["name"]); add_node(pmn, "method"); add_edge(src, "improves_on", pmn)
                    for lim in (pm.get("has_limitation") or []):
                        add_node(n_lim(lim), "limitation"); add_edge(pmn, "limited_by", n_lim(lim))
        for c in (fr.get("contributions") or []):
            if not isinstance(c, dict) or not c.get("name"): continue
            m = n_meth(c["name"]); add_node(m, "method"); add_edge(did, "contributes", m); contribs.append(m)
            for f in (c.get("achieves") or []): add_node(n_fun(f), "function"); add_edge(m, "achieves", n_fun(f))
            for o in (c.get("overcomes") or []): add_node(n_lim(o), "limitation"); add_edge(m, "overcomes", n_lim(o))
            for me in (c.get("mechanism") or []): add_node(n_mech(me), "mechanism"); add_edge(m, "works_via", n_mech(me))
            wire(m, c)
        for cf in (fr.get("causal_findings") or []):
            if not isinstance(cf, dict): continue
            eff, abt, cau = cf.get("effect") or "", cf.get("about") or "", cf.get("cause") or ""
            if not (eff or abt): continue
            fnode = f"[finding] {norm(eff or abt)[:70]}"; add_node(fnode, "finding")
            add_edge(did, "reports", fnode); contribs.append(fnode)
            if abt: add_node(n_fun(abt), "function"); add_edge(fnode, "concerns", n_fun(abt))
            if eff: add_node(n_lim(eff), "limitation"); add_edge(fnode, "explains", n_lim(eff))
            if cau: add_node(n_mech(cau), "mechanism"); add_edge(fnode, "works_via", n_mech(cau))
            wire(fnode, cf)
        for dep in (fr.get("depends_on") or []):
            if contribs:
                dm = n_meth(dep); add_node(dm, "method")
                for c in contribs: add_edge(c, "builds_on", dm)
        for pm in (fr.get("prior_methods") or []):
            if isinstance(pm, dict) and pm.get("name"):
                pmn = n_meth(pm["name"]); add_node(pmn, "method")
                for c in contribs: add_edge(c, "improves_on", pmn)
                for lim in (pm.get("has_limitation") or []):
                    add_node(n_lim(lim), "limitation"); add_edge(pmn, "limited_by", n_lim(lim))
    return node_type, edges


def embed_types(node_type, split, tau_canon, tau_edge, cap, k):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL)
    type_nodes = defaultdict(list)
    for n, t in node_type.items():
        if t in CTYPES: type_nodes[t].append(n)
    emb = {}
    for t in CTYPES:
        # scoped: this was f"{CACHE}/ds_{split}_emb_{t}.npy", so a SIR-4 build
        # loaded TOMATO concept embeddings whenever the shapes happened to fit.
        cp = f"{graph_cache_dir()}/ds_{split}_emb_{t}.npy"
        if os.path.exists(cp): emb[t] = np.load(cp)
        else:
            V = model.encode([strip(n) for n in type_nodes[t]], normalize_embeddings=True, batch_size=128, show_progress_bar=True)
            emb[t] = np.asarray(V, np.float32); np.save(cp, emb[t])
    return type_nodes, emb


def cluster(embm, tau, chunk=1024):
    n = len(embm); parent = list(range(n))
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i0 in range(0, n, chunk):
        sims = embm[i0:i0 + chunk] @ embm.T
        ii, jj = np.where(sims >= tau)
        for r, j in zip(ii, jj):
            a, b = find(i0 + int(r)), find(int(j))
            if a != b: parent[a] = b
    groups = defaultdict(list)
    for i in range(n): groups[find(i)].append(i)
    return list(groups.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--out", default=f"{KG}/data")
    ap.add_argument("--tau_canon", type=float, default=0.85)
    ap.add_argument("--no_canon", action="store_true",
                    help="skip 0.85 single-linkage merging entirely; every node stays "
                         "its own canonical node (soft-only graph)")
    ap.add_argument("--tau_edge", type=float, default=0.80)
    ap.add_argument("--cap", type=float, default=0.995)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--tau_snap", type=float, default=0.60)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--entity_seeds", action="store_true", default=True, help="add cross-type entity seeds (query terms -> nearest node)")
    ap.add_argument("--no_entity_seeds", dest="entity_seeds", action="store_false")
    ap.add_argument("--tau_ent", type=float, default=0.60)
    ap.add_argument("--topk_ent", type=int, default=3)
    add_dataset_arg(ap)
    ap.add_argument("--probe_seeds", action="store_true", default=True, help="add BGE document/probe seeds (from v6rel retrieval)")
    ap.add_argument("--no_probe_seeds", dest="probe_seeds", action="store_false")
    a = ap.parse_args()
    set_dataset(a.dataset)
    print(banner())

    docf, qf = frames_path("doc", a.split), frames_path("query", a.split)
    # The back-compat fallback to the bare frames_doc.jsonl is TOMATO-ONLY. It
    # used to fire for any dataset, which silently built a `sir4_cs_smoke_test`
    # graph out of 3,182 TOMATO queries -- named for one corpus, made of another,
    # and with no error. `frames_path()` already handles TOMATO's unsuffixed test
    # filename, so this only covers datasets predating that helper.
    if is_legacy() and a.split == "test" and not os.path.exists(docf):
        docf, qf = f"{CACHE}/frames_doc.jsonl", f"{CACHE}/frames_query.jsonl"
    assert os.path.exists(docf), f"missing {docf} -- run extract_frames.py --side doc for split={a.split}"

    def load(p): return {json.loads(l)["id"]: (json.loads(l).get("frame") or {}) for l in open(p)}
    docs, queries = load(docf), load(qf)
    node_type, edges = build_base(docs)
    print(f"[ds] base: {len(node_type)} nodes, {len(edges)} edges")

    type_nodes, emb = embed_types(node_type, a.split, a.tau_canon, a.tau_edge, a.cap, a.k)

    # consolidation
    docdeg = Counter()
    for s, rel, t in edges:
        if node_type.get(s) == "document": docdeg[t] += 1
        if node_type.get(t) == "document": docdeg[s] += 1
    canon, cent = {}, {}
    if a.no_canon:
        # SOFT-ONLY. Every node is its own canonical node, so `canon` stays
        # empty and cmap is the identity. This skips the O(n^2) single-linkage
        # scan outright (~12 min on cs_train) rather than running it with an
        # unreachable threshold. Seeds then snap to raw node embeddings instead
        # of cluster centroids, which is the intended soft-only behaviour.
        for t in CTYPES:
            for i, n in enumerate(type_nodes[t]):
                cent[n] = emb[t][i]
    else:
        for t in CTYPES:
            ns = type_nodes[t]
            for mem in cluster(emb[t], a.tau_canon):
                rep = ns[max(mem, key=lambda i: (docdeg[ns[i]], -len(ns[i])))]
                v = emb[t][mem].mean(0); v /= (np.linalg.norm(v) + 1e-9); cent[rep] = v
                for i in mem: canon[ns[i]] = rep
    cmap = lambda n: canon.get(n, n)

    new_type = {cmap(n): ty for n, ty in node_type.items()}
    G = set()
    for s, rel, t in edges:
        cs, ct = cmap(s), cmap(t)
        if cs != ct: G.add((cs, rel, ct))
    # soft edges (mutual-kNN)
    for t in CTYPES:
        ns = type_nodes[t]; e = emb[t]; n = len(e)
        top = [[] for _ in range(n)]
        for i0 in range(0, n, 1024):
            sims = e[i0:i0 + 1024] @ e.T
            for r in range(sims.shape[0]):
                gi = i0 + r; row = sims[r]; row[gi] = -1
                cand = np.argpartition(-row, min(a.k, n - 1))[:a.k]
                top[gi] = [int(j) for j in cand if a.tau_edge <= row[j] <= a.cap]
        tset = [set(x) for x in top]
        guard = t in POL_T
        pol = [polarity(strip(x)) for x in ns] if guard else None
        for i in range(n):
            for j in top[i]:
                if i < j and i in tset[j]:
                    if guard and pol[i] and pol[j] and pol[i] != pol[j]: continue
                    si, sj = cmap(ns[i]), cmap(ns[j])
                    if si != sj: G.add((si, REL[t], sj))
    print(f"[ds] {'soft-only (no canon)' if a.no_canon else 'soft+canon'}: "
          f"{len(new_type)} nodes, {len(G)} edges")

    # canonical centroids per type for snapping
    reps_by_type = defaultdict(list)
    for rep in cent: reps_by_type[new_type[rep]].append(rep)
    cmat = {t: np.stack([cent[r] for r in reps]) for t, reps in reps_by_type.items()}
    # ALL canonical component nodes (any type) — target for cross-type entity snapping.
    # Only the entity channel reads this, so under --no_entity_seeds it was a
    # dead ~1 GB copy of every node vector. Under --no_canon there is no merging
    # to shrink it first, which makes it the largest single allocation in the run.
    all_reps = ([r for reps in reps_by_type.values() for r in reps]
                if a.entity_seeds else [])
    all_mat = np.stack([cent[r] for r in all_reps]) if all_reps else None

    # query start_nodes via snapping
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL)
    SEED_T = ("task", "function", "method", "limitation")
    tests = json.load(open(f"{corpus_dir(a.split)}/raw/{a.split}.json"))
    qmeta = {t["id"]: t for t in tests}
    doc_nodes = {n for n, t in new_type.items() if t == "document"}

    # per-query raw frame phrases (built once)
    q_raw = {}
    for qid, fr in queries.items():
        raw = defaultdict(list)
        for tk in (fr.get("task") or []): raw["task"].append(n_task(tk))
        for f in (fr.get("needs") or []): raw["function"].append(n_fun(f))
        for cm in (fr.get("current_methods") or []):
            if isinstance(cm, dict):
                if cm.get("name"): raw["method"].append(n_meth(cm["name"]))
                for lim in (cm.get("has_limitation") or []): raw["limitation"].append(n_lim(lim))
        for lim in (fr.get("task_limitations") or []): raw["limitation"].append(n_lim(lim))
        q_raw[qid] = raw

    # cached batch-embed: phrases embedded once, reused across reruns (keyed by split+tag)
    def cached_embed(phrases, tag):
        phrases = sorted(set(phrases))
        # scoped: this was f"{CACHE}/ds_{a.split}_phremb_{tag}.npz", which is
        # TOMATO's directory. A SIR-4 build therefore LOADED the TOMATO phrase
        # cache and then wrote its own phrases back into it. The vectors were
        # not wrong -- the cache is keyed by phrase string and BGE is
        # deterministic -- but it broke corpus isolation and made the run
        # depend on a 209 MB TOMATO file it had no reason to touch.
        cp = f"{graph_cache_dir()}/ds_{a.split}_phremb_{tag}.npz"
        if os.path.exists(cp):
            z = np.load(cp, allow_pickle=True)
            d = {p: v for p, v in zip(z["ph"], z["emb"])}
            miss = [p for p in phrases if p not in d]
            if not miss:
                return {p: d[p] for p in phrases}
        else:
            d = {}
        todo = [p for p in phrases if p not in d]
        if todo:
            V = model.encode([strip(p) for p in todo], normalize_embeddings=True, batch_size=256, show_progress_bar=True)
            for p, v in zip(todo, np.asarray(V, np.float32)): d[p] = v
            allp = sorted(d)
            np.savez(cp, ph=np.array(allp, object), emb=np.stack([d[p] for p in allp]))
        return {p: d[p] for p in phrases}

    # BATCH-embed every unique frame phrase per type ONCE (cached across reruns)
    frame_emb = {}
    for ty in SEED_T:
        uphr = [p for raw in q_raw.values() for p in raw.get(ty, [])]
        frame_emb[ty] = cached_embed(uphr, f"frame_{ty}") if uphr else {}

    # ENTITY channel: v6rel's query-side entity terms + BGE document/probe seeds (both graph-agnostic)
    q_entities, ent_emb, q_probes = {}, {}, {}
    v6f = f"{KG}/data/{graph_name(a.split, "v6r")}/processed/stage1/{a.split}.json"
    if os.path.exists(v6f):
        for q in json.load(open(v6f)):
            q_entities[q["id"]] = [strip(e) for e in q.get("start_nodes", {}).get("entity", [])]
            q_probes[q["id"]] = [d for d in q.get("start_nodes", {}).get("document", []) if d in doc_nodes]
        if a.entity_seeds and all_mat is not None:
            uniq = [p for ps in q_entities.values() for p in ps]
            if uniq:
                ent_emb = cached_embed(uniq, "entity")
        print(f"[ds] entity+probe channel: entity phrases={len(ent_emb)}  mean probes/q={np.mean([len(v) for v in q_probes.values()]):.1f}")
    else:
        print(f"[ds] entity/probe channel skipped: missing {v6f}")

    # ---- precompute snapping ONCE per unique phrase (batched matmul), then O(1) lookup per query ----
    # ~40 entity terms/query over 9669 queries = ~387k occurrences, but only ~36k unique terms;
    # snapping each unique phrase once (vs per-occurrence) is the whole speedup.
    def batch_snap(emb_map, mat, reps, topk, tau):
        """phrase -> [rep,...]: the topk nearest reps above cosine tau, in batched matmuls."""
        out = {}
        if mat is None or not emb_map: return out
        phrases = list(emb_map)
        M = np.stack([emb_map[p] for p in phrases]).astype(np.float32)   # (P, d)
        matT = mat.T                                                    # (d, N)
        kk = min(topk, mat.shape[0] - 1)
        for i0 in tqdm(range(0, len(phrases), 1024), desc="snap-precompute", leave=False):
            S = M[i0:i0 + 1024] @ matT                                  # (b, N)
            idx = np.argpartition(-S, kk, axis=1)[:, :topk]             # (b, topk)
            for r in range(S.shape[0]):
                out[phrases[i0 + r]] = [reps[int(j)] for j in idx[r] if S[r, int(j)] >= tau]
        return out

    frame_snap = {ty: batch_snap(frame_emb.get(ty, {}), cmat.get(ty), reps_by_type.get(ty, []),
                                 a.topk, a.tau_snap) for ty in SEED_T}
    ent_snap = batch_snap(ent_emb, all_mat, all_reps, a.topk_ent, a.tau_ent) if a.entity_seeds else {}

    out_queries = []
    for qid, fr in tqdm(queries.items(), total=len(queries), desc="snap seeds"):
        start = defaultdict(list)
        # frame seeds (precomputed lookup)
        for ty in SEED_T:
            fs = frame_snap[ty]
            for p in set(q_raw[qid].get(ty, [])):
                start[ty].extend(fs.get(p, []))
        # entity seeds (precomputed lookup)
        if a.entity_seeds:
            for term in q_entities.get(qid, []):
                start["entity"].extend(ent_snap.get(term, []))
        # probe/document seeds: BGE top-k similar docs (same corpus/DOIs as v6rel)
        if a.probe_seeds:
            start["document"].extend(q_probes.get(qid, []))
        m = qmeta.get(qid, {})
        gold = m.get("supporting_documents") or []
        out_queries.append({
            "id": qid, "question": m.get("question", ""), "answer": m.get("answer", ""),
            "supporting_documents": gold, "stratum": m.get("stratum", "?"),
            "start_nodes": {k: sorted(set(v)) for k, v in start.items()},
            "target_nodes": {"document": gold if isinstance(gold, list) else [gold]},
        })

    # write stage1
    outdir = f"{a.out}/{graph_name(a.split)}/processed/stage1"
    os.makedirs(outdir, exist_ok=True)
    with open(f"{outdir}/nodes.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["name", "type", "attributes"])
        for n, t in new_type.items(): w.writerow([n, t, "{}"])
    rels = sorted({r for _, r, _ in G})
    with open(f"{outdir}/edges.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["source", "relation", "target", "attributes"])
        for s, r, t in sorted(G): w.writerow([s, r, t, "{}"])
    with open(f"{outdir}/relations.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["name", "attributes"])
        for r in rels: w.writerow([r, "{}"])
    json.dump(out_queries, open(f"{outdir}/{a.split}.json", "w"))
    print(f"[ds] wrote {outdir}  nodes={len(new_type)} edges={len(G)} relations={len(rels)} queries={len(out_queries)}")
    print(f"[ds] relations: {rels}")


if __name__ == "__main__":
    main()
