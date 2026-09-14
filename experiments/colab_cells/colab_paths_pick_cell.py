# ===== Path interpretations for HAND-PICKED (query, gold) pairs, CCMP gate on vs off, wider beam =====
# Paste into the showcase kernel AFTER cell 5 has run for the dataset (it uses ARMS, model_env, hydra_common,
# sh, SCAN_OUT, RUNS, PROBES, SECTIONS). Needs the engine: if cell 1 said NEED_ENGINE False, set
# NEED_ENGINE = True and re-run cells 2b-4 first. Outputs: SCAN_OUT/hops_pick_<arm>.json and a diff table.
PICK = {   # query id -> gold ids (SIR-4 CS examples; edit freely)
    "10.48550_arxiv.2601.22474": ["q:ac2856c33ccdf7be", "10.1037/h0061626"],   # LLM latent learning -> Tolman 1930 / 1948
    "10.48550_arxiv.2602.20408": ["10.3758/bf03202751"],                        # LLM idea diversity -> constraining effects of examples
    "10.48550_arxiv.2602.04572": ["__TITLE__Two-Person Cooperative Games"],     # GenAI + forum -> Nash 1953 (resolved by title below)
    "10.48550_arxiv.2601.20761": ["10.1561/3600000002"],                        # quantum tomography -> e-values
    "10.48550_arxiv.2602.21585": ["q:4f02a72cb830786c"],                        # test-time LLM optimisation -> Double Thompson
}
NUM_BEAM, PATH_TOPK, TOP_VIEWS = 24, 6, 5     # wider than the showcase run (6 / 3 / 3) so 5-6 hop routes surface
import os, json
assert os.path.isdir("/content/gfm-rag"), "engine not installed: set NEED_ENGINE = True in cell 1 and run cells 2b-4"
if "SEMF" not in globals():                     # component tables were skipped on a cached pass
    _r = get_ipython().run_cell(SECTIONS["components"], store_history=False); assert _r.success
docs = json.load(open(f"{DATA_ROOT}/{DATASET}_test/raw/documents.json"))
for q, gs in PICK.items():                      # resolve "__TITLE__..." golds
    PICK[q] = [next(d for d, t in docs.items() if t.startswith(g[9:])) if g.startswith("__TITLE__") else g for g in gs]
PK_Q = f"{SCAN_OUT}/qids_pick.json"; json.dump(sorted(PICK), open(PK_Q, "w"))
PK_G = f"{SCAN_OUT}/golds_pick.json"; json.dump(PICK, open(PK_G, "w"))

def paths_pick(name, ckpt, graph, skey, gate=None):
    out = f"{SCAN_OUT}/hops_pick_{name}.json"
    env_, info = model_env(ckpt, graph, skey, gate=gate)
    rl = f"{RUNS}/hops_pick_{name}"; os.makedirs(rl, exist_ok=True)
    print(f"[paths] {name}" + (f" gate={'on' if gate else 'off'}" if gate is not None else ""))
    rc = sh("python -u -m gfmrag.workflow.interpret_paths " + hydra_common(graph) +
            f"+interp.ckpt={ckpt} +interp.qids_file={PK_Q} +interp.out={out} " + (f"+interp.probes={PROBES} " if PROBES else "") +
            f"+interp.paths=1 +interp.golds_file={PK_G} +interp.max_golds=8 +interp.top_views={TOP_VIEWS} "
            f"+interp.num_beam={NUM_BEAM} +interp.path_topk={PATH_TOPK} hydra.run.dir={rl}",
            "/content/gfm-rag", extra=env_, log=f"{rl}/console.log", check=False)
    assert rc == 0 and os.path.exists(out), f"{name} failed (exit {rc}); read {rl}/console.log"
    return out
H = {name: paths_pick(name, ckpt, graph, skey, gate) for name, ckpt, graph, skey, gate in ARMS}

# ---- diff table: rank per channel per arm, then the routes with the gate on every hop and the on/off weight
def tab(p): return {(r["id"], t["doc"]): (r, t) for r in json.load(open(p)) for t in r["targets"]}
T = {n: tab(p) for n, p in H.items()}
def title(d): return docs.get(d, d).split(". ")[0][:80]
def lab(n): return f"[paper] {title(n)[:50]}" if n in docs else n[:55]
def fmt(p): return " -> ".join(f"{lab(h['head'])} --{h['rel'].replace('inverse_','inv.')}" + (f" <g={h['gate']:.2f}>" if "gate" in h and abs(h["gate"]-1) > 0.02 else "") for h in p["hops"]) + f" --> {lab(p['hops'][-1]['tail'])}"
for q, gs in PICK.items():
    for g in gs:
        print(f"\n########## {q}  ->  {title(g)}")
        print("| arm | fused | graph | scorer | dense | top path weight |"); print("|---|--:|--:|--:|--:|--:|")
        for n in T:
            if (q, g) not in T[n]: print(f"| {n} | -- | -- | -- | -- | -- |"); continue
            r, t = T[n][(q, g)]; rk = t["rank"]
            print(f"| {n} | {rk.get('fused')} | {rk.get('graph')} | {rk.get('scorer')} | {rk.get('dense')} | {t['paths'][0]['weight']:.2f} |" if t["paths"] else f"| {n} | {rk.get('fused')} | {rk.get('graph')} | {rk.get('scorer')} | {rk.get('dense')} | no valid path |")
        for n in T:
            if (q, g) in T[n]:
                r, t = T[n][(q, g)]
                for p in t["paths"][:PATH_TOPK]: print(f"  {n:15} w={p['weight']:7.3f}: {fmt(p)}")
        if (q, g) in T.get("frame_ccmp", {}):
            for v in T["frame_ccmp"][(q, g)][1].get("views", [])[:TOP_VIEWS]: print(f"  view {v['view']} match {v['match']:.2f}: {(v.get('text') or '')[:200]}")
print("\nwrote:", {n: os.path.relpath(p, DRIVE) for n, p in H.items()})
