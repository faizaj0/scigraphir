#!/usr/bin/env python3
"""
build_gate_decomp_notebook.py -- generate colab_ccmp_gate_decomp.ipynb from colab_showcase_all.ipynb
+ the current engine (gfm_overlay.zip, interpret_paths.py) + eval/gate_decomp_fig.py.

The notebook answers "which gates move a route's weight and the gold's rank?" on the CCMP arm: for
every interpreted (query, gold) it follows the top routes found under the full gate and re-evaluates
the SAME routes (directly along their edges) and the gold's graph / fused rank under six inference-time
gate masks on the SAME trained weights (trainer.gate_decomposition). Setup cells are the all-datasets
showcase notebook's, verbatim; the overlay cell is refreshed from the repo's gfm_overlay.zip and the
driver rewrites interpret_paths.py from the repo copy (the setup blob writes an older one).

    python3 prep/build_gate_decomp_notebook.py
"""
import base64
import copy
import json
import os
import re
import time

HERE = os.path.dirname(os.path.abspath(__file__))
S4 = os.path.dirname(HERE)
ROOT = os.path.dirname(S4)
SRC = f"{S4}/colab_showcase_all.ipynb"
OUT = f"{S4}/colab_ccmp_gate_decomp.ipynb"
STAMP = time.strftime("%Y-%m-%d %H:%M")


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


def b64(path):
    return base64.b64encode(open(path, "rb").read()).decode()


nb = json.load(open(SRC))
cells = nb["cells"]
assert "".join(cells[2]["source"]).lstrip().startswith("# 1. Datasets to run"), "cell 2 of the source notebook moved"
assert "".join(cells[24]["source"]).lstrip().startswith("# 4. interpret_paths.py"), "cell 24 of the source notebook moved"

# ---------------------------------------------------------------- spec cell
spec_src = "".join(cells[2]["source"])
spec_src = spec_src.replace('DATASETS = ["sir4_cs", "sir4_biology", "sir4_physics", "sir4_matsci", "tomato", "mir"]',
                            'DATASETS = ["sir4_biology", "sir4_cs"]     # CCMP gate decomposition on these (any dataset with a CCMP arm works)\n'
                            'GD_ARM   = "frame_ccmp"                    # the arm whose gate is decomposed\n'
                            'GD_MAX_GOLDS = 120                          # golds per dataset (cross-field first, deepest under cosine first)\n'
                            'GD_EXAMPLES = {"sir4_biology": ["10.1007/s11263-023-01831-9"]}   # golds always included (the OT example)')
assert "GD_ARM" in spec_src
head, _, _ = spec_src.partition("# --- resume plan:")
spec_src = head + '''# --- the CCMP arm only ---------------------------------------------------------------------------
for _d in SPEC:
    SPEC[_d].pop("extra", None)
    SPEC[_d]["arms"] = [a for a in SPEC[_d]["arms"] if a[0] == GD_ARM]
    assert SPEC[_d]["arms"], f"{_d}: no '{GD_ARM}' arm in the spec"

# --- resume plan --------------------------------------------------------------------------------------
# FORCE = "" (nothing), "fig" (redraw), "decomp" (redo the decomposition), or "all".
FORCE = ""
PLAN = {}
for d in DATASETS:
    o = f"{DRIVE}/outputs/scan/{d}"
    rel = SPEC[d]["arms"][0][1]; rels = rel if isinstance(rel, list) else [rel]
    PLAN[d] = {"ckpt": any(os.path.exists(f"{DRIVE}/{r}/model_best.pth") for r in rels),
               "hops": os.path.exists(f"{o}/hops_{GD_ARM}.json"),           # the routes to follow come from the showcase run
               "decomp": os.path.exists(f"{o}/gate_decomp_{d}.json") and FORCE not in ("decomp", "all"),
               "fig": os.path.exists(f"{o}/fig_gate_decomp_{d}.png") and FORCE == ""}
NEED_ENGINE = any(PLAN[d]["ckpt"] and PLAN[d]["hops"] and not PLAN[d]["decomp"] for d in DATASETS)
import numpy as np
print("datasets:", DATASETS, "| arm:", GD_ARM, "| FORCE:", FORCE or "(resume)")
for d in DATASETS:
    P = PLAN[d]
    print(f"  {d:14} checkpoint {'ok' if P['ckpt'] else 'MISSING'} | hops_{GD_ARM}.json {'ok' if P['hops'] else 'MISSING (run the showcase notebook first)'} "
          f"| decomposition {'cached' if P['decomp'] else 'to do'} | figure {'cached' if P['fig'] else 'to do'}")
print(f"engine + Qwen3 + component tables needed: {NEED_ENGINE}" + ("" if NEED_ENGINE else "  -> cells 2b-3g are no-ops; run cell 1, 2, 4, 5"))
'''

# ---------------------------------------------------------------- driver cell
driver = r'''# 5. Per dataset: graphs + checkpoint, component tables (engine runs only), pick the golds, run the gate
# decomposition (trainer.gate_decomposition via interpret_paths.py mode=gate_decomp), draw the figure.
import base64 as _b64
# the CURRENT interpret_paths.py (the setup blob writes an older one) and the figure script, from the repo
if os.path.isdir("/content/gfm-rag"):
    open("/content/gfm-rag/gfmrag/workflow/interpret_paths.py", "wb").write(_b64.b64decode("__INTERP_B64__"))
    print("interpret_paths.py written from the repo copy (gate_decomp mode)")
os.makedirs(f"{S4}/eval", exist_ok=True)
open(f"{S4}/eval/gate_decomp_fig.py", "wb").write(_b64.b64decode("__FIG_B64__"))
print("eval/gate_decomp_fig.py written; notebook built __STAMP__")

def set_dataset(d):
    global DATASET, S, SCAN_DATASET, CACHE, QUERIES, env
    DATASET = SCAN_DATASET = d; S = SPEC[d]
    CACHE = f"{DRIVE}/outputs/{d}/cache"; os.makedirs(CACHE, exist_ok=True)
    QUERIES = f"{DATA_ROOT}/{d}_test/raw/test.json"
    os.environ["CARGO_DATASET"] = d
    env = dict(os.environ, CARGO_ROOT=CARGO_ROOT, CARGO_DATASET=d, PYTHONUNBUFFERED="1")
    cp.set_dataset(d); print(cp.banner())
    for k in list(os.environ):
        if k.startswith(("CCMP", "ROUTE", "STRAT_", "CQIG", "RESID_", "MISS_W")): os.environ.pop(k)

GD = {}
GD["pick"] = r"""
# --- the golds to decompose: every (query, gold) the showcase run interpreted with a valid route on the CCMP arm,
# cross-field first, deepest under Qwen3 cosine first, capped at GD_MAX_GOLDS; the listed examples always included
_hops = json.load(open(f"{SCAN_OUT}/hops_{GD_ARM}.json"))
_cand = []
for r in _hops:
    seeds = set(r["seeds"])
    for t in r["targets"]:
        if any(p["hops"] and p["hops"][0]["head"] in seeds for p in t["paths"]):
            _cand.append((0 if (r.get("stratum") == "cross") else 1, -t["rank"]["dense"], r["id"], t["doc"]))
_cand.sort()
_ex = set(GD_EXAMPLES.get(DATASET, []))
_keep = [c for c in _cand if c[3] in _ex] + [c for c in _cand if c[3] not in _ex][:GD_MAX_GOLDS]
GD_PIN = {}
for _, _, q, g in _keep: GD_PIN.setdefault(q, []).append(g)
GD_QIDS = f"{SCAN_OUT}/qids_gate_decomp.json"; json.dump(sorted(GD_PIN), open(GD_QIDS, "w"))
GD_GOLDS = f"{SCAN_OUT}/golds_gate_decomp.json"; json.dump(GD_PIN, open(GD_GOLDS, "w"))
print(f"{DATASET}: {len(_keep)} golds on {len(GD_PIN)} queries (of {len(_cand)} interpreted with a route); examples present:",
      [g for g in _ex if any(c[3] == g for c in _keep)])
"""
GD["decomp"] = r"""
# --- run the decomposition (six gate masks per gold, same weights) --------------------------------------
GD_OUT = f"{SCAN_OUT}/gate_decomp_{DATASET}.json"
if os.path.exists(GD_OUT) and FORCE not in ("decomp", "all"): print("[cached]", os.path.relpath(GD_OUT, DRIVE))
else:
    name, ckpt, graph, skey, gate = ARMS[0]
    env_, info = model_env(ckpt, graph, skey, gate=True)
    print(f"[gate-decomp] {name}: {os.path.relpath(ckpt, DRIVE)} {info}")
    rl = f"{RUNS}/gate_decomp_{DATASET}"; os.makedirs(rl, exist_ok=True)
    rc = sh("python -u -m gfmrag.workflow.interpret_paths " + hydra_common(graph) +
            f"+interp.ckpt={ckpt} +interp.qids_file={GD_QIDS} +interp.out={GD_OUT} +interp.golds_file={GD_GOLDS} "
            f"+interp.mode=gate_decomp +interp.num_beam=10 +interp.path_topk=5 +interp.max_golds=4 hydra.run.dir={rl}",
            "/content/gfm-rag", extra=env_, log=f"{rl}/console.log", check=False)
    assert rc == 0 and os.path.exists(GD_OUT), f"gate decomposition failed (exit {rc}); read {rl}/console.log"
"""
GD["fig"] = r"""
# --- figure + table -------------------------------------------------------------------------------
GD_FIG = f"{SCAN_OUT}/fig_gate_decomp_{DATASET}"
cmd = [sys.executable, "-u", "eval/gate_decomp_fig.py", "--decomp", f"{SCAN_OUT}/gate_decomp_{DATASET}.json",
       "--docs", f"{DATA_ROOT}/{DATASET}_test/raw/documents.json", "--out", GD_FIG]
if GD_EXAMPLES.get(DATASET): cmd += ["--example", GD_EXAMPLES[DATASET][0]]
sh(cmd, S4)
from IPython.display import Image, display
display(Image(GD_FIG + ".png"))
print(open(GD_FIG + ".md").read()[:6000])
"""

def sections_for(d):
    P = PLAN[d]
    todo = ["graphs", "model_env"] if P["decomp"] else ["graphs", "components", "model_env"]
    todo += ["pick", "decomp", "fig"]
    return todo
ALL = dict(SECTIONS); ALL.update(GD)

DONE, FAILED = [], {}
for _d in DATASETS:
    print(f"\n\n#################### {_d} ####################")
    if not (PLAN[_d]["ckpt"] and PLAN[_d]["hops"]):
        FAILED[_d] = "checkpoint or hops_frame_ccmp.json missing on Drive"; print(f"[{_d}] SKIPPED: {FAILED[_d]}"); continue
    if not PLAN[_d]["decomp"] and not os.path.isdir("/content/gfm-rag"):
        FAILED[_d] = "needs the engine but it is not installed"; print(f"[{_d}] SKIPPED: {FAILED[_d]}; re-run cell 1 then cells 2b-3g"); continue
    set_dataset(_d)
    _todo = sections_for(_d); print("sections:", ", ".join(_todo))
    try:
        for _name in _todo:
            print(f"\n======== {_d}: {_name} ========")
            _r = get_ipython().run_cell(ALL[_name], store_history=False)
            assert _r.success, f"section '{_name}' failed; see the traceback above"
        DONE.append(_d)
    except Exception as _e:
        FAILED[_d] = str(_e)[:300]; print(f"[{_d}] FAILED: {_e}")
print("\nfinished:", DONE, "| failed:", FAILED)
'''.replace("__INTERP_B64__", b64(f"{ROOT}/kg-construction/gfm-rag/gfmrag/workflow/interpret_paths.py")) \
   .replace("__FIG_B64__", b64(f"{S4}/eval/gate_decomp_fig.py")).replace("__STAMP__", STAMP)

intro = f"""# CCMP gate decomposition: which gates move a route's weight?

The path interpretations show route weights changing between the CCMP gate on and off even when the gate
on every drawn hop is 1.00 (e.g. the protein-RNA / optimal-transport example on SIR-4 Biology: 9.28 → 12.44
with g = 0.9999 and 1.0015). A route's weight is the mean over its hops of ∂score/∂(edge weight) at the layer
the hop is attributed to, so the gate on the drawn hop enters only as a factor of 1; the change must come from
gates elsewhere: on the same senders at later layers (the frontier-mean responsibility falls from ≈0.48 at layer
0 to ≈0.17 at layer 5, so a sender at 0.42 is amplified ≈1.7× there), or on competing senders.

This notebook measures that directly on the same trained weights. For every interpreted (query, gold) it follows
the top routes found under the full gate and re-evaluates the **same routes** (directly along their edges) and the
gold's graph / fused rank under six inference-time gate masks:

| condition | gate applied to |
|---|---|
| gate_on | everything (the model as run) |
| gate_off | nothing (the "no CCMP" arm) |
| gate_layers_attributed | only the layers the route is attributed to (0 .. L−1) |
| gate_layers_later | only the later layers (L .. 5) |
| gate_route_senders_only | only the route's own sender nodes |
| gate_all_but_route_senders | every node except the route's senders |

Engine: `models.py` gains `_gate_layers` / `_gate_nodes` masks on the CCMP gate (interpretation only);
`fusion_trainer.py` gains `gate_decomposition()`; `interpret_paths.py` gains `+interp.mode=gate_decomp`. All
three are in the repo and in the embedded overlay. Outputs per dataset on Drive: `outputs/scan/<dataset>/
gate_decomp_<dataset>.json`, `fig_gate_decomp_<dataset>.{{pdf,png,md}}`. Needs `hops_frame_ccmp.json` from the
showcase run (the routes to follow) and the CCMP checkpoint. Generated by `prep/build_gate_decomp_notebook.py`
on {STAMP}.
"""

new = copy.deepcopy(nb)
new["cells"] = ([md(intro), md("## 1. Datasets, environment, checkpoint spec, resume plan"), code(spec_src)]
                + copy.deepcopy(cells[3:25])
                + [md("## 5. Run every dataset: pick golds, decompose the gate, draw the figure"), code(driver)])
_ovb = b64(f"{S4}/gfm_overlay.zip")
for c in new["cells"]:
    src = "".join(c["source"]) if isinstance(c.get("source"), list) else c.get("source", "")
    if c["cell_type"] == "code" and "gfm_overlay.zip from the repo" in src and "b64decode(" in src:
        c["source"] = re.sub(r'b64decode\("([A-Za-z0-9+/=]+)"\)', lambda m: f'b64decode("{_ovb}")', src, count=1)
        print("re-embedded gfm_overlay.zip in the overlay cell")
for c in new["cells"]:
    if isinstance(c.get("source"), str):
        c["source"] = c["source"].splitlines(keepends=True)
    if c["cell_type"] == "code":
        c["outputs"] = []; c["execution_count"] = None
json.dump(new, open(OUT, "w"), indent=1)
print(f"wrote {OUT}: {len(new['cells'])} cells")
