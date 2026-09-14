"""
build_showcase_cell.py -- one paste-in Colab cell: the graph-channel scan of colab_graph_scan_cell.py
followed by path interpretations for every cross-field gold (plus a same-field sample) under each
arm, then eval/showcase.py: the ranked candidate list of "amazing" cross-domain examples (markdown),
the GFM-RAG Table 4 style LaTeX, and the hop figure (GFM-RAG Fig. 6 analogue).

    python3 prep/build_showcase_cell.py      # -> colab_showcase_cell.py

Set DATASET at the top of the produced cell, paste it into a FRESH runtime, run.
"""
import ast
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

SHOWCASE = r'''

# ================= SHOWCASE: path interpretations for every cross-field gold + the hop figure =================
# Runs the gradient beam search (paths=1) for a random query sample (the unbiased hop figure) plus every
# candidate gold the scan flagged (pinned), under each arm, then eval/showcase.py ranks the candidates a reader would
# call amazing: cosine buries the gold, the graph channel or the full model recovers it, and the top
# route passes a function / limitation / method frame rather than a domain hub.
import random
SAMPLE_CROSS, SAMPLE_SAME = 80, 80   # random queries interpreted for the hop figure (unbiased); candidates are added on top
TOP, N_TEX = 30, 4                   # candidates listed in the markdown / examples in the LaTeX table
MIN_DENSE, MAX_GRAPH, MAX_FUSED = 25, 5, 25   # candidate filter, same as eval/showcase.py
_qs = json.load(open(QUERIES))
_cross = [q["id"] for q in _qs if q.get("stratum") == "cross"]
_same = [q["id"] for q in _qs if q.get("stratum", "same") != "cross"]
_rng = random.Random(0)
SAMPLE = (_rng.sample(_cross, min(SAMPLE_CROSS, len(_cross))) + _rng.sample(_same, min(SAMPLE_SAME, len(_same)))) if _cross \
         else _rng.sample([q["id"] for q in _qs], min(SAMPLE_CROSS + SAMPLE_SAME, len(_qs)))
# candidates from the scan of the showcased arm: cosine buries the gold, the graph channel or the model recovers it
_scan_main = next((SA[k] for k in ("frame_ccmp", "frame_nocc", "openie") if k in SA), None)
PIN = {}
for r in json.load(open(_scan_main)):
    if _cross and r.get("stratum") != "cross": continue
    for t in r["targets"]:
        rk = t["rank"]
        if rk["dense"] >= MIN_DENSE and (rk["graph"] <= MAX_GRAPH or rk["fused"] <= MAX_FUSED):
            PIN.setdefault(r["id"], []).append(t["doc"])
PATH_QIDS = sorted(set(SAMPLE) | set(PIN))
PQ_FILE = f"{SCAN_OUT}/qids_paths.json"; json.dump(PATH_QIDS, open(PQ_FILE, "w"))
GOLDS_FILE = f"{SCAN_OUT}/golds_paths.json"; json.dump(PIN, open(GOLDS_FILE, "w"))      # pins the candidate gold(s) per query
json.dump(SAMPLE, open(f"{SCAN_OUT}/qids_sample.json", "w"))                         # the unbiased subset for the hop figure
print(f"path search on {len(PATH_QIDS)} queries: {len(SAMPLE)} random ({min(SAMPLE_CROSS, len(_cross)) if _cross else 0} cross) "
      f"+ {len(PIN)} candidate queries with {sum(map(len, PIN.values()))} pinned golds; per arm")
def paths_all(name, ckpt, graph, skey, gate=None):
    out = f"{SCAN_OUT}/hops_{name}.json"
    if os.path.exists(out): print("[cached]", os.path.relpath(out, DRIVE)); return out
    env_, info = model_env(ckpt, graph, skey, gate=gate)
    print(f"[paths] {name}: {os.path.relpath(ckpt, DRIVE)}" + (f" gate={'on' if gate else 'off'}" if gate is not None else ""))
    rl = f"{RUNS}/hops_{name}"; os.makedirs(rl, exist_ok=True)
    rc = sh("python -u -m gfmrag.workflow.interpret_paths " + hydra_common(graph) +
            f"+interp.ckpt={ckpt} +interp.qids_file={PQ_FILE} +interp.out={out} " +
            (f"+interp.probes={PROBES} " if PROBES else "") +
            f"+interp.paths=1 +interp.golds_file={GOLDS_FILE} +interp.max_golds=4 +interp.top_views=3 "
            f"+interp.num_beam=6 +interp.path_topk=3 hydra.run.dir={rl}",
            "/content/gfm-rag", extra=env_, log=f"{rl}/console.log", check=False)
    assert rc == 0 and os.path.exists(out), f"{name} path search failed (exit {rc}); read {rl}/console.log"
    return out
HOPS = {}
for name, ckpt, graph, skey, gate in ARMS:
    HOPS[name] = paths_all(name, ckpt, graph, skey, gate)
print("interpreted:", sorted(HOPS))

os.makedirs(f"{S4}/eval", exist_ok=True)
open(f"{S4}/eval/showcase.py", "w").write(r"""__SHOWCASE_PY__""")
print("eval/showcase.py written from this notebook, built __SHOWCASE_BUILT__")
ARM_LABEL = {"frame_ccmp": "SciGraphIR (frame graph + CCMP)", "frame_ccmp_off": "frame graph, CCMP gate off",
             "frame_nocc": "frame graph, no CCMP", "openie": "OpenIE graph"}
BASE = f"{DRIVE}/outputs/baselines/{DATASET}"
SHOW = f"{SCAN_OUT}/showcase_{DATASET}"
cmd = [sys.executable, "-u", "eval/showcase.py", "--dataset", DATASET, "--queries", QUERIES,
       "--docs", f"{DATA_ROOT}/{DATASET}_test/raw/documents.json",
       "--edges", f"{DATA_ROOT}/{FRAME_TEST}/processed/stage1/edges.csv",
       "--out", SHOW, "--top", str(TOP), "--n-tex", str(N_TEX), "--sample-qids", f"{SCAN_OUT}/qids_sample.json"]
for name in ("frame_ccmp", "frame_ccmp_off", "frame_nocc", "openie"):
    if name in HOPS: cmd += ["--arm", f"{ARM_LABEL[name]}={HOPS[name]}"]
for tag in ("qwen3", "bge", "reasonir", "bm25"):
    p = f"{BASE}/predictions_{tag}_{DATASET}_test.json"
    if os.path.exists(p): cmd += ["--pred", f"{tag}={p}"]
sh(cmd, S4)
from IPython.display import Image, display
if os.path.exists(f"{SHOW}_hops.png"):
    display(Image(f"{SHOW}_hops.png"))
else:                       # showcase.py says why above; the markdown, LaTeX and _hops.json are written
    print(f"no hop figure for {DATASET} (see the message above); {os.path.basename(SHOW)}.md/.tex/_hops.json are on Drive")
print(open(f"{SHOW}.md").read()[:8000])
print("\nfiles:", sorted(f for f in os.listdir(SCAN_OUT) if f.startswith(("showcase_", "hops_"))))
'''


def main() -> int:
    scan = open(f"{ROOT}/colab_cells/colab_graph_scan_cell.py").read()
    show = open(f"{ROOT}/eval/showcase.py").read()
    assert '"""' not in show.replace('"""', "", 2) or True   # the module docstring uses """; the cell wraps it in r"""..."""? no:
    # The embedded file is wrapped in r"""...""", so its own triple double quotes would end the string.
    # Swap the docstring's delimiter to triple single quotes before embedding (it contains no ''').
    assert "'''" not in show, "showcase.py must not contain ''' (it is embedded inside r'''...''' of the cell builder)"
    show = show.replace('"""', "'''")
    head = ("# Showcase cell: graph-channel scan + path interpretations + 'amazing example' finder + hop figure, for ONE\n"
            "# dataset in ONE cell. Set DATASET, paste into a fresh runtime, run. Outputs go to outputs/scan/<dataset>/\n"
            "# (hops_<arm>.json, showcase_<dataset>.md/.tex/_hops.pdf). The first part is colab_graph_scan_cell.py verbatim.\n")
    body = scan.split("\n", 6)   # drop the scan cell's 6-line header comment
    import datetime
    stamp = datetime.datetime.fromtimestamp(os.path.getmtime(f"{ROOT}/eval/showcase.py")).strftime("%Y-%m-%d %H:%M")
    cell = head + "\n".join(body[6:]) + SHOWCASE.replace("__SHOWCASE_PY__", show).replace("__SHOWCASE_BUILT__", stamp)
    ast.parse(cell)
    out = f"{ROOT}/colab_cells/colab_showcase_cell.py"
    open(out, "w").write(cell)
    print(f"wrote {out}: {len(cell.splitlines())} lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
