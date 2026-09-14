"""
build_mir_scan_notebook.py -- graph-channel ranks of EVERY MIR gold under every MIR checkpoint.

The MIR counterpart of the SIR-4 CS "graph channel only, same golds, same queries" comparison:
for each test query and each of its golds, the rank of that gold under the graph channel alone,
the multi-view scorer alone, Qwen3 cosine, and the fused score, for

    frame_ccmp      SciAfford frame graph + CCMP            (outputs/mir/mir_qwenmlp_ccmp_e10_b2)
    frame_ccmp_off  the SAME weights with the CCMP gate off at inference
    frame_nocc      SciAfford frame graph, no CCMP, its own run (outputs/mir/mir_qwenmlp_graph_e10_b2)
    openie          OpenIE entity graph, no CCMP            (outputs/mir/mir_openie_qwenmlp_graph_e10_b2)

No training, no path search (interpret_paths with +interp.paths=0): a scan of 155 queries per
arm, minutes each. All golds are scanned (max_golds=8 > MIR's per-query maximum), so there is no
best-gold selection bias. Output: median rank, R@5/10/50 per channel per arm, and gold-by-gold
win rates (frame vs OpenIE, CCMP on vs off, CCMP vs the separate no-CCMP run), in
outputs/mir/scan/graph_channel_scan_mir.{json,md} on Drive.

Reuses build_mir_notebook.py's paths/unpack cells and build_rb_zeroshot_notebook.py's engine
cells; the scan cell mirrors build_qualitative_notebook.py's model_env/hydra_common.

Usage
-----
    python3 prep/build_mir_scan_notebook.py            # -> colab_mir_scan.ipynb
"""
from __future__ import annotations

import argparse
import ast
import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_mir_notebook as bm                # noqa: E402
import build_rb_zeroshot_notebook as rb        # noqa: E402

ROOT, CARGO, FORK = rb.ROOT, rb.CARGO, rb.FORK
md, code = rb.md, rb.code
FUSION_REL = rb.FUSION_REL + ["gfmrag/models/ultra/layers.py", "gfmrag/workflow/interpret_paths.py"]

HEADER = '''# MIR — graph-channel ranks on every gold, frame graph vs OpenIE, CCMP on vs off

The MIR version of the SIR-4 CS comparison. For every test query and every one of its golds:
the rank of the gold under the **graph channel alone**, the **scorer alone**, **Qwen3 cosine**
and the **fused** score, for four arms that all share the MIR scorer warm start:

| arm | checkpoint |
|---|---|
| frame graph + CCMP | `outputs/mir/mir_qwenmlp_ccmp_e10_b2` |
| same weights, CCMP gate OFF at inference | (same) |
| frame graph, no CCMP (own run) | `outputs/mir/mir_qwenmlp_graph_e10_b2` |
| OpenIE entity graph, no CCMP | `outputs/mir/mir_openie_qwenmlp_graph_e10_b2` |

Nothing trains and there is no path search; each arm is a scan of the 155 test queries
(minutes). All golds are scanned, so no best-gold selection bias. Needs the re-bundled
`mir_bundle.zip` (both graphs) on Drive and the three finished runs above.
'''

EXTRA_PATHS = '''
# --- the two test graphs and the three checkpoints ------------------------------------
FRAME_TEST  = f"{DATASET}_test_v16sc"              # SciAfford frame graph
OPENIE_TEST = f"{DATASET}_test"                    # OpenIE entity graph (same corpus, same queries)
FRAME_CCMP_RUN = f"{OUT_ROOT}/{DATASET}_qwenmlp_ccmp_e{EPOCHS}_b{BATCH}"
FRAME_NOCC_RUN = f"{OUT_ROOT}/{DATASET}_qwenmlp_graph_e{EPOCHS}_b{BATCH}"
OPENIE_RUN     = f"{OUT_ROOT}/{DATASET}_openie_qwenmlp_graph_e{EPOCHS}_b{BATCH}"
SCAN_OUT = f"{OUT_ROOT}/scan"
os.makedirs(SCAN_OUT, exist_ok=True)
for lab, d in (("frame + CCMP", FRAME_CCMP_RUN), ("frame, no CCMP", FRAME_NOCC_RUN), ("OpenIE", OPENIE_RUN)):
    print(f"  {'ok ' if os.path.exists(f'{d}/model_best.pth') else 'MISSING'}  {lab:15} {os.path.relpath(d, DRIVE)}/model_best.pth")
'''

GRAPHS_CHECK = '''# 2b. Both test graphs must be in the bundle. The OpenIE graph directory IS the corpus directory,
# so its raw/documents.json is already where the loader looks.
for g in (FRAME_TEST, OPENIE_TEST):
    assert os.path.exists(f"{DATA_ROOT}/{g}/processed/stage1/nodes.csv"), f"missing graph {g}: re-bundle after MIR_RUNBOOK.md section 9"
    dst = f"{DATA_ROOT}/{g}/raw/documents.json"
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy(f"{cp.corpus_dir('test')}/raw/documents.json", dst)
PROBES = cp.probes_path("test")
assert os.path.exists(PROBES), PROBES
print("graphs ok:", FRAME_TEST, OPENIE_TEST, "| probes", os.path.basename(PROBES))
'''

COMPONENTS = '''# 4. Component tables for BOTH test graphs (operator + scorer views), the scorer files from Drive,
# and the cached Qwen3 node indexes.
import numpy as np
if os.path.isdir(f"{CACHE}/op_emb"):
    shutil.copytree(f"{CACHE}/op_emb", f"{CARGO_ROOT}/outputs/caches/op_emb", dirs_exist_ok=True); print("restored embedding caches")
def restore_index(g):
    src = f"{CACHE}/index/{g}"
    if not os.path.isdir(src):
        print(f"  {g}: no cached index on Drive (built on first use, 5-15 min for a test graph)"); return
    for d in os.listdir(src):
        shutil.copytree(f"{src}/{d}", f"{DATA_ROOT}/{g}/processed/{d}", dirs_exist_ok=True)
    print(f"  {g}: index restored")
def save_index(g):
    pr = f"{DATA_ROOT}/{g}/processed"
    for d in os.listdir(pr):
        if d != "stage1": shutil.copytree(f"{pr}/{d}", f"{CACHE}/index/{g}/{d}", dirs_exist_ok=True)
def _sem_ok(p):
    if not os.path.exists(p): return False
    zz = np.load(p, allow_pickle=True)
    return os.path.exists(str(zz["h_path"])) and "qwen" in str(zz["encoder"]).lower()
def opc(g):  return f"{DATA_ROOT}/{g}/operator_components{OP_SLUG}.npz"
def semc(g): return f"{DATA_ROOT}/{g}/semantic_components{OP_SLUG}.npz"

SEM = f"{S4}/results/semantic_{DATASET}"; SEM_DRIVE = f"{OUT_ROOT}/semantic"
SEM_CKPT = f"{SEM}/params_semantic_mlp_fixedloss_{DATASET}.json"
SEM_POP  = f"{SEM}/popnet_semantic_mlp_fixedloss_{DATASET}.pt"
os.makedirs(SEM, exist_ok=True)
if os.path.isdir(SEM_DRIVE): shutil.copytree(SEM_DRIVE, SEM, dirs_exist_ok=True)
for p in (SEM_CKPT, SEM_POP): assert os.path.exists(p), f"scorer file missing on Drive: {p}"
print("scorer: jmax", json.load(open(SEM_CKPT))["jmax"])

for g in (FRAME_TEST, OPENIE_TEST):
    restore_index(g)
    if not os.path.exists(opc(g)):
        c = f"{CACHE}/{g}_operator_components{OP_SLUG}.npz"
        if os.path.exists(c): shutil.copy(c, opc(g))
        else:
            sh(f"python3 -u experiments/probe_greasoner/precompute_operator_components.py "
               f"--dataset {DATASET} --graph {g} --split test --model {OP_MODEL}", KGDIR)
            shutil.copy(opc(g), c)
    if not _sem_ok(semc(g)):      # the H memmap does not survive a runtime reset; seconds for a test split
        sh(f"python3 -u experiments/probe_greasoner/precompute_semantic_components.py "
           f"--dataset {DATASET} --model {OP_MODEL} --graph {g} --split test", KGDIR)
    zz = np.load(semc(g), allow_pickle=True)
    print(f"  {g:18} H {tuple(int(x) for x in zz['h_shape'])}  Jmax={int(zz['Jmax'])}")
shutil.copytree(f"{CARGO_ROOT}/outputs/caches/op_emb", f"{CACHE}/op_emb", dirs_exist_ok=True)
'''

MODEL_ENV = '''# 5. Environment for a checkpoint on a graph (mirrors build_qualitative_notebook.py).
import torch
def inspect_ckpt(ckpt):
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
    resp = [k for k in sd if "resp_" in k]; sem = [k for k in sd if k.startswith("sem_")]
    hid = next((int(sd[k].shape[0]) for k in resp if k.endswith("resp_proj.0.weight")), None)
    jmax = next((int(sd[k].shape[1]) - 2 for k in sem if k.endswith("sem_net.0.weight")), None)
    return {"tensors": len(sd), "resp_keys": len(resp), "sem_keys": len(sem), "ccmp_hid": hid, "jmax": jmax}

def model_env(ckpt, graph, gate=None):
    info = inspect_ckpt(ckpt)
    st = json.load(open(SEM_CKPT))
    assert int(st["jmax"]) == info["jmax"], f"scorer width mismatch: ckpt jmax={info['jmax']} vs scorer jmax={st['jmax']}"
    env_ = dict(WANDB_MODE="disabled", HYDRA_FULL_ERROR="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                OPERATOR_COMPONENTS=opc(graph), OPERATOR_COMPONENTS_TEST=opc(graph),
                SEMANTIC_COMPONENTS=semc(graph), SEMANTIC_COMPONENTS_TEST=semc(graph),
                SEMANTIC_CKPT=SEM_CKPT, SEMANTIC_POPNET=SEM_POP, SEM_POP_LAMBDA="1.0",
                FUSION_OBJECTIVE="hardneg", HARDNEG_HUB="50", HARDNEG_RAND="50", AUX_W="1.0",
                PER_GOLD="1", HARDNEG_GRAPH="50", STRAT_TEST=QUERIES)
    for k in list(os.environ):
        if k.startswith(("CCMP", "ROUTE")): os.environ.pop(k)
    if info["resp_keys"]:
        env_.update(CCMP="1", CCMP_HID=str(info["ccmp_hid"]), CCMP_GATE="1", CCMP_GATE_NORM="1", CCMP_ETA="0.5")
        aj = f"{os.path.dirname(ckpt)}/arm.json"
        if os.path.exists(aj):
            a_ = json.load(open(aj))
            if a_.get("ccmp_eta") is not None: env_["CCMP_ETA"] = str(a_["ccmp_eta"])
        if gate is not None: env_["CCMP_GATE"] = "1" if gate else "0"
    return env_, info

def hydra_common(graph):
    return ("--config-path config/gfm_reasoner --config-name sft_training_fusion text_emb_model=qwen3_st "
            f"datasets.cfgs.root={DATA_ROOT} datasets.cfgs.force_reload=False "
            f"datasets.train_names=[{graph}] datasets.valid_names=[{graph}] model.semantic=mlp model.cqig=false ")
print("model_env ready")
'''

SCAN = '''# 6. Scan every gold of every test query under each arm: per-channel ranks, no path search.
QIDS_FILE = f"{SCAN_OUT}/qids_all.json"
json.dump([q["id"] for q in json.load(open(QUERIES))], open(QIDS_FILE, "w"))

def scan_all(name, ckpt, graph, gate=None):
    out = f"{SCAN_OUT}/scan_all_{name}.json"
    if os.path.exists(out): print("[cached]", os.path.relpath(out, DRIVE)); return out
    env_, info = model_env(ckpt, graph, gate=gate)
    print(f"[ckpt] {name}: {os.path.relpath(ckpt, DRIVE)} {info}" + (f" gate={'on' if gate else 'off'}" if gate is not None else ""))
    rl = f"{RUNS}/scan_all_{name}"; os.makedirs(rl, exist_ok=True)
    rc = sh("python -u -m gfmrag.workflow.interpret_paths " + hydra_common(graph) +
            f"+interp.ckpt={ckpt} +interp.qids_file={QIDS_FILE} +interp.out={out} +interp.probes={PROBES} "
            f"+interp.paths=0 +interp.max_golds=8 +interp.top_views=1 hydra.run.dir={rl}",
            "/content/gfm-rag", extra=env_, log=f"{rl}/console.log", check=False)
    assert rc == 0 and os.path.exists(out), f"{name} failed (exit {rc}); read {rl}/console.log"
    save_index(graph)
    return out

ARMS = [("frame_ccmp",     f"{FRAME_CCMP_RUN}/model_best.pth", FRAME_TEST,  True),
        ("frame_ccmp_off", f"{FRAME_CCMP_RUN}/model_best.pth", FRAME_TEST,  False),
        ("frame_nocc",     f"{FRAME_NOCC_RUN}/model_best.pth", FRAME_TEST,  None),
        ("openie",         f"{OPENIE_RUN}/model_best.pth",     OPENIE_TEST, None)]
SA = {}
for name, ckpt, graph, gate in ARMS:
    if not os.path.exists(ckpt):
        print(f"[skip] {name}: no checkpoint at {os.path.relpath(ckpt, DRIVE)}"); continue
    SA[name] = scan_all(name, ckpt, graph, gate)
print("scanned:", sorted(SA))
'''

TABLE = '''# 7. The table: every gold of every query, graph channel and the other channels, per arm; then
# gold-by-gold win rates. MIR has one stratum, so one block.
import numpy as np
T = {k: {(r["id"], t["doc"]): t["rank"] for r in json.load(open(v)) for t in r["targets"]} for k, v in SA.items()}
common = sorted(set.intersection(*[set(T[k]) for k in T]))
n_q = len({q for q, _ in common})
CH = ["graph", "scorer", "dense", "fused"]
LAB = {"frame_ccmp": "frame graph + CCMP", "frame_ccmp_off": "frame graph, CCMP gate off (same weights)",
       "frame_nocc": "frame graph, no CCMP (own run)", "openie": "OpenIE entity graph"}
lines = []
def out(s=""): print(s); lines.append(s)
out(f"# MIR graph-channel scan: {len(common)} golds of {n_q} test queries, every gold, no path search\\n")
out("| arm | channel | median rank | R@5 | R@10 | R@50 | R@100 |"); out("|---|---|--:|--:|--:|--:|--:|")
res = {}
for arm in SA:
    for ch in CH:
        v = np.array([T[arm][k].get(ch, np.nan) for k in common], dtype=float)
        if np.all(np.isnan(v)): continue
        res[(arm, ch)] = v
        out(f"| {LAB.get(arm, arm)} | {ch} | {np.nanmedian(v):.0f} | {100*np.nanmean(v<=5):.1f} | {100*np.nanmean(v<=10):.1f} | "
            f"{100*np.nanmean(v<=50):.1f} | {100*np.nanmean(v<=100):.1f} |")
def wins(a, b, ch="graph"):
    x, y = res[(a, ch)], res[(b, ch)]
    return 100*np.mean(x<y), 100*np.mean(x==y), 100*np.mean(x>y)
out("\\nGold-by-gold, graph channel (rank lower is better):")
for a, b, lab in (("frame_ccmp", "openie", "frame + CCMP vs OpenIE"), ("frame_nocc", "openie", "frame no-CCMP vs OpenIE"),
                  ("frame_ccmp", "frame_ccmp_off", "CCMP gate on vs off (same weights)"), ("frame_ccmp", "frame_nocc", "CCMP run vs no-CCMP run")):
    if (a, "graph") in res and (b, "graph") in res:
        w, t, l = wins(a, b); out(f"- {lab}: first better {w:.0f}%  tie {t:.0f}%  second better {l:.0f}%")
out("\\nmedian = median rank of the gold in the 4,857-document corpus. graph = graph channel alone; scorer = multi-view "
    "scorer alone; dense = Qwen3 cosine; fused = the model's output ranking. Compare with the SIR-4 CS table "
    "(frame graph median 31/16, OpenIE 1,307/770).")
json.dump({"n_golds": len(common), "n_queries": n_q,
           "stats": {f"{a}/{c}": {"median": float(np.nanmedian(v)), "r5": float(np.nanmean(v<=5)), "r10": float(np.nanmean(v<=10)),
                                  "r50": float(np.nanmean(v<=50)), "r100": float(np.nanmean(v<=100))} for (a, c), v in res.items()},
           "ranks": {f"{a}/{c}": {f"{q}|{d}": float(x) for (q, d), x in zip(common, v)} for (a, c), v in res.items()}},
          open(f"{SCAN_OUT}/graph_channel_scan_mir.json", "w"), indent=1)
open(f"{SCAN_OUT}/graph_channel_scan_mir.md", "w").write("\\n".join(lines) + "\\n")
print("\\nwrote", os.path.relpath(f"{SCAN_OUT}/graph_channel_scan_mir.md", DRIVE))
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    src = json.load(open(rb.SRC_NB))["cells"]
    built = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    fusion_files = {f"/content/gfm-rag/{rel}": open(f"{FORK}/{rel}").read() for rel in FUSION_REL}
    assert not any("'''" in v for v in fusion_files.values())
    assert "def interpret(" in fusion_files["/content/gfm-rag/gfmrag/trainers/fusion_trainer.py"], "trainer lacks interpret()"
    files_cell = code(
        f"# === write the CARGO-fusion files into the fork (generated from the repo copies {built}) ===\n"
        "import json, os\n"
        f"FILES = json.loads(r'''{json.dumps(fusion_files)}''')\n"
        "for p, c in FILES.items():\n"
        "    os.makedirs(os.path.dirname(p), exist_ok=True)\n"
        "    open(p, 'w').write(c)\n"
        "    print('wrote', p, f'({len(c)} bytes)')\n"
        "import importlib, sys\n"
        "sys.path.insert(0, '/content/gfm-rag')\n"
        "for m in ['gfmrag.models.fusion_reasoner', 'gfmrag.trainers.fusion_trainer', 'gfmrag.workflow.interpret_paths']:\n"
        "    importlib.import_module(m); print('import OK:', m)\n"
        "print('fusion files ready (path interpretation included)')\n")
    overlay = {rel: open(f"{CARGO}/{rel}").read() for rel in rb.OVERLAY_REL}
    assert not any("'''" in v for v in overlay.values())

    paths = (bm.PATHS.replace("__EPOCHS__", str(a.epochs)).replace("__BATCH__", str(a.batch))
             .replace("__CCMP__", "standard").replace("__GRAPH__", "frame")) + EXTRA_PATHS
    cells = [md(HEADER), md("## 1. GPU + Drive + paths"), code(paths),
             md("## 2. Unpack + install the current scripts"),
             code(bm.UNPACK.replace("__OVERLAY__", json.dumps(overlay)).replace("__BUILT__", built)),
             code(GRAPHS_CHECK),
             md("## 2c. Qwen3-Embedding-0.6B (cached on Drive)"), src[rb.QWEN_CELLS[1]],
             md("## 3. Engine + fusion sources (adds `interpret_paths.py`)")]
    for i in rb.ENGINE_CELLS:
        cells.append(files_cell if i == 7 else src[i])
    cells += [md("## 4. Component tables for both graphs + scorer files"), code(COMPONENTS),
              md("## 5. Checkpoint environment"), code(MODEL_ENV),
              md("## 6. Scan every gold under every arm"), code(SCAN),
              md("## 7. The table"), code(TABLE)]

    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        outl, mag = [], False
        for ln in "".join(c["source"]).split("\n"):
            if ln.lstrip().startswith(("!", "%")) or mag:
                mag = ln.rstrip().endswith("\\"); outl.append("pass")
            else:
                outl.append(ln)
        try:
            ast.parse("\n".join(outl))
        except SyntaxError as e:
            raise SystemExit(f"cell {i} line {e.lineno}: {e.msg}")
    out = a.out or f"{ROOT}/colab_mir_scan.ipynb"
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                                       "language_info": {"name": "python"}, "accelerator": "GPU"},
          "nbformat": 4, "nbformat_minor": 5}
    json.dump(nb, open(out, "w"), indent=1)
    print(f"wrote {out}: {len(cells)} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
