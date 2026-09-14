"""
build_sir4_hyb_notebook.py -- SIR-4, all four fields in ONE notebook: SciGraphIR on the merged graph
(frame + entity seeds + mention edges + mechanism shortcuts, prep/build_hybrid_graph.py) without and
with CCMP, compared against the existing OpenIE-graph row and the scorer alone.

Fields run quickest first (matsci, physics, biology, cs). Every (field, arm) is its OWN cell, finished
runs skip, and each field ends with its thesis table (markdown with the graph-channel diagnostics, and
LaTeX). No routing arms. Needs on Drive: sir4_<field>_bundle.zip x4, sir4_hyb_bundle.zip, the scorer
files under outputs/sir4_<field>/semantic, and the OpenIE row's score files under outputs/sir4_openie/scores.

    python3 prep/build_sir4_hyb_notebook.py            # -> colab_sir4_hyb.ipynb
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
import build_rb_zeroshot_notebook as rb          # noqa: E402
import build_sir4_openie_notebook as oi          # noqa: E402

ROOT, CARGO, FORK = rb.ROOT, rb.CARGO, rb.FORK
md, code = rb.md, rb.code
ORDER = ["matsci", "physics", "biology", "cs"]     # quickest first (1.3k, 3.1k, 4.0k, 5.4k training queries)
TITLE = {"cs": "CS", "biology": "Bio", "physics": "Phys.", "matsci": "MS"}

HEADER = '''# SIR-4, merged graph: SciGraphIR without / with CCMP vs the OpenIE-graph row, four fields, one notebook

The merged graph (`sir4_<field>_{train,test}_hyb`) is the SciAffordGraph plus the query's named entities as
extra seeds, entity->paper mention edges from the OpenIE graph (entities in at most 30 papers) and direct
paper->function / limitation / mechanism shortcut edges. Every arm shares the field's multi-view scorer
warm start, the graph reasoner, the fusion gate and the ranking loss; the rows differ only in the graph
and in the CCMP head. No routing arms.

| field | training queries | est. per epoch (batch 2) | arms |
|---|--:|--:|---|
| matsci | 1,304 | ~11 min | merged (no CCMP), merged + CCMP |
| physics | 3,087 | ~27 min | same |
| biology | 3,954 | ~35 min | same |
| cs | 5,445 | ~48 min (batch 1: ~95 min) | same |

Each (field, arm) is its own cell; a finished run on Drive is skipped, so the notebook can be re-run after a
disconnect. After both arms of a field, that field's table cell scores them, pulls the OpenIE row and the
scorer-alone row, runs the paired bootstrap on the cross slice and writes the LaTeX table to
`outputs/sir4_hyb/table_sir4_<field>.tex`. The last cell assembles the four-field table with the macro gap.
'''

ARMS_CELL = '''# 4b. The two arms. CCMP_ENV is the standard CCMP recipe of the thesis rows (gate on, gate norm, eta 0.5,
# 64 negatives, 2000 targets, pool 256, residual off, head lr 2e-3). The control has no CCMP variable at all.
CCMP_ENV = dict(CCMP="1", CCMP_W="1.0", CCMP_NEG="64", CCMP_M="2000", CCMP_LR="2e-3",
                CCMP_GATE="1", CCMP_GATE_NORM="1", CCMP_ETA="0.5",
                CCMP_POOL="256", CCMP_M_POS="512", CCMP_M_NEG="512",
                CCMP_RESIDUAL="0", CCMP_IDENTITY_W="1.0")
ARM_DEFS = {"control": ("SciGraphIR, merged graph (no CCMP)", {}),
            "ccmp":    ("SciGraphIR, merged graph + CCMP",   dict(CCMP_ENV))}
EXPECT = {"control": (["semantic='mlp' warm start"], ["[ccmp] responsibility head", "[ccmp targets]"]),
          "ccmp":    (["semantic='mlp' warm start", "[ccmp] responsibility head", "[ccmp targets]"], [])}

def run_cmd(d, run_dir, epochs, max_steps=None, predict=True):
    return ("python -u -m gfmrag.workflow.sft_training "
            "--config-path config/gfm_reasoner --config-name sft_training_fusion text_emb_model=qwen3_st "
            f"datasets.cfgs.root={DATA_ROOT} datasets.cfgs.force_reload=False "
            f"datasets.train_names=[{g_of(d, 'train')}] datasets.valid_names=[{g_of(d, 'test')}] "
            "model.semantic=mlp model.cqig=false "
            f"trainer.args.num_epoch={epochs} trainer.args.train_batch_size={batch_of(d)} "
            f"+trainer.args.eval_batch_size={batch_of(d)} "
            f"+trainer.args.do_predict={'true' if predict else 'false'} +trainer.args.predict_top_k=300 "
            + (f"trainer.args.max_steps_per_epoch={max_steps} " if max_steps else "")
            + f"hydra.run.dir={run_dir}")

def run_name(d, key): return f"sir4_{d}_hyb_{key}_e{EPOCHS}_b{batch_of(d)}"
RUN = {}
def train_arm(d, key):
    """Smoke (1 epoch, 3 steps, log asserted), then the full run; local dir + 10-min Drive sync; skips a finished run."""
    name = run_name(d, key); local, drive = f"{RUNS}/{name}", f"{OUT_ROOT}/{name}"
    pred_drive = f"{drive}/predictions_{g_of(d, 'test')}.json"
    if os.path.exists(pred_drive) and os.path.exists(f"{drive}/model_best.pth"):
        print(f"[skip] finished: {os.path.relpath(drive, DRIVE)}"); RUN[(d, key)] = drive; return drive
    for k in list(os.environ):
        if k.startswith(("CCMP", "ROUTE", "STRAT_", "CQIG", "RESID_", "MISS_W")): os.environ.pop(k)
    ex = dict(prep_field(d), **ARM_DEFS[key][1])
    smoke = f"{local}_smoke"; os.makedirs(smoke, exist_ok=True)
    rc = sh(run_cmd(d, smoke, epochs=1, max_steps=3, predict=False), "/content/gfm-rag", extra=ex, log=f"{smoke}/console.log", check=False)
    log = open(f"{smoke}/console.log", errors="ignore").read()
    assert rc == 0, f"{d}/{key}: smoke failed (exit {rc}); read {smoke}/console.log"
    must, must_not = EXPECT[key]
    for s_ in must: assert s_ in log, f"{d}/{key}: smoke log lacks {s_!r}"
    for s_ in must_not: assert s_ not in log, f"{d}/{key}: smoke log contains {s_!r}, wrong arm"
    for s in ("train", "test"): save_index(g_of(d, s), S4_CACHE[d])
    print(f"{d}/{key}: smoke passed, starting the {EPOCHS}-epoch run ({os.path.relpath(drive, DRIVE)})")
    os.makedirs(local, exist_ok=True); os.makedirs(drive, exist_ok=True)
    json.dump({"arm": ARM_DEFS[key][0], "field": d, "graph": "hyb", "train": g_of(d, "train"), "valid": g_of(d, "test"),
               "epochs": EPOCHS, "batch": batch_of(d), "semantic": "mlp", "ccmp": key == "ccmp",
               "ccmp_eta": 0.5 if key == "ccmp" else None, "ccmp_gate": key == "ccmp",
               "started": time.strftime("%Y-%m-%dT%H:%M:%S")}, open(f"{local}/arm.json", "w"), indent=1)
    finish = start_sync(local, drive, every=600)
    try:
        rc = sh(run_cmd(d, local, epochs=EPOCHS), "/content/gfm-rag", extra=ex, log=f"{local}/console.log", check=False)
    finally:
        finish()
    assert rc == 0, f"{d}/{key}: training failed, exit {rc} (-9 = host RAM, CUDA OOM = lower BATCH_OF[{d!r}])"
    assert os.path.exists(f"{local}/predictions_{g_of(d, 'test')}.json"), f"{d}/{key}: no predictions written"
    sync_dir(local, drive); RUN[(d, key)] = drive
    return drive
print("arms:", {k: v[0] for k, v in ARM_DEFS.items()})
'''

TRAIN_CELL = '''# {field}: {label}
train_arm("{field}", "{key}")
'''

TABLE_CELL = '''# __FIELD__: table. Scores the two merged-graph arms, reads the OpenIE row's score files and scores the scorer alone,
# parses the graph-channel diagnostics from the console logs, bootstraps on the cross slice, writes md + LaTeX.
d = "__FIELD__"
SCORES = f"{OUT_ROOT}/scores"; BOOT = f"{OUT_ROOT}/bootstrap"; os.makedirs(SCORES, exist_ok=True); os.makedirs(BOOT, exist_ok=True)
def final_diag(log_path):
    if not os.path.exists(log_path): return {}
    log = open(log_path, errors="ignore").read()
    tail = log[log.rfind("Running final evaluation"):] if "Running final evaluation" in log else log
    o = {}
    m = re.search(r"\\[diag\\] nDCG@5\\s+semantic ([\\d.]+) \\| graph ([\\d.]+) \\| fused ([\\d.]+)\\s+gamma mean ([\\d.]+)", tail)
    if m: o.update(sem=float(m[1]), graph=float(m[2]), gamma=float(m[4]))
    m = re.search(r"\\[prior\\] parameter-free walk nDCG@5 ([\\d.]+)", tail)
    if m: o.update(walk=float(m[1]))
    m = re.search(r"1k\\+ ([\\d.]+) \\(n=(\\d+)\\)", tail)
    if m: o.update(buried=float(m[1]), buried_n=int(m[2]))
    best = re.findall(r"New best model! document_ndcg@5: ([\\d.]+) at epoch (\\d+)", log)
    if best: o.update(best_epoch=int(best[-1][1]))
    return o
def score_pred(key, pred, label):
    js, pq = f"{SCORES}/{key}_scores.json", f"{SCORES}/{key}_perquery.json"
    if not os.path.exists(js):
        preds = json.load(open(pred)); preds = list(preds.values()) if isinstance(preds, dict) else preds
        have = {r["id"] for r in preds}
        for x in json.load(open(QUERIES[d])):
            if x["id"] not in have:     # a seedless test query the loader dropped counts as a miss
                preds.append({"id": x["id"], "stratum": x.get("stratum"), "supporting_documents": x["supporting_documents"], "predictions": {"document": []}})
        filled = f"{SCORES}/{key}_predictions.json"; json.dump(preds, open(filled, "w"))
        sh([sys.executable, "-u", "eval/score_sir4.py", "--pred", filled, "--queries", QUERIES[d], "--sets", SETS[d],
            "--name", label, "--cols", STD_COLS, "--json-out", js, "--per-query-out", pq], S4)
    return json.load(open(js))
ROWS = []      # (label, scores, diag, perquery key or None)
_sp = f"{DRIVE}/outputs/sir4_{d}/semantic/predictions_semantic_mlp_fixedloss_sir4_{d}_test.json"
if os.path.exists(_sp): ROWS.append(("Multi-View Semantic Scorer", score_pred(f"{d}_scorer", _sp, "scorer alone"), {}, f"{d}_scorer"))
_oj = f"{DRIVE}/outputs/sir4_openie/scores/{d}_openie_graph_scores.json"; _opq = f"{DRIVE}/outputs/sir4_openie/scores/{d}_openie_graph_perquery.json"
if os.path.exists(_oj):
    if os.path.exists(_opq): shutil.copy(_opq, f"{SCORES}/{d}_openie_perquery.json")
    ROWS.append(("SciGraphIR, OpenIE graph (no CCMP)", json.load(open(_oj)),
                 final_diag(f"{DRIVE}/outputs/sir4_openie/sir4_{d}_openie_qwenmlp_graph_e10_b{batch_of(d)}/console.log"), f"{d}_openie"))
else:
    print("no OpenIE score file for", d, "(run colab_sir4_openie_ablation.ipynb); row omitted")
for key in ("control", "ccmp"):
    drv = RUN.get((d, key)) or f"{OUT_ROOT}/{run_name(d, key)}"
    pred = f"{drv}/predictions_{g_of(d, 'test')}.json"
    if not os.path.exists(pred): print(f"{d}/{key}: not finished, row omitted"); continue
    ROWS.append((ARM_DEFS[key][0], score_pred(f"{d}_hyb_{key}", pred, ARM_DEFS[key][0]), final_diag(f"{drv}/console.log"), f"{d}_hyb_{key}"))
def paired(a, b, tag, sl="cross"):
    pa, pb = f"{SCORES}/{a}_perquery.json", f"{SCORES}/{b}_perquery.json"
    js = f"{BOOT}/{tag}_{sl}.json"
    if os.path.exists(pa) and os.path.exists(pb) and not os.path.exists(js):
        sh([sys.executable, "-u", "transfer/paired_bootstrap.py", "--a", pa, "--b", pb, "--a-name", a, "--b-name", b,
            "--metrics", "ndcg@5,recall@5", "--slice", sl, "--iters", "10000", "--json-out", js], S4, check=False)
    if not os.path.exists(js): return None
    v = json.load(open(js))["metrics"].get("ndcg@5")
    return None if not v else (100 * v["diff"], 100 * v["ci95"][0], 100 * v["ci95"][1], not v["crosses_zero"])
CI = {}
CI["control_vs_openie"] = paired(f"{d}_hyb_control", f"{d}_openie", f"{d}_hyb_control_vs_openie")
CI["ccmp_vs_control"] = paired(f"{d}_hyb_ccmp", f"{d}_hyb_control", f"{d}_hyb_ccmp_vs_control")
CI["ccmp_vs_openie"] = paired(f"{d}_hyb_ccmp", f"{d}_openie", f"{d}_hyb_ccmp_vs_openie")
def ci_txt(v): return "--" if v is None else f"{v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]" + ("*" if v[3] else "")
lines = [f"## SIR-4 {d}: merged graph vs OpenIE graph ({len(json.load(open(QUERIES[d])))} test queries)\\n",
         "| row | nDCG@5 same | nDCG@5 cross | gap | R@5 same | R@5 cross | graph nDCG@5 | walk | gamma | buried R@10 | best ep | vs OpenIE (cross nDCG@5, 95% CI) |",
         "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|"]
f = lambda x: "--" if x is None else f"{100 * x:.2f}"
for label, sc, dg, pk in ROWS:
    ns, nc = sc["same"]["ndcg@5"], sc["cross"]["ndcg@5"]
    ci = CI["control_vs_openie"] if pk and pk.endswith("hyb_control") else (CI["ccmp_vs_openie"] if pk and pk.endswith("hyb_ccmp") else None)
    lines.append(f"| {label} | {100*ns:.2f} | {100*nc:.2f} | {100*(nc-ns)/ns:+.1f}% | {100*sc['same']['recall@5']:.2f} | {100*sc['cross']['recall@5']:.2f} | "
                 f"{f(dg.get('graph'))} | {f(dg.get('walk'))} | {dg.get('gamma', float('nan')):.3f} | {f(dg.get('buried'))} | {dg.get('best_epoch', '--')} | {ci_txt(ci)} |")
lines.append(f"\\nCCMP vs no CCMP on the merged graph (cross nDCG@5, paired bootstrap): {ci_txt(CI['ccmp_vs_control'])}")
lines.append("graph = graph channel alone at the best epoch; walk = zero-parameter walk prior on the same seeds; buried = graph recall@10 on golds the scorer ranked past 1,000. * = 95% CI excludes zero.")
print("\\n".join(lines)); open(f"{OUT_ROOT}/table_sir4_{d}.md", "w").write("\\n".join(lines) + "\\n")
BS = chr(92); EOL = " " + BS + BS
tex = [BS + "begin{table}[t]", BS + "centering", BS + "small", BS + "setlength{" + BS + "tabcolsep}{5pt}", BS + "renewcommand{" + BS + "arraystretch}{1.2}",
       BS + "begin{tabular}{@{}l cc c cc c@{}}", BS + "toprule",
       "& " + BS + "multicolumn{2}{c}{nDCG@5} & & " + BS + "multicolumn{2}{c}{Recall@5} & $" + BS + "Delta$ vs." + BS + " OpenIE graph" + EOL,
       BS + "cmidrule(lr){2-3}" + BS + "cmidrule(lr){5-6}",
       BS + "textbf{Method} & Same & Cross & $" + BS + "Delta$ & Same & Cross & (cross nDCG@5, 95" + BS + "% CI)" + EOL, BS + "midrule"]
for label, sc, dg, pk in ROWS:
    ns, nc = sc["same"]["ndcg@5"], sc["cross"]["ndcg@5"]
    ci = CI["control_vs_openie"] if pk and pk.endswith("hyb_control") else (CI["ccmp_vs_openie"] if pk and pk.endswith("hyb_ccmp") else None)
    cit = "--" if ci is None else f"${ci[0]:+.2f}$ [{ci[1]:+.2f}, {ci[2]:+.2f}]" + ("$^{*}$" if ci[3] else "")
    lab = label.replace("SciGraphIR", BS + "textsc{SciGraphIR}")
    tex.append(f"{lab} & {100*ns:.2f} & {100*nc:.2f} & ${100*(nc-ns)/ns:+.2f}" + BS + f"%$ & {100*sc['same']['recall@5']:.2f} & {100*sc['cross']['recall@5']:.2f} & {cit}" + EOL)
tex += [BS + "bottomrule", BS + "end{tabular}",
        BS + "caption{Graph construction on SIR-4 " + TITLE[d] + " (nDCG@5 and Recall@5, " + BS + "%). All rows share the multi-view scorer, "
        "graph reasoner, fusion gate and ranking loss; the merged graph adds the query's named entities as seeds, entity--paper mention "
        "edges and direct paper--mechanism edges to the SciAffordGraph. $" + BS + "Delta$ is the relative same-to-cross reduction; the last "
        "column is the paired bootstrap over cross-field queries against the OpenIE-graph row; $^{*}$ marks a 95" + BS + "% CI that excludes zero.}",
        BS + "label{tab:hyb-sir4-" + d + "}", BS + "end{table}"]
open(f"{OUT_ROOT}/table_sir4_{d}.tex", "w").write("\\n".join(tex) + "\\n"); print("\\n" + "\\n".join(tex)); print("\\nwrote", f"{OUT_ROOT}/table_sir4_{d}.tex")
'''

ALL_TABLE = '''# All-fields table (nDCG@5 same / cross per field, macro gap), from the score files written above.
ROWS4 = [("Multi-View Semantic Scorer", "{d}_scorer"), ("SciGraphIR, OpenIE graph (no CCMP)", "{d}_openie_fromdrive"),
         ("SciGraphIR, merged graph (no CCMP)", "{d}_hyb_control"), ("SciGraphIR, merged graph + CCMP", "{d}_hyb_ccmp")]
def load_scores(d, pat):
    if pat == "{d}_openie_fromdrive":
        p = f"{DRIVE}/outputs/sir4_openie/scores/{d}_openie_graph_scores.json"
    else:
        p = f"{OUT_ROOT}/scores/{pat.format(d=d)}_scores.json"
    return json.load(open(p)) if os.path.exists(p) else None
lines = []
def out(s=""): print(s); lines.append(s)
for metric, mname in (("ndcg@5", "nDCG@5"), ("recall@5", "R@5")):
    out(f"### {mname} (%)"); out("| Method | " + " | ".join(f"same {TITLE[d]}" for d in FIELDS) + " | " + " | ".join(f"cross {TITLE[d]}" for d in FIELDS) + " | macro gap |")
    out("|---|" + "--:|" * (2 * len(FIELDS) + 1))
    for lab, pat in ROWS4:
        same, cross = [], []
        for d in FIELDS:
            s = load_scores(d, pat)
            same.append(100 * s["same"][metric] if s else None); cross.append(100 * s["cross"][metric] if s else None)
        ok = [i for i, v in enumerate(same) if v is not None]
        if not ok: out(f"| {lab} | " + " | ".join("--" for _ in range(2 * len(FIELDS))) + " | -- |"); continue
        gap = (sum(cross[i] for i in ok) / len(ok) - sum(same[i] for i in ok) / len(ok)) / (sum(same[i] for i in ok) / len(ok))
        out(f"| {lab} | " + " | ".join(f"{v:.2f}" if v is not None else "--" for v in same) + " | "
            + " | ".join(f"{v:.2f}" if v is not None else "--" for v in cross) + f" | {100 * gap:+.2f}% |")
    out()
out("Macro gap = (mean over the fields present of cross - mean of same) / mean of same.")
open(f"{OUT_ROOT}/table_sir4_hyb_all.md", "w").write("\\n".join(lines) + "\\n"); print("wrote", f"{OUT_ROOT}/table_sir4_hyb_all.md")
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", default=",".join(ORDER))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--out", default=f"{ROOT}/notebooks/colab_sir4_hyb.ipynb")
    a = ap.parse_args()
    fields = [f.strip() for f in a.fields.split(",") if f.strip()]
    ds_of = {f: f"sir4_{f}" for f in fields}

    src = json.load(open(rb.SRC_NB))["cells"]
    built = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    fusion_files = {f"/content/gfm-rag/{rel}": open(f"{FORK}/{rel}").read() for rel in rb.FUSION_REL}
    assert not any("'''" in v for v in fusion_files.values())
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
        "for m in ['gfmrag.models.fusion_reasoner', 'gfmrag.trainers.fusion_trainer']:\n"
        "    importlib.import_module(m); print('import OK:', m)\n"
        "print('fusion files ready')\n")
    overlay = {rel: open(f"{CARGO}/{rel}").read() for rel in rb.OVERLAY_REL}
    assert not any("'''" in v for v in overlay.values())
    cfg_cell = dict(src[9]); cfg_src = "".join(cfg_cell["source"])
    guard = '    assert "sir4" not in t, f"a sir4 dataset reference survived in {cfg}"\n'
    assert guard in cfg_src
    cfg_cell["source"] = cfg_src.replace(guard, "").splitlines(True)

    paths = (oi.PATHS.replace("__FIELDS__", json.dumps(fields)).replace("__DS_OF__", json.dumps(ds_of))
             .replace("__LABEL___openie", "sir4_hyb").replace("__TRAIN__", f"sir4_{fields[0]}_train_hyb").replace("__TEST__", f"sir4_{fields[0]}_test_hyb")
             .replace("__EPOCHS__", str(a.epochs)).replace("__BATCH__", str(a.batch))
             .replace('print("fields     ", FIELDS, "| OpenIE graphs", {d: f"{DS_OF[d]}_{{train,test}}" for d in FIELDS})',
                      'print("fields     ", FIELDS, "| merged graphs", {d: f"{DS_OF[d]}_{{train,test}}_hyb" for d in FIELDS})\n'
                      'HYB_BUNDLE = f"{DRIVE}/sir4_hyb_bundle.zip"      # the four merged graphs, one add-on zip\n'
                      f'TITLE = {json.dumps(TITLE)}      # field -> short name, used by the table cells'))
    # the merged biology graph (15.6k document nodes + entities) OOMs an 80 GB A100 at batch 2 once the
    # CCMP head retains all six hidden layers (2026-09-09); both arms of a field share the batch
    paths = paths.replace('BATCH_OF = {"cs": 1}', 'BATCH_OF = {"cs": 1, "biology": 1}')
    assert "sir4_hyb" in paths and "HYB_BUNDLE" in paths and "TITLE = " in paths and '"biology": 1' in paths
    unpack = (oi.UNPACK.replace("__OVERLAY__", json.dumps(overlay)).replace("__BUILT__", built).replace("__SETS_MAP__", json.dumps(oi.SETS_MAP))
              .replace('if os.path.isdir(PARK):\n    os.makedirs(os.path.dirname(KEEP), exist_ok=True); shutil.move(PARK, KEEP); print("restored caches")',
                       'assert os.path.exists(HYB_BUNDLE) and zipfile.is_zipfile(HYB_BUNDLE), f"{HYB_BUNDLE} missing or not a zip (upload sir4_hyb_bundle.zip to cargo-gfmrag/)"\n'
                       'zipfile.ZipFile(HYB_BUNDLE).extractall(SCIGRAPHIR_ROOT); print("unpacked", os.path.basename(HYB_BUNDLE))\n'
                       'if os.path.isdir(PARK):\n    os.makedirs(os.path.dirname(KEEP), exist_ok=True); shutil.move(PARK, KEEP); print("restored caches")')
              .replace('def g_of(d, s): return f"{DS_OF[d]}_{s}"        # the OpenIE graphs: NO _v16sc suffix',
                       'def g_of(d, s): return f"{DS_OF[d]}_{s}_hyb"    # the merged graphs')
              .replace('assert "entity" in types and "document" in types, f"{g_of(d, s)} is not an entity+document graph: {dict(types)}"',
                       'assert "entity" in types and "document" in types and "function" in types, f"{g_of(d, s)} is not a merged graph: {dict(types)}"')
              .replace('        print(f"  {g_of(d, s):22} entity {types[\'entity\']:>8,}  document {types[\'document\']:>7,}  "',
                       '        print(f"  {g_of(d, s):24} entity {types[\'entity\']:>7,}  frames {sum(v for k, v in types.items() if k not in (\'entity\', \'document\')):>8,}  document {types[\'document\']:>7,}  "'))
    assert "_hyb" in unpack and "HYB_BUNDLE" in unpack and '"function" in types' in unpack
    helpers = oi.HELPERS.replace("# 4a. Helpers: cache restore/save (graph-keyed, so the OpenIE graphs never collide with the\n# frame graphs' artefacts), and the per-field preparation.",
                                 "# 4a. Helpers: cache restore/save (graph-keyed, so the merged graphs get their own index and component\n# files), and the per-field preparation (embeddings, index, operator + scorer components, scorer warm start).")

    cells = [md(HEADER), md("## 1. GPU + Drive + paths"), code(paths),
             md("## 2. Unpack the field bundles + the merged-graph bundle + current scripts"), code(unpack),
             md("## 2b. Qwen3-Embedding-0.6B (cached on Drive)"), src[rb.QWEN_CELLS[1]],
             md("## 3. Engine + fusion sources")]
    for i in rb.ENGINE_CELLS:
        cells.append(files_cell if i == 7 else (cfg_cell if i == 9 else src[i]))
    cells += [md("## 4. Per-field preparation + the two arms"), code(helpers), code(ARMS_CELL)]
    for n, d in enumerate(fields, 5):
        cells += [md(f"## {n}. {TITLE[d]} (`sir4_{d}`)\nTwo training cells, then the table. A finished run on Drive is skipped."),
                  code(TRAIN_CELL.format(field=d, label="merged graph, no CCMP", key="control")),
                  code(TRAIN_CELL.format(field=d, label="merged graph + CCMP", key="ccmp")),
                  code(TABLE_CELL.replace("__FIELD__", d))]
    cells += [md(f"## {len(fields) + 5}. All fields"), code(ALL_TABLE)]

    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"]); outl, mag = [], False
        for ln in s.split("\n"):
            if ln.lstrip().startswith(("!", "%")) or mag:
                mag = ln.rstrip().endswith("\\"); outl.append("pass")
            else:
                outl.append(ln)
        try:
            ast.parse("\n".join(outl))
        except SyntaxError as e:
            raise SystemExit(f"cell {i} line {e.lineno}: {e.msg}: {outl[e.lineno - 1][:120]}")
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                                       "language_info": {"name": "python"}, "accelerator": "GPU"},
          "nbformat": 4, "nbformat_minor": 5}
    json.dump(nb, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}: {len(cells)} cells | fields {fields} | epochs {a.epochs} batch {a.batch} (cs: 1)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
