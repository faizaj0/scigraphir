# MIR graph-channel scan in ONE cell, from a fresh runtime. It replays the setup cells (paths, bundle,
# Qwen3, engine, fusion sources, patches) of a MIR notebook you already have in Drive's "Colab Notebooks"
# folder, so nothing new is uploaded, then scans every gold under the four arms and prints the table.
import os, re, json, glob, shutil, time
import sys
# Colab now runs Python 3.13 and the engine's editable install refuses it, so subprocesses must be
# told where gfmrag is and which interpreter to use. Set BEFORE the replayed paths cell, which
# snapshots os.environ for its sh() helper.
os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
os.environ["PYTHONPATH"] = "/content/gfm-rag" + os.pathsep + os.environ.get("PYTHONPATH", "")
from google.colab import drive
drive.mount('/content/drive')
NB_PATH = ""   # optional: full path of a colab_mir_*.ipynb to replay; empty = first match below
_cands = [p for name in ("colab_mir_all", "colab_mir_openie", "colab_mir_standard", "colab_mir_zeroshot", "colab_mir_scan")
          for p in glob.glob(f"/content/drive/MyDrive/**/{name}*.ipynb", recursive=True)]
def _has_interpret(p):  # only notebooks built on/after 6 Sep carry trainer.interpret() in their fusion blob
    try: return any("def interpret(" in ''.join(c["source"]) for c in json.load(open(p))["cells"] if c["cell_type"] == "code")
    except Exception: return False
_cands = [p for p in _cands if _has_interpret(p)]
NB_PATH = NB_PATH or (_cands[0] if _cands else "")
assert NB_PATH, "no colab_mir_*.ipynb with interpret() found under /content/drive/MyDrive (colab_mir_all.ipynb has it); set NB_PATH"
print("replaying setup cells from", NB_PATH)
_code = [''.join(c["source"]) for c in json.load(open(NB_PATH))["cells"] if c["cell_type"] == "code"]
# Colab saves pasted cells into the notebook; a pasted replay cell mentions every marker below, so drop it.
_code = [s for s in _code if "replaying setup cells from" not in s and "def _pick(" not in s]
def _pick(pred, which="first"):
    hits = [s for s in _code if pred(s)]
    assert hits, f"no setup cell matched {pred.__doc__}"
    return hits[0] if which == "first" else hits[-1]
def _run(label, src, fatal=True):
    print(f"\n======== replay: {label} ========")
    r = get_ipython().run_cell(src, store_history=False)
    if not r.success and not fatal:
        print(f"[warn] setup cell '{label}' failed; continuing (its check is advisory here)"); return
    assert r.success, f"setup cell '{label}' failed; see the traceback above"
_run("paths",  _pick(lambda s: s.startswith("!nvidia-smi")))
_run("unpack", _pick(lambda s: s.startswith("# 2. Unpack the MIR bundle")))
_run("qwen3",  _pick(lambda s: s.startswith("import os, shutil") and "qwen3" in s.lower()))
_run("engine", _pick(lambda s: s.startswith("import os, sys, torch") and "gfm-rag-adapted.zip" in s))
_pins = [s for s in _code if '_im.version("wandb")' in s or 'wandb.__version__.startswith("0.18.")' in s]
if _pins: _run("pins", _pins[0], fatal=False)
_run("fusion sources", _pick(lambda s: s.startswith("# === write the SciGraphIR-fusion files")))
_run("config defaults", _pick(lambda s: s.startswith("# Cell 3a is reused VERBATIM")))
_run("PyG version fix", _pick(lambda s: s.startswith("# The ULTRA layers vendored"), "last"))
_run("torchvision shim", _pick(lambda s: s.startswith("# THE TRAINING SUBPROCESS IS A FRESH PYTHON"), "last"))
_run("diagnostics patch", _pick(lambda s: s.startswith("STF = ")))
_run("CCMP_LR patch", _pick(lambda s: s.startswith("# Idempotent: re-running is a no-op")))
assert "def interpret(" in open("/content/gfm-rag/gfmrag/trainers/fusion_trainer.py").read(), "engine lacks interpret(): the replayed notebook is older than 6 Sep"
# the interpretation entry point (not in the training notebooks' fusion blob)
open("/content/gfm-rag/gfmrag/workflow/interpret_paths.py", "w").write(r'''"""
interpret_paths.py -- path interpretations for a trained fusion checkpoint.

Same construction as sft_training (config, datasets, model, trainer) with no training: the
checkpoint is loaded, then FusionSFTTrainer.interpret() runs the NBFNet-style gradient beam
search from each requested query's seed nodes to its best-ranked gold and records the CCMP
responsibility along every path. Output: one JSON.

    python -m gfmrag.workflow.interpret_paths --config-path config/gfm_reasoner \\
        --config-name sft_training_fusion text_emb_model=qwen3_st \\
        datasets.cfgs.root=... datasets.train_names=[G] datasets.valid_names=[G] \\
        model.semantic=mlp model.cqig=false \\
        +interp.ckpt=/path/model_best.pth +interp.qids_file=/path/qids.json \\
        +interp.out=/path/paths.json +interp.answers=/path/probes_test.jsonl \\
        hydra.run.dir=/path/run
"""
try:  # same torchvision shim as sft_training
    import torchvision.io as _tvio
    if not hasattr(_tvio, "VideoReader"):
        class _NoVideoReader:
            def __init__(self, *a, **k):
                raise RuntimeError("torchvision video API removed")
        _tvio.VideoReader = _NoVideoReader
except Exception:
    pass
import json
import logging
import os

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from gfmrag import utils
from gfmrag.graph_index_datasets import GraphDatasetLoader
from gfmrag.trainers.sft_trainer import SFTLoss

logger = logging.getLogger(__name__)


@hydra.main(config_path="config/gfm_rag", config_name="sft_training", version_base=None)
def main(cfg: DictConfig) -> None:
    utils.init_distributed_mode(cfg.timeout)
    torch.manual_seed(cfg.seed)
    output_dir = HydraConfig.get().runtime.output_dir
    logger.info(f"Config:\n {OmegaConf.to_yaml(cfg)}")
    feat_dim = set(utils.init_multi_dataset(cfg, 1, 0))
    assert len(feat_dim) == 1
    model = instantiate(cfg.model, feat_dim=feat_dim.pop())
    ckpt = cfg.interp.ckpt
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
    res = model.load_state_dict(sd, strict=False)
    print(f"[interpret] loaded {ckpt}: {len(sd)} tensors, missing {len(res.missing_keys)}, "
          f"unexpected {len(res.unexpected_keys)}", flush=True)
    if res.unexpected_keys:
        print("[interpret] unexpected keys (first 5):", res.unexpected_keys[:5], flush=True)
    valid_loader = GraphDatasetLoader(cfg.datasets, cfg.datasets.valid_names, shuffle=False,
                                      max_datasets_in_memory=cfg.datasets.max_datasets_in_memory,
                                      data_loading_workers=cfg.datasets.data_loading_workers)
    optimizer = instantiate(cfg.optimizer, model.parameters())
    loss_functions = [SFTLoss(name=lc.name, loss_fn=instantiate(lc.loss), weight=lc.weight,
                              target_node_type=lc.target_node_type,
                              is_distillation_loss=lc.get("is_distillation_loss", False))
                      for lc in cfg.losses]
    trainer = instantiate(cfg.trainer, output_dir=output_dir, model=model, optimizer=optimizer,
                          loss_functions=loss_functions, train_graph_dataset_loader=valid_loader,
                          eval_graph_dataset_loader=valid_loader)
    qids = json.load(open(cfg.interp.qids_file))
    golds = json.load(open(cfg.interp.golds_file)) if cfg.interp.get("golds_file") else None
    kw = dict(answers_path=cfg.interp.get("answers", cfg.interp.get("probes")),
              num_beam=int(cfg.interp.get("num_beam", 10)), path_topk=int(cfg.interp.get("path_topk", 5)),
              max_golds=int(cfg.interp.get("max_golds", 2)), top_views=int(cfg.interp.get("top_views", 3)),
              do_paths=bool(int(cfg.interp.get("paths", 1))), golds=golds,
              necessity=bool(int(cfg.interp.get("necessity", 0))), distractor=bool(int(cfg.interp.get("distractor", 0))),
              dump_k=int(cfg.interp.get("dump_scores", 0)))
    import inspect   # older fusion blobs (the qualitative notebooks) have interpret() without the later keywords
    accepted = inspect.signature(trainer.interpret).parameters
    dropped = [k for k in kw if k not in accepted]
    if dropped:
        print(f"[interpret] this trainer's interpret() lacks {dropped}; not passed", flush=True)
    trainer.interpret(qids, cfg.interp.out, **{k: v for k, v in kw.items() if k in accepted})
    valid_loader.shutdown()
    utils.synchronize()
    utils.cleanup()


if __name__ == "__main__":
    main()
''')
print("wrote interpret_paths.py")
for _n in ("DRIVE", "DATASET", "OUT_ROOT", "CACHE", "DATA_ROOT", "S4", "KGDIR", "RUNS", "OP_MODEL", "OP_SLUG", "QUERIES", "sh", "cp", "SCIGRAPHIR_ROOT"):
    assert _n in globals(), f"{_n} is not defined after the replay"
EPOCHS, BATCH = globals().get("EPOCHS", 10), globals().get("BATCH", 2)

# --- the two test graphs and the three checkpoints ------------------------------------
FRAME_TEST  = f"{DATASET}_test_v16sc"              # SciAfford SciAfford graph
OPENIE_TEST = f"{DATASET}_test"                    # OpenIE entity graph (same corpus, same queries)
FRAME_CCMP_RUN = f"{OUT_ROOT}/{DATASET}_qwenmlp_ccmp_e{EPOCHS}_b{BATCH}"
FRAME_NOCC_RUN = f"{OUT_ROOT}/{DATASET}_qwenmlp_graph_e{EPOCHS}_b{BATCH}"
OPENIE_RUN     = f"{OUT_ROOT}/{DATASET}_openie_qwenmlp_graph_e{EPOCHS}_b{BATCH}"
SCAN_OUT = f"{OUT_ROOT}/scan"
os.makedirs(SCAN_OUT, exist_ok=True)
for lab, d in (("SciAfford graph + CCMP", FRAME_CCMP_RUN), ('affordance representation, no CCMP', FRAME_NOCC_RUN), ("OpenIE", OPENIE_RUN)):
    print(f"  {'ok ' if os.path.exists(f'{d}/model_best.pth') else 'MISSING'}  {lab:15} {os.path.relpath(d, DRIVE)}/model_best.pth")

# 2b. Both test graphs must be in the bundle. The OpenIE graph directory IS the corpus directory,
# so its raw/documents.json is already where the loader looks.
for g in (FRAME_TEST, OPENIE_TEST):
    assert os.path.exists(f"{DATA_ROOT}/{g}/processed/stage1/nodes.csv"), f"missing graph {g}: re-bundle after MIR_RUNBOOK.md section 9"
    dst = f"{DATA_ROOT}/{g}/raw/documents.json"
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy(f"{cp.corpus_dir('test')}/raw/documents.json", dst)
ANSWERS = cp.answers_path("test")
assert os.path.exists(ANSWERS), ANSWERS
print("graphs ok:", FRAME_TEST, OPENIE_TEST, '| answers', os.path.basename(ANSWERS))

# 4. Component tables for BOTH test graphs (handcrafted scorer + scorer views), the scorer files from Drive,
# and the cached Qwen3 node indexes.
import numpy as np
if os.path.isdir(f"{CACHE}/op_emb"):
    shutil.copytree(f"{CACHE}/op_emb", f"{SCIGRAPHIR_ROOT}/outputs/caches/op_emb", dirs_exist_ok=True); print("restored embedding caches")
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
            sh(f"python3 -u precompute/precompute_handcrafted_components.py "
               f"--dataset {DATASET} --graph {g} --split test --model {OP_MODEL}", KGDIR)
            shutil.copy(opc(g), c)
    if not _sem_ok(semc(g)):      # the H memmap does not survive a runtime reset; seconds for a test split
        sh(f"python3 -u precompute/precompute_semantic_components.py "
           f"--dataset {DATASET} --model {OP_MODEL} --graph {g} --split test", KGDIR)
    zz = np.load(semc(g), allow_pickle=True)
    print(f"  {g:18} H {tuple(int(x) for x in zz['h_shape'])}  Jmax={int(zz['Jmax'])}")
shutil.copytree(f"{SCIGRAPHIR_ROOT}/outputs/caches/op_emb", f"{CACHE}/op_emb", dirs_exist_ok=True)

# 5. Environment for a checkpoint on a graph (mirrors build_qualitative_notebook.py).
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
                HANDCRAFTED_COMPONENTS=opc(graph), HANDCRAFTED_COMPONENTS_TEST=opc(graph),
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

# 6. Scan every gold of every test query under each arm: per-channel ranks, no path search.
QIDS_FILE = f"{SCAN_OUT}/qids_all.json"
json.dump([q["id"] for q in json.load(open(QUERIES))], open(QIDS_FILE, "w"))

def scan_all(name, ckpt, graph, gate=None):
    out = f"{SCAN_OUT}/scan_all_{name}.json"
    if os.path.exists(out): print("[cached]", os.path.relpath(out, DRIVE)); return out
    env_, info = model_env(ckpt, graph, gate=gate)
    print(f"[ckpt] {name}: {os.path.relpath(ckpt, DRIVE)} {info}" + (f" gate={'on' if gate else 'off'}" if gate is not None else ""))
    rl = f"{RUNS}/scan_all_{name}"; os.makedirs(rl, exist_ok=True)
    rc = sh("python -u -m gfmrag.workflow.interpret_paths " + hydra_common(graph) +
            f"+interp.ckpt={ckpt} +interp.qids_file={QIDS_FILE} +interp.out={out} +interp.answers={ANSWERS} "
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

# 7. The table: every gold of every query, graph channel and the other channels, per arm; then
# gold-by-gold win rates. MIR has one stratum, so one block.
import numpy as np
T = {k: {(r["id"], t["doc"]): t["rank"] for r in json.load(open(v)) for t in r["targets"]} for k, v in SA.items()}
common = sorted(set.intersection(*[set(T[k]) for k in T]))
n_q = len({q for q, _ in common})
CH = ["graph", "scorer", "dense", "fused"]
LAB = {"frame_ccmp": 'SciAfford graph + CCMP', "frame_ccmp_off": 'SciAfford graph, CCMP gate off (same weights)',
       "frame_nocc": 'SciAfford graph, no CCMP (own run)', "openie": "OpenIE entity graph"}
lines = []
def out(s=""): print(s); lines.append(s)
out(f"# MIR graph-channel scan: {len(common)} golds of {n_q} test queries, every gold, no path search\n")
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
out("\nGold-by-gold, graph channel (rank lower is better):")
for a, b, lab in (("frame_ccmp", "openie", 'affordance representation + CCMP vs OpenIE'), ("frame_nocc", "openie", 'affordance representation no-CCMP vs OpenIE'),
                  ("frame_ccmp", "frame_ccmp_off", "CCMP gate on vs off (same weights)"), ("frame_ccmp", "frame_nocc", "CCMP run vs no-CCMP run")):
    if (a, "graph") in res and (b, "graph") in res:
        w, t, l = wins(a, b); out(f"- {lab}: first better {w:.0f}%  tie {t:.0f}%  second better {l:.0f}%")
out("\nmedian = median rank of the gold in the 4,857-document corpus. graph = graph channel alone; scorer = multi-view "
    "scorer alone; dense = Qwen3 cosine; fused = the model's output ranking. Compare with the SIR-4 CS table "
    '(SciAfford graph median 31/16, OpenIE 1,307/770).')
json.dump({"n_golds": len(common), "n_queries": n_q,
           "stats": {f"{a}/{c}": {"median": float(np.nanmedian(v)), "r5": float(np.nanmean(v<=5)), "r10": float(np.nanmean(v<=10)),
                                  "r50": float(np.nanmean(v<=50)), "r100": float(np.nanmean(v<=100))} for (a, c), v in res.items()},
           "ranks": {f"{a}/{c}": {f"{q}|{d}": float(x) for (q, d), x in zip(common, v)} for (a, c), v in res.items()}},
          open(f"{SCAN_OUT}/graph_channel_scan_mir.json", "w"), indent=1)
open(f"{SCAN_OUT}/graph_channel_scan_mir.md", "w").write("\n".join(lines) + "\n")
print("\nwrote", os.path.relpath(f"{SCAN_OUT}/graph_channel_scan_mir.md", DRIVE))
