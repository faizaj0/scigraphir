# Graph-channel scan for ONE dataset in ONE cell, from a fresh runtime. Set DATASET, paste, run.
# It replays the setup cells (paths, bundle, Qwen3, engine, fusion sources, patches) of a notebook
# you already have in Drive's "Colab Notebooks" folder, so nothing new is uploaded, then ranks EVERY
# gold of every test query under each arm (graph channel alone, scorer alone, Qwen3 cosine, fused;
# no training, no path search) and prints the table per stratum. Same numbers as the SIR-4 CS
# "graph channel only, same golds, same queries" comparison, for any dataset.
DATASET = "sir4_cs"     # tomato | mir | sir4_cs | sir4_biology | sir4_physics | sir4_matsci
import os, re, json, glob, shutil, time, sys
# Colab runs Python 3.13 and the engine's editable install refuses it, so subprocesses must be told
# where gfmrag is and which interpreter to use. Set BEFORE the replayed paths cell snapshots os.environ.
os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
os.environ["PYTHONPATH"] = "/content/gfm-rag" + os.pathsep + os.environ.get("PYTHONPATH", "")
from google.colab import drive
drive.mount('/content/drive')

# --- per-dataset spec: source notebook to replay, test graphs, scorer files, checkpoints (Drive-relative) ---
def _sir4(field):
    pair = "physics+biology" if field in ("physics", "biology") else "cs+matsci"
    warm = "biology" if field in ("physics", "biology") else "cs"
    d = f"sir4_{field}"
    return dict(nb=f"colab_qualitative_{field}", frame=f"{d}_test_v16sc", openie=f"{d}_test",
                sem={"field": (f"outputs/{d}/semantic", d),                                  # the field's own scorer
                     "frame": (f"outputs/sir4_zeroshot/semantic_sir4_{warm}", f"sir4_{warm}")},  # the CCMP run's warm start
                arms=[("frame_ccmp",     f"outputs/sir4_zeroshot/scigraphir_{pair}_qwenmlp_ccmp_e10_b2", "frame",  "frame", True),
                      ("frame_ccmp_off", f"outputs/sir4_zeroshot/scigraphir_{pair}_qwenmlp_ccmp_e10_b2", "frame",  "frame", False),
                      ("frame_nocc",     [f"outputs/routing/{d}/{d}_route_control_e10_b2_s1024",
                                          f"outputs/{d}/{d}_fusion_qwenmlp_epoch10_b2"],                     "frame",  "field", None),
                      ("openie",         f"outputs/sir4_openie/{d}_openie_qwenmlp_graph_e10_b2",          "openie", "field", None)])
SPEC = {
    "mir": dict(nb="colab_mir_all", frame="mir_test_v16sc", openie="mir_test",
                extra={"hyb": "mir_test_hyb"},             # merged graph (prep/build_hybrid_graph.py; mir_hyb_bundle.zip on Drive)
                sem={"field": ("outputs/mir/semantic", "mir")},
                arms=[("frame_ccmp",     "outputs/mir/mir_qwenmlp_ccmp_e10_b2",         "frame",  "field", True),
                      ("frame_ccmp_off", "outputs/mir/mir_qwenmlp_ccmp_e10_b2",         "frame",  "field", False),
                      ("frame_nocc",     "outputs/mir/mir_qwenmlp_graph_e10_b2",        "frame",  "field", None),
                      ("openie",         "outputs/mir/mir_openie_qwenmlp_graph_e10_b2", "openie", "field", None),
                      ("hyb_nocc",       "outputs/routing_hyb/mir/mir_hyb_route_control_e10_b2_s1024", "hyb", "field", None),
                      ("hyb_ccmp",       "outputs/routing_hyb/mir/mir_hyb_route_ccmp_e10_b2_s1024",    "hyb", "field", True),
                      ("hyb_ccmp_off",   "outputs/routing_hyb/mir/mir_hyb_route_ccmp_e10_b2_s1024",    "hyb", "field", False)]),
    "tomato": dict(nb="colab_routing_tomato", frame="tomato_test_v16sc", openie="tomato_test",   # OpenIE arm: colab_tomato_openie_ablation.ipynb
                   extra={"hyb": "tomato_test_hyb"},        # merged graph (prep/build_hybrid_graph.py; tomato_hyb_bundle.zip on Drive)
                   sem={"field": ("outputs/tomato_ablations_v1/tomato/semantic", "tomato")},
                   arms=[("frame_ccmp",     "outputs/tomato_ablations_v1/tomato/tomato_fusion_qwenmlp_ccmp_epoch10_b2", "frame", "field", True),
                         ("frame_ccmp_off", "outputs/tomato_ablations_v1/tomato/tomato_fusion_qwenmlp_ccmp_epoch10_b2", "frame", "field", False),
                         ("frame_nocc",     ["outputs/tomato_ablations_v1/tomato/tomato_fusion_qwenmlp_epoch10_b2",
                                             "outputs/routing/tomato/tomato_route_control_e10_b2_s1024"],              "frame", "field", None),
                         ("openie",         "outputs/tomato_openie/tomato_openie_qwenmlp_graph_e10_b2",                  "openie", "field", None),
                         ("hyb_nocc",       "outputs/routing_hyb/tomato/tomato_hyb_route_control_e10_b2_s1024",          "hyb",    "field", None),
                         ("hyb_ccmp",       "outputs/routing_hyb/tomato/tomato_hyb_route_ccmp_e10_b2_s1024",             "hyb",    "field", True),
                         ("hyb_ccmp_off",   "outputs/routing_hyb/tomato/tomato_hyb_route_ccmp_e10_b2_s1024",             "hyb",    "field", False)]),
}
for _f in ("cs", "biology", "physics", "matsci"): SPEC[f"sir4_{_f}"] = _sir4(_f)
assert DATASET in SPEC, f"DATASET must be one of {sorted(SPEC)}"
S = SPEC[DATASET]; SCAN_DATASET = DATASET

# --- replay the source notebook's setup cells ------------------------------------------------------
NB_PATH = ""   # optional: full path of the notebook to replay; empty = first match of the spec's name
def _has_interpret(p):  # only notebooks built on/after 6 Sep carry trainer.interpret() in their fusion blob
    try: return any("def interpret(" in ''.join(c["source"]) for c in json.load(open(p))["cells"] if c["cell_type"] == "code")
    except Exception: return False
_cands = [p for p in sorted(glob.glob(f"/content/drive/MyDrive/**/{S['nb']}*.ipynb", recursive=True)) if _has_interpret(p)]
NB_PATH = NB_PATH or (_cands[0] if _cands else "")
assert NB_PATH, f"no {S['nb']}*.ipynb with interpret() found under /content/drive/MyDrive; upload it once or set NB_PATH"
print("replaying setup cells from", NB_PATH)
_code = [''.join(c["source"]) for c in json.load(open(NB_PATH))["cells"] if c["cell_type"] == "code"]
# Colab saves pasted cells into the notebook; a pasted replay cell mentions every marker below, so drop it.
_code = [s for s in _code if "replaying setup cells from" not in s and "def _pick(" not in s]
def _pick(pred, which="first"):
    hits = [s for s in _code if pred(s)]
    assert hits, "no setup cell matched; the notebook on Drive is not one this cell knows"
    return hits[0] if which == "first" else hits[-1]
def _run(label, src, fatal=True):
    print(f"\n======== replay: {label} ========")
    r = get_ipython().run_cell(src, store_history=False)
    if not r.success and not fatal:
        print(f"[warn] setup cell '{label}' failed; continuing (its check is advisory here)"); return
    assert r.success, f"setup cell '{label}' failed; see the traceback above"
_run("paths",  _pick(lambda s: s.startswith("!nvidia-smi")))
_run("unpack", _pick(lambda s: s.startswith("# 2. Unpack")))
_run("qwen3",  _pick(lambda s: s.startswith("import os, shutil") and "qwen3" in s.lower()))
_run("engine", _pick(lambda s: s.startswith("import os, sys, torch") and "gfm-rag-adapted.zip" in s))
_pins = [s for s in _code if '_im.version("wandb")' in s or 'wandb.__version__.startswith("0.18.")' in s]
if _pins: _run("pins", _pins[0], fatal=False)
_run("fusion sources", _pick(lambda s: s.startswith("# === write the SciGraphIR-fusion files")))
# The notebook's blob may predate the current engine (an old interpret() has no do_paths, so a "scan"
# silently runs the slow path search). gfm_overlay.zip on Drive carries the current fusion sources.
import zipfile as _zf
_ov = f"/content/drive/MyDrive/cargo-gfmrag/gfm_overlay.zip"
if os.path.exists(_ov):
    _zf.ZipFile(_ov).extractall("/content/gfm-rag"); print("applied", _ov, "->", _zf.ZipFile(_ov).namelist())
_ft = open("/content/gfm-rag/gfmrag/trainers/fusion_trainer.py").read()
assert all(k in _ft for k in ("def interpret(", "do_paths", "golds=None", "min_hops")), (
    "the replayed notebook's engine is older than the current interpret(); upload experiments/gfm_overlay.zip "
    "to MyDrive/cargo-gfmrag/ and rerun (without it the scan runs the slow path search on every gold)")
_run("config defaults", _pick(lambda s: s.startswith("# Cell 3a is reused VERBATIM")))
_run("PyG version fix", _pick(lambda s: s.startswith("# The ULTRA layers vendored"), "last"))
_run("torchvision shim", _pick(lambda s: s.startswith("# THE TRAINING SUBPROCESS IS A FRESH PYTHON"), "last"))
_run("diagnostics patch", _pick(lambda s: s.startswith("STF = ")))
_run("CCMP_LR patch", _pick(lambda s: s.startswith("# Idempotent: re-running is a no-op")))
assert DATASET == SCAN_DATASET, f"the replayed notebook is for {DATASET}, not {SCAN_DATASET}"
for _n in ("DRIVE", "CACHE", "DATA_ROOT", "S4", "KGDIR", "RUNS", "OP_MODEL", "OP_SLUG", "QUERIES", "sh", "SCIGRAPHIR_ROOT"):
    assert _n in globals(), f"{_n} is not defined after the replay"
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

# --- graphs, checkpoints, output dir ------------------------------------------------------------------
FRAME_TEST, OPENIE_TEST = S["frame"], S["openie"]
GNAME = {"frame": FRAME_TEST, "openie": OPENIE_TEST, **S.get("extra", {})}
# extra graphs (merged graph) come from an add-on bundle: unpack it here if it is on Drive and not yet unpacked
for _k, _g in S.get("extra", {}).items():
    _z = f"{DRIVE}/{DATASET}_{_k}_bundle.zip"
    if not os.path.exists(f"{DATA_ROOT}/{_g}/processed/stage1/nodes.csv") and os.path.exists(_z):
        import zipfile as _zf; _zf.ZipFile(_z).extractall(SCIGRAPHIR_ROOT); print("unpacked", os.path.basename(_z))
    if not os.path.exists(f"{DATA_ROOT}/{_g}/processed/stage1/nodes.csv"):
        print(f"[skip] extra graph {_g}: not in the bundle and no {os.path.basename(_z)} on Drive"); GNAME.pop(_k)
GRAPHS = [g for k, g in GNAME.items() if g]
for g in GRAPHS:
    assert os.path.exists(f"{DATA_ROOT}/{g}/processed/stage1/nodes.csv"), f"graph {g} is not in the bundle"
    dst = f"{DATA_ROOT}/{g}/raw/documents.json"
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy(f"{DATA_ROOT}/{DATASET}_test/raw/documents.json", dst)
ANSWERS = None
try: ANSWERS = cp.answers_path("test")
except Exception: pass
if not (ANSWERS and os.path.exists(ANSWERS)): ANSWERS = None
SCAN_OUT = f"{DRIVE}/outputs/scan/{DATASET}"; os.makedirs(SCAN_OUT, exist_ok=True)
ARMS = []
for name, rel, gkey, skey, gate in S["arms"]:
    rels = rel if isinstance(rel, list) else [rel]          # candidates in preference order; first existing wins
    rel = next((r for r in rels if os.path.exists(f"{DRIVE}/{r}/model_best.pth")), rels[0])
    ckpt = f"{DRIVE}/{rel}/model_best.pth"; ok = os.path.exists(ckpt)
    print(f"  {'ok ' if ok else 'MISSING'}  {name:15} {rel}/model_best.pth" + ("" if ok or len(rels) == 1 else f"  (also tried {rels[1:]})"))
    if ok and gkey not in GNAME: print(f"  [skip] {name}: its graph is not available"); ok = False
    if ok: ARMS.append((name, ckpt, GNAME[gkey], skey, gate))
assert ARMS, "no checkpoint found for any arm"
print("test queries", len(json.load(open(QUERIES))), "| graphs", GRAPHS, "| writes", os.path.relpath(SCAN_OUT, DRIVE))

# --- component tables for the test graphs (handcrafted scorer + scorer views) and the scorer files -----------
import numpy as np, fnmatch, torch
def copy_new(src, dst, pattern="*"):
    if not os.path.isdir(src): return 0
    n = 0
    for root, _, files in os.walk(src):
        rel = os.path.relpath(root, src)
        for f in files:
            if not fnmatch.fnmatch(f, pattern): continue
            s_, d_ = f"{root}/{f}", f"{dst}/{rel}/{f}"
            if os.path.exists(d_) and os.path.getsize(d_) == os.path.getsize(s_): continue
            os.makedirs(os.path.dirname(d_), exist_ok=True); shutil.copy(s_, d_); n += 1
    return n
t0 = time.time()
EMB_LOCAL = f"{SCIGRAPHIR_ROOT}/outputs/caches/op_emb"
print(f"embedding cache: {copy_new(f'{CACHE}/op_emb', EMB_LOCAL, 'test_*')} test-split files restored ({time.time()-t0:.0f}s)")
def restore_index(g):
    src = f"{CACHE}/index/{g}"
    if not os.path.isdir(src): print(f"  {g}: no cached index on Drive (built on first use, 5-15 min)"); return
    n = sum(copy_new(f"{src}/{d}", f"{DATA_ROOT}/{g}/processed/{d}") for d in os.listdir(src))
    print(f"  {g}: index restored ({n} files)")
def save_index(g):
    pr = f"{DATA_ROOT}/{g}/processed"
    for d in os.listdir(pr):
        if d != "stage1": copy_new(f"{pr}/{d}", f"{CACHE}/index/{g}/{d}")
def _sem_ok(p):
    if not os.path.exists(p): return False
    zz = np.load(p, allow_pickle=True)
    return os.path.exists(str(zz["h_path"])) and "qwen" in str(zz["encoder"]).lower()
def opc(g):  return f"{DATA_ROOT}/{g}/operator_components{OP_SLUG}.npz"
def semc(g): return f"{DATA_ROOT}/{g}/semantic_components{OP_SLUG}.npz"

SEMF = {}   # scorer key -> (params json, popnet pt)
for key, (rel, nm) in S["sem"].items():
    src = f"{DRIVE}/{rel}"
    if key == "field":   # the pipeline expects the field's own scorer under results/semantic_<dataset>
        loc = f"{S4}/results/semantic_{nm}"; os.makedirs(loc, exist_ok=True); copy_new(src, loc)
    else:
        loc = src
    SEMF[key] = (f"{loc}/params_semantic_mlp_fixedloss_{nm}.json", f"{loc}/popnet_semantic_mlp_fixedloss_{nm}.pt")
    for p in SEMF[key]: assert os.path.exists(p), f"scorer file missing: {p}"
    print(f"  scorer '{key}': jmax {json.load(open(SEMF[key][0]))['jmax']}  ({rel})")

for g in GRAPHS:
    restore_index(g)
    if not os.path.exists(opc(g)):
        c = f"{CACHE}/{g}_operator_components{OP_SLUG}.npz"
        if os.path.exists(c): shutil.copy(c, opc(g))
        else:
            sh(f"python3 -u precompute/precompute_handcrafted_components.py "
               f"--dataset {DATASET} --graph {g} --split test --model {OP_MODEL}", KGDIR)
            shutil.copy(opc(g), c)
    if not _sem_ok(semc(g)):      # the H memmap never survives a runtime reset; seconds to rebuild for a test split
        sh(f"python3 -u precompute/precompute_semantic_components.py "
           f"--dataset {DATASET} --model {OP_MODEL} --graph {g} --split test", KGDIR)
    zz = np.load(semc(g), allow_pickle=True)
    print(f"  {g:24} H {tuple(int(x) for x in zz['h_shape'])}  Jmax={int(zz['Jmax'])}")
print(f"{copy_new(EMB_LOCAL, f'{CACHE}/op_emb', 'test_*')} new embedding files written back to Drive")

# --- checkpoint environment ---------------------------------------------------------------------------
def inspect_ckpt(ckpt):
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
    resp = [k for k in sd if "resp_" in k]; sem = [k for k in sd if k.startswith("sem_")]
    hid = next((int(sd[k].shape[0]) for k in resp if k.endswith("resp_proj.0.weight")), None)
    jmax = next((int(sd[k].shape[1]) - 2 for k in sem if k.endswith("sem_net.0.weight")), None)
    return {"tensors": len(sd), "resp_keys": len(resp), "sem_keys": len(sem), "ccmp_hid": hid, "jmax": jmax}
def model_env(ckpt, graph, skey, gate=None):
    info = inspect_ckpt(ckpt); sem_ckpt, sem_pop = SEMF[skey]
    st = json.load(open(sem_ckpt))
    assert int(st["jmax"]) == info["jmax"], f"scorer width mismatch: ckpt jmax={info['jmax']} vs scorer '{skey}' jmax={st['jmax']}"
    env_ = dict(WANDB_MODE="disabled", HYDRA_FULL_ERROR="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                HANDCRAFTED_COMPONENTS=opc(graph), HANDCRAFTED_COMPONENTS_TEST=opc(graph),
                SEMANTIC_COMPONENTS=semc(graph), SEMANTIC_COMPONENTS_TEST=semc(graph),
                SEMANTIC_CKPT=sem_ckpt, SEMANTIC_POPNET=sem_pop, SEM_POP_LAMBDA="1.0",
                FUSION_OBJECTIVE="hardneg", HARDNEG_HUB="50", HARDNEG_RAND="50", AUX_W="1.0",
                PER_GOLD="1", HARDNEG_GRAPH="50", STRAT_TEST=QUERIES)
    for k in list(os.environ):
        if k.startswith(("CCMP", "ROUTE", "STRAT_", "CQIG", "RESID_", "MISS_W")): os.environ.pop(k)
    if info["resp_keys"]:
        env_.update(CCMP="1", CCMP_HID=str(info["ccmp_hid"]), CCMP_GATE="1", CCMP_GATE_NORM="1", CCMP_ETA="0.5")
        aj = f"{os.path.dirname(ckpt)}/arm.json"
        if os.path.exists(aj):
            a_ = json.load(open(aj))
            if a_.get("ccmp_eta") is not None: env_["CCMP_ETA"] = str(a_["ccmp_eta"])
        if gate is not None: env_["CCMP_GATE"] = "1" if gate else "0"
    else:
        assert gate is None, f"{os.path.relpath(ckpt, DRIVE)} has no CCMP head; the gate on/off arm needs a CCMP checkpoint"
    return env_, info
def hydra_common(graph):
    return ("--config-path config/gfm_reasoner --config-name sft_training_fusion text_emb_model=qwen3_st "
            f"datasets.cfgs.root={DATA_ROOT} datasets.cfgs.force_reload=False "
            f"datasets.train_names=[{graph}] datasets.valid_names=[{graph}] model.semantic=mlp model.cqig=false ")

# --- scan every gold of every test query under each arm -------------------------------------------
QIDS_FILE = f"{SCAN_OUT}/qids_all.json"
json.dump([q["id"] for q in json.load(open(QUERIES))], open(QIDS_FILE, "w"))
def scan_all(name, ckpt, graph, skey, gate=None):
    out = f"{SCAN_OUT}/scan_all_{name}.json"
    if os.path.exists(out): print("[cached]", os.path.relpath(out, DRIVE)); return out
    env_, info = model_env(ckpt, graph, skey, gate=gate)
    print(f"[ckpt] {name}: {os.path.relpath(ckpt, DRIVE)} {info}" + (f" gate={'on' if gate else 'off'}" if gate is not None else ""))
    rl = f"{RUNS}/scan_all_{name}"; os.makedirs(rl, exist_ok=True)
    rc = sh("python -u -m gfmrag.workflow.interpret_paths " + hydra_common(graph) +
            f"+interp.ckpt={ckpt} +interp.qids_file={QIDS_FILE} +interp.out={out} " +
            (f"+interp.answers={ANSWERS} " if ANSWERS else "") +
            f"+interp.paths=0 +interp.max_golds=8 +interp.top_views=1 hydra.run.dir={rl}",
            "/content/gfm-rag", extra=env_, log=f"{rl}/console.log", check=False)
    assert rc == 0 and os.path.exists(out), f"{name} failed (exit {rc}); read {rl}/console.log"
    save_index(graph)
    return out
SA = {}
for name, ckpt, graph, skey, gate in ARMS:
    SA[name] = scan_all(name, ckpt, graph, skey, gate)
print("scanned:", sorted(SA))

# --- the table: every gold, per stratum, graph channel and the other channels, per arm; win rates ----
STRAT = {q["id"]: q.get("stratum", "same") for q in json.load(open(QUERIES))}
T = {k: {(r["id"], t["doc"]): t["rank"] for r in json.load(open(v)) for t in r["targets"]} for k, v in SA.items()}
common = sorted(set.intersection(*[set(T[k]) for k in T]))
n_docs = len(json.load(open(f"{DATA_ROOT}/{DATASET}_test/raw/documents.json")))
CH = ["graph", "scorer", "dense", "fused"]
LAB = {"frame_ccmp": 'SciAfford graph + CCMP', "frame_ccmp_off": 'SciAfford graph, CCMP gate off (same weights)',
       "frame_nocc": 'SciAfford graph, no CCMP (own run)', "openie": "OpenIE entity graph",
       "hyb_nocc": "merged graph, no CCMP", "hyb_ccmp": "merged graph + CCMP", "hyb_ccmp_off": "merged graph, CCMP gate off (same weights)"}
lines = []
def out(s=""): print(s); lines.append(s)
strata = ["all"] + sorted({STRAT[q] for q, _ in common}, key=lambda s: (s != "same", s))
if len(strata) == 2: strata = ["all"]
res = {}
out(f"# {DATASET} graph-channel scan: {len(common)} golds of {len({q for q, _ in common})} test queries, "
    f"every gold, no path search, corpus {n_docs:,} documents\n")
for st in strata:
    keys = [k for k in common if st == "all" or STRAT[k[0]] == st]
    out(f"## {st}: {len(keys)} golds of {len({q for q, _ in keys})} queries\n")
    out("| arm | channel | median rank | R@5 | R@10 | R@50 | R@100 |"); out("|---|---|--:|--:|--:|--:|--:|")
    for arm in SA:
        for ch in CH:
            v = np.array([T[arm][k].get(ch, np.nan) for k in keys], dtype=float)
            if np.all(np.isnan(v)): continue
            res[(st, arm, ch)] = v
            out(f"| {LAB.get(arm, arm)} | {ch} | {np.nanmedian(v):.0f} | {100*np.nanmean(v<=5):.1f} | {100*np.nanmean(v<=10):.1f} | "
                f"{100*np.nanmean(v<=50):.1f} | {100*np.nanmean(v<=100):.1f} |")
    out("\nGold-by-gold, graph channel (rank lower is better):")
    for a, b, lab in (("frame_ccmp", "openie", 'affordance representation + CCMP vs OpenIE'), ("frame_nocc", "openie", 'affordance representation no-CCMP vs OpenIE'),
                      ("frame_ccmp", "frame_ccmp_off", "CCMP gate on vs off (same weights)"), ("frame_ccmp", "frame_nocc", "CCMP run vs no-CCMP run"),
                      ("hyb_nocc", "frame_nocc", 'merged graph vs SciAfford graph (no CCMP)'), ("hyb_nocc", "openie", "merged graph vs OpenIE (no CCMP)"),
                      ("hyb_ccmp", "frame_ccmp", 'merged + CCMP vs affordance representation + CCMP'), ("hyb_ccmp", "hyb_nocc", "merged: CCMP run vs no-CCMP run")):
        if (st, a, "graph") in res and (st, b, "graph") in res:
            x, y = res[(st, a, "graph")], res[(st, b, "graph")]
            out(f"- {lab}: first better {100*np.mean(x<y):.0f}%  tie {100*np.mean(x==y):.0f}%  second better {100*np.mean(x>y):.0f}%")
    out()
out("median = median rank of the gold in the test corpus. graph = graph channel alone; scorer = multi-view scorer alone; "
    "dense = Qwen3 cosine; fused = the model's output ranking.")
json.dump({"dataset": DATASET, "n_golds": len(common), "n_queries": len({q for q, _ in common}), "n_docs": n_docs,
           "stats": {f"{s}/{a}/{c}": {"median": float(np.nanmedian(v)), "r5": float(np.nanmean(v<=5)), "r10": float(np.nanmean(v<=10)),
                                      "r50": float(np.nanmean(v<=50)), "r100": float(np.nanmean(v<=100)), "n": int(len(v))}
                     for (s, a, c), v in res.items()},
           "ranks": {f"{a}/{c}": {f"{q}|{d}": float(x) for (q, d), x in zip(common, v)} for (s, a, c), v in res.items() if s == "all"},
           "strata": {q: STRAT[q] for q, _ in common}},
          open(f"{SCAN_OUT}/graph_channel_scan_{DATASET}.json", "w"), indent=1)
open(f"{SCAN_OUT}/graph_channel_scan_{DATASET}.md", "w").write("\n".join(lines) + "\n")
print("\nwrote", os.path.relpath(f"{SCAN_OUT}/graph_channel_scan_{DATASET}.md", DRIVE))
