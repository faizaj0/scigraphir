# Showcase cell: graph-channel scan + path interpretations + 'amazing example' finder + hop figure, for ONE
# dataset in ONE cell. Set DATASET, paste into a fresh runtime, run. Outputs go to outputs/scan/<dataset>/
# (hops_<arm>.json, showcase_<dataset>.md/.tex/_hops.pdf). The first part is colab_graph_scan_cell.py verbatim.
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
_run("fusion sources", _pick(lambda s: s.startswith("# === write the CARGO-fusion files")))
# The notebook's blob may predate the current engine (an old interpret() has no do_paths, so a "scan"
# silently runs the slow path search). gfm_overlay.zip on Drive carries the current fusion sources.
import zipfile as _zf
_ov = f"/content/drive/MyDrive/cargo-gfmrag/gfm_overlay.zip"
if os.path.exists(_ov):
    _zf.ZipFile(_ov).extractall("/content/gfm-rag"); print("applied", _ov, "->", _zf.ZipFile(_ov).namelist())
_ft = open("/content/gfm-rag/gfmrag/trainers/fusion_trainer.py").read()
assert all(k in _ft for k in ("def interpret(", "do_paths", "golds=None", "min_hops")), (
    "the replayed notebook's engine is older than the current interpret(); upload sir4-retrieval/gfm_overlay.zip "
    "to MyDrive/cargo-gfmrag/ and rerun (without it the scan runs the slow path search on every gold)")
_run("config defaults", _pick(lambda s: s.startswith("# Cell 3a is reused VERBATIM")))
_run("PyG version fix", _pick(lambda s: s.startswith("# The ULTRA layers vendored"), "last"))
_run("torchvision shim", _pick(lambda s: s.startswith("# THE TRAINING SUBPROCESS IS A FRESH PYTHON"), "last"))
_run("diagnostics patch", _pick(lambda s: s.startswith("STF = ")))
_run("CCMP_LR patch", _pick(lambda s: s.startswith("# Idempotent: re-running is a no-op")))
assert DATASET == SCAN_DATASET, f"the replayed notebook is for {DATASET}, not {SCAN_DATASET}"
for _n in ("DRIVE", "CACHE", "DATA_ROOT", "S4", "KGDIR", "RUNS", "OP_MODEL", "OP_SLUG", "QUERIES", "sh", "CARGO_ROOT"):
    assert _n in globals(), f"{_n} is not defined after the replay"
# the interpretation entry point (not in the training notebooks' fusion blob)
open("/content/gfm-rag/gfmrag/workflow/interpret_paths.py", "w").write(r'''"""
interpret_paths.py -- path interpretations for a trained fusion checkpoint.

Same construction as sft_training (config, datasets, model, trainer) with no training: the
checkpoint is loaded, then FusionSFTTrainer.interpret() runs the NBFNet-style gradient beam
search from each requested query's seed frames to its best-ranked gold and records the CCMP
responsibility along every path. Output: one JSON.

    python -m gfmrag.workflow.interpret_paths --config-path config/gfm_reasoner \\
        --config-name sft_training_fusion text_emb_model=qwen3_st \\
        datasets.cfgs.root=... datasets.train_names=[G] datasets.valid_names=[G] \\
        model.semantic=mlp model.cqig=false \\
        +interp.ckpt=/path/model_best.pth +interp.qids_file=/path/qids.json \\
        +interp.out=/path/paths.json +interp.probes=/path/probes_test.jsonl \\
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
    kw = dict(probes_path=cfg.interp.get("probes"),
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
        import zipfile as _zf; _zf.ZipFile(_z).extractall(CARGO_ROOT); print("unpacked", os.path.basename(_z))
    if not os.path.exists(f"{DATA_ROOT}/{_g}/processed/stage1/nodes.csv"):
        print(f"[skip] extra graph {_g}: not in the bundle and no {os.path.basename(_z)} on Drive"); GNAME.pop(_k)
GRAPHS = [g for k, g in GNAME.items() if g]
for g in GRAPHS:
    assert os.path.exists(f"{DATA_ROOT}/{g}/processed/stage1/nodes.csv"), f"graph {g} is not in the bundle"
    dst = f"{DATA_ROOT}/{g}/raw/documents.json"
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy(f"{DATA_ROOT}/{DATASET}_test/raw/documents.json", dst)
PROBES = None
try: PROBES = cp.probes_path("test")
except Exception: pass
if not (PROBES and os.path.exists(PROBES)): PROBES = None
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

# --- component tables for the test graphs (operator + scorer views) and the scorer files -----------
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
EMB_LOCAL = f"{CARGO_ROOT}/outputs/caches/op_emb"
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
            sh(f"python3 -u experiments/probe_greasoner/precompute_operator_components.py "
               f"--dataset {DATASET} --graph {g} --split test --model {OP_MODEL}", KGDIR)
            shutil.copy(opc(g), c)
    if not _sem_ok(semc(g)):      # the H memmap never survives a runtime reset; seconds to rebuild for a test split
        sh(f"python3 -u experiments/probe_greasoner/precompute_semantic_components.py "
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
                OPERATOR_COMPONENTS=opc(graph), OPERATOR_COMPONENTS_TEST=opc(graph),
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
            (f"+interp.probes={PROBES} " if PROBES else "") +
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
LAB = {"frame_ccmp": "frame graph + CCMP", "frame_ccmp_off": "frame graph, CCMP gate off (same weights)",
       "frame_nocc": "frame graph, no CCMP (own run)", "openie": "OpenIE entity graph",
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
    for a, b, lab in (("frame_ccmp", "openie", "frame + CCMP vs OpenIE"), ("frame_nocc", "openie", "frame no-CCMP vs OpenIE"),
                      ("frame_ccmp", "frame_ccmp_off", "CCMP gate on vs off (same weights)"), ("frame_ccmp", "frame_nocc", "CCMP run vs no-CCMP run"),
                      ("hyb_nocc", "frame_nocc", "merged graph vs frame graph (no CCMP)"), ("hyb_nocc", "openie", "merged graph vs OpenIE (no CCMP)"),
                      ("hyb_ccmp", "frame_ccmp", "merged + CCMP vs frame + CCMP"), ("hyb_ccmp", "hyb_nocc", "merged: CCMP run vs no-CCMP run")):
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
open(f"{S4}/eval/showcase.py", "w").write(r"""#!/usr/bin/env python3
'''
showcase.py -- find the worked examples that show cross-domain reasoning, and draw the hop figure.

Reads interpret_paths.py outputs with paths (one file per arm; the FIRST --arm is the model being
showcased, normally the frame graph + CCMP) and produces, for one dataset:

  <out>.md               candidates ranked for a reader: cross-field golds the dense retrievers bury
                         that the full model ranks at the top, with a readable multi-hop route through
                         a mechanism frame (function / limitation / method), not a domain hub. Per
                         candidate: the query, the gold and its field, every rank, the top routes on
                         every graph with the CCMP gate per hop.
  <out>.tex              the top --n-tex candidates in the GFM-RAG Table 4 layout (query / inspiration /
                         ranks / paths), one block per example.
  <out>_candidates.json  the ranked candidate list (machine-readable).
  <out>_hops.{pdf,png}   the hop figure (GFM-RAG Fig. 6 analogue): share of golds by length of the
                         top path under each arm, against the shortest seed->gold route on the graph
                         (a structural floor, NOT a ground-truth reasoning path; the caption must say so),
                         one panel per stratum, with the mean gap to the floor printed per arm.
  <out>_hops.json        the distribution behind the figure, for --combine.

--combine a_hops.json b_hops.json ... draws one row of panels (one per dataset, cross stratum) into
--out.{pdf,png}, the multi-dataset figure for the thesis.

usage (one dataset):
  showcase.py --dataset sir4_cs --queries raw/test.json --docs raw/documents.json \\
      --edges processed/stage1/edges.csv \\
      --arm "SciGraphIR (frame graph + CCMP)=hops_frame_ccmp.json" \\
      --arm "frame graph, CCMP gate off=hops_frame_ccmp_off.json" --arm "OpenIE graph=hops_openie.json" \\
      [--pred qwen3=predictions_qwen3_sir4_cs_test.json --pred bge=...] [--quartet eval.json] \\
      --out results/qualitative/showcase_sir4_cs --top 30 --n-tex 4
'''
from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics as st
from collections import Counter, defaultdict

BIG = 10 ** 6
MECH = ("function", "limitation", "method", "finding")     # frame types that carry a mechanism
SYS_LABEL = {"bm25": "BM25", "bge": "BGE-large", "qwen3": "Qwen3-Emb.", "specter2": "SPECTER2", "scincl": "SciNCL",
             "reasonir": "ReasonIR-8B", "dense": "Qwen3 cosine", "scorer": "multi-view scorer", "graph": "graph channel",
             "fused": "SciGraphIR"}


# ----------------------------------------------------------------------------------------------- helpers
def tex_escape(s: str) -> str:
    return re.sub(r"([&%$#_{}])", r"\\\1", str(s)).replace("~", "\\textasciitilde{}").replace("^", "\\textasciicircum{}")


def short(s: str, n: int) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def title_of(doc_id: str, docs: dict, n: int = 110) -> str:
    return short(docs.get(doc_id, doc_id).split(". ")[0], n)


def ntype(name: str, docs: dict) -> str:
    if name in docs:
        return "paper"
    if name.startswith("[") and "]" in name:
        return name[1 : name.index("]")]
    return "entity"


def node_label(name: str, docs: dict, n: int = 60) -> str:
    return "[paper] " + title_of(name, docs, n) if name in docs else short(name, n)


def rel_label(r: str) -> str:
    return r.replace("inverse_", "inv. ").replace("_", " ")


def valid(p: dict, seeds: set) -> bool:
    h = p.get("hops") or []
    return bool(h) and h[0]["head"] in seeds and all(a["tail"] == b["head"] for a, b in zip(h, h[1:]))


def fmt_path(p: dict, docs: dict, gate: bool) -> str:
    parts = []
    for h in p["hops"]:
        g = f" (gate {h['gate']:.2f})" if gate and "gate" in h else ""
        parts.append(f"{node_label(h['head'], docs)}{g} --{rel_label(h['rel'])}-->")
    parts.append(node_label(p["hops"][-1]["tail"], docs))
    return " ".join(parts)


def fmt_path_tex(p: dict, docs: dict, gate: bool) -> str:
    parts = []
    for h in p["hops"]:
        g = f" {{\\scriptsize({h['gate']:.2f})}}" if gate and "gate" in h else ""
        parts.append(f"{tex_escape(node_label(h['head'], docs, 48))}{g} $\\xrightarrow{{\\text{{{tex_escape(rel_label(h['rel']))}}}}}$")
    parts.append(tex_escape(node_label(p["hops"][-1]["tail"], docs, 48)))
    return " ".join(parts)


def ranked_docs(rec) -> list:
    p = rec.get("predictions", rec)
    d = p.get("document", p) if isinstance(p, dict) else p
    return [x[0] if isinstance(x, (list, tuple)) else x for x in d]


def fnum(x, spec: str = ".2f") -> str:
    '''Format a number, or 'n/a' when a statistic is undefined (no valid route, or no min_hops in the file).'''
    return "n/a" if x is None else format(x, spec)


def load_arm(path: str) -> dict:
    '''{(qid, gold): target-with-context} for one interpret_paths output.'''
    out = {}
    for r in json.load(open(path)):
        seeds = set(r.get("seeds", []))
        for t in r.get("targets", []):
            ps = [p for p in t.get("paths", []) if valid(p, seeds)]
            out[(r["id"], t["doc"])] = {"rank": t["rank"], "min_hops": t.get("min_hops"), "paths": ps,
                                        "n_raw_paths": len(t.get("paths", [])), "views": t.get("views", []),
                                        "dense_cos": t.get("dense_cos"), "stratum": r.get("stratum") or "same",
                                        "question": r.get("question", ""), "seeds": r.get("seeds", [])}
    if out and all(t["min_hops"] is None for t in out.values()):
        print(f"[showcase] WARNING {os.path.basename(path)}: no target carries min_hops; the engine that wrote it predates "
              "the structural floor (apply gfm_overlay.zip and rerun the path stage). The hop figure will show no floor.")
    return out


def doc_domains(edges_csv: str | None) -> dict:
    '''document id -> '[domain] ...' node names from the frame graph's in_field edges.'''
    dom = defaultdict(list)
    if not edges_csv or not os.path.exists(edges_csv):
        return dom
    import csv
    with open(edges_csv, newline="") as fh:
        for row in csv.reader(fh):
            if len(row) >= 3 and row[1] == "in_field":
                dom[row[0]].append(row[2].replace("[domain] ", ""))
    return dom


def quartet_fields(path: str | None) -> dict:
    '''(qid, gold) -> 'Computer Science -> Engineering' from the QUARTET export, when available.'''
    out = {}
    if not path or not os.path.exists(path):
        return out
    for x in json.load(open(path)):
        for d, m in (x.get("quartet", {}).get("per_document") or {}).items():
            if m.get("field_pair"):
                out[(x["id"], d)] = {"field_pair": m["field_pair"], "stratum": m.get("stratum")}
    return out


def route_kind(p: dict | None, docs: dict) -> str:
    '''bridge = passes a mechanism frame; hub = only papers/domain/task/entity nodes; none = no valid path.'''
    if not p:
        return "none"
    inner = [ntype(h["head"], docs) for h in p["hops"][1:]] + [ntype(p["hops"][0]["head"], docs)]
    if any(t in MECH for t in inner):
        return "bridge"
    if any(t == "domain" for t in inner):
        return "hub"
    return "other"


# ----------------------------------------------------------------------------------------------- hop figure
def hop_stats(arms: list, docs: dict, max_hops: int) -> dict:
    '''Per stratum and arm: distribution of top-path length, the shortest-route floor, mean gap, unreachable.'''
    strata = sorted({t["stratum"] for _, tab in arms for t in tab.values()}, key=lambda s: (s != "same", s))
    out = {"max_hops": max_hops, "strata": strata, "arms": [lab for lab, _ in arms], "dist": {}}
    for s in strata:
        out["dist"][s] = {}
        first = arms[0][1]
        floor = [t["min_hops"] for t in first.values() if t["stratum"] == s and t["min_hops"] is not None]
        cf = Counter(min(h, max_hops) for h in floor)
        out["dist"][s]["floor"] = {"n": len(floor), "pct": [100 * cf.get(h, 0) / max(1, len(floor)) for h in range(1, max_hops + 1)],
                                   "mean": st.mean(floor) if floor else None}
        for lab, tab in arms:
            sub = [t for t in tab.values() if t["stratum"] == s]
            hops = [len(t["paths"][0]["hops"]) for t in sub if t["paths"]]
            gaps = [len(t["paths"][0]["hops"]) - t["min_hops"] for t in sub if t["paths"] and t["min_hops"] is not None]
            c = Counter(min(h, max_hops) for h in hops)
            out["dist"][s][lab] = {"n_golds": len(sub), "n_paths": len(hops), "unreachable": len(sub) - len(hops),
                                   "pct": [100 * c.get(h, 0) / max(1, len(hops)) for h in range(1, max_hops + 1)],
                                   "mean": st.mean(hops) if hops else None, "mean_gap": st.mean(gaps) if gaps else None,
                                   "mae": st.mean(abs(g) for g in gaps) if gaps else None}
    return out


def draw_hops(panels: list, out_prefix: str, title_fs: float = 10) -> None:
    '''panels: [(title, stats_dict, stratum)] -> one row of panels, GFM-RAG Fig. 6 style.

    Colour and marker are keyed on the ARM LABEL, not its position, because datasets do not all have
    the same arms (a missing checkpoint drops one) and a shared legend taken from the first panel would
    then mislabel every other panel's curves. The legend is the union over panels.

    Every number printed here can be undefined (an arm with no valid route has no mean gap, a file
    written before min_hops existed has no floor), so all of them go through fnum() and a panel with
    nothing to draw is dropped rather than crashing the run.
    '''
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [(t, S, s) for t, S, s in panels if s in (S.get("dist") or {})]
    if not panels:
        print("[showcase] no stratum carries any interpreted gold; the hop figure is skipped")
        return
    labels = []                       # arm labels in first-seen order: the colour key
    for _, S, _s in panels:
        for lab in S.get("arms") or []:
            if lab not in labels:
                labels.append(lab)
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.3 * n + 0.4, 3.0), squeeze=False)
    mk = ["s", "D", "^", "v", "P", "X"]
    col = ["#1f77b4", "#7f7f7f", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2"]
    for ax, (title, S, s) in zip(axes[0], panels):
        H = S.get("max_hops") or 6; xs = list(range(1, H + 1)); D = S["dist"][s]
        fl = D.get("floor") or {}
        if fl.get("n"):
            ax.plot(xs, fl["pct"], marker="o", color="#2ca02c", lw=1.8, ms=6,
                    label=f"shortest route on the {(S.get('arms') or [''])[0].split('(')[-1].rstrip(')').split('+')[0].strip() or 'first'} graph (floor)")
        drawn = 0
        for lab in S.get("arms") or []:
            d = D.get(lab)
            if not d or not d.get("n_paths"):
                continue
            k = labels.index(lab)
            ax.plot(xs, d["pct"], marker=mk[k % len(mk)], color=col[k % len(col)], lw=1.6, ms=5.5, label=lab)
            ax.text(0.03, 0.95 - 0.085 * drawn,
                    f"{short(lab, 30)}: mean gap {fnum(d.get('mean_gap'), '+.2f')}, no valid route {d.get('unreachable', 0)}",
                    transform=ax.transAxes, fontsize=6.6, va="top", color=col[k % len(col)])
            drawn += 1
        ax.set_xticks(xs); ax.set_xlabel("hops"); ax.set_ylabel("share of golds (%)")
        # n = golds behind the floor when there is one, otherwise golds interpreted under the first arm
        n_g = fl.get("n") or max((d.get("n_golds", 0) for k_, d in D.items() if k_ != "floor"), default=0)
        ax.set_title(f"{title} ({s}-field, n={n_g}{'' if fl.get('n') else ', no floor'})", fontsize=title_fs)
        ax.grid(True, ls="--", alpha=0.4); ax.set_ylim(0, max(50, ax.get_ylim()[1]))
    seen, h, l = {}, [], []           # union legend, deduplicated by label
    for ax in axes[0]:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in seen:
                seen[ll] = 1; h.append(hh); l.append(ll)
    if l:
        fig.legend(h, l, loc="upper center", ncol=min(4, len(l)), fontsize=8, frameon=True, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_prefix}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--combine", nargs="*", default=None, help="<out>_hops.json files; draws the multi-dataset figure to --out")
    ap.add_argument("--dataset", default="")
    ap.add_argument("--queries"); ap.add_argument("--docs"); ap.add_argument("--edges", default=None)
    ap.add_argument("--arm", action="append", default=[], help="label=hops json; the first is the showcased model")
    ap.add_argument("--pred", action="append", default=[], help="name=predictions json of a baseline (rank of the gold)")
    ap.add_argument("--quartet", default=None, help="QUARTET eval.json for the field pair of each gold (SIR-4 only)")
    ap.add_argument("--stratum", default="auto", help="cross | same | any | auto (cross when the dataset has it)")
    ap.add_argument("--max-fused", type=int, default=25); ap.add_argument("--max-graph", type=int, default=5)
    ap.add_argument("--min-dense", type=int, default=25, help="the gold must be at least this deep under raw cosine")
    ap.add_argument("--min-hops", type=int, default=2); ap.add_argument("--max-hops", type=int, default=6)
    ap.add_argument("--top", type=int, default=30); ap.add_argument("--n-tex", type=int, default=4)
    ap.add_argument("--paths-per-arm", type=int, default=2)
    ap.add_argument("--sample-qids", default=None, help="json list of query ids; the hop figure uses only these (unbiased sample)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.combine is not None:
        panels = []
        for p in a.combine:
            S = json.load(open(p))
            if not S.get("strata"):
                print(f"[showcase] {os.path.basename(p)} has no interpreted gold; skipped")
                continue
            s = "cross" if "cross" in S["strata"] else S["strata"][0]
            panels.append((S.get("dataset", os.path.basename(p)), S, s))
        draw_hops(panels, a.out)
        print("wrote", a.out + ".pdf")
        return 0

    assert a.arm and a.queries and a.docs, "--arm, --queries and --docs are required"
    docs = json.load(open(a.docs))
    queries = {q["id"]: q for q in json.load(open(a.queries))}
    arms = []
    for spec in a.arm:
        lab, path = spec.split("=", 1)
        arms.append((lab, load_arm(path)))
    main_lab, main_tab = arms[0]
    preds = {}
    for spec in a.pred:
        nm, path = spec.split("=", 1)
        if os.path.exists(path):
            preds[nm] = {r["id"]: ranked_docs(r) for r in json.load(open(path))}
    dom = doc_domains(a.edges)
    qf = quartet_fields(a.quartet)
    strata_present = {t["stratum"] for t in main_tab.values()}
    stratum = a.stratum if a.stratum != "auto" else ("cross" if "cross" in strata_present else "any")

    # ---- candidates ------------------------------------------------------------------------------
    cands = []
    for (qid, gold), t in main_tab.items():
        # a "cross" query can carry same-field golds too; use the gold's own label when QUARTET has it
        g_strat = (qf.get((qid, gold)) or {}).get("stratum") or t["stratum"]
        if stratum != "any" and g_strat != stratum:
            continue
        rk = t["rank"]
        gk = lambda c: rk.get(c, BIG)      # a channel the scan did not write = "did not retrieve it", never a KeyError
        # eligible: cosine buries the gold and either the graph channel or the full model recovers it
        if gk("dense") < a.min_dense or (gk("graph") > a.max_graph and gk("fused") > a.max_fused):
            continue
        p = t["paths"][0] if t["paths"] else None
        kind = route_kind(p, docs)
        hops = len(p["hops"]) if p else 0
        gates = [h.get("gate") for h in p["hops"]] if p else []
        row = {"id": qid, "gold": gold, "stratum": g_strat, "question": queries.get(qid, {}).get("question", t["question"]),
               "gold_title": title_of(gold, docs), "gold_domain": "; ".join(dom.get(gold, [])) or None,
               "field_pair": (qf.get((qid, gold)) or {}).get("field_pair"), "ranks": dict(rk), "route": kind, "hops": hops,
               "max_gate": max((g for g in gates if g is not None), default=None), "min_hops": t["min_hops"],
               "dense_cos": t["dense_cos"], "n_golds": len(queries.get(qid, {}).get("supporting_documents", []) or [gold])}
        for lab, tab in arms[1:]:
            o = tab.get((qid, gold))
            row["ranks"][f"{lab}|fused"] = o["rank"].get("fused") if o else None
            row["ranks"][f"{lab}|graph"] = o["rank"].get("graph") if o else None
        for nm, tab in preds.items():
            lst = tab.get(qid)
            row["ranks"][nm] = (lst.index(gold) + 1) if lst and gold in lst else (BIG if lst else None)
        # amazingness: a real route through a mechanism, deep under cosine, top under the model,
        # and missed by the OpenIE graph and the dense baselines when we know them
        openie = [v for k, v in row["ranks"].items() if k.endswith("|fused") and "OpenIE" in k and v]
        base = [v for k, v in row["ranks"].items() if k in preds and v]
        row["tier"] = "A" if gk("fused") <= 10 else ("B" if gk("graph") <= 5 else "C")   # A: model top-10; B: graph top-5 only
        row["score"] = ((2 if kind == "bridge" else 0) + (1 if hops >= a.min_hops else 0)
                        + (3 if gk("fused") <= 5 else 2 if gk("fused") <= 10 else 1 if gk("fused") <= 25 else 0)
                        + (1 if gk("graph") <= 5 else 0)
                        + min(3.0, __import__("math").log10(max(1, gk("dense"))))
                        + (1 if openie and min(openie) > 25 else 0) + (1 if base and min(base) > 25 else 0)
                        + (0.5 if row["max_gate"] and row["max_gate"] > 1.05 else 0))
        cands.append(row)
    cands.sort(key=lambda r: (-r["score"], -r["ranks"].get("dense", BIG), r["ranks"].get("fused", BIG)))
    top = cands[: a.top]

    # ---- markdown -------------------------------------------------------------------------------
    md = [f"# {a.dataset or 'dataset'}: showcase candidates ({stratum}-field golds; cosine >= {a.min_dense} and "
          f"(graph <= {a.max_graph} or fused <= {a.max_fused}); tier A = model top-10, B = graph top-5 only, C = rest)", "",
          f"{len(main_tab)} golds with interpretations under '{main_lab}'; {len(cands)} pass the filter; "
          f"routes: {Counter(r['route'] for r in cands)}", "",
          "Read the top rows first. 'bridge' = the top route passes a function / limitation / method / finding frame; "
          "'hub' = it only passes papers and a domain node (the failure signature); gates > 1 are hops CCMP amplified.", ""]
    # the extra columns (other arms, then baselines) are built as one list: with no baseline predictions
    # on disk, joining two groups with " | " left a stray empty column and a malformed markdown table
    extra_hdr = [short(lab, 18) for lab, _ in arms[1:]] + [SYS_LABEL.get(n, n) for n in preds]
    md.append("| # | score | tier | route | hops | cosine | scorer | graph | fused | "
              + "".join(h + " | " for h in extra_hdr) + "gold | field | query |")
    md.append("|--:|--:|---|---|--:|--:|--:|--:|--:|" + "---:|" * len(extra_hdr) + "---|---|---|")
    for i, r in enumerate(top, 1):
        rk = r["ranks"]; f = lambda v: "--" if v is None else (">300" if v >= BIG else str(v))
        extra = [f(rk.get(f"{lab}|fused")) for lab, _ in arms[1:]] + [f(rk.get(n)) for n in preds]
        md.append(f"| {i} | {r['score']:.1f} | {r['tier']} | {r['route']} | {r['hops']} | {f(rk.get('dense'))} | {f(rk.get('scorer'))} | {f(rk.get('graph'))} | {f(rk.get('fused'))} | "
                  + "".join(c + " | " for c in extra)
                  + f"{short(r['gold_title'], 60)} | {short(r['field_pair'] or r['gold_domain'] or '', 40)} | {short(r['question'], 80).replace('|', '/')} |")
    md.append("")
    for i, r in enumerate(top, 1):
        qid, gold = r["id"], r["gold"]
        md += [f"## {i}. {r['id']}  ->  {r['gold_title']}", "",
               f"**Field:** {r['field_pair'] or r['gold_domain'] or 'unknown'} | **stratum:** {r['stratum']} | "
               f"**golds for this query:** {r['n_golds']} | **route:** {r['route']}, {r['hops']} hops (shortest {r['min_hops']})", "",
               f"**Query:** {short(r['question'], 700)}", "",
               f"**Inspiration (gold):** {short(docs.get(gold, gold), 500)}", "",
               "**Ranks of this gold:** " + ", ".join(f"{SYS_LABEL.get(k, k)} {v if v < BIG else '>300'}" for k, v in r["ranks"].items() if v is not None), ""]
        t = main_tab[(qid, gold)]
        if t["views"]:
            md.append("**Scorer views that matched best (hypothetical answers written for the query):**")
            for v in t["views"]:
                md.append(f"- view {v['view']} match {v['match']:.3f}: {short(v.get('text') or '(text unavailable)', 240)}")
            md.append("")
        for lab, tab in arms:
            o = tab.get((qid, gold))
            md.append(f"**{lab}**" + (f" (ranks: {', '.join(f'{k} {v}' for k, v in o['rank'].items())})" if o else ": no interpretation"))
            if o:
                gate = any("gate" in h for p in o["paths"] for h in p["hops"])
                if not o["paths"]:
                    md.append("- (no valid path within the reasoner's depth)")
                for p in o["paths"][: a.paths_per_arm]:
                    md.append(f"- w={p['weight']:.2f}: {fmt_path(p, docs, gate)}")
            md.append("")
    open(a.out + ".md", "w").write("\n".join(md) + "\n")
    json.dump({"dataset": a.dataset, "stratum": stratum, "n_interpreted": len(main_tab), "n_pass": len(cands), "candidates": cands},
              open(a.out + "_candidates.json", "w"), indent=1)

    # ---- LaTeX (GFM-RAG Table 4 layout, one block per example) ------------------------------------
    tex = ["% generated by showcase.py; edit freely",
           "\\begin{table}[t]\n\\centering\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\\renewcommand{\\arraystretch}{1.15}",
           "\\begin{tabular}{@{}p{0.13\\linewidth} p{0.85\\linewidth}@{}}\n\\toprule"]
    for r in top[: a.n_tex]:
        qid, gold = r["id"], r["gold"]
        field = r["field_pair"] or r["gold_domain"] or ""
        rk = r["ranks"]
        rank_txt = ", ".join(f"{SYS_LABEL.get(k, k)} {v if v < BIG else '$>$300'}" for k, v in rk.items()
                             if v is not None and k in ("dense", "scorer", "graph", "fused", "qwen3", "bge"))
        for lab, _ in arms[1:]:
            v = rk.get(f"{lab}|fused")
            if v:
                rank_txt += f", {tex_escape(lab)} {v}"
        tex.append(f"\\textbf{{Query}} & {tex_escape(short(r['question'], 420))} \\\\")
        tex.append(f"\\textbf{{Inspiration}} & {tex_escape(r['gold_title'])}" + (f" {{\\scriptsize({tex_escape(field)})}}" if field else "") + " \\\\")
        tex.append(f"\\textbf{{Rank}} & {{\\scriptsize {rank_txt}}} \\\\")
        lines = []
        for lab, tab in arms:
            o = tab.get((qid, gold))
            if not o:
                continue
            gate = any("gate" in h for p in o["paths"] for h in p["hops"])
            for p in o["paths"][: (a.paths_per_arm if lab == main_lab else 1)]:
                lines.append(f"{{\\scriptsize {tex_escape(lab)}}} {p['weight']:.2f}: {fmt_path_tex(p, docs, gate)}")
            if not o["paths"]:
                lines.append(f"{{\\scriptsize {tex_escape(lab)}}}: no route within {a.max_hops} hops")
        tex.append("\\textbf{Paths} & " + " \\newline ".join(lines) + " \\\\\n\\midrule")
    if tex[-1].endswith("\\midrule"):
        tex[-1] = tex[-1][: -len("\n\\midrule")]
    tex.append("\\bottomrule\n\\end{tabular}")
    tex.append(f"\\caption{{Path interpretations on {tex_escape(a.dataset)}: cross-field queries whose gold inspiration the dense "
               "retrievers bury and SciGraphIR ranks at the top. Paths are the highest-weighted routes from a query seed frame to "
               "the gold under each graph (gradient beam search over per-layer edge weights, as in NBFNet and GFM-RAG); numbers in "
               "parentheses are the CCMP gate on each hop's sender (1 = frontier mean; $>$1 amplified). Ranks are the position of "
               "the gold under each channel and system.}")
    tex.append(f"\\label{{tab:showcase-{a.dataset.replace('_', '-')}}}\n\\end{{table}}")
    open(a.out + ".tex", "w").write("\n".join(tex) + "\n")

    # ---- hop figure ------------------------------------------------------------------------------
    if a.sample_qids and os.path.exists(a.sample_qids):
        keep = set(json.load(open(a.sample_qids)))
        arms_fig = [(lab, {k: v for k, v in tab.items() if k[0] in keep}) for lab, tab in arms]
    else:
        arms_fig = arms
    S = hop_stats(arms_fig, docs, a.max_hops); S["dataset"] = a.dataset; S["sampled"] = bool(a.sample_qids)
    json.dump(S, open(a.out + "_hops.json", "w"), indent=1)
    try:
        draw_hops([(a.dataset, S, s) for s in S["strata"]], a.out + "_hops")
    except Exception as e:      # the markdown, the LaTeX and the distribution are already on disk
        print(f"[showcase] the hop figure failed ({type(e).__name__}: {e}); {a.out}.md/.tex/_hops.json are written, "
              "redraw with --combine once the cause is fixed")
    print("\n".join(md[:8 + min(len(top), 12)]))
    print("\nhop figure:")
    for s in S["strata"]:
        D = S["dist"][s]
        print(f"  {s}: floor mean {fnum(D['floor']['mean'])} (n={D['floor']['n']})" + "".join(
            f" | {lab}: mean {fnum(d['mean'])} gap {fnum(d['mean_gap'], '+.2f')} no valid route {d['unreachable']}"
            for lab, d in D.items() if lab != "floor" and d["n_paths"]))
    print(f"\nwrote {a.out}.md, .tex, _candidates.json, _hops.json, _hops.pdf/.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
""")
print("eval/showcase.py written from this notebook, built 2026-09-08 17:41")
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
