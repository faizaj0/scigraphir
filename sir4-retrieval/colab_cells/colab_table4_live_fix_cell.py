# ===== LIVE FIX for colab_table4_openie.ipynb (paste into the RUNNING kernel after cell 5 failed) =====
# 1. the SIR-4 CS no-CCMP frame arm lives under outputs/ccmp_control/ (found on Drive 9 Sep), not where the spec looked;
# 2. the table section no longer asserts on the OpenIE arm: whichever arms have hops files are rendered, in T4_ARMS order;
# 3. SIR-4 CS gets a fresh pin + path search (the cached hops_t4_*.json only cover the golds pinned from two scans);
# 4. TOMATO's OpenIE arm is the July v1 fusion on the OpenIE graph (operator scorer, no multi-view table): the engine
#    is patched in place so interpret() runs in operator mode, and the checkpoint's 1-feature gate is loaded as-is;
# 5. both datasets are re-run with the same driver logic as cell 5 (new scans, new pins, path search, tables);
# 6. qualitative reporting uses official SIR-4 field labels and marks TOMATO OpenIE as a legacy,
#    unmatched system rather than presenting it as a clean graph-construction ablation.
import os, sys, json, shutil

_EXTRA_CKPT = {("sir4_cs", "frame_nocc"): ["outputs/ccmp_control/sir4_cs/sir4_cs_fusion_qwenmlp_ccmpctrl_epoch10_b1",
                                            "outputs/sir4_cs/sir4_cs_fusion_qwenmlp_epoch20_b1"],
               ("tomato", "openie"):      ["outputs/v1/v1_fusion_epoch20"]}   # July v1 fusion on the OpenIE graph (operator scorer)
for _d in SPEC:
    SPEC[_d]["arms"] = [(n, [r_ for r_ in ((r if isinstance(r, list) else [r]) + _EXTRA_CKPT.get((_d, n), []))], g, s_, gt)
                        for n, r, g, s_, gt in SPEC[_d]["arms"]]
    # dedupe while keeping order
    SPEC[_d]["arms"] = [(n, list(dict.fromkeys(r)), g, s_, gt) for n, r, g, s_, gt in SPEC[_d]["arms"]]

ENGINE = "/content/gfm-rag/gfmrag"
def patch_engine_t4():
    """Operator-scorer checkpoints (the July v1 fusion) in interpret(): idempotent string patches on the
    engine files this kernel installed; the same edits live in kg-construction/gfm-rag (gfm_overlay.zip)."""
    import re as _re
    ft = f"{ENGINE}/trainers/fusion_trainer.py"; ip = f"{ENGINE}/workflow/interpret_paths.py"
    if not os.path.exists(ft): print("[engine] not installed in this kernel (cached run); nothing to patch"); return
    s = open(ft).read(); n = 0
    if 'getattr(self.model, "semantic", "mlp") == "operator"' not in s:
        old = ('        sem = self.model._s_op[0].float()\n        tag, row = self.model._sem_row[str(batch["id"][0])]\n')
        new = ('        sem = self.model._s_op[0].float()\n'
               '        if getattr(self.model, "semantic", "mlp") == "operator":\n'
               '            tag, row = self.model._row[str(batch["id"][0])]\n'
               '            dense = self.model._dense[tag][row].float().to(fused.device)\n'
               '            return did, {"fused": fused, "graph": graph_raw, "scorer": sem, "dense": dense}, (None, None)\n'
               '        tag, row = self.model._sem_row[str(batch["id"][0])]\n')
        assert s.count(old) == 1, "fusion_trainer._channel_scores not where expected"; s = s.replace(old, new); n += 1
    if "Hq, vmask = None, None" not in s:
        old = ('                Hq = np.asarray(tab["H"][row])[:, tab["col"]].astype(np.float32)   # [J, n_doc]\n'
               '                vmask = tab["mask"][row].numpy() > 0\n')
        new = ('                if tab is not None:\n'
               '                    Hq = np.asarray(tab["H"][row])[:, tab["col"]].astype(np.float32)   # [J, n_doc]\n'
               '                    vmask = tab["mask"][row].numpy() > 0\n'
               '                else:\n                    Hq, vmask = None, None\n')
        assert s.count(old) == 1, "fusion_trainer.interpret views not where expected"; s = s.replace(old, new); n += 1
        old = '                    tv = [(int(a), float(Hq[a, j])) for a in np.argsort(-Hq[:, j]) if vmask[a]][:top_views]\n'
        assert s.count(old) == 1; s = s.replace(old, old.rstrip("\n") + " if Hq is not None else []\n"); n += 1
    open(ft, "w").write(s)
    t = open(ip).read()
    if "shape differs from this model" not in t:
        old = '    res = model.load_state_dict(sd, strict=False)\n'
        new = ('    msd = model.state_dict()\n'
               '    bad = [k for k, v in sd.items() if k in msd and tuple(msd[k].shape) != tuple(v.shape)]\n'
               '    for k in bad:\n        sd.pop(k)\n'
               '    if bad:\n        print(f"[interpret] dropped {len(bad)} tensors whose shape differs from this model: {bad[:5]}", flush=True)\n'
               '    res = model.load_state_dict(sd, strict=False)\n')
        assert t.count(old) == 1, "interpret_paths load not where expected"; t = open(ip, "w").write(t.replace(old, new)); n += 1
    print(f"[engine] operator-scorer support: {n} edits applied" if n else "[engine] operator-scorer support already present")
patch_engine_t4()

T4["model_env_t4"] = r"""
# --- operator-scorer checkpoints: no sem_ keys (jmax None) -> model.semantic=operator, 1-feature gate ---------
_SEM_MODE = {"mode": "mlp"}
if "_model_env_mlp" not in globals(): _model_env_mlp = model_env
def model_env(ckpt, graph, skey, gate=None):
    info = inspect_ckpt(ckpt)
    if info["jmax"] is None:
        assert gate is None and not info["resp_keys"], f"{os.path.relpath(ckpt, DRIVE)}: operator-scorer checkpoint with a CCMP head?"
        _SEM_MODE["mode"] = "operator"
        env_ = dict(WANDB_MODE="disabled", HYDRA_FULL_ERROR="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                    OPERATOR_COMPONENTS=opc(graph), OPERATOR_COMPONENTS_TEST=opc(graph), FUSION_ROUTER="cov", STRAT_TEST=QUERIES)
        for k in list(os.environ):
            if k.startswith(("CCMP", "ROUTE", "STRAT_", "CQIG", "RESID_", "MISS_W")): os.environ.pop(k)
        print(f"[operator-scorer checkpoint] {os.path.relpath(ckpt, DRIVE)}: no multi-view scorer; model.semantic=operator, FUSION_ROUTER=cov")
        return env_, info
    _SEM_MODE["mode"] = "mlp"
    return _model_env_mlp(ckpt, graph, skey, gate)
def hydra_common(graph):
    return ("--config-path config/gfm_reasoner --config-name sft_training_fusion text_emb_model=qwen3_st "
            f"datasets.cfgs.root={DATA_ROOT} datasets.cfgs.force_reload=False "
            f"datasets.train_names=[{graph}] datasets.valid_names=[{graph}] model.semantic={_SEM_MODE['mode']} model.cqig=false ")
"""
T4["table4"] = r"""
# --- eval/table4.py: best examples, markdown + GFM-RAG Table-4 LaTeX + candidates json ----------------
ARM_LABEL = {"openie": "OpenIE graph", "frame_nocc": "SciGraphIR", "frame_ccmp": "SciGraphIR + CCMP"}
if DATASET == "tomato":
    ARM_LABEL["openie"] = "OpenIE graph (legacy scorer/fusion)"
HOPS_T4 = {a[0]: f"{SCAN_OUT}/hops_t4_{a[0]}.json" for a in ARMS if os.path.exists(f"{SCAN_OUT}/hops_t4_{a[0]}.json")}
assert HOPS_T4, f"{DATASET}: no hops_t4_*.json on Drive (run the t4paths section)"
T4_ORDER = [n for n in T4_ARMS if n in HOPS_T4]
if T4_MAIN not in HOPS_T4: print(f"[{DATASET}] no '{T4_MAIN}' checkpoint on Drive; the table shows {T4_ORDER}")
FRAME_EDGES = f"{DATA_ROOT}/{S['frame']}/processed/stage1/edges.csv"
_field = DATASET.split("_", 1)[1] if DATASET.startswith("sir4_") else DATASET
QUARTET_FILE = next((p for p in [f"{DRIVE}/quartet/{_field}_test_final/eval.json", f"{DRIVE}/quartet/{DATASET}/eval.json",
                                 f"{CARGO_ROOT}/quartet/data.nosync/benchmark/{_field}_test_final/eval.json",
                                 f"{S4}/results/qualitative/quartet_cache/{_field}_test_final.eval.json"]
                     if os.path.exists(p)), None)
BASE = f"{DRIVE}/outputs/baselines/{DATASET}"
T4_OUT = f"{SCAN_OUT}/table4_{DATASET}"
cmd = [sys.executable, "-u", "eval/table4.py", "--dataset", DATASET, "--queries", QUERIES,
       "--docs", f"{DATA_ROOT}/{DATASET}_test/raw/documents.json", "--edges", FRAME_EDGES,
       "--min-dense", "25", "--max-graph", "5", "--max-fused", "10", "--min-hops", "2", "--max-hops", "6",
       "--top", "30", "--n-tex", "3", "--paths", "3", "--recover", "any", "--out", T4_OUT]
for name in T4_ORDER: cmd += ["--arm", f"{ARM_LABEL.get(name, name)}={HOPS_T4[name]}"]
if DATASET.startswith("sir4_") and not QUARTET_FILE:
    raise FileNotFoundError(
        f"{DATASET}: official QUARTET eval.json is required for verified same-/cross-field labels. "
        "Place it at one of: "
        f"{DRIVE}/quartet/{_field}_test_final/eval.json, "
        f"{DRIVE}/quartet/{DATASET}/eval.json, or "
        f"{CARGO_ROOT}/quartet/data.nosync/benchmark/{_field}_test_final/eval.json; "
        f"the packaged cache {S4}/results/qualitative/quartet_cache/{_field}_test_final.eval.json "
        "was also checked"
    )
if QUARTET_FILE:
    cmd += ["--quartet", QUARTET_FILE]
for tag in ("qwen3", "bge", "reasonir", "bm25"):
    p = f"{BASE}/predictions_{tag}_{DATASET}_test.json"
    if os.path.exists(p): cmd += ["--pred", f"{tag}={p}"]
sh(cmd, S4)
print(open(f"{T4_OUT}.md").read()[:14000])
print("\nfiles:", sorted(f for f in os.listdir(SCAN_OUT) if f.startswith(("table4_", "hops_t4_"))))
"""
ALL = dict(SECTIONS); ALL.update(T4)

# --- the pinned set changes once the third scan exists, so the two-arm hops files are parked and redone ---
for _d in ("sir4_cs", "tomato"):
    _o = f"{DRIVE}/outputs/scan/{_d}"
    for _f in os.listdir(_o) if os.path.isdir(_o) else []:
        if _f.startswith("hops_t4_") and _f.endswith(".json"):
            shutil.move(f"{_o}/{_f}", f"{_o}/{_f[:-5]}_2arm.json"); print(f"parked {_d}/{_f} -> {_f[:-5]}_2arm.json")

# --- recompute the plan with the corrected spec ---------------------------------------------------------
PLAN = {}
for d in DATASETS:
    o = f"{DRIVE}/outputs/scan/{d}"; arms = _arms_with_ckpt(d)
    PLAN[d] = {"arms": arms,
               "scan":    bool(arms) and all(os.path.exists(f"{o}/scan_all_{a}.json") for a in arms),
               "t4paths": bool(arms) and all(os.path.exists(f"{o}/hops_t4_{a}.json") for a in arms),
               "table4":  False}          # always re-render
for d in DATASETS:
    P = PLAN[d]; print(f"  {d:14} arms: {','.join(P['arms'])}  ->  to do: "
                       + ", ".join(k for k in ("scan", "t4paths", "table4") if not P[k]))

def sections_for(d):
    P = PLAN[d]
    todo = ["graphs", "model_env", "model_env_t4"]
    if not (P["scan"] and P["t4paths"]): todo.insert(1, "components")
    return todo + ["t4scan", "t4paths", "table4"]

DONE, FAILED = [], {}
for _d in DATASETS:
    print(f"\n\n#################### {_d} ####################")
    if not (PLAN[_d]["scan"] and PLAN[_d]["t4paths"]) and not os.path.isdir("/content/gfm-rag"):
        FAILED[_d] = "needs a scan or a path search but the engine is not installed"; print(f"[{_d}] SKIPPED: {FAILED[_d]}"); continue
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
