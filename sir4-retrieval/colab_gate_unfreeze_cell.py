# ===== LIVE PATCH: make the fusion gate (gamma) train, and watch it train =====
# Paste into the RUNNING colab_sir4_hyb kernel BEFORE calling train_arm(...). Idempotent.
# What it does to /content/gfm-rag/gfmrag/workflow/sft_training.py (the training subprocess):
#   1. moves the gate's parameters (gate.*, i.e. gamma_q = softplus(gate(phi_q))) into their own AdamW group with
#      lr FUSION_GATE_LR (default 5e-3, 10x the model lr) and no weight decay, built AFTER the dtype cast;
#   2. wraps optimizer.step so every 100 steps it prints the gate's output bias, its last-layer weight norm and the
#      GRADIENT norms. If "grad bias" prints None or 0.0 at step 1 and 101, the gate receives no gradient and a
#      higher lr cannot help: stop the run and report the line.
# gamma_mean in the [diag] line must move away from 0.0100 within the first epoch if the gate is training.
import os, re
STF = "/content/gfm-rag/gfmrag/workflow/sft_training.py"
src = open(STF).read()
MARK = "# === GATE UNFREEZE + MONITOR (live patch) ==="
if MARK in src:
    print("sft_training.py already patched")
else:
    anchor = "    trainer = instantiate(\n"
    assert src.count(anchor) == 1, f"expected one trainer construction in sft_training.py, found {src.count(anchor)}"
    block = '''    ''' + MARK + '''
    import torch as _t
    _gate_params = [p for n, p in model.named_parameters() if n.startswith("gate.")]
    assert _gate_params, "no gate.* parameters on the model; is this the fusion reasoner?"
    _gate_ids = {id(p) for p in _gate_params}
    for _g in optimizer.param_groups:
        _g["params"] = [p for p in _g["params"] if id(p) not in _gate_ids]
    optimizer.param_groups = [g for g in optimizer.param_groups if g["params"]]
    _gate_lr = float(os.environ.get("FUSION_GATE_LR", "5e-3"))
    optimizer.add_param_group({"params": _gate_params, "lr": _gate_lr, "weight_decay": 0.0})
    for _p in _gate_params:
        _p.requires_grad_(True)
    print(f"[gate] {len(_gate_params)} gate tensors in their own AdamW group, lr {_gate_lr}, wd 0; "
          f"bias {float(model.gate[-1].bias.flatten()[0]):.4f} dtype {model.gate[-1].bias.dtype}", flush=True)
    _step0 = optimizer.step
    _cnt = {"n": 0}
    def _gate_step(*a, **k):
        _cnt["n"] += 1
        if _cnt["n"] % 100 == 1:
            _b, _w = model.gate[-1].bias, model.gate[-1].weight
            _gb = None if _b.grad is None else float(_b.grad.norm())
            _gw = None if _w.grad is None else float(_w.grad.norm())
            print(f"[gate] step {_cnt['n']}  bias {float(_b.flatten()[0]):.4f}  gamma(bias) {float(_t.nn.functional.softplus(_b.flatten()[0])):.4f}  "
                  f"|w_out| {float(_w.norm()):.3e}  grad bias {_gb}  grad w_out {_gw}", flush=True)
        return _step0(*a, **k)
    optimizer.step = _gate_step
'''
    src = src.replace(anchor, block + anchor)
    open(STF, "w").write(src)
    print("patched sft_training.py: gate in its own optimizer group (FUSION_GATE_LR, default 5e-3) + [gate] monitor every 100 steps")

# optional: a different gate learning rate for the next train_arm(...) call
os.environ["FUSION_GATE_LR"] = os.environ.get("FUSION_GATE_LR", "5e-3")
# the notebook's set_dataset() pops env keys starting with FUSION_ before snapshotting the subprocess env,
# so inject the value through sh() instead of relying on os.environ:
if "_sh_before_gate" not in globals():
    _sh_before_gate = sh
    def sh(cmd, cwd, extra=None, **kw):
        extra = dict(extra or {}); extra.setdefault("FUSION_GATE_LR", os.environ["FUSION_GATE_LR"])
        return _sh_before_gate(cmd, cwd, extra=extra, **kw)
print("sh() now passes FUSION_GATE_LR =", os.environ["FUSION_GATE_LR"], "to every training subprocess")
print("next: run the biology smoke (3 steps) and read the two [gate] lines; then train_arm('biology', 'control')")
