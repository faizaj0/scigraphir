"""
build_showcase_all_notebook.py -- ONE notebook for every dataset: the qualitative showcase (graph-channel
scan of every gold, path interpretations for every cross-field gold, the ranked "amazing example" list,
the GFM-RAG Table-4 LaTeX, the Fig-6-style hop figure), looping over DATASETS in one runtime.

Setup is done once (all bundles unpacked side by side, Qwen3, engine, fusion sources with
interpret()); the analysis sections of colab_showcase_cell.py are stored as source strings and run
per dataset. Finished arms are cached on Drive, so a re-run skips them.

    python3 prep/build_showcase_all_notebook.py      # -> colab_showcase_all.ipynb
"""
import ast
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import overlay_cell  # noqa: E402
from build_showcase_notebook import tolerate_pyg_fix  # noqa: E402
SRC_NB = f"{ROOT}/notebooks/colab_mir_all.ipynb"      # its fusion blob carries interpret(); overlay = the newest repo scripts
SETUP = [("import os, shutil", "first", "2b. Qwen3-Embedding (cached on Drive)"),
         ("import os, sys, torch", "first", "3. Engine"),
         ("# === write the CARGO-fusion files", "first", "3b. Fusion sources (with interpret())"),
         ("# Cell 3a is reused VERBATIM", "first", "3c. Config defaults"),
         ("# The ULTRA layers vendored", "last", "3d. PyG version fix"),
         ("# THE TRAINING SUBPROCESS IS A FRESH PYTHON", "last", "3e. torchvision shim"),
         ("STF = ", "first", "3f. Diagnostics patch"),
         ("# Idempotent: re-running is a no-op", "first", "3g. CCMP_LR patch")]
SECTIONS = [("# --- graphs, checkpoints, output dir", "graphs"), ("# --- component tables", "components"),
            ("# --- checkpoint environment", "model_env"), ("# --- scan every gold", "scan"), ("# --- the table", "table"),
            ("# ================= SHOWCASE", "paths"), ("os.makedirs(f\"{S4}/eval\", exist_ok=True)", "showcase")]


def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}


def demagic(src: str) -> str:
    """The cell source with `!`/`%` lines replaced by `pass` AT THE SAME INDENT, so it can be parsed."""
    out, cont = [], False
    for ln in src.split("\n"):
        if cont:                                  # a backslash continuation of the magic above: drop it
            cont = ln.rstrip().endswith("\\")
            continue
        if ln.lstrip().startswith(("!", "%")):
            cont = ln.rstrip().endswith("\\")
            out.append(" " * (len(ln) - len(ln.lstrip())) + "pass")
        else:
            out.append(ln)
    return "\n".join(out)


SKIP_MSG = "print('skipped: every scan and path file is already on Drive, so the engine is not needed')"
GUARD_HEAD = ("# Only runs when cell 1 found a scan or a path search still to do; drawing the figures from\n"
              "# cached interpretations needs neither the engine nor Qwen3.\n")


def guard(src: str) -> str:
    """Run a setup cell only when cell 1 decided the engine is needed.

    Indenting the source under `if NEED_ENGINE:` keeps it readable, and IPython still transforms the
    `!` lines inside the block -- but it would also indent the CONTENT of a multi-line string, and some
    of these cells write module-level Python into a file from exactly such a string. Those cells are
    run verbatim through run_cell instead.
    """
    import base64 as b64
    import io
    import tokenize
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(demagic(src)).readline))
        spans_lines = any(t.type == tokenize.STRING and t.start[0] != t.end[0] for t in toks)
    except tokenize.TokenError:
        spans_lines = True
    if spans_lines:      # verbatim, so nothing inside a multi-line literal moves
        blob = b64.b64encode(src.encode()).decode()
        return (GUARD_HEAD + "# (run verbatim: this cell writes indentation-sensitive source into a file)\n"
                "import base64 as _b64\n"
                f'_src = _b64.b64decode("{blob}").decode()\n'
                "if NEED_ENGINE:\n"
                "    _r = get_ipython().run_cell(_src, store_history=False)\n"
                "    assert _r.success, 'setup cell failed; see the traceback above'\n"
                f"else:\n    {SKIP_MSG}\n")
    body = "\n".join(("    " + ln) if ln.strip() else ln for ln in src.rstrip("\n").split("\n"))
    return GUARD_HEAD + "if NEED_ENGINE:\n" + body + f"\nelse:\n    {SKIP_MSG}\n"


PATHS = r'''# 1. Datasets to run, environment, Drive, paths, the checkpoint spec. Edit DATASETS to run a subset.
DATASETS = ["sir4_cs", "sir4_biology", "sir4_physics", "sir4_matsci", "tomato", "mir"]
import os, sys, re, json, glob, shutil, time, zipfile, subprocess, threading, random
# Colab runs Python 3.13 and the engine's editable install refuses it, so subprocesses must be told
# where gfmrag is and which interpreter to use. Must precede every sh() snapshot of os.environ.
os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
os.environ["PYTHONPATH"] = "/content/gfm-rag" + os.pathsep + os.environ.get("PYTHONPATH", "")
from google.colab import drive
drive.mount('/content/drive')
DRIVE      = "/content/drive/MyDrive/cargo-gfmrag"
SCIGRAPHIR_ROOT = "/content/scigraphir"
DATA_ROOT  = f"{SCIGRAPHIR_ROOT}/retriever/data"
S4         = f"{SCIGRAPHIR_ROOT}/experiments"
KGDIR      = f"{SCIGRAPHIR_ROOT}/retriever"
RUNS       = "/content/runs"
OP_MODEL   = "/content/qwen3"
OP_SLUG    = "_content-qwen3"
TRAIN = TEST = "mir_test_v16sc"        # config DEFAULT names only; every command overrides them on the CLI
os.environ["SCIGRAPHIR_ROOT"] = SCIGRAPHIR_ROOT
try:
    from google.colab import userdata
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
except Exception:
    os.environ.setdefault("HF_TOKEN", "")
os.makedirs(RUNS, exist_ok=True)

def sh(cmd, cwd, extra=None, log=None, check=True):
    """Run a command with LIVE output (raw chunks, so tqdm bars show)."""
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=cwd, env=dict(env, **(extra or {})), shell=isinstance(cmd, str),
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    fh = open(log, "w") if log else None
    while True:
        c = os.read(p.stdout.fileno(), 8192)
        if not c: break
        s = c.decode("utf-8", "replace")
        sys.stdout.write(s); sys.stdout.flush()
        if fh: fh.write(s); fh.flush()
    p.wait()
    if fh: fh.close()
    print(f"\n[{time.time()-t0:.0f}s, exit {p.returncode}]")
    if check:
        assert p.returncode == 0, f"FAILED (exit {p.returncode}): {cmd if isinstance(cmd, str) else ' '.join(map(str, cmd))}"
    return p.returncode
env = dict(os.environ, SCIGRAPHIR_ROOT=SCIGRAPHIR_ROOT, PYTHONUNBUFFERED="1")

__SPEC__
for d in DATASETS: assert d in SPEC, f"{d}: DATASETS must be from {sorted(SPEC)}"

# --- resume plan: what is already on Drive, so a re-run only does what is missing --------------------
# FORCE = "" (nothing), "showcase" (redraw the markdown/figures), "paths", "scan", or "all".
FORCE = ""
_ORDER = ["scan", "paths", "showcase"]
def _force(stage): return FORCE == "all" or (FORCE in _ORDER and _ORDER.index(stage) >= _ORDER.index(FORCE))
def _arms_with_ckpt(d):
    out = []
    for name, rel, _g, _s, _gate in SPEC[d]["arms"]:
        rels = rel if isinstance(rel, list) else [rel]
        if any(os.path.exists(f"{DRIVE}/{r}/model_best.pth") for r in rels): out.append(name)
    return out
PLAN = {}
for d in DATASETS:
    o = f"{DRIVE}/outputs/scan/{d}"; arms = _arms_with_ckpt(d)
    PLAN[d] = {"arms": arms,
               "scan":  bool(arms) and all(os.path.exists(f"{o}/scan_all_{a}.json") for a in arms) and not _force("scan"),
               "paths": bool(arms) and all(os.path.exists(f"{o}/hops_{a}.json") for a in arms) and not _force("paths"),
               "showcase": (os.path.exists(f"{o}/showcase_{d}.md") and os.path.exists(f"{o}/showcase_{d}_hops.png")
                            and not _force("showcase")),
               "table": os.path.exists(f"{o}/graph_channel_scan_{d}.md") and not _force("scan")}
# The engine, Qwen3 and the component tables are needed ONLY to run a scan or a path search. When every
# dataset has both cached, cells 2b-3g and the per-dataset component build are skipped and the notebook
# goes straight from the bundles to eval/showcase.py (minutes instead of the better part of an hour).
NEED_ENGINE = any(not (PLAN[d]["scan"] and PLAN[d]["paths"]) for d in DATASETS)
import numpy as np      # the table section needs it whether or not the engine cells run
print("datasets:", DATASETS, "| FORCE:", FORCE or "(resume)")
for d in DATASETS:
    P = PLAN[d]
    todo = [k for k in ("scan", "paths", "showcase") if not P[k]] or ["nothing (all cached)"]
    print(f"  {d:14} arms {len(P['arms'])}: {','.join(P['arms']) or 'NO CHECKPOINT'}  ->  to do: {', '.join(todo)}")
print(f"engine + Qwen3 + component tables needed: {NEED_ENGINE}"
      + ("" if NEED_ENGINE else "  -> cells 2b-3g are no-ops; run cell 2, then 4, 5, 6"))
'''

UNPACK = r'''# 2. Unpack every bundle side by side into one local root (caches kept) and install the CURRENT
# repo scripts. Bundles: <dataset>_bundle.zip on Drive, each with the frame graph and (except TOMATO
# before its OpenIE run) the OpenIE graph of its test split.
# Re-running this cell in the same runtime (after fixing something downstream) re-uses what is already
# unpacked; set FORCE_UNPACK = True to wipe /content/scigraphir and start over.
FORCE_UNPACK = False
MARK = f"{SCIGRAPHIR_ROOT}/.unpacked.json"
_have = set(json.load(open(MARK))) if (os.path.exists(MARK) and not FORCE_UNPACK) else set()
_todo = [d for d in DATASETS if d not in _have]
if _todo:
    KEEP, PARK = f"{SCIGRAPHIR_ROOT}/outputs/caches", "/content/_caches_keep"
    if _have:                       # incremental: keep what is unpacked, add the missing bundles
        for d in _todo:
            z = f"{DRIVE}/{d}_bundle.zip"
            assert os.path.exists(z) and zipfile.is_zipfile(z), f"missing or corrupt {z}"
            t0 = time.time(); zipfile.ZipFile(z).extractall(SCIGRAPHIR_ROOT); print(f"unpacked {os.path.basename(z)} ({time.time()-t0:.0f}s)")
    else:
        if os.path.isdir(KEEP):
            shutil.rmtree(PARK, ignore_errors=True); shutil.move(KEEP, PARK)
        if os.path.exists(SCIGRAPHIR_ROOT):
            shutil.rmtree(SCIGRAPHIR_ROOT)
        os.makedirs(SCIGRAPHIR_ROOT, exist_ok=True)
        for d in DATASETS:
            z = f"{DRIVE}/{d}_bundle.zip"
            assert os.path.exists(z) and zipfile.is_zipfile(z), f"missing or corrupt {z}"
            t0 = time.time(); zipfile.ZipFile(z).extractall(SCIGRAPHIR_ROOT); print(f"unpacked {os.path.basename(z)} ({time.time()-t0:.0f}s)")
        if os.path.isdir(PARK):
            os.makedirs(os.path.dirname(KEEP), exist_ok=True); shutil.move(PARK, KEEP); print("restored caches")
    json.dump(sorted(_have | set(DATASETS)), open(MARK, "w"))
else:
    print("bundles already unpacked:", sorted(_have), "(FORCE_UNPACK = True to redo)")
OVERLAY = json.loads(r"""__OVERLAY__""")
ov = f"{DRIVE}/code_overlay"
if os.path.isdir(ov):
    shutil.copytree(ov, SCIGRAPHIR_ROOT, dirs_exist_ok=True); print("applied Drive code_overlay")
for rel, src in OVERLAY.items():
    p = f"{SCIGRAPHIR_ROOT}/{rel}"
    os.makedirs(os.path.dirname(p), exist_ok=True); open(p, "w").write(src)
print(f"installed {len(OVERLAY)} repo scripts captured __BUILT__")
sys.path.insert(0, SCIGRAPHIR_ROOT)
import scigraphir_paths as cp
for d in DATASETS:
    for g in [SPEC[d]["frame"]] + ([SPEC[d]["openie"]] if SPEC[d]["openie"] else []):
        ok = os.path.exists(f"{DATA_ROOT}/{g}/processed/stage1/nodes.csv")
        print(f"  {'ok ' if ok else 'MISSING'}  {d:13} graph {g}")
'''

DRIVER = r'''# 5. Run the analysis for every dataset in DATASETS. Each section is the corresponding cell of the
# per-dataset showcase notebook. Anything cell 1's PLAN found on Drive is skipped, so a re-run after a
# failure (or after adding a checkpoint) only does the missing work; set FORCE in cell 1 to redo a stage.
def set_dataset(d):
    global DATASET, S, SCAN_DATASET, CACHE, QUERIES, env
    DATASET = SCAN_DATASET = d; S = SPEC[d]
    CACHE = f"{DRIVE}/outputs/{d}/cache"; os.makedirs(CACHE, exist_ok=True)
    QUERIES = f"{DATA_ROOT}/{d}_test/raw/test.json"
    os.environ["SCIGRAPHIR_DATASET"] = d
    env = dict(os.environ, SCIGRAPHIR_ROOT=SCIGRAPHIR_ROOT, SCIGRAPHIR_DATASET=d, PYTHONUNBUFFERED="1")
    cp.set_dataset(d); print(cp.banner())
    for k in list(os.environ):
        if k.startswith(("CCMP", "ROUTE", "STRAT_", "CQIG", "RESID_", "MISS_W")): os.environ.pop(k)

def sections_for(d):
    """The sections this dataset still needs. 'components' builds the operator/semantic tables and
    restores the index over Drive FUSE (minutes); it is only ever used by a scan or a path search."""
    P = PLAN[d]
    todo = ["graphs", "model_env"]
    if not (P["scan"] and P["paths"]): todo.insert(1, "components")
    todo += ["scan"]                                   # cached arms print [cached] and cost nothing
    if not P["table"]: todo += ["table"]
    todo += ["paths", "showcase"]
    return todo

DONE, SKIPPED, FAILED = [], [], {}
for _d in DATASETS:
    if PLAN[_d]["showcase"] and PLAN[_d]["scan"] and PLAN[_d]["paths"]:
        SKIPPED.append(_d); print(f"\n#################### {_d}: complete on Drive, skipped (FORCE in cell 1 to redo)")
        continue
    print(f"\n\n#################### {_d} ####################")
    if not (PLAN[_d]["scan"] and PLAN[_d]["paths"]) and not os.path.isdir("/content/gfm-rag"):
        FAILED[_d] = "needs a scan or a path search but the engine is not installed"
        print(f"[{_d}] SKIPPED: {FAILED[_d]}; re-run cell 1 (so NEED_ENGINE is recomputed) then cells 2b-3g")
        continue
    set_dataset(_d)
    _todo = sections_for(_d)
    print("sections:", ", ".join(_todo), "| skipped:", ", ".join(k for k in SECTIONS if k not in _todo) or "none")
    if "table" not in _todo:      # cached: show it instead of recomputing it from the scan files
        print("".join(open(f"{DRIVE}/outputs/scan/{_d}/graph_channel_scan_{_d}.md").readlines()[:16]))
    try:
        for _name in _todo:
            print(f"\n======== {_d}: {_name} ========")
            _r = get_ipython().run_cell(SECTIONS[_name], store_history=False)
            assert _r.success, f"section '{_name}' failed; see the traceback above"
        DONE.append(_d)
    except Exception as _e:
        FAILED[_d] = str(_e)[:300]; print(f"[{_d}] FAILED: {_e}")
print("\nfinished:", DONE, "| skipped (already done):", SKIPPED, "| failed:", FAILED)
if FAILED: print("re-run this cell after fixing the cause: everything that succeeded is cached on Drive and will be skipped")
'''

COMBINE = r'''# 6. The multi-dataset hop figure (one panel per dataset, cross-field stratum) and where everything is.
hops = sorted(glob.glob(f"{DRIVE}/outputs/scan/*/showcase_*_hops.json"))
if hops:
    sh([sys.executable, "-u", "eval/showcase.py", "--combine", *hops, "--out", f"{DRIVE}/outputs/scan/fig_hops_all"], S4)
    from IPython.display import Image, display
    if os.path.exists(f"{DRIVE}/outputs/scan/fig_hops_all.png"):
        display(Image(f"{DRIVE}/outputs/scan/fig_hops_all.png"))
    else:
        print("no combined figure (see the message above)")
for d in DATASETS:
    p = f"{DRIVE}/outputs/scan/{d}"
    print(f"{d:13}", sorted(f for f in os.listdir(p) if f.startswith(("showcase_", "graph_channel_scan"))) if os.path.isdir(p) else "no output")
'''


def main() -> int:
    cell = open(f"{ROOT}/colab_cells/colab_showcase_cell.py").read()
    spec = cell[cell.index("# --- per-dataset spec"): cell.index("assert DATASET in SPEC")]
    tail = cell[cell.index("# the interpretation entry point"):]
    interp = tail[: tail.index("print(\"wrote interpret_paths.py\")")] + 'print("wrote interpret_paths.py")\n'
    idx = [tail.index(m) for m, _ in SECTIONS]
    assert idx == sorted(idx)
    parts = {name: tail[a:b] for (_, name), a, b in zip(SECTIONS, idx, idx[1:] + [len(tail)])}
    import base64   # the sections contain both kinds of triple quotes, so they travel base64-encoded
    blob = base64.b64encode(json.dumps({n: parts[n] for _, n in SECTIONS}).encode()).decode()
    # interpret_paths.py goes INTO the engine, so it is written only when the engine cells ran; the
    # SECTIONS dict below is always needed (the driver runs it even on a cached, figures-only pass).
    sections_cell = ("# 4. interpret_paths.py (the interpretation entry point) and the per-dataset analysis sections,\n"
                     "# stored as source (base64 JSON, they contain triple quotes) and run by the driver below per dataset.\n"
                     + guard(interp) + "import base64 as _b64\n"
                     f"SECTIONS = json.loads(_b64.b64decode(\"{blob}\").decode())\n"
                     "print('sections:', list(SECTIONS))\n")

    src = [''.join(c["source"]) for c in json.load(open(SRC_NB))["cells"] if c["cell_type"] == "code"]
    unpack_src = next(s for s in src if s.startswith("# 2. Unpack"))
    m = re.search(r"OVERLAY = json\.loads\(r'''(.*?)'''\)", unpack_src, re.S)
    overlay_json = m.group(1)
    built = re.search(r"repo scripts captured ([0-9: -]+)", unpack_src).group(1).strip()
    assert '"""' not in overlay_json
    setup = []
    for mark, which, title in SETUP:
        hits = [s for s in src if s.startswith(mark)]
        assert hits, f"no cell starts with {mark!r}"
        setup.append((title, tolerate_pyg_fix(hits[0] if which == "first" else hits[-1])))
    header = md("# Qualitative showcase, all datasets\n\n"
                "One runtime, every dataset in `DATASETS` (SIR-4 CS / biology / physics / materials science, TOMATO-Star, MIR):\n\n"
                "1. every gold of every test query ranked under the graph channel alone, the multi-view scorer, Qwen3 cosine and the fused "
                "score, per arm (frame graph + CCMP, same weights with the CCMP gate off, no-CCMP control, OpenIE graph; arms whose "
                "checkpoint is missing are skipped);\n"
                "2. NBFNet-style gradient beam search from the query's seed frames to every gold of every cross-field query (plus 120 same-field "
                "queries), with the CCMP gate on every hop;\n"
                "3. `eval/showcase.py`: candidates ranked for a reader (cosine buries the gold, the model recovers it through a mechanism route), "
                "the GFM-RAG Table-4 LaTeX, the Fig-6-style hop figure; then one multi-dataset figure.\n\n"
                "Needs on Drive: `<dataset>_bundle.zip` for each dataset, the finished checkpoints (listed by cell 1), the scorer files, "
                "`gfm-rag-adapted.zip`, `qwen3-embedding-0.6b/`. Outputs: `outputs/scan/<dataset>/` and `outputs/scan/fig_hops_all.pdf`. "
                "**Re-running is cheap.** Cell 1 checks Drive and prints, per dataset, what is still to do; a dataset whose scan, "
                "paths and showcase files are all there is skipped, the component build is skipped whenever the scan and the path "
                "search are cached, and when no dataset needs either, cells 2b-3g (Qwen3 and the engine) are no-ops -- so a re-run "
                "that only has to redraw the figures takes minutes rather than the better part of an hour. Set `FORCE` in cell 1 to "
                "`\"showcase\"`, `\"paths\"`, `\"scan\"` or `\"all\"` to redo a stage anyway.")
    cells = [header, md("## 1. Datasets, environment, checkpoint spec"), code(PATHS.replace("__SPEC__", spec.rstrip() + "\n")),
             md("## 2. Unpack every bundle + current scripts"),
             code(UNPACK.replace("__OVERLAY__", overlay_json).replace("__BUILT__", built))]
    for title, s in setup:
        cells += [md(f"## {title}"), code(guard(s))]
        if title.startswith("3b."):
            cells += [md("## 3b-bis. Current fusion sources (embedded overlay)"), code(guard(overlay_cell.cell()))]
    cells += [md("## 4. interpret_paths.py + analysis sections"), code(sections_cell),
              md("## 5. Run every dataset"), code(DRIVER),
              md("## 6. Multi-dataset hop figure"), code(COMBINE)]
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        try:
            ast.parse(demagic(c["source"]))
        except SyntaxError as e:
            raise SystemExit(f"cell {i} line {e.lineno}: {e.msg}")
    for n in parts:   # each section must parse on its own too
        ast.parse(parts[n])
    out = f"{ROOT}/notebooks/colab_showcase_all.ipynb"
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                                       "language_info": {"name": "python"}, "accelerator": "GPU"},
          "nbformat": 4, "nbformat_minor": 5}
    json.dump(nb, open(out, "w"), indent=1)
    print(f"wrote {out}: {len(cells)} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
