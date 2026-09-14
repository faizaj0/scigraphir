"""
build_showcase_refix_cell.py -- ONE paste-in Colab cell that fixes a RUNNING kernel.

Rewrites eval/showcase.py with the current repo version and redraws every dataset from the
hops_*.json already on Drive: no engine, no Qwen3, no path search. Use it instead of re-uploading
the notebook when the kernel is still alive.

    python3 prep/build_showcase_refix_cell.py     # -> colab_showcase_refix_cell.py
"""
import ast
import datetime
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

CELL = r'''# ===== LIVE FIX: rewrite eval/showcase.py and redraw every dataset from the cached hops files =====
# Paste into the RUNNING kernel and run. No engine, no Qwen3, no path search: it only re-reads the
# hops_*.json already on Drive. Fixes in this version:
#   * the hop figure's legend is keyed on the arm LABEL, so an arm keeps its colour across panels and
#     the shared legend no longer mislabels the datasets that have fewer arms;
#   * the panel title shows the real gold count and says "no floor" when min_hops is absent;
#   * the candidate table no longer emits a stray empty column when no baseline predictions are found;
#   * a failure in the figure can no longer destroy the .md / .tex / _hops.json.
import os, sys, json, glob, time, subprocess
S4        = globals().get("S4",        "/content/cargo/sir4-retrieval")
DRIVE     = globals().get("DRIVE",     "/content/drive/MyDrive/cargo-gfmrag")
DATA_ROOT = globals().get("DATA_ROOT", "/content/cargo/kg-construction/data")
DATASETS  = globals().get("DATASETS") or [globals().get("DATASET")]
DATASETS  = [d for d in DATASETS if d]
assert DATASETS, "no DATASETS / DATASET in this kernel; set DATASETS = ['mir', ...] before running"
TOP, N_TEX = 30, 4

os.makedirs(f"{S4}/eval", exist_ok=True)
open(f"{S4}/eval/showcase.py", "w").write(r"""__SHOWCASE_PY__""")
print(f"wrote {S4}/eval/showcase.py (repo version of __SHOWCASE_BUILT__)\n")

ARM_LABEL = {"frame_ccmp": "SciGraphIR (frame graph + CCMP)", "frame_ccmp_off": "frame graph, CCMP gate off",
             "frame_nocc": "frame graph, no CCMP", "openie": "OpenIE graph"}

def _run(cmd):
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=S4, env=dict(os.environ, PYTHONUNBUFFERED="1"),
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while True:
        c = os.read(p.stdout.fileno(), 8192)
        if not c: break
        sys.stdout.write(c.decode("utf-8", "replace")); sys.stdout.flush()
    p.wait(); print(f"[{time.time()-t0:.0f}s, exit {p.returncode}]")
    return p.returncode

done, missing = [], []
for d in DATASETS:
    out = f"{DRIVE}/outputs/scan/{d}"
    hops = {n: f"{out}/hops_{n}.json" for n in ARM_LABEL if os.path.exists(f"{out}/hops_{n}.json")}
    if not hops:
        missing.append(d); print(f"[skip] {d}: no hops_*.json under {out}"); continue
    print(f"\n======== {d}: {', '.join(hops)} ========")
    show = f"{out}/showcase_{d}"
    cmd = [sys.executable, "-u", "eval/showcase.py", "--dataset", d,
           "--queries", f"{DATA_ROOT}/{d}_test/raw/test.json",
           "--docs",    f"{DATA_ROOT}/{d}_test/raw/documents.json",
           "--edges",   f"{DATA_ROOT}/{d}_test_v16sc/processed/stage1/edges.csv",
           "--out", show, "--top", str(TOP), "--n-tex", str(N_TEX)]
    if os.path.exists(f"{out}/qids_sample.json"):
        cmd += ["--sample-qids", f"{out}/qids_sample.json"]
    for n in ("frame_ccmp", "frame_ccmp_off", "frame_nocc", "openie"):
        if n in hops: cmd += ["--arm", f"{ARM_LABEL[n]}={hops[n]}"]
    for tag in ("qwen3", "bge", "reasonir", "bm25"):
        p_ = f"{DRIVE}/outputs/baselines/{d}/predictions_{tag}_{d}_test.json"
        if os.path.exists(p_): cmd += ["--pred", f"{tag}={p_}"]
    done.append(d) if _run(cmd) == 0 else missing.append(d)

# the multi-dataset figure, then show everything
hops_json = sorted(glob.glob(f"{DRIVE}/outputs/scan/*/showcase_*_hops.json"))
if hops_json:
    print("\n======== combined hop figure ========")
    _run([sys.executable, "-u", "eval/showcase.py", "--combine", *hops_json,
          "--out", f"{DRIVE}/outputs/scan/fig_hops_all"])
from IPython.display import Image, display
for p_ in [f"{DRIVE}/outputs/scan/fig_hops_all.png"] + [f"{DRIVE}/outputs/scan/{d}/showcase_{d}_hops.png" for d in done]:
    if os.path.exists(p_): print(os.path.relpath(p_, DRIVE)); display(Image(p_))
print("\nredrawn:", done, "| skipped or failed:", missing)
'''


def main() -> int:
    show = open(f"{ROOT}/eval/showcase.py").read()
    assert "'''" not in show, "showcase.py must not contain ''' (it is embedded inside r'''...''' here)"
    stamp = datetime.datetime.fromtimestamp(os.path.getmtime(f"{ROOT}/eval/showcase.py")).strftime("%Y-%m-%d %H:%M")
    cell = CELL.replace("__SHOWCASE_PY__", show.replace('"""', "'''")).replace("__SHOWCASE_BUILT__", stamp)
    ast.parse(cell)
    out = f"{ROOT}/colab_showcase_refix_cell.py"
    open(out, "w").write(cell)
    print(f"wrote {out}: {len(cell.splitlines())} lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
