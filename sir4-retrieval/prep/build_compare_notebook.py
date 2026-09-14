#!/usr/bin/env python3
"""Emit ONE Colab notebook that reads every arm, in every domain, from Drive.

This exists because the arms were being compared by scrolling four training logs and
remembering which number came from which run. The five-arm table below is the whole
experiment; anything not in it is an ablation and belongs in a different table.

    arm  semantic      graph      CQIG        fusion
    1    multi-view    off        --          --
    2A   multi-view    standard   off         additive
    2B   multi-view    standard   off         mixture
    3A   multi-view    standard   lam = 0.9   additive
    3B   multi-view    standard   lam = 0.9   mixture

FOUR TRAINED RUNS PER DOMAIN, one per cell of the 2x2, plus arm 1 from section 5d.

The fusion form is a TRAINING difference, not an eval-time re-scoring. relu(z(g)) zeroes
the graph gradient for every document below the graph's own mean and the mixture does not,
so the two forms fit different graph channels; scoring one form off the other's checkpoint
compares combiners, not methods. The [sweep]/[gsweep] lines still print and are still read
below, but as a secondary paired diagnostic, never as the headline.

Arm 1 is 5d's standalone `mlp` score, NOT the `[diag] semantic` column of a fusion run.
That column belongs to a scorer that kept training under the fused ranking loss, so using
it as the baseline would subtract part of the graph's own contribution from every delta.

Usage:  python build_compare_notebook.py [--epochs 10] [--batch 1]
Writes: ../colab_compare_sir4.ipynb
"""
import argparse
import json
import os

ALL_DOMAINS = ["matsci", "biology", "cs", "physics"]

_ap = argparse.ArgumentParser()
# ONE NOTEBOOK PER DOMAIN is the default, matching the training notebooks: a session mounts
# Drive, works one domain, and a table that is three-quarters MISSING is not informative
# about the domain you are actually looking at. `--domain all` still builds the cross-domain
# view, which is the one to read at the end when the four have landed, not during.
_ap.add_argument("--domain", default="matsci", choices=ALL_DOMAINS + ["all"])
_ap.add_argument("--run-set", default="ladder",
                 help="read outputs/<NAME>/<dataset>/. Defaults to the clean `ladder` namespace so a report never mixes new runs with the exploratory ones.")
_ap.add_argument("--epochs", type=int, default=10)
_ap.add_argument("--batch", type=int, default=1)
# NOT `colab_compare_sir4*.ipynb`: `colab_compare_sir4_arms.ipynb` already exists and does
# something else entirely (it rescores the thesis table from saved prediction files). Two
# notebooks whose names differ by a suffix is how the wrong one gets opened.
_ap.add_argument("--out", default=None)
_a = _ap.parse_args()
DOMAINS = ALL_DOMAINS if _a.domain == "all" else [_a.domain]
if _a.out is None:
    # matches build_notebook.py's scheme: sir4_<scope>_<role>.ipynb, so a domain's ladder
    # and its report sort next to each other instead of under a shared `colab_` prefix.
    _a.out = os.path.join(os.path.dirname(__file__), os.pardir,
                          f"sir4_{_a.domain}_report.ipynb")


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.rstrip("\n").splitlines(keepends=True)}


def md(s):
    return {"cell_type": "markdown", "metadata": {},
            "source": s.rstrip("\n").splitlines(keepends=True)}


cells = []

_SCOPE = ("every domain, one table" if _a.domain == "all"
          else f"**{_a.domain}**, all five arms")

cells.append(md(f"""# SIR-4 arms: {_SCOPE}

Reads the run directories already on Drive. Trains nothing, so it is safe to re-run at any
point and it will simply show more rows as runs land.

| arm | semantic | graph | CQIG | fusion |
|---|---|---|---|---|
| 1 | multi-view | off | — | — |
| 2A | multi-view | standard | off | additive |
| 2B | multi-view | standard | off | mixture |
| 3A | multi-view | standard | λ = 0.9 | additive |
| 3B | multi-view | standard | λ = 0.9 | mixture |

**Four trained runs, one per cell of the 2×2.** Arm 1 is the `semantic` column every run
prints, so it costs nothing. Each of 2A, 2B, 3A, 3B is its own training under its own fusion
form.

**Why not score both forms off one checkpoint.** `relu(z(g))` zeroes the graph gradient for
every document below the graph's own mean; the mixture does not. The two forms therefore fit
different graph channels, and applying one form to the other's checkpoint would compare
combiners rather than methods. The `[sweep]`/`[gsweep]` lines are still parsed and shown
further down as a secondary paired read, but the table reports trained arms only."""))

cells.append(code('''from google.colab import drive
drive.mount("/content/drive")'''))

cells.append(code(f'''import os, re, json, glob, math, statistics as st

ROOT     = "/content/drive/MyDrive/cargo-gfmrag"
RUN_SET  = "{_a.run_set}"          # clean namespace; "" = the old flat layout
OUT      = f"{{ROOT}}/outputs" + (f"/{{RUN_SET}}" if RUN_SET else "")
REPORTS  = f"{{ROOT}}/reports"          # this notebook's own output lands here
os.makedirs(REPORTS, exist_ok=True)

EPOCHS, BATCH = {_a.epochs}, {_a.batch}
DOMAINS = {DOMAINS!r}          # one domain per notebook; --domain all builds the joint view

# FOUR TRAINED RUNS PER DOMAIN. Arm 1 is not one of them: it is 5d's standalone scorer,
# read from its own file below. Anything else on Drive (cqiglink09, cqigctr*, cqigmuz*)
# is an ABLATION and is listed separately further down so it stays visible without
# contaminating the headline table.
RUNS = {{
    "2A": f"fusion_qwenmlp_epoch{{EPOCHS}}_b{{BATCH}}",
    "2B": f"fusion_qwenmlp_mix_epoch{{EPOCHS}}_b{{BATCH}}",
    "3A": f"fusion_qwenmlp_cqiglam09_epoch{{EPOCHS}}_b{{BATCH}}",
    "3B": f"fusion_qwenmlp_cqiglam09_mix_epoch{{EPOCHS}}_b{{BATCH}}",
}}
# FOUR TRAINED RUNS, one per cell of the 2x2. The fusion form is NOT scored off a shared
# checkpoint: relu(z(g)) zeroes the graph gradient below the graph's own mean and the
# mixture does not, so the two forms fit different graph channels and only a trained
# comparison measures the method rather than the combiner.
ARMS = [("1",  None, "semantic5d", "multi-view only, NO graph"),
        ("2A", "2A", "fused",      "graph, no gate, additive"),
        ("2B", "2B", "fused",      "graph, no gate, mixture"),
        ("3A", "3A", "fused",      "graph, lam=0.9, additive"),
        ("3B", "3B", "fused",      "graph, lam=0.9, mixture")]

# ARM 1 COMES FROM 5d, NOT FROM 2A. Arm 1 is "graph reasoning: off", and 5d fits and
# scores the multi-view scorer with no graph in the objective at all. A fusion run's
# `semantic` column is a different quantity: that scorer keeps training under the FUSED
# ranking loss, so it has already been shaped by the graph. Using it as the baseline
# would subtract part of the graph's contribution from every delta above it, i.e. bias
# the ladder against the thing the ladder exists to measure.
LOSSD = "fixedloss"

def sem5d_path(domain):
    # section 5d copies its whole results dir to Drive at the end of the cell
    return (f"{{OUT}}/sir4_{{domain}}/semantic/scores_semantic_mlp_{{LOSSD}}.json")

def arm1_value(domain):
    p = sem5d_path(domain)
    if not os.path.isfile(p):
        return None
    return json.load(open(p)).get("all", {{}}).get("ndcg@5")

def run_dir(domain, key):
    return f"{{OUT}}/sir4_{{domain}}/sir4_{{domain}}_{{RUNS[key]}}"
print("root exists:", os.path.isdir(OUT))'''))

cells.append(md("""## 1. Inventory first

Before any number, what is actually on Drive. The fragmentation is the finding here: a
missing cell is not a bad result, it is an unrun experiment, and the two must never be
read the same way."""))

cells.append(code('''def _stat(p):
    """(exists, n_epoch_evals, has_stdout, has_per_query)."""
    if not os.path.isdir(p):
        return (False, 0, False, False)
    log = os.path.join(p, "sft_training.log")
    n = 0
    if os.path.isfile(log):
        # per-epoch eval lines; hydra's file logger captures these
        n = sum(1 for L in open(log, errors="ignore") if "/document_ndcg@5:" in L)
    # [diag] / [sweep] / [gsweep] / [form] go to STDOUT, which the file logger does not
    # capture. They survive only if the training cell teed stdout to console.log.
    con = any(os.path.isfile(os.path.join(p, f)) for f in ("console.log", "stdout.log"))
    pq = os.path.isfile(os.path.join(p, "per_query_ndcg5.json"))
    return (True, n, con, pq)

print(f"{'domain':9} {'run':12} {'dir':6} {'evals':6} {'stdout':7} {'per-query':9}")
print("-" * 56)
for d in DOMAINS:
    for k in RUNS:
        e, n, c, q = _stat(run_dir(d, k))
        print(f"{d:9} {k:12} {'yes' if e else 'MISSING':6} {n:<6} "
              f"{'yes' if c else 'no':7} {'yes' if q else 'no':9}")
print()
print("stdout=no  -> the fusion-form comparison ([sweep]/[gsweep]/[form]) was not captured;")
print("              the per-epoch table still works, the A/B columns do not.")
print("per-query=no -> pre-dates the dump, so no bootstrap for that run.")'''))

cells.append(md("""## 2. Parse

Two sources, deliberately kept separate because they have different reliability. The hydra
log is always there and holds the per-epoch aggregate. The stdout capture holds the
component split and both fusion curves, and only exists if the training cell teed it."""))

cells.append(code(r'''_RE_EP   = re.compile(r"/document_ndcg@5:\s*([0-9.]+)")
_RE_BEST = re.compile(r"Current best document_ndcg@5:\s*([0-9.]+)\s*at epoch\s*(\d+)")
_RE_DIAG = re.compile(r"\[diag\] nDCG@5\s+\w+\s+([0-9.]+)\s*\|\s*\w+\s+([0-9.]+)"
                      r"\s*\|\s*\w+\s+([0-9.]+)\s+gamma mean\s+([0-9.]+)")
_RE_SWB  = re.compile(r"\[sweep\] best a=([0-9.e-]+) -> ([0-9.]+)\s+vs semantic ([0-9.]+)")
_RE_GSWB = re.compile(r"\[gsweep\] best gamma=([0-9.e-]+) -> ([0-9.]+)\s+vs semantic ([0-9.]+)")
_RE_CURV = re.compile(r"\[g?sweep\] fused nDCG@5 vs constant (a|gamma)[^:]*:\s+(.*)")

def parse(p):
    """Everything one run directory can tell us. Missing pieces come back as None/[]."""
    r = {"epochs": [], "best": None, "best_ep": None, "diag": [],
         "sweep_a": None, "sweep_g": None, "curve_a": None, "curve_g": None}
    log = os.path.join(p, "sft_training.log")
    if os.path.isfile(log):
        for L in open(log, errors="ignore"):
            m = _RE_EP.search(L)
            if m and "Current best" not in L:
                r["epochs"].append(float(m.group(1)))
            m = _RE_BEST.search(L)
            if m:
                r["best"], r["best_ep"] = float(m.group(1)), int(m.group(2))
    for f in ("console.log", "stdout.log"):
        q = os.path.join(p, f)
        if not os.path.isfile(q):
            continue
        for L in open(q, errors="ignore"):
            m = _RE_DIAG.search(L)
            if m:
                r["diag"].append(tuple(float(x) for x in m.groups()))
            m = _RE_SWB.search(L)
            if m:
                r["sweep_a"] = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
            m = _RE_GSWB.search(L)
            if m:
                r["sweep_g"] = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
            m = _RE_CURV.search(L)
            if m:
                pts = re.findall(r"[ag]=([0-9.e-]+)\s+([0-9.]+)", m.group(2))
                r["curve_" + ("a" if m.group(1) == "a" else "g")] = \
                    [(float(x), float(y)) for x, y in pts]
        break
    return r

CACHE = {(d, k): parse(run_dir(d, k)) for d in DOMAINS for k in RUNS}
print("parsed", sum(1 for v in CACHE.values() if v["epochs"]), "runs with epoch data")'''))

cells.append(md("""## 3. The table

One row per arm, one column per domain. `--` means the run is absent; `n/a` means the run
exists but that particular number was not captured (almost always: stdout was not teed, so
the fusion curves are gone even though the aggregate survived)."""))

cells.append(code('''def arm_value(domain, key, kind):
    """nDCG@5 for one arm. 'semantic5d' is the standalone scorer; 'fused' is a run."""
    if kind == "semantic5d":
        return arm1_value(domain)         # 5d, no graph anywhere in its objective
    r = CACHE[(domain, key)]
    if not r["epochs"]:
        return None                       # run absent
    if kind == "semantic":
        # kept for ablations; NOT what arm 1 uses -- see the note beside ARMS
        return r["diag"][-1][0] if r["diag"] else None
    if kind == "fused":
        # the run's own best checkpoint, i.e. the model as TRAINED under its fusion form.
        # sweep_a / sweep_g are still parsed and printed below as a secondary paired read,
        # but they are not what this table reports.
        return r["best"]

def fmt(v, exists):
    return f"{v:.4f}" if v is not None else ("n/a" if exists else "--")

hdr = f"{'arm':4} {'description':28}" + "".join(f"{d:>11}" for d in DOMAINS)
print(hdr); print("-" * len(hdr))
TABLE = {}
for arm, key, kind, desc in ARMS:
    row = []
    for d in DOMAINS:
        # arm 1 has key None: it is not a run, so "exists" means 5d's file is on Drive
        ex = os.path.isfile(sem5d_path(d)) if key is None else bool(CACHE[(d, key)]["epochs"])
        v = arm_value(d, key, kind)
        TABLE[(arm, d)] = v
        row.append(fmt(v, ex))
    print(f"{arm:4} {desc:28}" + "".join(f"{c:>11}" for c in row))

print()
print("deltas over arm 1 (multi-view alone), same domain:")
for arm, key, kind, desc in ARMS[1:]:
    cs = []
    for d in DOMAINS:
        a1, av = TABLE.get(("1", d)), TABLE.get((arm, d))
        cs.append(f"{av - a1:+.4f}" if (a1 is not None and av is not None) else "--")
    print(f"{arm:4} {desc:28}" + "".join(f"{c:>11}" for c in cs))'''))

cells.append(md("""## 4. Trajectories

Per-epoch test nDCG@5. The point of plotting these rather than quoting the best epoch is
that **best-epoch selection hides instability**: the mixture arm peaked at epoch 6 and
declined for three epochs afterwards while its mixing weight kept ramping, which a single
number cannot show. A flat-then-noisy trajectory and a rising one deserve different
amounts of trust in the same final score."""))

cells.append(code('''import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, len(DOMAINS), figsize=(4.2 * len(DOMAINS), 3.4), sharey=True)
axes = axes if len(DOMAINS) > 1 else [axes]
for ax, d in zip(axes, DOMAINS):
    any_line = False
    # KEYED OFF RUNS, not off literals. These used to be ("gate_off", "gate_lam09"),
    # which were the run names before the 2x2 existed; CACHE is built from RUNS, so
    # the cell raised KeyError on the first domain and took the whole report with it.
    # Deriving the loop from RUNS means a renamed or added arm cannot desynchronise.
    STYLE = {"2A": ("--", "tab:blue"), "2B": ("--", "tab:orange"),
             "3A": ("-", "tab:blue"),  "3B": ("-", "tab:orange")}
    for key in RUNS:
        r = CACHE[(d, key)]
        style, colour = STYLE.get(key, ("-", None))
        if r["epochs"]:
            ax.plot(range(1, len(r["epochs"]) + 1), r["epochs"], style, marker="o",
                    ms=3, color=colour, label=f"{key} fused")
            any_line = True
        if r["diag"]:
            ax.plot(range(1, len(r["diag"]) + 1), [x[0] for x in r["diag"]], ":",
                    lw=1, alpha=.7, color=colour, label=f"{key} semantic")
    ax.set_title(d); ax.set_xlabel("epoch"); ax.grid(alpha=.3)
    if not any_line:
        ax.text(.5, .5, "no runs", ha="center", va="center", transform=ax.transAxes)
axes[0].set_ylabel("test nDCG@5")
axes[-1].legend(fontsize=7)
plt.tight_layout()
plt.savefig(f"{REPORTS}/trajectories.png", dpi=150)
plt.show()

# The mixing weight per epoch, which is what explains a declining trajectory: gamma on
# the additive arms, a_q on the mixture arms. Printed for every arm, not just one, since
# the two forms are exactly what this report is comparing.
for d in DOMAINS:
    for key in RUNS:
        r = CACHE[(d, key)]
        if r["diag"]:
            g = [x[3] for x in r["diag"]]
            what = "a_q  " if "mix" in RUNS[key] else "gamma"
            print(f"{d:9} {key:3} {what} per epoch: " + " ".join(f"{x:.3f}" for x in g))'''))

cells.append(md("""## 5. Paired bootstrap

The reason this cell exists: on n≈331 an unpaired comparison of two training runs has a
standard error around 0.027, and every arm measured so far sits inside one such interval of
every other. Ranking four numbers 0.013 apart is not a result. Paired over the same queries
the interval is far tighter, which is the only way any of these differences become readable.

Needs `per_query_ndcg5.json`, which runs built after this change write next to the
checkpoint. Older runs will report as missing rather than being silently skipped."""))

cells.append(code('''import numpy as np

def boot(x, y, n=20000, seed=0):
    """Paired bootstrap of mean(x) - mean(y). Returns (delta, lo, hi, P(delta>0)).

    Resamples the per-query DIFFERENCE, not the two arms independently: the arms are scored
    on the same queries off the same checkpoint, and that pairing is most of the precision.
    Resampling them separately would throw it away and reproduce the ~0.027 unpaired
    interval that makes every arm look identical to every other.
    """
    d = np.asarray(x, float) - np.asarray(y, float)
    assert d.ndim == 1 and d.size, d.shape
    idx = np.random.default_rng(seed).integers(0, d.size, size=(n, d.size))
    reps = d[idx].mean(1)
    return (d.mean(), *np.percentile(reps, [2.5, 97.5]), float((reps > 0).mean()))

# THE PRIMARY COMPARISON IS BETWEEN THE FOUR TRAINED RUNS, each read on its own `fused`
# column. It used to pick the best a-cell and the best gamma-cell WITHIN one run and
# compare those, which is an eval-time combiner sweep on additive-fitted channels, and
# optimistic twice over (best-of-8 selected on the same data it is then tested on).
PAIRS = [("2B", "2A", "FUSION FORM, gate off"),
         ("3B", "3A", "FUSION FORM, gate on"),
         ("3A", "2A", "CQIG, additive held fixed"),
         ("3B", "2B", "CQIG, mixture held fixed")]

def per_query(d, key):
    p = os.path.join(run_dir(d, key), "per_query_ndcg5.json")
    return json.load(open(p)) if os.path.isfile(p) else None

for d in DOMAINS:
    PQ = {k: per_query(d, k) for k in RUNS}
    for k, v in PQ.items():
        if v is None:
            print(f"{d:9} {k:3} per_query_ndcg5.json missing -> excluded from every pair")

    for a, b, why in PAIRS:
        if PQ.get(a) is None or PQ.get(b) is None:
            continue
        xa, xb = PQ[a]["fused"], PQ[b]["fused"]
        # Pairing is only valid if both runs scored the same queries in the same order.
        # Same test graph, same eval loop, so they do -- but a silent length mismatch
        # would make the difference meaningless rather than merely noisy, so check.
        if len(xa) != len(xb):
            print(f"{d:9} {a}-{b}  LENGTH MISMATCH {len(xa)} vs {len(xb)}, not paired")
            continue
        o, lo, hi, pg = boot(xa, xb)
        print(f"{d:9} {a}-{b:3} {why:28} {o:+.4f}  [{lo:+.4f}, {hi:+.4f}]  "
              f"P>0={pg:.3f}  n={len(xa)}")

    # Graph-adds, paired WITHIN each run against its own semantic channel. This is not
    # arm 1: that scorer trained jointly with the graph. Reported as a diagnostic of
    # whether the graph moved anything at all, not as the ladder's 1-to-2A delta.
    print()
    for k in RUNS:
        if PQ.get(k) is None:
            continue
        o, lo, hi, pg = boot(PQ[k]["fused"], PQ[k]["semantic"])
        print(f"{d:9} {k:3} fused - own semantic (diagnostic) {o:+.4f}  "
              f"[{lo:+.4f}, {hi:+.4f}]  P>0={pg:.3f}")
    print()'''))

cells.append(md("""## 6. Ablations, kept out of the headline table

Everything else on Drive. These are single-domain variants of one knob and they do not
belong in the five-arm table, but they should stay visible so a result is not rediscovered
or a failed arm re-run by accident."""))

cells.append(code('''known = {v for v in RUNS.values()}
for d in DOMAINS:
    base = f"{OUT}/sir4_{d}"
    if not os.path.isdir(base):
        continue
    for p in sorted(glob.glob(f"{base}/sir4_{d}_*")):
        suf = os.path.basename(p)[len(f"sir4_{d}_"):]
        if suf in known:
            continue
        r = parse(p)
        b = f"{r['best']:.4f} @ep{r['best_ep']}" if r["best"] is not None else "no result"
        print(f"{d:9} {suf:48} {b}")'''))

cells.append(code('''# one CSV, so the table can be pasted into the paper without re-running anything
import csv
with open(f"{REPORTS}/arms.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["arm", "description", *DOMAINS])
    for arm, key, kind, desc in ARMS:
        w.writerow([arm, desc, *[TABLE.get((arm, d)) for d in DOMAINS]])
print("wrote", f"{REPORTS}/arms.csv")
print("wrote", f"{REPORTS}/trajectories.png")'''))

nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3",
                                                  "display_name": "Python 3"},
                                   "language_info": {"name": "python"},
                                   "colab": {"provenance": []}},
      "nbformat": 4, "nbformat_minor": 0}

out = os.path.abspath(_a.out)
with open(out, "w") as fh:
    json.dump(nb, fh, indent=1)
print(f"wrote {out}  ({len(cells)} cells, {os.path.getsize(out) // 1024} KB)")
