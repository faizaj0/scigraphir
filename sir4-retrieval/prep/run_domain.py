"""
run_domain.py -- drive one or more SIR-4 domains through the local pipeline.

WHY THIS EXISTS. The alternative is six commands per domain with the domain name
substituted into each, run four times. That is twenty-four chances to type
`sir4_cs` into a physics run, and the failure mode is silent: every script
happily processes whatever corpus it is pointed at.

THREE PROPERTIES THAT MATTER:

  IDEMPOTENT.  Every step checks its own outputs before running. Re-running the
      whole pipeline after a crash costs nothing and repeats nothing. This is
      what makes `--phase all` safe to just run again.

  PAID WORK IS GATED.  `extract` and `probes` cost money. They refuse to run
      without --spend, and print an estimate first. Everything else is free and
      runs unprompted.

  GRAPH BUILDS ARE SERIAL BY CONSTRUCTION.  Extraction is I/O-bound and two
      domains can overlap happily. The graph build is BGE encoding plus
      all-pairs matmuls, which is what drove a 17 GB machine into swap
      exhaustion earlier in this project. This script builds graphs one at a
      time and refuses to start if another build is already running.

PHASE-MAJOR, NOT DOMAIN-MAJOR. Prefer running one phase across all domains
before starting the next:

    python3 prep/run_domain.py --domain all --phase stage        # free
    python3 prep/run_domain.py --domain all --phase extract --spend
    python3 prep/run_domain.py --domain all --phase graph        # free, serial
    python3 prep/run_domain.py --domain all --phase ship         # audit+bundle+nb

That way all the spending happens in one reviewable block, every corpus is
verified before any money is committed, and a data problem in biology surfaces
before physics has had a graph built for it.

Usage
-----
    python3 prep/run_domain.py --status
    python3 prep/run_domain.py --domain physics --phase all --dry-run
    python3 prep/run_domain.py --domain physics,matsci --phase extract --spend
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CARGO = os.path.dirname(ROOT)
KG, V16 = f"{CARGO}/kg-construction", f"{CARGO}/kg-construction-v16"
sys.path.insert(0, HERE)
sys.path.insert(0, CARGO)
from stage_sir4 import BENCH, DOMAINS                       # noqa: E402

SPLITS = ("train", "test")
# CS cost $9.55 for 24,384 docs + 6,497 queries. Linear in items is crude but
# honest, and the point is a sanity check before spending, not accounting.
COST_PER_ITEM = 9.55 / (24_384 + 6_497)
TAU_CANON = 0.95


def ds(dom: str) -> str:
    return f"sir4_{dom}"


def corpus_raw(dom: str, split: str) -> str:
    return f"{KG}/data/{ds(dom)}_{split}/raw"


def counts(dom: str, split: str) -> tuple[int, int]:
    """(documents, queries) from the STAGED corpus, or the export if unstaged."""
    r = corpus_raw(dom, split)
    if os.path.exists(f"{r}/documents.json"):
        return (len(json.load(open(f"{r}/documents.json"))),
                len(json.load(open(f"{r}/{split}.json"))))
    exp = f"{BENCH}/{DOMAINS[dom][0 if split == 'train' else 1]}"
    if os.path.isdir(exp):
        return (len(json.load(open(f"{exp}/raw/documents.json"))),
                len(json.load(open(f"{exp}/eval.json"))))
    return (0, 0)


def nlines(p: str) -> int:
    if not os.path.exists(p):
        return 0
    with open(p) as f:
        return sum(1 for _ in f)


def state(dom: str) -> dict:
    """What is already done. Every check reads real outputs, never a marker file:
    a marker can outlive the thing it claims."""
    st = {}
    st["staged"] = all(os.path.exists(f"{corpus_raw(dom, s)}/documents.json") for s in SPLITS)
    fr, pr = {}, {}
    for s in SPLITS:
        d, q = counts(dom, s)
        # 98%: a handful of items legitimately never extract (title-only records,
        # or ones that exhausted their retries and were deliberately not cached).
        fr[s] = (nlines(f"{V16}/cache/{ds(dom)}/frames_doc_{s}.jsonl") >= 0.98 * d > 0 and
                 nlines(f"{V16}/cache/{ds(dom)}/frames_query_{s}.jsonl") >= 0.98 * q > 0)
        pr[s] = nlines(f"{KG}/construct_v2/cache/{ds(dom)}/probes_{s}.jsonl") >= 0.98 * q > 0
    st["frames"] = all(fr.values())
    st["probes"] = all(pr.values())
    st["graphs"] = all(os.path.exists(
        f"{KG}/data/{ds(dom)}_{s}_v16sc/processed/stage1/nodes.csv") for s in SPLITS)
    st["bundle"] = os.path.exists(f"{ROOT}/{ds(dom)}_bundle.zip")
    st["notebook"] = os.path.exists(f"{ROOT}/colab_train_{ds(dom)}_fusion.ipynb")
    return st


def run(cmd: str, cwd: str, dry: bool) -> None:
    print(f"    $ {cmd}")
    if dry:
        return
    t0 = time.time()
    p = subprocess.run(cmd, cwd=cwd, shell=True)
    print(f"      [{time.time() - t0:.0f}s, exit {p.returncode}]")
    if p.returncode != 0:
        raise SystemExit(f"FAILED: {cmd}")


GRAPH_LOCK = "/tmp/.sir4_graph_build.lock"


def graph_lock():
    """One graph build at a time. Two concurrent builds is how this project
    exhausted swap and lost an hour; a lock is cheaper than remembering."""
    if os.path.exists(GRAPH_LOCK):
        pid = open(GRAPH_LOCK).read().strip()
        alive = subprocess.run(f"ps -p {pid}", shell=True,
                               capture_output=True).returncode == 0
        if alive:
            raise SystemExit(
                f"another graph build is running (pid {pid}). Graph builds are "
                f"memory-heavy and must not overlap. Wait, or remove {GRAPH_LOCK} "
                f"if that pid is stale.")
        os.remove(GRAPH_LOCK)
    open(GRAPH_LOCK, "w").write(str(os.getpid()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="all",
                    help="comma separated, or 'all'")
    ap.add_argument("--phase", default="all",
                    choices=["stage", "extract", "probes", "graph", "audit",
                             "bundle", "notebook", "ship", "all"],
                    help="'ship' = audit + bundle + notebook")
    ap.add_argument("--spend", action="store_true",
                    help="required for the paid phases (extract, probes)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=512,
                    help="tier 5 allows 30,000 RPM; 512 concurrent at ~1.5s/call "
                         "is ~20,000 RPM, which leaves headroom for backoff")
    ap.add_argument("--tau-canon", type=float, default=TAU_CANON)
    ap.add_argument("--batch", type=int, default=1, help="Colab train batch size")
    ap.add_argument("--force", action="store_true",
                    help="re-run steps that are already complete")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()

    doms = sorted(DOMAINS) if a.domain == "all" else [d.strip() for d in a.domain.split(",")]
    for d in doms:
        if d not in DOMAINS:
            return print(f"unknown domain {d!r}; known: {sorted(DOMAINS)}") or 2

    if a.status:
        keys = ["staged", "frames", "probes", "graphs", "bundle", "notebook"]
        print(f"\n{'domain':9} {'docs':>7} {'queries':>8} | " +
              " ".join(f"{k:>8}" for k in keys) + "   remaining $")
        print("-" * 96)
        tot = 0.0
        for d in sorted(DOMAINS):
            st = state(d)
            dt = sum(counts(d, s)[0] for s in SPLITS)
            qt = sum(counts(d, s)[1] for s in SPLITS)
            due = 0.0 if st["frames"] and st["probes"] else (dt + qt) * COST_PER_ITEM
            tot += due
            print(f"{d:9} {dt:7,} {qt:8,} | " +
                  " ".join(f"{'yes' if st[k] else '-':>8}" for k in keys) +
                  f"   {'' if due == 0 else f'~${due:.2f}'}")
        print(f"\nremaining extraction cost: ~${tot:.2f}")
        return 0

    phases = (["stage", "extract", "probes", "graph", "audit", "bundle", "notebook"]
              if a.phase == "all" else
              ["audit", "bundle", "notebook"] if a.phase == "ship" else [a.phase])

    # Cost gate: one estimate for everything about to be spent, up front.
    if not a.dry_run and any(p in phases for p in ("extract", "probes")):
        todo = [d for d in doms if a.force or not (state(d)["frames"] and state(d)["probes"])]
        est = sum((sum(counts(d, s)[0] for s in SPLITS) +
                   sum(counts(d, s)[1] for s in SPLITS)) * COST_PER_ITEM for d in todo)
        if todo and not a.spend:
            print(f"\nPAID phases requested for {todo}: ~${est:.2f}")
            print("re-run with --spend to authorise, or --dry-run to see the commands")
            return 1
        if todo:
            print(f"\n[spend] {todo}  estimated ~${est:.2f}\n")

    for dom in doms:
        st = state(dom)
        print(f"\n{'='*70}\n  {ds(dom)}   " +
              "  ".join(f"{k}={'y' if v else 'n'}" for k, v in st.items()) +
              f"\n{'='*70}")
        for ph in phases:
            done = {"stage": st["staged"], "extract": st["frames"], "probes": st["probes"],
                    "graph": st["graphs"], "bundle": st["bundle"],
                    "notebook": st["notebook"], "audit": False}[ph]
            if done and not a.force:
                print(f"  [skip] {ph}: already complete")
                continue
            print(f"  [run ] {ph}")

            if ph == "stage":
                run(f"python3 {HERE}/stage_sir4.py --domain {dom}", CARGO, a.dry_run)
            elif ph == "extract":
                for s in SPLITS:
                    for side in ("doc", "query"):
                        run(f"python3 extract_frames.py --dataset {ds(dom)} --side {side} "
                            f"--split {s} --workers {a.workers}", V16, a.dry_run)
            elif ph == "probes":
                for s in SPLITS:
                    run(f"python3 construct_v2/gen_probes.py --dataset {ds(dom)} "
                        f"--split {s} --workers {a.workers}", KG, a.dry_run)
            elif ph == "graph":
                if not a.dry_run:
                    graph_lock()
                try:
                    for s in SPLITS:
                        run(f"python3 build_greasoner_dataset.py --dataset {ds(dom)} "
                            f"--split {s} --tau_canon {a.tau_canon} --no_entity_seeds "
                            f"--no_probe_seeds", V16, a.dry_run)
                finally:
                    if not a.dry_run and os.path.exists(GRAPH_LOCK):
                        os.remove(GRAPH_LOCK)
            elif ph == "audit":
                for s in SPLITS:
                    run(f"python3 eval/audit_graph.py --dataset {ds(dom)} --split {s} "
                        f"--sample 40", ROOT, a.dry_run)
            elif ph == "bundle":
                run(f"python3 prep/bundle.py --dataset {ds(dom)}", ROOT, a.dry_run)
            elif ph == "notebook":
                run(f"python3 prep/build_notebook.py --domain {dom} --batch {a.batch}",
                    ROOT, a.dry_run)
            st = state(dom)

        if not a.dry_run:
            man = f"{ROOT}/data/{dom}/PIPELINE.json"
            os.makedirs(os.path.dirname(man), exist_ok=True)
            json.dump({"domain": dom, "state": state(dom), "tau_canon": a.tau_canon,
                       "workers": a.workers, "batch": a.batch}, open(man, "w"), indent=1)

    print("\ndone. `--status` for the matrix across all domains.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
