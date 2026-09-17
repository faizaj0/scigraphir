#!/usr/bin/env python3
"""Run SIR-4 stages 2 to 5 end to end for one (domain, split).

    fulltext   02_fulltext.py        fetch arXiv LaTeX / PMC XML          free
    audit      audit_fulltext.py     refuse to continue on a fatal defect  free
    refs       02b_openalex_refs.py  structured reference lists            free
    decompose  03_new_method.py      hypothesis -> ordered inspirations    PAID
    resolve    04_resolve.py         titles -> real papers with abstracts  free
    export     05_export.py          documents.json / eval.json / sets.json free

THE PILOT FIRST, THEN THE DOMAIN.

    python3 build/run_pipeline.py --domain cs --split test --sample 100 --tag pilot
    python3 build/run_pipeline.py --domain cs --split test --headroom 1.4

The pilot is not a different pipeline. It is the same six stages over a
deterministic subset, writing to tagged paths so nothing it produces can be
mistaken for, or overwrite, the real run. Everything it reveals about prompts,
resolution rates and export shape applies unchanged at scale.

WHAT IS SHARED AND WHAT IS NOT.

Stage 2 output is SHARED between the pilot and the full run, because fetching is
free, resumable and additive: the 100 papers the pilot fetches are 100 the full
run does not have to. Stages 3 to 5 are tagged, because stage 3 costs money and
its output is what everything downstream is scored against, so a pilot artefact
sitting in the real path is the one mistake that would be expensive to notice
late.

To spend the pilot's decomposition money only once, seed the full run's output
with it before starting. Stage 3 resumes by DOI, so those papers are skipped:

    cp data/03_decomposed/cs_test_pilot.jsonl data/03_decomposed/cs_test.jsonl

RESUMING. Every stage resumes on its own, so rerunning the same command after a
failure or a Ctrl-C continues rather than restarting. To skip stages that
already succeeded, use --from:

    python3 build/run_pipeline.py --domain cs --split test --from resolve

COST. Only `decompose` spends anything. The runner counts the papers that stage
would actually process, after resume, and asks before starting. --yes skips the
question, for an unattended run.

    python3 build/run_pipeline.py --domain cs --split test --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"

STAGES = ["fulltext", "audit", "refs", "decompose", "resolve", "export"]
PAID = {"decompose"}

# MEASURED, not estimated: the cs_train run billed $151.10 for 7,737 papers with
# the uniqueness sweep, at effort=medium and off-peak DeepSeek pricing.
#
# The previous value here was a pre-run projection of $97 for 9,052 papers, which
# quoted 45% under what the same workload actually cost. A gate that under-quotes
# is worse than no gate: it is the number you approve the spend against. Replaced
# with the invoice. The two cs splits agree closely, so this is not one outlier:
#
#     cs test pilot   200 papers    $3.25    $0.0164/paper
#     cs test       1,118 papers   $20.46    $0.0183/paper
#     cs train      7,737 papers  $151.10    $0.0195/paper
#
# Still an estimate for any OTHER domain, and --cost-per-paper overrides it. Peak
# hours (09:00-12:00 and 14:00-18:00 Beijing) are charged at 2x, which this does
# NOT try to detect. Stage 4's OpenAlex spend is separate and is roughly
# 3.45 titles/paper at $0.0001, so about 1.8% on top.
COST_PER_PAPER = 151.10 / 7737


def _stage3_model() -> str:
    """Whatever Stage 3 will actually call, resolved the same way it resolves it."""
    env = os.environ.get("NEW_MODEL")
    if env:
        return f"{env}  (from NEW_MODEL)"
    try:
        txt = (BUILD / "03_decompose.py").read_text()
        m = re.search(r'^MODEL_DECOMPOSE\s*=\s*"([^"]+)"', txt, re.M)
        return m.group(1) if m else "unknown"
    except OSError:
        return "unknown"


def _count_lines(p: Path) -> int:
    try:
        return sum(1 for l in open(p) if l.strip())
    except OSError:
        return 0


def say(msg: str = "") -> None:
    print(msg, flush=True)


def rule(title: str = "") -> None:
    say(f"\n{'=' * 74}")
    if title:
        say(title)
        say("=" * 74)


class Failed(Exception):
    """A stage exited non-zero. Carries what to run to pick up from there."""

    def __init__(self, stage: str, code: int, log: Path | None):
        super().__init__(stage)
        self.stage, self.code, self.log = stage, code, log


# ============================================================== the sampled subset
def make_subset(src: Path, dst: Path, n: int, seed: int) -> int:
    """A deterministic n-paper view of the Stage 2 output.

    Stage 2's own --sample draws from Stage 1, so it only bounds what gets
    FETCHED. Once the shared fulltext file has more papers in it than the pilot
    wants, from an earlier or a later run, that is no longer the same thing as
    bounding what gets DECOMPOSED, and decomposition is the part that costs
    money. Sampling again here, off the file stage 3 will actually read, makes
    the pilot's size exact and its membership reproducible from the seed.

    Sorted by DOI before sampling so the subset does not depend on the order
    Stage 2's workers happened to finish in.

    SHUFFLE AND TAKE, not random.sample, because the samples must NEST: with the
    same seed, the 200-paper pilot has to contain the 100-paper one. Escalating
    100 -> 200 -> full then costs only the difference, since Stage 3 resumes by
    DOI and skips what is already decomposed. random.sample gives no such
    guarantee, so every escalation would pay for the whole sample again.
    """
    rows = [l for l in open(src) if l.strip()]
    if len(rows) <= n:
        dst.write_text("".join(rows))
        return len(rows)
    rows.sort(key=lambda l: json.loads(l).get("doi") or "")
    random.Random(seed).shuffle(rows)
    dst.write_text("".join(rows[:n]))
    return n


# ==================================================================== the commands
def build_plan(a, paths: dict) -> list[tuple[str, list[str]]]:
    py = sys.executable
    d, s = a.domain, a.split
    plan: list[tuple[str, list[str]]] = []

    ft = [py, str(BUILD / "02_fulltext.py"), "--domain", d, "--split", s,
          "--workers", str(a.fetch_workers)]
    if a.sample:
        ft += ["--sample", str(a.sample), "--seed", str(a.seed)]
    elif a.headroom:
        ft += ["--headroom", str(a.headroom)]
    plan.append(("fulltext", ft))

    plan.append(("audit", [py, str(BUILD / "audit_fulltext.py"),
                           "--domain", d, "--split", s]))

    plan.append(("refs", [py, str(BUILD / "02b_openalex_refs.py"),
                          "--domain", d, "--split", s,
                          "--workers", str(a.fetch_workers)]))

    dec = [py, str(BUILD / ("03_decompose.py" if a.arm == "baseline"
                            else "03_new_method.py")),
           "--domain", d, "--split", s,
           "--input", str(paths["stage3_input"]),
           "--output", str(paths["decomposed"]),
           "--workers", str(a.llm_workers)]
    plan.append(("decompose", dec))

    res = [py, str(BUILD / "04_resolve.py"),
           "--input", str(paths["decomposed"]),
           "--output", str(paths["resolved"]),
           "--workers", str(a.resolve_workers)]
    if a.no_arxiv:
        res.append("--no-arxiv")
    if a.s2_first:
        res.append("--s2-first")
    plan.append(("resolve", res))

    exp = [py, str(BUILD / "05_export.py"),
           "--input", str(paths["resolved"]),
           "--out", str(paths["benchmark"]),
           "--split", f"{d}_{s}" + (f"_{a.tag}" if a.tag else "")]
    if a.drop_title_only:
        exp.append("--drop-title-only")
    plan.append(("export", exp))

    return plan


# ==================================================================== preflight
def preflight(a, paths: dict) -> list[str]:
    """Everything that can be checked without spending anything. Fatal issues raise."""
    problems, notes = [], []

    collected = ROOT / "data" / "01_collected" / f"{a.domain}_{a.split}.jsonl"
    if not collected.exists() and "fulltext" in a.stages:
        problems.append(f"no Stage 1 output at {collected}\n"
                        f"      run: python3 build/01_collect.py "
                        f"--domain {a.domain} --split {a.split}")

    if "decompose" in a.stages and not os.environ.get("DEEPSEEK_API_KEY"):
        problems.append("DEEPSEEK_API_KEY is not set, and decompose needs it")
    if "resolve" in a.stages and not (os.environ.get("OPENALEX_MAILTO")
                                      or os.environ.get("OPENALEX_API_KEY")):
        problems.append("neither OPENALEX_MAILTO nor OPENALEX_API_KEY is set. "
                        "The anonymous pool 429s immediately")
    if "refs" in a.stages and not os.environ.get("OPENALEX_API_KEY"):
        notes.append("OPENALEX_API_KEY unset: Stage 2b uses the free pool, which "
                     "is shared by IP and will stop early on a full split")
    if "resolve" in a.stages and not os.environ.get("S2_API_KEY"):
        notes.append("S2_API_KEY unset: the Semantic Scholar /paper/batch endpoint "
                     "is refused anonymously, so abstract backfill falls back to "
                     "one request per paper")

    # A stage that reads something an earlier, SKIPPED stage was to produce.
    # Existence is not the test: `data/03_decomposed/cs_test.jsonl` sat there as a
    # 0-byte placeholder, passed an exists() check, and let --from resolve run the
    # whole tail of the pipeline over nothing before failing at export.
    def has_rows(p: Path) -> bool:
        return p.exists() and any(l.strip() for l in open(p))

    if "decompose" not in a.stages and "resolve" in a.stages \
            and not has_rows(paths["decomposed"]):
        problems.append(f"--from {a.stages[0]} skips decompose, but "
                        f"{paths['decomposed']} is missing or empty")
    if "resolve" not in a.stages and "export" in a.stages \
            and not has_rows(paths["resolved"]):
        problems.append(f"export needs {paths['resolved']}, "
                        f"which is missing or empty")
    # Checked against the fulltext file, not stage3_input: on a pilot the subset
    # is cut from it later, immediately before decompose, so it does not exist
    # yet on the first run and its absence proves nothing.
    if "fulltext" not in a.stages and "decompose" in a.stages \
            and not has_rows(paths["fulltext"]):
        problems.append(f"decompose needs {paths['fulltext']}, "
                        f"which is missing or empty")

    # The matcher tests. Stage 4 writes the benchmark's gold documents, and the
    # last audit found six wrong ones shipped by a scoring rule that looked fine;
    # 40 seconds here is cheaper than finding out after the decomposition spend.
    if not a.skip_tests:
        for t in ("test_matching.py", "verify_resolve.py"):
            r = subprocess.run([sys.executable, str(BUILD / t)],
                               capture_output=True, text=True)
            if r.returncode != 0:
                tail = (r.stdout or r.stderr).strip().splitlines()[-6:]
                problems.append(f"{t} FAILED:\n      " + "\n      ".join(tail))

    if problems:
        # A dry run is how you inspect the plan before committing to it, often
        # from a shell that has not exported the keys yet. Reporting the problems
        # and still showing the plan is more useful than refusing to do either.
        rule("PREFLIGHT " + ("WARNINGS (dry run)" if a.dry_run else "FAILED"))
        for p in problems:
            say(f"  !! {p}")
        if not a.dry_run:
            raise SystemExit(2)
        say("\n  These are fatal in a real run.")
    return notes


def count_todo(a, paths: dict) -> tuple[int, int]:
    """(papers stage 3 would process, papers already done). Drives the cost gate."""
    src = paths["stage3_input"]
    if not src.exists():
        return (a.sample or 0, 0)
    total = sum(1 for l in open(src) if l.strip())
    done = 0
    if paths["decomposed"].exists():
        done = len({json.loads(l).get("doi")
                    for l in open(paths["decomposed"]) if l.strip()})
    return max(total - done, 0), done


# ========================================================================== run
def run_stage(name: str, cmd: list[str], logdir: Path, dry: bool) -> float:
    rule(f"{name.upper()}")
    say("  " + " ".join(cmd))
    if dry:
        return 0.0
    log = logdir / f"{name}.log"
    t0 = time.monotonic()
    # Tee, reading BYTES not lines.
    #
    # `for line in proc.stdout` blocks until a newline arrives, and a tqdm bar
    # emits "\r" without one. Piped through a line reader, every bar in the
    # pipeline appeared frozen until its stage finished, which is the opposite of
    # what a progress bar is for. Reading raw chunks passes the carriage returns
    # straight through to this terminal, where they render normally.
    #
    # PYTHONUNBUFFERED, because a child whose stdout is a pipe buffers by block
    # rather than by line, so even correct output arrives in 8KB bursts.
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with open(log, "w") as fh:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, env=env, bufsize=0)
        fd = proc.stdout.fileno()
        pending = ""
        while True:
            # os.read returns as soon as ANY bytes are available, so a partial
            # line reaches the terminal immediately. read(1) would also work and
            # would cost one syscall per byte over a multi-megabyte run.
            data = os.read(fd, 8192)
            if not data:
                break
            text = data.decode("utf-8", "replace")
            sys.stdout.write(text)
            sys.stdout.flush()
            # The log keeps finished lines only. A bar that redraws 4,000 times
            # would otherwise put 4,000 near-identical lines in the file; keeping
            # the last state before each newline gives one line per real event.
            for ch in text:
                if ch == "\n":
                    fh.write(pending + "\n")
                    pending = ""
                elif ch == "\r":
                    pending = ""
                else:
                    pending += ch
            fh.flush()
        if pending:
            fh.write(pending + "\n")
        code = proc.wait()
    dt = time.monotonic() - t0
    say(f"\n  {name}: {'ok' if code == 0 else f'EXIT {code}'} in {dt / 60:.1f} min"
        f"   log -> {log}")
    if code != 0:
        raise Failed(name, code, log)
    return dt


def run_both(a, ap) -> int:
    """--split both. Test first, then train, sharing one cost decision.

    Implemented by re-running this script per split rather than by looping
    inside main(), so the single-split path stays exactly the code that has been
    run and cannot drift from it. The children are given --yes because the
    question was already asked here, for both splits at once.

    SEQUENTIAL, not parallel: both splits draw on the same DeepSeek account
    concurrency, so running them together would not be faster, and a failure in
    the first should stop the second rather than burn its budget too.
    """
    splits = ["test", "train"]
    counts, missing = {}, []
    for sp in splits:
        ft = ROOT / "data" / "02_fulltext" / f"{a.domain}_{sp}.jsonl"
        n = _count_lines(ft)
        counts[sp] = n
        if not n:
            missing.append(str(ft))
    if missing:
        rule("PREFLIGHT FAILED")
        for m in missing:
            say(f"  !! no Stage 2 output at {m}")
        return 2

    total = sum(counts.values())
    rule(f"SIR-4  {a.domain}  BOTH SPLITS")
    for sp in splits:
        say(f"  {sp:<8}{counts[sp]:>7,} papers")
    say(f"  {'total':<8}{total:>7,} papers")
    say(f"  model     {_stage3_model()}")
    if "decompose" in a.stages:
        eff = min(a.llm_workers, max(counts.values())) or 1
        mins = sum(max(273.3 * n / eff, 273.3) for n in counts.values()) / 60
        say(f"  workers   {a.llm_workers}   ~{mins / 60:.1f} h total, "
            f"from measured throughput")
        say(f"  estimated ${total * a.cost_per_paper:.2f}, off-peak "
            f"(2x during 09:00-12:00 and 14:00-18:00 Beijing)")
        say(f"  papers already decomposed are skipped, so the real cost is lower")
        if not a.yes and not a.dry_run:
            if input("\n  proceed with BOTH splits? [y/N] ").strip().lower() \
                    not in ("y", "yes"):
                return 1

    rc = 0
    t0 = time.monotonic()
    for sp in splits:
        rule(f"SPLIT {sp.upper()}  ({counts[sp]:,} papers)")
        cmd = [sys.executable, str(Path(__file__).resolve()),
               "--domain", a.domain, "--split", sp,
               "--from", a.stages[0], "--to", a.stages[-1],
               "--arm", a.arm, "--llm-workers", str(a.llm_workers),
               "--resolve-workers", str(a.resolve_workers),
               "--fetch-workers", str(a.fetch_workers),
               "--cost-per-paper", str(a.cost_per_paper), "--yes"]
        if a.tag:
            cmd += ["--tag", a.tag]
        if a.sample:
            cmd += ["--sample", str(a.sample), "--seed", str(a.seed)]
        if a.headroom:
            cmd += ["--headroom", str(a.headroom)]
        for flag in ("no_arxiv", "drop_title_only", "skip_tests", "dry_run"):
            if getattr(a, flag):
                cmd.append("--" + flag.replace("_", "-"))
        rc = subprocess.call(cmd, cwd=ROOT)
        if rc != 0:
            rule(f"STOPPED IN {sp.upper()}")
            say(f"  exit {rc}. The other split was NOT started.")
            say(f"  Everything resumes, so fix the cause and rerun the same "
                f"command:\n    the finished split will be skipped.")
            return rc
    rule("BOTH SPLITS DONE")
    say(f"  {(time.monotonic() - t0) / 60:.0f} min total")
    for sp in splits:
        say(f"  {a.domain}_{sp} -> data/benchmark/{a.domain}_{sp}"
            + (f"_{a.tag}" if a.tag else ""))
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", required=True,
                    help="cs | physics | biology | maths | matsci")
    ap.add_argument("--split", required=True, choices=["train", "test", "both"],
                    help="'both' runs test then train, in that order, as one "
                         "command: one cost decision, one run directory, two "
                         "sets of outputs. Test first, deliberately, because it "
                         "is the split you score and it is six times smaller, so "
                         "a problem surfaces on the cheap half")
    ap.add_argument("--tag", default=None,
                    help="suffix for stages 3-5, so a pilot cannot overwrite a real "
                         "run. Implied by --sample if not given")
    ap.add_argument("--sample", type=int, metavar="N",
                    help="pilot on a deterministic N-paper subset")
    ap.add_argument("--seed", type=int, default=42, help="subset seed (default 42)")
    ap.add_argument("--headroom", type=float, default=None, metavar="X",
                    help="Stage 2 fetch target as a multiple of the benchmark "
                         "target, e.g. 1.4. Ignored when --sample is given")
    ap.add_argument("--arm", choices=["new", "baseline"], default="new",
                    help="new = 03_new_method.py, the ordered decomposition with the "
                         "uniqueness sweep. baseline = 03_decompose.py")
    ap.add_argument("--from", dest="start", choices=STAGES, default=STAGES[0])
    ap.add_argument("--to", dest="stop", choices=STAGES, default=STAGES[-1])
    ap.add_argument("--only", choices=STAGES, help="run exactly one stage")
    ap.add_argument("--fetch-workers", type=int, default=12)
    ap.add_argument("--llm-workers", type=int, default=128,
                    help="concurrent PAPERS in Stage 3 (default 128). DeepSeek "
                         "limits by concurrency, not rate: flash allows 2,500 "
                         "simultaneous. Each paper is ~12 sequential calls at "
                         "~22s, so this is the only lever on wall time. Measured "
                         "on 80 papers: 273 call-seconds each. Nothing in the "
                         "call path serialises, so this can go as high as the "
                         "paper count; beyond that it does nothing. The floor is "
                         "one paper's own chain, about 4.6 min. Lower it only if "
                         "429s appear")
    ap.add_argument("--resolve-workers", type=int, default=8)
    ap.add_argument("--s2-first", action="store_true",
                    help="Stage 4 asks Semantic Scholar for the title search "
                         "before OpenAlex. Free instead of $0.001 per title, but "
                         "1 request/second. An overnight setting")
    ap.add_argument("--no-arxiv", action="store_true",
                    help="skip Stage 4's arXiv abstract fallback. One request every "
                         "3s, so minutes on a pilot and hours on a full split")
    ap.add_argument("--drop-title-only", action="store_true",
                    help="Stage 5: exclude golds with no abstract anywhere")
    ap.add_argument("--cost-per-paper", type=float, default=COST_PER_PAPER)
    ap.add_argument("--yes", action="store_true", help="do not ask before spending")
    ap.add_argument("--skip-tests", action="store_true",
                    help="skip the Stage 4 matcher tests in preflight")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and the cost, run nothing")
    a = ap.parse_args()

    if a.only:
        a.start = a.stop = a.only
    i, j = STAGES.index(a.start), STAGES.index(a.stop)
    if i > j:
        ap.error(f"--from {a.start} comes after --to {a.stop}")
    a.stages = STAGES[i:j + 1]
    if a.sample and not a.tag:
        a.tag = f"pilot{a.sample}"

    # After a.stages exists, which run_both reads. Placing this before that line
    # crashed on the first --split both invocation.
    if a.split == "both":
        raise SystemExit(run_both(a, ap))

    sfx = f"_{a.tag}" if a.tag else ""
    stem = f"{a.domain}_{a.split}"
    paths = {
        "fulltext":  ROOT / "data" / "02_fulltext" / f"{stem}.jsonl",
        "subset":    ROOT / "data" / "02_fulltext" / f"{stem}__{a.tag}.jsonl",
        "decomposed": ROOT / "data" / "03_decomposed" / f"{stem}{sfx}.jsonl",
        "resolved":  ROOT / "data" / "04_resolved" / f"{stem}{sfx}.jsonl",
        "benchmark": ROOT / "data" / "benchmark" / f"{stem}{sfx}",
    }
    paths["stage3_input"] = paths["subset"] if a.sample else paths["fulltext"]
    run_name = f"{stem}{sfx}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    logdir = ROOT / "data" / "_runs" / run_name
    for p in ("decomposed", "resolved"):
        paths[p].parent.mkdir(parents=True, exist_ok=True)
    if not a.dry_run:
        logdir.mkdir(parents=True, exist_ok=True)

    rule(f"SIR-4  {a.domain}/{a.split}"
         + (f"  [{a.tag}]" if a.tag else "  [FULL]"))
    say(f"  stages      {' -> '.join(a.stages)}")
    say(f"  arm         {a.arm}"
        + ("   (ordered decomposition + uniqueness sweep)" if a.arm == "new" else ""))
    # Printed, not assumed. Stage 3 takes its model from a constant that an env
    # var can override, and the only thing worse than paying pro prices is
    # paying them without noticing.
    say(f"  model       {_stage3_model()}   (decompose and gates)")
    if a.sample:
        say(f"  pilot       {a.sample} papers, seed {a.seed}")
    if "decompose" in a.stages:
        # Measured, not guessed: 21,862 call-seconds over 80 papers in
        # data/_snap/run_B.log, same arm, same model.
        n = a.sample or _count_lines(paths["fulltext"])
        # 273.3 call-seconds per paper, measured over 80 papers in
        # data/_snap/run_B.log. The floor is ONE paper's chain: a paper is ~12
        # SEQUENTIAL calls, so no amount of concurrency finishes it faster than
        # about 4.6 minutes, and workers above the paper count do nothing at all.
        eff = min(a.llm_workers, n) or 1
        mins = max(273.3 * n / eff, 273.3) / 60
        note = ""
        if a.llm_workers > n:
            note = f"  ({a.llm_workers} requested, only {n} papers to run)"
        elif eff >= n:
            note = "  (every paper at once; this is the floor)"
        say(f"  workers     {eff} papers at once{note}")
        say(f"              ~{mins:.0f} min for {n} papers, from measured "
            f"throughput (273s/paper of call time)")
    say(f"  fulltext    {paths['fulltext']}      (shared, never tagged)")
    say(f"  decomposed  {paths['decomposed']}")
    say(f"  resolved    {paths['resolved']}")
    say(f"  benchmark   {paths['benchmark']}/")
    say(f"  logs        {logdir}/")

    notes = preflight(a, paths)
    for n in notes:
        say(f"\n  note: {n}")

    # The subset is cut between fulltext and decompose, so it sees whatever
    # stage 2 has just fetched rather than whatever was there before.
    def cut_subset() -> None:
        if not a.sample or a.dry_run:
            return
        if not paths["fulltext"].exists():
            raise SystemExit(f"no Stage 2 output at {paths['fulltext']}")
        n = make_subset(paths["fulltext"], paths["subset"], a.sample, a.seed)
        say(f"\n  pilot subset: {n} papers -> {paths['subset'].name}")

    plan = build_plan(a, paths)
    plan = [(n, c) for n, c in plan if n in a.stages]

    # ---------------------------------------------------------------- cost gate
    if "decompose" in a.stages:
        if a.sample and STAGES.index(a.start) <= STAGES.index("fulltext"):
            todo, done = a.sample, 0        # subset not cut yet
        else:
            todo, done = count_todo(a, paths)
        est = todo * a.cost_per_paper
        rule("COST")
        say(f"  decompose is the only stage that spends anything.")
        say(f"  {todo} papers to process"
            + (f", {done} already done and skipped by resume" if done else ""))
        say(f"  estimated ${est:.2f} at ${a.cost_per_paper:.4f}/paper, off-peak.")
        say(f"  DeepSeek charges 2x during 09:00-12:00 and 14:00-18:00 Beijing time.")
        if not a.yes and not a.dry_run:
            if input("\n  proceed? [y/N] ").strip().lower() not in ("y", "yes"):
                raise SystemExit("stopped before spending anything")

    if a.dry_run:
        rule("PLAN (dry run, nothing will run)")
        for n, c in plan:
            say(f"\n  {n}")
            say("    " + " ".join(c))
        if a.sample:
            say(f"\n  plus: cut {a.sample} papers from {paths['fulltext'].name} "
                f"-> {paths['subset'].name}")
        return

    # -------------------------------------------------------------------- run
    timings: dict[str, float] = {}
    t0 = time.monotonic()
    try:
        for name, cmd in plan:
            if name == "decompose":
                cut_subset()
            timings[name] = run_stage(name, cmd, logdir, a.dry_run)
    except Failed as f:
        rule("PIPELINE STOPPED")
        say(f"  {f.stage} exited {f.code}")
        if f.log and f.log.exists():
            say(f"\n  last lines of {f.log}:")
            for line in f.log.read_text().splitlines()[-12:]:
                say("    " + line)
        say(f"\n  every stage resumes, so fix the cause and rerun from there:")
        say(f"    python3 build/run_pipeline.py --domain {a.domain} "
            f"--split {a.split}" + (f" --tag {a.tag}" if a.tag else "")
            + (f" --sample {a.sample}" if a.sample else "")
            + f" --from {f.stage}")
        raise SystemExit(1)
    except KeyboardInterrupt:
        rule("INTERRUPTED")
        say("  Stage output is written as it goes and every stage resumes by id,")
        say("  so nothing already paid for is lost. Rerun the same command.")
        raise SystemExit(130)

    # ------------------------------------------------------------------ report
    total = time.monotonic() - t0
    rule("DONE")
    for name, dt in timings.items():
        say(f"  {name:<12}{dt / 60:>8.1f} min")
    say(f"  {'total':<12}{total / 60:>8.1f} min")

    manifest = {
        "run": run_name, "domain": a.domain, "split": a.split, "tag": a.tag,
        "arm": a.arm, "sample": a.sample, "seed": a.seed,
        "stages": a.stages, "minutes": {k: round(v / 60, 2) for k, v in timings.items()},
        "paths": {k: str(v) for k, v in paths.items()},
        "finished": f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}",
    }
    bench_manifest = paths["benchmark"] / "manifest.json"
    if bench_manifest.exists():
        manifest["benchmark"] = json.loads(bench_manifest.read_text())
    (logdir / "run.json").write_text(json.dumps(manifest, indent=1))
    say(f"\n  manifest -> {logdir / 'run.json'}")
    if paths["benchmark"].exists():
        say(f"  benchmark -> {paths['benchmark']}/")
        for f in ("raw/documents.json", "eval.json", "sets.json"):
            p = paths["benchmark"] / f
            if p.exists():
                say(f"      {f:<24}{p.stat().st_size:>12,} bytes")


if __name__ == "__main__":
    main()
