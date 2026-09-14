"""One-command robustness driver: re-run BOTH judge stages under a stronger judge.

The composer is unchanged (gpt-4o-mini compositions are reused as-is); only the
JUDGES are swapped.  Outputs are namespaced by `--tag` (default = judge name) so
the new run sits ALONGSIDE the gpt-4o-mini results instead of clobbering them:

  matched_{arm}__{tag}.jsonl    idea_arena__{tag}.jsonl    summary__{tag}.json

Stages (all resumable -- re-run the same command to continue after an interrupt):
  1. Matched-Score  for none, bm25, moose_chem, oracle   (4 x 500 = 2000 calls)
  2. Idea Arena     round-robin over none, bm25, moose_chem, oracle, gt
                    (C(5,2)=10 pairs x 500 = 5000 records x 2 orders = 10000 calls)
  3. aggregate      prints both ladders + the 5-arm arena leaderboard

The arena includes the `gt` arm (the REAL paper hypothesis), so this single run
also delivers the composer-vs-judge diagnostic under the stronger judge.  Build it
first if you haven't:  python -m analysis.downstream.build_gt_arm

Needs OPENAI_API_KEY.  On gpt-4o expect ~12000 calls total, roughly $50-65, with
the arena stage the long pole (~45-60 min wallclock at --workers 20).

Run:
  cd TOMATO-Star
  python -m analysis.downstream.run_robustness --judge gpt-4o --workers 20
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

MATCHED_ARMS = ["none", "bm25", "moose_chem", "oracle"]   # gt is reference, not scored
ARENA_ARMS = ["none", "bm25", "moose_chem", "oracle", "gt"]


def _run(cmd: list[str]) -> None:
    print("\n" + "=" * 72)
    print(">>> " + " ".join(cmd))
    print("=" * 72, flush=True)
    t0 = time.time()
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"\nStage FAILED ({r.returncode}): {' '.join(cmd)}\n"
                 f"Fix the cause and re-run the SAME driver command -- every stage "
                 f"is resumable, so completed work is not repeated.")
    print(f"[stage done in {time.time() - t0:.0f}s]", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="gpt-4o",
                    help="judge model for BOTH stages (default gpt-4o)")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--tag", default=None,
                    help="output namespace; defaults to the judge name")
    ap.add_argument("--arena-arms", default=",".join(ARENA_ARMS),
                    help="comma-separated arms for the arena round-robin")
    ap.add_argument("--skip-matched", action="store_true",
                    help="skip the Matched-Score stage")
    ap.add_argument("--skip-arena", action="store_true",
                    help="skip the Idea Arena stage")
    args = ap.parse_args()

    tag = args.tag or args.judge
    py = [sys.executable, "-m"]
    w = str(args.workers)

    print(f"Robustness re-run: judge={args.judge}  tag={tag}  workers={w}")
    print(f"  matched arms: {', '.join(MATCHED_ARMS)}")
    print(f"  arena arms  : {args.arena_arms}")

    t_all = time.time()
    if not args.skip_matched:
        for arm in MATCHED_ARMS:
            _run(py + ["analysis.downstream.score_matched", "--arm", arm,
                       "--model", args.judge, "--tag", tag, "--workers", w])
    if not args.skip_arena:
        _run(py + ["analysis.downstream.idea_arena", "--arms", args.arena_arms,
                   "--judge", args.judge, "--tag", tag, "--workers", w])
    _run(py + ["analysis.downstream.aggregate", "--tag", tag])

    print(f"\nAll stages done in {time.time() - t_all:.0f}s.  "
          f"Compare results/downstream/summary.json (gpt-4o-mini) "
          f"vs results/downstream/summary__{tag}.json (this run).")


if __name__ == "__main__":
    main()
