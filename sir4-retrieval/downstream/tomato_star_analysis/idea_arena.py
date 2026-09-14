"""Idea Arena: round-robin pairwise comparison of composed hypotheses across arms,
using the OFFICIAL Chain-of-Ideas (CoI) judge.

Source: DAMO-NLP-SG/CoI-Agent (paper arXiv:2410.13185). We import their exact
judge prompt (`get_judge_idea_all_prompt`) and tag parser (`extract`) directly from
the cloned repo so the protocol is theirs, not a reimplementation:

  - Criteria (5): Novelty, Significance, Feasibility, Clarity, Effectiveness.
  - Judge outputs <criterion>0|1|2</criterion> where 0 = idea0 better,
    1 = idea1 better, 2 = tie.
  - Position bias is cancelled by judging BOTH orders of every pair (their
    round-robin uses an `i != j` double loop).
  - Aggregation mirrors their `change_winner_to_score`: +1 to the winner, +0.5 to
    each on a tie / unparseable, summed over both orders and all criteria.

Default arms entering the round-robin: none, bm25, moose_chem, oracle.  Every
unordered pair is judged in both orders, so with >2 arms the downstream Elo
leaderboard (see aggregate.py) is a genuine multi-player rating, not the
degenerate 2-player case.  Override the field with --arms.

Reference-free -> no GT hypothesis is shown. CoI's prompt takes a "topic"; we pass
the research_question (the concise problem statement) as that topic.

Needs OPENAI_API_KEY.  Resumable (per query x arm-pair).

Run:
  cd TOMATO-Star
  python -m analysis.downstream.idea_arena --judge gpt-4o-mini --workers 20

Output: results/downstream/idea_arena.jsonl  (one row per query x arm-pair)
  {query_id, stratum, arm0, arm1,
   orderA:{criterion->'0'|'1'|'2'},   # idea0 = arm0, idea1 = arm1
   orderB:{criterion->'0'|'1'|'2'},   # idea0 = arm1, idea1 = arm0
   failed}
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools

from analysis.downstream._common import (
    OUT_DIR,
    INPUTS_PATH,
    PROJ,
    JsonlWriter,
    chat,
    index_by,
    read_jsonl,
    run_concurrent,
)

# CoI's 5 idea criteria, in the order their prompt emits them.
CRITERIA = ["novelty", "significance", "feasibility", "clarity", "effectiveness"]
CRIT_DISPLAY = {"novelty": "Novelty", "significance": "Significance",
                "feasibility": "Feasibility", "clarity": "Clarity",
                "effectiveness": "Effectiveness"}

DEFAULT_ARMS = ["none", "bm25", "moose_chem", "oracle"]


def _load_coi():
    """Load CoI's judge prompt + tag parser directly from the cloned repo by file
    path (avoids importing their heavy `prompts`/`searcher` packages)."""
    coi = PROJ.parent / "CoI-Agent"
    jp = coi / "prompts" / "juder_prompts.py"
    ut = coi / "utils.py"
    if not jp.exists() or not ut.exists():
        raise SystemExit(
            f"CoI-Agent repo not found at {coi}. Clone it first:\n"
            f"  git clone https://github.com/DAMO-NLP-SG/CoI-Agent.git {coi}")

    def _imp(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    jp_mod = _imp("coi_juder_prompts", jp)
    ut_mod = _imp("coi_utils", ut)
    return jp_mod.get_judge_idea_all_prompt, ut_mod.extract


GET_JUDGE_PROMPT, COI_EXTRACT = _load_coi()


def _judge_one(idea0: str, idea1: str, topic: str, model: str,
               temperature: float, max_tokens: int) -> dict:
    """One CoI judge call -> {criterion: '0'|'1'|'2'}.  Uses CoI's own prompt +
    extractor; unparseable choices are recorded as '2' (tie), matching their
    change_winner_to_score fallback."""
    prompt = GET_JUDGE_PROMPT(idea0, idea1, topic)
    resp = chat(prompt, model=model, temperature=temperature, max_tokens=max_tokens)
    out = {}
    for crit in CRITERIA:
        raw = COI_EXTRACT(resp, crit)  # CoI extractor (hard=True default)
        val = (raw or "").strip()
        out[crit] = val if val in ("0", "1", "2") else "2"
    return out


def _resume_keys(path):
    """Set of (query_id, arm0, arm1) already written (round-robin schema)."""
    return {(r["query_id"], r["arm0"], r["arm1"])
            for r in read_jsonl(path) if r.get("arm0") and r.get("arm1")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="gpt-4o-mini")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS),
                    help="comma-separated arms entering the round-robin")
    ap.add_argument("--tag", default="",
                    help="namespace output by judge (e.g. gpt-4o); default "
                         "keeps idea_arena.jsonl")
    args = ap.parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    if len(arms) < 2:
        raise SystemExit("Need >=2 arms for the arena.")

    inputs = index_by(read_jsonl(INPUTS_PATH))
    comps = {}
    for a in arms:
        rows = index_by(read_jsonl(OUT_DIR / f"compositions_{a}.jsonl"))
        if not rows:
            raise SystemExit(
                f"Missing compositions_{a}.jsonl -- run: "
                f"python -m analysis.downstream.compose --arm {a}")
        comps[a] = rows
    suffix = f"__{args.tag}" if args.tag else ""
    out_path = OUT_DIR / f"idea_arena{suffix}.jsonl"

    # Guard against silently appending round-robin rows onto an old 2-arm file.
    existing = read_jsonl(out_path)
    if existing and not any(r.get("arm0") for r in existing):
        raise SystemExit(
            f"{out_path} is in the old 2-arm format. Remove it and re-run:\n"
            f"  rm {out_path}")

    qids = [q for q in inputs if all(q in comps[a] for a in arms)]
    pairs = list(itertools.combinations(arms, 2))      # arm0 precedes arm1 in `arms`
    items = [(q, a0, a1) for q in qids for (a0, a1) in pairs]

    done = _resume_keys(out_path)
    todo = [it for it in items if it not in done]
    print(f"arms={arms}  pairs/query={len(pairs)}  queries={len(qids)}  "
          f"matchups={len(items)}  done={len(done)}  todo={len(todo)}")
    if not todo:
        print("Nothing to do.")
        return

    writer = JsonlWriter(out_path)

    def work(item: tuple) -> dict:
        qid, a0, a1 = item
        topic = inputs[qid]["research_question"]
        h0 = (comps[a0][qid].get("delta_hypothesis") or "").strip()
        h1 = (comps[a1][qid].get("delta_hypothesis") or "").strip()
        if not h0 or not h1:
            return {"query_id": qid, "stratum": inputs[qid]["stratum"],
                    "arm0": a0, "arm1": a1,
                    "orderA": None, "orderB": None, "failed": True}
        orderA = _judge_one(h0, h1, topic, args.judge, args.temperature, args.max_tokens)
        orderB = _judge_one(h1, h0, topic, args.judge, args.temperature, args.max_tokens)
        return {"query_id": qid, "stratum": inputs[qid]["stratum"],
                "arm0": a0, "arm1": a1,
                "orderA": orderA, "orderB": orderB, "failed": False}

    n = run_concurrent(todo, work, writer, workers=args.workers, desc="arena")
    writer.close()
    print(f"\nWrote {n} matchups -> {out_path}")


if __name__ == "__main__":
    main()
