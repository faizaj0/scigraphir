"""Matched-Score for ALL arms in one consistent judge pass, via the OpenAI Batch API.

Why a separate script:
  * one judge configuration for every arm (the live runs mixed judges)
  * Batch API = 50% off; a short-output instruction caps the judge's reasoning,
    which is the expensive side of gpt-4o
  * identical (query, delta-hypothesis) texts across arms are judged ONCE and the
    score is fanned out, so shared compositions cost nothing twice
  * --samples N averages N judge samples per composition (temperature 0 is not
    deterministic; N=3 shrinks judge noise ~40%)

Needs OPENAI_API_KEY.

Steps (from TOMATO-Star):
  python -m analysis.downstream.score_matched_batch build  --tag gpt-4o-batch   # writes the request file + cost estimate, no API
  python -m analysis.downstream.score_matched_batch submit --tag gpt-4o-batch   # uploads, creates the batch, saves its id
  python -m analysis.downstream.score_matched_batch fetch  --tag gpt-4o-batch   # polls; when complete writes matched_{arm}__{tag}.jsonl
  python -m analysis.downstream.aggregate --tag gpt-4o-batch
Fallback if your org has no Batch access:
  python -m analysis.downstream.score_matched_batch live --tag gpt-4o-live      # same prompts, concurrent live calls

Files: results/downstream/batch/{tag}_requests.jsonl, {tag}_manifest.json,
       {tag}_raw_output.jsonl; results/downstream/matched_{arm}__{tag}.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict

from analysis.downstream._common import (
    OUT_DIR, INPUTS_PATH, RERANKER_PROMPT_TEMPLATE, SCORING_RUBRIC,
    JsonlWriter, chat, client, index_by, parse_scores, read_jsonl, run_concurrent,
)
from analysis.downstream.compose import ARM_FIELD

DEFAULT_ARMS = ["none", "random", "qwen3", "reasonir", "moose_chem", "lattice",
                "scigraphir", "oracle"]
DIMS = ("motivation", "mechanism", "methodology")

# Appended to the MOOSE-Star rubric prompt.  The rubric text itself is unchanged.
SHORT_SUFFIX = ("\n\nKeep your reasoning to at most one sentence per dimension, "
                "then give the three score lines exactly in the format above.")

BATCH_DIR = OUT_DIR / "batch"
BATCH_DIR.mkdir(parents=True, exist_ok=True)

# gpt-4o list price per 1M tokens (live); batch = half.  Update if the price page moves.
PRICE_IN, PRICE_OUT = 2.50, 10.00


def _prompt(gt: str, gen: str) -> str:
    return RERANKER_PROMPT_TEMPLATE.format(
        gt_hypothesis=gt, generated_hypothesis=gen,
        scoring_rubric=SCORING_RUBRIC) + SHORT_SUFFIX


def _collect(arms: list[str]):
    """-> units: {uid: {qid, gen, gt, members:[(arm, stratum)]}}, zeros: [(arm,qid,stratum)]"""
    inputs = index_by(read_jsonl(INPUTS_PATH))
    units, zeros = {}, []
    for arm in arms:
        comps = read_jsonl(OUT_DIR / f"compositions_{arm}.jsonl")
        if not comps:
            raise SystemExit(f"No compositions for arm '{arm}'. Run compose --arm {arm} first.")
        field = ARM_FIELD.get(arm)
        for c in comps:
            qid = c["query_id"]
            gen = (c.get("delta_hypothesis") or "").strip()
            if not gen:
                zeros.append((arm, qid, c["stratum"]))
                continue
            rec = inputs[qid]
            # Judge reference.  Default: the query's gt_delta (TOMATO: the step's
            # component).  An arm's input field may carry its own reference
            # (SIR-4, --reference component: the fed GOLD's delta when the top-1
            # is a gold); then the unit id includes the reference hash so the
            # same generated text judged against two references is two units.
            gt = rec["gt_delta"]
            fed = rec.get(field) if field else None
            if isinstance(fed, dict) and fed.get("gt_delta"):
                gt = fed["gt_delta"]
            h = hashlib.sha1(gen.encode()).hexdigest()[:12]
            uid = f"{qid}|{h}"
            if gt != rec["gt_delta"]:
                uid += "|g" + hashlib.sha1(gt.encode()).hexdigest()[:8]
            u = units.setdefault(uid, {"qid": qid, "gen": gen, "gt": gt, "members": []})
            u["members"].append((arm, c["stratum"]))
    return units, zeros


def _write_scores(tag, arms, units, zeros, scores_by_uid):
    """scores_by_uid: uid -> list of {dim: int} (one per successful sample)."""
    for a in arms:                       # start clean (batch results are complete)
        (OUT_DIR / f"matched_{a}__{tag}.jsonl").write_text("")
    writers = {a: JsonlWriter(OUT_DIR / f"matched_{a}__{tag}.jsonl") for a in arms}
    n_fail = 0
    for uid, u in units.items():
        samples = scores_by_uid.get(uid, [])
        if not samples:
            n_fail += len(u["members"])
            for arm, st in u["members"]:
                writers[arm].write({"query_id": u["qid"], "arm": arm, "stratum": st,
                                    "scores": None, "total": None, "failed": True})
            continue
        sc = {d: sum(s[d] for s in samples) / len(samples) for d in DIMS}
        tot = sum(sc.values())
        for arm, st in u["members"]:
            writers[arm].write({"query_id": u["qid"], "arm": arm, "stratum": st,
                                "scores": sc, "total": tot, "failed": False,
                                "n_samples": len(samples)})
    for arm, qid, st in zeros:
        writers[arm].write({"query_id": qid, "arm": arm, "stratum": st,
                            "scores": {d: 0 for d in DIMS}, "total": 0,
                            "failed": False, "n_samples": 0})
    for w in writers.values():
        w.close()
    print(f"wrote matched_{{arm}}__{tag}.jsonl for {len(arms)} arms; "
          f"{n_fail} arm-query rows failed to parse")


def _scored_uids(tag) -> dict:
    """uid -> [scores] already present in this tag's raw output (previous batches)."""
    raw_path = BATCH_DIR / f"{tag}_raw_output.jsonl"
    scores = defaultdict(list)
    if not raw_path.exists():
        return scores
    for line in raw_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        uid = r["custom_id"].rsplit("|s", 1)[0]
        try:
            txt = r["response"]["body"]["choices"][0]["message"]["content"]
        except (KeyError, TypeError, IndexError):
            continue
        sc = parse_scores(txt or "")
        if sc is not None:
            scores[uid].append(sc)
    return scores


def cmd_build(a):
    units, zeros = _collect(a.arms)
    have = _scored_uids(a.tag)
    new_units = {uid: u for uid, u in units.items() if uid not in have}
    if have:
        print(f"{len(have)} compositions already judged under tag '{a.tag}'; "
              f"building requests for the {len(new_units)} new ones only")
    req_path = BATCH_DIR / f"{a.tag}_requests.jsonl"
    n_tok_in = 0
    with req_path.open("w") as f:
        for uid, u in new_units.items():
            p = _prompt(u["gt"], u["gen"])
            n_tok_in += len(p) // 4
            for s in range(a.samples):
                f.write(json.dumps({
                    "custom_id": f"{uid}|s{s}",
                    "method": "POST", "url": "/v1/chat/completions",
                    "body": {"model": a.model, "temperature": 0.0,
                             "max_tokens": a.max_tokens,
                             "messages": [{"role": "user", "content": p}]},
                }) + "\n")
    manifest = {"tag": a.tag, "arms": a.arms, "model": a.model,
                "samples": a.samples, "max_tokens": a.max_tokens,
                "n_units": len(units), "n_zero": len(zeros), "batch_id": None}
    (BATCH_DIR / f"{a.tag}_manifest.json").write_text(json.dumps(manifest, indent=2))
    n_req = len(new_units) * a.samples
    n_pairs = sum(len(u["members"]) for u in units.values())
    est_in = n_tok_in * a.samples
    est_out = n_req * 150
    live = est_in / 1e6 * PRICE_IN + est_out / 1e6 * PRICE_OUT
    print(f"arms {a.arms}")
    print(f"arm-query pairs {n_pairs + len(zeros)}  ->  unique compositions {len(units)}  "
          f"(+{len(zeros)} empty, scored 0 without a call);  to judge now: {len(new_units)}")
    if not new_units:
        print("nothing new to judge: run  write  to (re)generate the matched files")
    print(f"requests {n_req}  est input tokens {est_in:,}  est output {est_out:,}")
    print(f"est cost: live ${live:.2f}   batch ${live / 2:.2f}")
    print(f"wrote {req_path}")


def cmd_submit(a):
    man_path = BATCH_DIR / f"{a.tag}_manifest.json"
    man = json.loads(man_path.read_text())
    req_path = BATCH_DIR / f"{a.tag}_requests.jsonl"
    up = client().files.create(file=req_path.open("rb"), purpose="batch")
    b = client().batches.create(input_file_id=up.id, endpoint="/v1/chat/completions",
                                completion_window="24h",
                                metadata={"tag": a.tag, "purpose": "matched-score"})
    man["batch_id"] = b.id
    man_path.write_text(json.dumps(man, indent=2))
    print(f"submitted batch {b.id}  status={b.status}")


def cmd_fetch(a):
    man_path = BATCH_DIR / f"{a.tag}_manifest.json"
    man = json.loads(man_path.read_text())
    if not man.get("batch_id"):
        raise SystemExit("no batch_id in manifest; run submit first")
    while True:
        b = client().batches.retrieve(man["batch_id"])
        c = b.request_counts
        print(f"status={b.status}  done {c.completed}/{c.total}  failed {c.failed}")
        if b.status in ("completed", "failed", "expired", "cancelled"):
            break
        if not a.wait:
            return
        time.sleep(60)
    if b.status != "completed" or not b.output_file_id:
        raise SystemExit(f"batch ended with status {b.status}; error file {b.error_file_id}")
    raw = client().files.content(b.output_file_id).text
    raw_path = BATCH_DIR / f"{a.tag}_raw_output.jsonl"
    done_ids = {json.loads(l)["custom_id"] for l in raw_path.read_text().splitlines()
                if l.strip()} if raw_path.exists() else set()
    new_lines = [l for l in raw.splitlines()
                 if l.strip() and json.loads(l)["custom_id"] not in done_ids]
    with raw_path.open("a") as f:            # accumulate across batches
        for l in new_lines:
            f.write(l + "\n")
    print(f"appended {len(new_lines)} new judge outputs to {raw_path.name}")
    man["batch_id"] = None                   # this batch is consumed
    man_path.write_text(json.dumps(man, indent=2))
    cmd_write(a, arms=man["arms"])


def cmd_write(a, arms=None):
    """Rebuild matched_{arm}__{tag}.jsonl from the accumulated raw output (no API)."""
    arms = arms or a.arms
    scores = _scored_uids(a.tag)
    units, zeros = _collect(arms)
    missing = [uid for uid in units if uid not in scores]
    print(f"judged {len(scores)} compositions; {len(missing)} of {len(units)} "
          f"needed by arms {arms} are unjudged" + ("  (run build/submit/fetch)" if missing else ""))
    _write_scores(a.tag, arms, units, zeros, scores)


def cmd_live(a):
    """Same prompts and short output, concurrent live calls (no batch discount)."""
    units, zeros = _collect(a.arms)
    raw_path = BATCH_DIR / f"{a.tag}_live_output.jsonl"
    done = {r["uid"] for r in read_jsonl(raw_path)}
    todo = [(uid, u) for uid, u in units.items() if uid not in done]
    print(f"unique compositions {len(units)}  done {len(done)}  todo {len(todo)}")
    if todo:
        w = JsonlWriter(raw_path)

        def work(item):
            uid, u = item
            samples = []
            for _ in range(a.samples):
                txt = chat(_prompt(u["gt"], u["gen"]), model=a.model,
                           temperature=0.0, max_tokens=a.max_tokens)
                sc = parse_scores(txt)
                if sc:
                    samples.append(sc)
            return {"uid": uid, "samples": samples}
        run_concurrent(todo, work, w, workers=a.workers, desc="judge")
        w.close()
    scores = {r["uid"]: r["samples"] for r in read_jsonl(raw_path)}
    _write_scores(a.tag, a.arms, units, zeros, scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "submit", "fetch", "write", "live"])
    ap.add_argument("--tag", default="gpt-4o-batch")
    ap.add_argument("--arms", nargs="+", default=None,
                    help="default: every DEFAULT_ARMS arm that has a compositions file")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--max_tokens", type=int, default=300)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--wait", action="store_true", help="fetch: poll until done")
    a = ap.parse_args()
    if a.arms is None:
        a.arms = [x for x in DEFAULT_ARMS
                  if (OUT_DIR / f"compositions_{x}.jsonl").exists()]
        print(f"arms (auto: compositions present under {OUT_DIR.name}): {a.arms}")
    {"build": cmd_build, "submit": cmd_submit, "fetch": cmd_fetch,
     "write": cmd_write, "live": cmd_live}[a.cmd](a)


if __name__ == "__main__":
    main()
