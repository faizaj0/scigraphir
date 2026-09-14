"""Idea Arena (official CoI judge) through the OpenAI Batch API.

Same protocol as idea_arena.py: every unordered arm pair, both presentation orders,
one call per order covering all five criteria, CoI's own prompt and tag extractor.
Output rows have the identical round-robin schema, so aggregate.py --tag <tag> reads them.

Needs OPENAI_API_KEY.  Steps (from TOMATO-Star):
  python -m analysis.downstream.idea_arena_batch build  --tag gpt-4o-batch --arms scigraphir,reasonir,moose_chem,oracle
  python -m analysis.downstream.idea_arena_batch submit --tag gpt-4o-batch
  python -m analysis.downstream.idea_arena_batch fetch  --tag gpt-4o-batch --wait
  python -m analysis.downstream.aggregate --tag gpt-4o-batch
Incremental: build skips matchups whose two orders are already in {tag}_arena_raw_output.jsonl,
so adding an arm later only judges the new pairs.

Files: results/downstream/batch/{tag}_arena_requests.jsonl, {tag}_arena_manifest.json,
       {tag}_arena_raw_output.jsonl;  results/downstream/idea_arena__{tag}.jsonl
"""
from __future__ import annotations

import argparse
import itertools
import json
import time

from analysis.downstream._common import (
    OUT_DIR, INPUTS_PATH, JsonlWriter, client, index_by, read_jsonl,
)
from analysis.downstream.idea_arena import CRITERIA, GET_JUDGE_PROMPT, COI_EXTRACT

BATCH_DIR = OUT_DIR / "batch"
BATCH_DIR.mkdir(parents=True, exist_ok=True)
PRICE_IN, PRICE_OUT = 2.50, 10.00        # gpt-4o list, per 1M tokens; batch = half


def _parse(resp: str) -> dict:
    out = {}
    for crit in CRITERIA:
        val = (COI_EXTRACT(resp, crit) or "").strip()
        out[crit] = val if val in ("0", "1", "2") else "2"
    return out


def _matchups(arms):
    inputs = index_by(read_jsonl(INPUTS_PATH))
    comps = {}
    for a in arms:
        rows = index_by(read_jsonl(OUT_DIR / f"compositions_{a}.jsonl"))
        if not rows:
            raise SystemExit(f"Missing compositions_{a}.jsonl -- run compose --arm {a}")
        comps[a] = rows
    qids = [q for q in inputs if all(q in comps[a] for a in arms)]
    pairs = list(itertools.combinations(arms, 2))
    items = []
    for q in qids:
        for a0, a1 in pairs:
            h0 = (comps[a0][q].get("delta_hypothesis") or "").strip()
            h1 = (comps[a1][q].get("delta_hypothesis") or "").strip()
            items.append({"qid": q, "stratum": inputs[q]["stratum"], "a0": a0, "a1": a1,
                          "topic": inputs[q]["research_question"], "h0": h0, "h1": h1})
    return items, pairs


def _raw(tag) -> dict:
    """custom_id -> judge text, accumulated over batches."""
    p = BATCH_DIR / f"{tag}_arena_raw_output.jsonl"
    out = {}
    if p.exists():
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            try:
                out[r["custom_id"]] = r["response"]["body"]["choices"][0]["message"]["content"] or ""
            except (KeyError, TypeError, IndexError):
                pass
    return out


def _cid(it, order):
    return f"{it['qid']}|{it['a0']}|{it['a1']}|{order}"


def cmd_build(a):
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    items, pairs = _matchups(arms)
    have = _raw(a.tag)
    req_path = BATCH_DIR / f"{a.tag}_arena_requests.jsonl"
    n_req = n_tok = n_skip = 0
    with req_path.open("w") as f:
        for it in items:
            if not it["h0"] or not it["h1"]:
                n_skip += 1
                continue
            for order, (x, y) in (("A", (it["h0"], it["h1"])), ("B", (it["h1"], it["h0"]))):
                cid = _cid(it, order)
                if cid in have:
                    continue
                p = GET_JUDGE_PROMPT(x, y, it["topic"])
                n_tok += len(p) // 4
                n_req += 1
                f.write(json.dumps({
                    "custom_id": cid, "method": "POST", "url": "/v1/chat/completions",
                    "body": {"model": a.model, "temperature": 0.0,
                             "max_tokens": a.max_tokens,
                             "messages": [{"role": "user", "content": p}]}}) + "\n")
    man = {"tag": a.tag, "arms": arms, "model": a.model, "max_tokens": a.max_tokens,
           "n_requests": n_req, "batch_id": None}
    (BATCH_DIR / f"{a.tag}_arena_manifest.json").write_text(json.dumps(man, indent=2))
    est_out = n_req * 350
    live = n_tok / 1e6 * PRICE_IN + est_out / 1e6 * PRICE_OUT
    print(f"arms {arms}  pairs/query {len(pairs)}  matchups {len(items)}  "
          f"(skipped {n_skip} with an empty hypothesis)")
    print(f"already judged {len(have)} calls; new requests {n_req}  "
          f"est input tokens {n_tok:,}  est output {est_out:,}")
    print(f"est cost: live ${live:.2f}   batch ${live / 2:.2f}")
    print(f"wrote {req_path}")


def cmd_submit(a):
    mp = BATCH_DIR / f"{a.tag}_arena_manifest.json"
    man = json.loads(mp.read_text())
    up = client().files.create(file=(BATCH_DIR / f"{a.tag}_arena_requests.jsonl").open("rb"),
                               purpose="batch")
    b = client().batches.create(input_file_id=up.id, endpoint="/v1/chat/completions",
                                completion_window="24h",
                                metadata={"tag": a.tag, "purpose": "idea-arena"})
    man["batch_id"] = b.id
    mp.write_text(json.dumps(man, indent=2))
    print(f"submitted batch {b.id}  status={b.status}")


def cmd_fetch(a):
    mp = BATCH_DIR / f"{a.tag}_arena_manifest.json"
    man = json.loads(mp.read_text())
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
    rp = BATCH_DIR / f"{a.tag}_arena_raw_output.jsonl"
    have = set(_raw(a.tag))
    new = [l for l in raw.splitlines() if l.strip() and json.loads(l)["custom_id"] not in have]
    with rp.open("a") as f:
        for l in new:
            f.write(l + "\n")
    print(f"appended {len(new)} judge outputs to {rp.name}")
    man["batch_id"] = None
    mp.write_text(json.dumps(man, indent=2))
    a.arms = ",".join(man["arms"])
    cmd_write(a)


def cmd_write(a):
    """Rebuild idea_arena__{tag}.jsonl for these arms from the raw output (no API).
    Rows for pairs outside this arm set are kept."""
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    items, _ = _matchups(arms)
    have = _raw(a.tag)
    out_path = OUT_DIR / f"idea_arena__{a.tag}.jsonl"
    keep = [r for r in read_jsonl(out_path)
            if not (r.get("arm0") in arms and r.get("arm1") in arms)]
    out_path.write_text("")
    w = JsonlWriter(out_path)
    for r in keep:
        w.write(r)
    n_ok = n_fail = n_missing = 0
    for it in items:
        row = {"query_id": it["qid"], "stratum": it["stratum"],
               "arm0": it["a0"], "arm1": it["a1"]}
        if not it["h0"] or not it["h1"]:
            row.update(orderA=None, orderB=None, failed=True); n_fail += 1
        elif _cid(it, "A") in have and _cid(it, "B") in have:
            row.update(orderA=_parse(have[_cid(it, "A")]),
                       orderB=_parse(have[_cid(it, "B")]), failed=False); n_ok += 1
        else:
            n_missing += 1
            continue
        w.write(row)
    w.close()
    print(f"wrote {out_path.name}: {n_ok} judged matchups, {n_fail} empty-hypothesis, "
          f"{n_missing} not yet judged, {len(keep)} rows kept from other pairs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "submit", "fetch", "write"])
    ap.add_argument("--tag", default="gpt-4o-batch")
    ap.add_argument("--arms", default="scigraphir,reasonir,moose_chem,oracle")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--wait", action="store_true")
    a = ap.parse_args()
    {"build": cmd_build, "submit": cmd_submit, "fetch": cmd_fetch, "write": cmd_write}[a.cmd](a)


if __name__ == "__main__":
    main()
