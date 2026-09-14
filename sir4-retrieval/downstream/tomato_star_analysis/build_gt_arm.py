"""Build a 'gt' arm for the Idea Arena from the REAL ground-truth hypotheses.

API-FREE.  The Oracle arm is the composer's *reconstruction* given the gold
inspiration; this arm is the ACTUAL paper hypothesis (`gt_delta`) itself, entered
into the round-robin as a 5th contestant.  This disentangles the Oracle's arena
loss:
  - if GT also loses to the model arms  -> the reference-free judge prefers
    confabulation over real science (judge problem);
  - if GT wins but Oracle loses         -> the composer fails to exploit the gold
    inspiration (composer problem).

It only reformats existing data (no generation), so there is nothing to score
against -- GT is a pure arena contestant, not a Matched-Score arm.

Run:
  cd TOMATO-Star
  python -m analysis.downstream.build_gt_arm

Then judge only the 4 new GT-vs-X pairs (existing 6 pairs resume untouched):
  python -m analysis.downstream.idea_arena \
      --arms none,bm25,moose_chem,oracle,gt --judge gpt-4o-mini --workers 20
  python -m analysis.downstream.aggregate

Output: results/downstream/compositions_gt.jsonl
  {query_id, arm:'gt', stratum, insp_key:'__gt__', is_gold:True,
   delta_hypothesis, raw_response}
"""
from __future__ import annotations

from analysis.downstream._common import (
    OUT_DIR,
    INPUTS_PATH,
    JsonlWriter,
    read_jsonl,
)


def main() -> None:
    rows = read_jsonl(INPUTS_PATH)
    if not rows:
        raise SystemExit(f"No inputs at {INPUTS_PATH}. Run build_inputs first.")

    out_path = OUT_DIR / "compositions_gt.jsonl"
    writer = JsonlWriter(out_path)
    n = n_empty = 0
    for r in rows:
        gt = (r.get("gt_delta") or "").strip()
        if not gt:
            n_empty += 1
        writer.write({
            "query_id": r["query_id"],
            "arm": "gt",
            "stratum": r["stratum"],
            "insp_key": "__gt__",
            "is_gold": True,
            "delta_hypothesis": gt,
            "raw_response": gt,
        })
        n += 1
    writer.close()
    print(f"Wrote {n} GT-arm rows -> {out_path}"
          + (f"  ({n_empty} empty gt_delta!)" if n_empty else ""))


if __name__ == "__main__":
    main()
