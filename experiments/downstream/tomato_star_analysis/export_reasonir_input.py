"""Write the query + corpus bundle that reasonir_rank_colab.py consumes.

API-free.  Output: outputs/caches/reasonir/reasonir_input_tomato.json
  {"queries": [{"id", "text", "gold": [key]}], "docs": [{"id", "text"}]}

Query text follows the CARGO delta protocol (question + background survey, plus the
hypothesis so far for step>=1), i.e. the same text the Qwen3 arm was ranked with.
The corpus is the full TOMATO test inspiration pool (3,033 docs).

Run (from TOMATO-Star):
  python -m analysis.downstream.export_reasonir_input
Then upload the json to Colab and run analysis/downstream/reasonir_rank_colab.py there.
"""
from __future__ import annotations

import json

from analysis.downstream._common import INPUTS_PATH, PROJ, read_jsonl
from analysis.downstream.build_inputs import _corpus_meta_and_records

OUT = PROJ / "outputs" / "caches" / "reasonir" / "reasonir_input_tomato.json"


def main() -> None:
    recs = read_jsonl(INPUTS_PATH)
    meta, _ = _corpus_meta_and_records()
    queries = []
    for r in recs:
        text = (r["research_question"] + " " + r["background_survey"]).strip()
        prev = r.get("prev_hypothesis") or ""
        if r["step_idx"] > 0 and prev and prev != "No previous hypothesis.":
            text += "  [HYPOTHESIS SO FAR]  " + " ".join(prev.split())
        queries.append({"id": r["query_id"], "text": text, "gold": [r["gold_key"]]})
    docs = [{"id": k, "text": (t + " " + a).strip()} for k, (t, a) in meta.items()
            if (t or a)]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"queries": queries, "docs": docs}, ensure_ascii=False))
    print(f"Wrote {OUT}: {len(queries)} queries, {len(docs)} docs")


if __name__ == "__main__":
    main()
