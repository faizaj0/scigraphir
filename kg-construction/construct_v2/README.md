# Hypothetical answers ("probes")

`gen_probes.py` makes the single query-side LLM call of the semantic branch (thesis
Chapter 6). For each research problem it asks `gpt-4o-mini` for up to 8 short passages:

- 3 to 4 **methods**: a specific, named technique the problem would need, written as it
  would appear in that method paper's abstract;
- 3 to 4 **analogs**: a 1 to 2 sentence hypothetical abstract from another field whose
  mechanism is structurally analogous. This is the cross-domain bridge.

The multi-view scorer compares every candidate paper to each answer separately
(`sir4-retrieval/eval/semantic_scorer.py`); the handcrafted operator uses their sum and
max (`../eval/cargo_operator.py`). The cache is resumable JSONL at
`cache/<dataset>/probes_<split>.jsonl` (git-ignored).

```bash
export OPENAI_API_KEY=...
cd $CARGO_ROOT/kg-construction
python3 construct_v2/gen_probes.py --dataset sir4_physics --split test  --workers 128
python3 construct_v2/gen_probes.py --dataset sir4_physics --split train --workers 128
```

The directory name is historical (second construction iteration); only the probe
generator from it is part of the final system.
