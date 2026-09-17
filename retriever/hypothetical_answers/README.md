# Hypothetical answers

[Retriever](../README.md) · [Semantic code guide](../../docs/README.md#2-multi-view-semantic-scoring)

[generate_answers.py](generate_answers.py) generates hypothetical answers for the semantic branch.
Each query receives up to eight passages:
specific methods that could address the problem and analogous mechanisms from other fields.

The [multi-view scorer](../../experiments/eval/semantic_scorer.py) compares every document
to each answer separately. The [handcrafted scorer](../eval/README.md) uses their sum and
maximum instead.

## Generate answers

After [setup and corpus staging](../../docs/SETUP.md), run from the repository root.
Uncached queries make paid LLM calls using `OPENAI_API_KEY`.

```bash
python retriever/hypothetical_answers/generate_answers.py --dataset sir4_physics --split train --workers 16
python retriever/hypothetical_answers/generate_answers.py --dataset sir4_physics --split test --workers 16
```

Outputs are resumable JSONL records containing `id` and `answers`. The
`answers_path(split)` helper in [scigraphir_paths.py](../../scigraphir_paths.py) resolves
the dataset cache, retaining existing storage names. Older record keys remain readable;
complete caches are reused without another generation call.
