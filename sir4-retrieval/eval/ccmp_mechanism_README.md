# CCMP: why did the displayed path weights change?

Open `sir4-retrieval/colab_table4_openie.ipynb` in Colab, choose an A100 or L4 GPU, and run all cells. The existing Drive directory is `/content/drive/MyDrive/cargo-gfmrag`. The notebook embeds its analysis and engine overlay; no extra code upload is needed.

Defaults are the biology optimal-transport example and the CS creativity and reconsolidation examples, with two **fixed** paths per case. Both bfloat16 and float32 run from fresh loads of the same checkpoint. This is evaluation only. Float32 changes arithmetic precision; it does not recover information already rounded in the saved checkpoint.

`PREFER_MERGED = True` selects merged-graph CCMP checkpoints on Drive where available:

| Dataset | Preferred checkpoint under `outputs/sir4_hyb/` | Evaluation graph |
|---|---|---|
| Biology | `sir4_biology_hyb_ccmp_e10_b2/model_best.pth` | `sir4_biology_test_hyb` |
| CS | `sir4_cs_hyb_ccmp_e10_b1/model_best.pth` | `sir4_cs_test_hyb` |

Other available merged CCMP epoch/batch runs are considered after the preferred recipe. Control runs, smoke runs and other fields are excluded. The corresponding field scorer is restored, preferring `outputs/sir4_hyb/semantic_<dataset>/`. Missing merged graph files are extracted from the shared `sir4_hyb_bundle.zip` on Drive. All six fixed paths and their starting seeds have been verified in the local merged graphs.

If **no merged checkpoint is available**, the previous frame checkpoint and `_test_v16sc` graph are used with an explicit fallback message. A present merged checkpoint with missing or incompatible assets stops the run rather than silently switching experiments. Set `PREFER_MERGED = False` only to deliberately reproduce the previous frame setup. Selection happens once per dataset and is recorded in the manifest/report; all interventions and precisions use that same checkpoint/graph.

The notebook tests seven conditions: off, native CCMP, frozen native gates, path-node gates only, outside-node gates only, outside suppression only, and outside amplification only. Every condition uses the same model weights, query, graph, seeds, target and paths. Outside gates are frozen from the native run and modified **after normalization without renormalization**.

By default, every node on either selected path is protected at **every layer**, including the gold node. Its gate stays 1 during outside interventions. Because CCMP gates sender nodes, this protects all messages those nodes send. `PROTECTION = "sender_layer"` is an optional, narrower sensitivity analysis that protects just the displayed sender/layer pairs. Neither definition makes other messages irrelevant by assumption.

## What answers the question

`outside_suppress − off` tests whether suppression outside the protected paths is sufficient to increase the fixed paths' attribution. `native − frozen` measures the contribution of differentiating through gates; their forward scores must match. Path-only, outside-full and amplification-only controls expose other effects. The report also gives nonlinear interaction terms. These contrasts do not justify assigning percentages of a total effect to independent mechanisms.

Gold graph/fused scores, ranks, ties, gold margins and the same five off-run non-gold competitors are tracked separately. An attribution increase does not imply a retrieval improvement. The cases were selected qualitatively and are not a population-level significance test.

The path weight retains the engine's `visualize()` convention: mean edge gradients along the exact path. Successive layer edge-weight clones are linked, so these are the legacy attribution variables, not independent layer-local edge perturbations. Gates are not multiplied to obtain the path weight. Both raw and actually applied, dtype-cast gates are exported.

## Outputs and checks

Results live under `outputs/ccmp_mechanism/<dataset>/<merged-or-frame>/<precision>/<protection>/<manifest-hash>/` on Drive:

- `report.md`: treatment table and cautious interpretation per path.
- `results.json`: scores, ranks, paths, per-hop gradients/gates, frontier gate statistics, effects, historical comparison and checks.
- `paths.csv`, `hops.csv`, `effects.csv`: measured values for analysis and figures.
- `tikz_values.tsv`, `tikz_index.json`: numeric TikZ/PGF inputs and label mapping; indices are zero based.
- `gates_*.pt`: full native raw gate tensors, message dtypes, protected masks and resolved fixed paths.
- `manifest.json`: checkpoint, data, source and configuration hashes, precision and runtime versions.

The workflow loads checkpoints strictly, validates exact paths and gold membership, rejects duplicate edges, and checks frozen replay, repeated forward scores, unchanged semantic scores, and agreement with uninstrumented inference. Frozen replay must apply gate tensors that are bit-identical to the saved native gates. CUDA sparse reductions can vary between otherwise identical bfloat16 passes, so score equality is evaluated using the measured repeat variation and an explicitly recorded dtype-and-score-scale allowance. Float32 retains the tight configured tolerance. Attribution repeat variation contributes to a separate numerical screening tolerance; neither tolerance is a confidence interval. Missing or failed cases stop the run. Resume only accepts matching manifests and intact gate archives. Old scan/interpretation JSON files never satisfy the new experiment cache.

The historical on/off values flag reproduction differences for the frame setup only. For merged runs, they are labelled as reference values from a different checkpoint/graph, and are not treated as a numerical reproduction target. They are never substituted for fresh measurements.

## Maintain and validate

```sh
python3 sir4-retrieval/prep/build_table4_notebook.py
python3 -m unittest discover -s sir4-retrieval/eval -p 'test_ccmp*.py' -v
```

The CPU tests execute the real patched `QueryNBFNet.bellmanford` with tiny differentiable layers, and the workflow with small local data and mocked Hydra/trainer infrastructure. They cover gate policies, dtype rounding, matched replay, derivative effects, fixed paths, cleanup on failure, strict loading, output generation and cache invalidation. They do not validate the Drive checkpoints or CUDA extension runtime. The actual scientific result requires the Colab runs.

The prior notebook and builder are preserved under `results/qualitative/notebook_archive/before_ccmp_mechanism/`.
