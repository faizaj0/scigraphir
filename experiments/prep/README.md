# Dataset preparation and notebook generation

[Experiments](../README.md) · [Setup](../../docs/SETUP.md) · [Notebooks](../notebooks/README.md)

## Prepare inputs

| Script | Purpose |
|---|---|
| [prepare_data.py](../../sir-4/prepare_data.py) | Verify and unpack the included SIR-4 dataset. |
| [stage_sir4.py](stage_sir4.py) | Validate and stage the SIR-4 exports into retrieval corpora. |
| [stage_mir.py](stage_mir.py) | Stage MIR in the shared corpus layout. |
| [run_domain.py](run_domain.py) | Drive staging, extraction, answers, graph construction, audit, bundling, and notebook generation. Paid phases require `--spend`. |
| [build_hybrid_graph.py](build_hybrid_graph.py) | Assemble the default SciAfford graph from affordance structure, OpenIE entity context and paper-to-affordance links. |
| [make_smoke.py](make_smoke.py) | Create a small corpus for a preparation rehearsal. |
| [filter_titleonly.py](filter_titleonly.py) | Prepare a title-only-document filtering variant. |

The [SIR-4 runbook](../RUNBOOK.md) explains the preparation sequence. Unpack the
[included dataset](../../sir-4/dataset/README.md) before staging it; the shared path resolver
then keeps each dataset’s corpora, graphs and caches separate.

## Package artifacts

| Script | Package |
|---|---|
| [bundle.py](bundle.py) | Dataset-specific code, corpora, graphs, and caches for Colab. `--slim` supports semantic-only work; `--no-emb` omits embeddings. |
| [make_release_data.py](make_release_data.py) | Existing SIR-4 corpora, benchmark exports, and extraction caches for a data release. |

`bundle.py` packages the affordance component through its retained path default.
The full-method notebook also expects the `_hyb` graphs in `sir4_hyb_bundle.zip`;
see the [default graph workflow](../RUNBOOK.md#default-graph-workflow).
The dataset bundle does not include the separate `gfm-rag-adapted.zip` engine archive.
Review the bundler's missing-artifact messages before uploading a bundle.

## Generate an experiment notebook

| Generator | Experiment family |
|---|---|
| [build_notebook.py](build_notebook.py) | Earlier SIR-4/TOMATO component-graph and scorer experiments. |
| [build_baselines_notebook.py](build_baselines_notebook.py) | Retrieval baselines. |
| [build_sir4_zeroshot_notebook.py](build_sir4_zeroshot_notebook.py) | Transfer between SIR-4 fields. |
| [build_rb_zeroshot_notebook.py](build_rb_zeroshot_notebook.py) | ResearchBench transfer. |
| [build_mir_all_notebook.py](build_mir_all_notebook.py) | MIR experiments. |
| [build_sir4_hyb_notebook.py](build_sir4_hyb_notebook.py) | Default SIR-4 graph workflow, with and without CCMP. |
| [build_routing_notebook.py](build_routing_notebook.py) | CCMP versus routing controls. |
| [build_scientific_paths_notebook.py](build_scientific_paths_notebook.py) | Discovery and display of scientific paths. |
| [build_table4_notebook.py](build_table4_notebook.py), [build_path_sender_notebook.py](build_path_sender_notebook.py), [build_gate_decomp_notebook.py](build_gate_decomp_notebook.py) | Gate and path interventions. |

Other `build_*_notebook.py` files cover additional graph controls, figures, and worked
examples. Choose the experiment from the [notebook index](../notebooks/README.md), then
inspect the matching generator and its arguments.

Generators reuse [the training template](../../retriever/train/README.md) and embed selected
engine sources. Regenerating a notebook can therefore change the model code it executes;
record the generator arguments and Git revision with each new notebook.
