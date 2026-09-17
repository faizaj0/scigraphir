#!/usr/bin/env bash
# Build the OpenIE component used by the default SciAfford graph.
# It can also be used alone for the OpenIE graph control. Accepts any staged dataset.
#
# Usage:  bash run_index.sh sir4_physics_train  # then sir4_physics_test
#         FORCE=true bash run_index.sh <name>  # force clean rebuild; default resumes (skips built graph, continues NER cache)
# Needs:  OPENAI_API_KEY in the environment; gfmrag installed (see install steps).
#
# Overrides baked in:
#   dataset.root=<retriever>/data        where the raw/ inputs live
#   dataset.data_name=$1                        which dataset to index
#   el_model=dpr_el_model                       BGE-large entity linker (CPU/MPS, no ColBERT/Qdrant)
#   graph_constructor.add_title=false           doc names are DOIs; title already in the text
# (num_processes is already 10 in the configs -> OpenIE/NER run ~10-way parallel.)
#
# Outputs:
#   data/$1/processed/stage1/{nodes,relations,edges}.csv + {train,test}.json   (the KG index)
#   tmp/{kg_construction,qa_construction}/<hash>/                              (per-step artifacts to audit)
#   logs/index_$1.log                                                          (full run log)
set -euo pipefail

DATA_NAME="${1:?usage: bash run_index.sh <tomato_mini|tomato_train|tomato_test> [threshold]}"
THRESHOLD="${2:-0.9}"    # entity-resolution cosine threshold (BGE linker): 0.8 over-merges; 0.9 chosen; 0.95 = stricter
# WORKERS: thread-pool width for the two LLM stages (document OpenIE, query NER). The OpenIE
# stage is API-latency bound (~2.6 docs/s at 10 threads on 18 Aug), and no retry/backoff
# exists in the vendored OpenIE model, so the ceiling is the OpenAI rate limit, not the code:
# 64 is comfortable on tier 5 (gpt-4o-mini 30k RPM). NOTE the value is part of the
# graph_constructor config, so it changes the tmp/kg_construction/<fingerprint> directory;
# 10 reproduces the SIR-4/TOMATO fingerprint 1d434bd5..., anything else gets a new one.
WORKERS="${WORKERS:-10}"
HERE="$(cd "$(dirname "$0")" && pwd)"          # retriever/
: "${OPENAI_API_KEY:?set OPENAI_API_KEY first}"

cd "$HERE"   # gfmrag is pip-installed (editable), so it runs from here; keeps tmp/ + outputs/ under retriever/
mkdir -p logs

# INTERPRETER. gfmrag is installed (editable) in the conda env `gfmrag` (the one the 18 Aug SIR-4
# indexes ran from; `gfmrag-mac` is an older June copy), NOT in base. Running from a shell that
# has not activated it fails with "No module named 'gfmrag'", so pick a python that imports it:
# the current one if it does, else the env's, else stop. Override with PYTHON=/path/to/python.
if [ -z "${PYTHON:-}" ]; then
  if python -c "import gfmrag" 2>/dev/null; then
    PYTHON=python
  else
    for _cand in /opt/homebrew/Caskroom/miniforge/base/envs/gfmrag/bin/python \
                 "$HOME/miniforge3/envs/gfmrag/bin/python" "$HOME/miniconda3/envs/gfmrag/bin/python"; do
      if [ -x "$_cand" ] && "$_cand" -c "import gfmrag" 2>/dev/null; then PYTHON="$_cand"; break; fi
    done
  fi
fi
: "${PYTHON:?no python imports gfmrag; run 'conda activate gfmrag' or set PYTHON=<env>/bin/python}"

echo ">> indexing '$DATA_NAME' (root=$HERE/data, el_model=dpr, add_title=false, threshold=$THRESHOLD, workers=$WORKERS, python=$PYTHON)"
HYDRA_FULL_ERROR=1 "$PYTHON" -m gfmrag.workflow.index_dataset \
  dataset.root="$HERE/data" \
  dataset.data_name="$DATA_NAME" \
  el_model=dpr_el_model \
  graph_constructor.add_title=false \
  graph_constructor.threshold="$THRESHOLD" \
  graph_constructor.num_processes="$WORKERS" \
  sft_constructor.num_processes="$WORKERS" \
  dataset.force="${FORCE:-false}" \
  2>&1 | tee "logs/index_${DATA_NAME}.log"

echo ">> done -> $HERE/data/$DATA_NAME/processed/stage1/"
