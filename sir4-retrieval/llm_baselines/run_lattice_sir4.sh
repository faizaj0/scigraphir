#!/usr/bin/env bash
# run_lattice_sir4.sh -- LATTICE (gpt-4o-mini, top-down tree, NumI=40, beam 2) on one SIR-4 field.
#
#   export OPENAI_API_KEY=sk-...
#   bash llm_baselines/run_lattice_sir4.sh cs            # data -> tree -> search -> score
#   bash llm_baselines/run_lattice_sir4.sh cs tree       # stop after the tree (check keywords.jsonl)
#   BEAM=5 bash llm_baselines/run_lattice_sir4.sh cs     # the stronger beam-5 variant
#   CONC=200 bash llm_baselines/run_lattice_sir4.sh cs   # more concurrent search calls (tier 5)
#   BATCH=1 bash llm_baselines/run_lattice_sir4.sh cs   # OpenAI Batch API for the search: half price, slower (one batch job per iteration)
#
# Tree: ceil(N/20) keyword calls + ~0.17 N cluster calls, ~$1-3 per field. Search: 2 prompts per
# query per iteration x 40 iterations = 80 calls/query, ~$0.03/query. Resumable: both steps
# cache (keywords.jsonl; results pkl + --load_existing).
set -euo pipefail
FIELD=${1:?field: cs biology physics matsci}
STOP=${2:-all}
BEAM=${BEAM:-2}
CONC=${CONC:-40}
SUBSET=${SUBSET:-default}          # or 'none' for every query
BATCH=${BATCH:-0}                  # 1 = --openai_batch on the search (tree build stays live, it is cheap)
BATCH_FLAG=""; [ "$BATCH" = "1" ] && BATCH_FLAG="--openai_batch --openai_batch_poll 30"
HERE="$(cd "$(dirname "$0")" && pwd)"
LATTICE="$HOME/Desktop/Project_Week9/llm-guided-hierarchical-search"
cd "$HERE"

python3 sir4_to_lattice.py --field "$FIELD" --subset "$SUBSET" --patch-prompts
N=$(wc -l < "$LATTICE/data/SIR-4/$FIELD/examples.jsonl" | tr -d ' ')

cd "$LATTICE"
mkdir -p "trees/SIR-4/$FIELD" "results/SIR-4/$FIELD"
if [ ! -f "trees/SIR-4/$FIELD/tree-top-down.pkl" ]; then
  python3 src/build_tree.py --corpus "data/SIR-4/$FIELD/documents.jsonl" \
      --out "trees/SIR-4/$FIELD/tree-top-down.pkl" \
      --max_branching 15 --min_branching 8 --llm gpt-4o-mini --concurrent 20 \
      2>&1 | tee "trees/SIR-4/$FIELD/build.log"
fi
[ "$STOP" = "tree" ] && exit 0

WANDB_MODE=offline python3 src/run.py --dataset SIR-4 --subset "$FIELD" --tree_version top-down \
    --traversal_prompt_version 5 --reasoning_in_traversal_prompt -1 \
    --num_leaf_calib 10 --pl_tau 5.0 --relevance_chain_factor 0.5 \
    --llm_api_backend openai --llm gpt-4o-mini \
    --num_iters 40 --num_eval_samples "$N" --max_beam_size "$BEAM" \
    --llm_max_concurrent_calls "$CONC" --llm_api_timeout 120 --llm_api_max_retries 4 \
    --load_existing $BATCH_FLAG 2>&1 | tee -a "results/SIR-4/$FIELD/run_beam$BEAM.log"

cd "$HERE"
python3 score_lattice_sir4.py --field "$FIELD" --max-beam-size "$BEAM"
