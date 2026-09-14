#!/usr/bin/env bash
# Build the Qwen3-Embedding top-500 pool per query for MOOSE-Chem's --pool mode, locally
# (Apple MPS or CPU; the 0.6B model is already in the HF cache). Same model, pooling and
# instruction as the Qwen3 baseline row, so the pool IS that row's ranking, cut at 500.
# Usage: bash make_pools.sh            # all five: mir + the four SIR-4 fields (~20-30 min on MPS)
#        bash make_pools.sh mir        # one dataset
set -euo pipefail
cd "$(dirname "$0")/.."
QWEN_INSTRUCT=$'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:'
DS=${@:-mir sir4_matsci sir4_physics sir4_biology sir4_cs}
for ds in $DS; do
  out="data/pool_qwen3_${ds}_test.json"
  [ -f "$out" ] && { echo "[skip] $out exists"; continue; }
  python3 eval/baselines_sir4.py --dataset "$ds" --split test --model Qwen/Qwen3-Embedding-0.6B \
      --pooling st --instruct "$QWEN_INSTRUCT" --tag qwen3 --topk 500 --batch 16 --out "$out"
done
ls -la data/pool_qwen3_*_test.json
