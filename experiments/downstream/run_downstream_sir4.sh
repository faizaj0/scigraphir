#!/usr/bin/env bash
# SIR-4 downstream hypothesis-quality experiment (thesis Table 9.3 layout), all four subsets.
# Usage:  bash run_downstream_sir4.sh <stage>
#   build        API-free: inputs from the SIR-4 records + saved rankings (re-run after downloads)
#   compose      gpt-4o-mini compositions, one call per (query, arm), resumable   (~$7)
#   judge-build  API-free: batch request file + cost estimate
#   judge-submit submit the gpt-4o batch                                          (~$28)
#   judge-fetch  poll until done, write matched files, aggregate
#   table        API-free: Table 9.3-style table, pooled + per field (console, .tex, .json)
#   all          every stage above in order (stops at the first error; ~$35)
set -euo pipefail
export DOWNSTREAM_DATASET=sir4
TAG=${TAG:-gpt-4o-batch}
ARMS=${ARMS:-"none qwen3 reasonir moose_chem lattice scigraphir oracle"}
WORKERS=${WORKERS:-20}
cd "${EXTERNAL_REPOS:-$(cd "$(dirname "$0")/../.." && pwd)/external}/TOMATO-Star"
case "${1:-}" in
  build)        python -m analysis.downstream.build_inputs_sir4 --arm qwen3 --arm moose_chem --arm lattice --arm reasonir --arm scigraphir ;;
  compose)      for a in $ARMS; do python -m analysis.downstream.compose --arm "$a" --workers "$WORKERS"; done ;;
  judge-build)  python -m analysis.downstream.score_matched_batch build  --tag "$TAG" --arms $ARMS ;;
  judge-submit) python -m analysis.downstream.score_matched_batch submit --tag "$TAG" --arms $ARMS ;;
  judge-fetch)  python -m analysis.downstream.score_matched_batch fetch  --tag "$TAG" --arms $ARMS --wait
                python -m analysis.downstream.aggregate --tag "$TAG" ;;
  table)        python -m analysis.downstream.downstream_table --tag "$TAG" --latex --by-field --arms $ARMS ;;
  all)          for st in build compose judge-build judge-submit judge-fetch table; do
                  echo; echo "################ stage: $st"; bash "$0" "$st"; done ;;
  *) sed -n 2,10p "$0"; exit 1 ;;
esac
