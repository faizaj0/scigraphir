#!/usr/bin/env bash
# MOOSE-Chem, Qwen3 top-100 pool, SIR-4 subset (all cross + 250 same), four fields in PARALLEL,
# each scored when done. Usage: export OPENAI_API_KEY=sk-...; bash launch_moosechem_pool.sh
#   SUBSET=none bash launch_moosechem_pool.sh     # full test sets instead (~$23)
#   POOL=300 ...                                   # bigger pool
# Logs: logs/moosechem_<field>.log. Rerun = resume (only missing/failed queries are redone).
set -u
cd "$(dirname "$0")"
[ -n "${OPENAI_API_KEY:-}" ] || { echo "export OPENAI_API_KEY first"; exit 1; }
POOL=${POOL:-100}; SUBSET=${SUBSET:-default}; WORKERS=${WORKERS:-24}
mkdir -p logs
for f in matsci physics biology cs; do
  pred="../data/predictions_qwen3_sir4_${f}_test.json"
  [ -f "$pred" ] || { echo "missing $pred"; continue; }
  nohup bash -c "python3 run_moose_chem_sir4.py --field $f --subset $SUBSET --pool $POOL --pool-pred $pred --workers $WORKERS --window-workers 16 \
    && python3 score_rankings_sir4.py --field $f --subset $SUBSET --rankings outputs/moose_chem/${f}_pool${POOL}/rankings.json --method 'MOOSE-Chem (pool $POOL)'" \
    > "logs/moosechem_$f.log" 2>&1 &
  echo "$f started, pid $!"
done
echo "check: bash check_moosechem.sh"
