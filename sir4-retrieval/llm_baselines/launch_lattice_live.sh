#!/usr/bin/env bash
# Switch the SIR-4 LATTICE subset runs from Batch to LIVE calls: stop the batch pollers, relaunch
# the four fields in the background with live API calls (CONC concurrent per field, default 25).
# Trees are cached; finished iterations are reloaded, so nothing already paid for is redone.
# Usage: export OPENAI_API_KEY=sk-...; bash launch_lattice_live.sh        (CONC=40 bash ... for tier-5 keys)
set -u
cd "$(dirname "$0")"
[ -n "${OPENAI_API_KEY:-}" ] || { echo "export OPENAI_API_KEY first"; exit 1; }
CONC=${CONC:-25}
FIELDS=${@:-matsci physics biology cs}       # optional: bash launch_lattice_live.sh physics cs
for f in $FIELDS; do pkill -f "run_lattice_sir4.sh $f" 2>/dev/null; pkill -f "src/run.py --dataset SIR-4 --subset $f" 2>/dev/null; done; sleep 2
mkdir -p logs
for f in $FIELDS; do
  BATCH=0 CONC=$CONC nohup bash run_lattice_sir4.sh "$f" > "logs/lattice_$f.log" 2>&1 &
  echo "$f started LIVE at $CONC concurrent, pid $!"
done
echo "check: bash check_lattice.sh"
