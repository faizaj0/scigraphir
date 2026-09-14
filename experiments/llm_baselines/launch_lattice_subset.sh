#!/usr/bin/env bash
# Launch LATTICE on the SIR-4 subset for all four fields in the background (Batch API).
# Usage: export OPENAI_API_KEY=sk-...; bash launch_lattice_subset.sh
# Logs: logs/lattice_<field>.log. Each job scores its field at the end. Rerun = resume.
set -u
cd "$(dirname "$0")"
[ -n "${OPENAI_API_KEY:-}" ] || { echo "export OPENAI_API_KEY first"; exit 1; }
pkill -f "run_lattice_sir4.sh" 2>/dev/null; pkill -f "src/run.py --dataset SIR-4" 2>/dev/null; sleep 2   # stop any live/batch run first
mkdir -p logs
for f in matsci physics biology cs; do
  BATCH=1 nohup bash run_lattice_sir4.sh "$f" > "logs/lattice_$f.log" 2>&1 &
  echo "$f started, pid $!"
done
echo "check: bash check_lattice.sh"
