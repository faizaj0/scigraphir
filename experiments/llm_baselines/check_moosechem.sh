#!/usr/bin/env bash
cd "$(dirname "$0")"
POOL=${POOL:-100}
for f in matsci physics biology cs; do
  log="logs/moosechem_$f.log"; [ -f "$log" ] || { echo "== $f: not started"; continue; }
  alive=$(pgrep -f "run_moose_chem_sir4.py --field $f " >/dev/null && echo running || echo stopped)
  fails=$(grep -c "FAIL" "$log"); done_=$(grep -o "\[[0-9]*/[0-9]*\]" "$log" | tail -1)
  echo "== $f: $alive | $done_ | failures $fails | $( [ -f outputs/moose_chem/${f}_pool${POOL}/scores.json ] && echo DONE)"
  [ -f "outputs/moose_chem/${f}_pool${POOL}/scores.json" ] && grep "^| MOOSE-Chem" "$log" | tail -1
done
