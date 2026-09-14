#!/usr/bin/env bash
# Progress of the background LATTICE jobs. Shows the resume point ("loaded K iterations"),
# the current iteration, quota/rate errors, and DONE once the field is scored.
cd "$(dirname "$0")"
for f in matsci physics biology cs; do
  log="logs/lattice_$f.log"
  [ -f "$log" ] || { echo "== $f: not started"; continue; }
  alive=$(pgrep -f "run_lattice_sir4.sh $f" >/dev/null && echo running || echo stopped)
  loaded=$(grep -o "Loaded existing experiment with [0-9]* eval samples and [0-9]* eval metric dfs" "$log" | tail -1 | grep -o "and [0-9]*" | grep -o "[0-9]*")
  cur=$(grep -o "INFO - -* Iter [0-9]*" "$log" | grep -o "[0-9]*$" | tail -1)
  quota=$(tr '\r' '\n' < "$log" | grep -c "no credits remaining")
  r429=$(tr '\r' '\n' < "$log" | grep -o "429s=[0-9]*" | tail -1)
  echo "== $f: $alive | resumed with ${loaded:-?} iterations saved | now on iteration $(( ${cur:--1} + 1 )) of 40 | no-credit errors $quota | $r429 | $( [ -f outputs/lattice/$f/scores.json ] && echo DONE)"
done
