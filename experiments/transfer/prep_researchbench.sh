#!/usr/bin/env bash
# prep_researchbench.sh -- Phase 1 of the transfer experiment. Run once.
#
# Produces everything both checkpoints share: frames, probes, the master graph and
# the full-pool operator components. Nothing here is specific to a checkpoint, and
# nothing here reads a gold label.
#
# RESEARCHBENCH IS EVALUATION-ONLY. There is no train split and none is created.
# Every command below is --split test. If you find yourself typing --split train,
# stop: that would be training on the benchmark this experiment claims never to
# train on.
#
# TWO GATES, because the steps fail in two different expensive ways.
#
#   --spend   steps 1-2 call a paid API (~$7 total).
#   --build   steps 3-5 encode 20,322 documents with BGE locally. That is the
#             workload that exhausted swap on a 17 GB machine earlier in this
#             project. Gated separately so extraction (safe anywhere, it is all
#             network I/O) can run locally while the encoding moves to Colab.
#
# EVERY STEP SKIPS CLEANLY WHEN ITS INPUT IS MISSING rather than aborting. A first
# run with no flags is a pure preview; re-running after extraction picks up where it
# left off. Nothing here is destructive, so re-running is always safe.
#
#   bash transfer/prep_researchbench.sh                    # preview, costs nothing
#   bash transfer/prep_researchbench.sh --spend            # extraction only
#   bash transfer/prep_researchbench.sh --spend --build    # everything, locally
set -euo pipefail

CARGO="${SCIGRAPHIR_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
KG="$CARGO/retriever"
V16="$CARGO/sciafford"
S4="$CARGO/experiments"
DS=researchbench
GRAPH="${DS}_test_v16sc"
WORKERS="${WORKERS:-512}"
SPEND=0; BUILD=0
for arg in "$@"; do
  case "$arg" in
    --spend) SPEND=1 ;;
    --build) BUILD=1 ;;
    *) echo "unknown flag $arg (expected --spend and/or --build)" >&2; exit 2 ;;
  esac
done

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
skip() { echo "   [skip] $*"; }
paid() {
  if [[ $SPEND -eq 1 ]]; then eval "$1"
  else skip "needs --spend:  $1"; fi
}

DOCF="$V16/cache/$DS/frames_doc_test.jsonl"
QRYF="$V16/cache/$DS/frames_query_test.jsonl"
PROBF="$KG/probes/cache/$DS/probes_test.jsonl"
NODES="$KG/data/$GRAPH/processed/stage1/nodes.csv"

# Guard for the heavy local steps: are the inputs there, and did the user opt in?
ready() {                       # ready <needed-file> <what-produces-it>
  [[ -f "$1" ]] || { skip "waiting on $2"; return 1; }
  [[ $BUILD -eq 1 ]] || { skip "needs --build (encodes 20,322 docs locally)"; return 1; }
  return 0
}

# ---------------------------------------------------------------- step 0
# The pool builder writes data/researchbench/raw, but scigraphir_paths resolves a corpus
# as data/{DATASET}_{split}. Without this move every later command silently looks in
# a directory that does not exist, or worse, finds a stale one.
say "0. corpus location"
if [[ -d "$KG/data/$DS" && ! -d "$KG/data/${DS}_test" ]]; then
  mv "$KG/data/$DS" "$KG/data/${DS}_test"
  echo "   moved data/$DS -> data/${DS}_test"
fi
test -f "$KG/data/${DS}_test/raw/documents.json" || {
  echo "FATAL: no corpus. Run researchbench/build_researchbench_pool.py first." >&2
  exit 1; }
python3 - <<PY
import json
b="$KG/data/${DS}_test/raw"
d=json.load(open(f"{b}/documents.json")); q=json.load(open(f"{b}/test.json"))
c=json.load(open(f"{b}/candidates.json"))
g=[x for x in q for _ in x["supporting_documents"]]
inn=sum(1 for x in q for s in x["supporting_documents"] if s in set(c.get(x["id"],[])))
print(f"   {len(d):,} documents  {len(q):,} queries  {len(g):,} gold rows")
print(f"   golds inside their candidate set: {inn}/{len(g)}")
assert inn==len(g), "a gold is outside its candidate set; the 75-protocol is invalid"
PY

# ---------------------------------------------------------------- step 1
say "1. frames  (PAID, ~\$7, the dominant cost)"
for side in doc query; do
  paid "cd '$V16' && python3 extract_frames.py --dataset $DS --side $side --split test --workers $WORKERS"
done

# ---------------------------------------------------------------- step 2
say "2. probes  (PAID, small)"
paid "cd '$KG' && python3 probes/gen_probes.py --dataset $DS --split test --workers $WORKERS"

# ---------------------------------------------------------------- step 3
# tau_canon 0.95 is not the default and matters: at 0.85 two thirds of one node type
# merged into a single node on SIR-4 CS. It must also MATCH the setting the
# checkpoints were trained under, or the graphs differ in kind and the transfer
# measures the threshold rather than the benchmark.
say "3. master graph  (free, but BGE-heavy)"
if [[ -f "$NODES" ]]; then echo "   already built"
elif ready "$DOCF" "step 1 (document frames)"; then
  cd "$V16" && python3 build_greasoner_dataset.py --dataset $DS --split test \
      --tau_canon 0.95 --no_entity_seeds --no_probe_seeds
fi

# ---------------------------------------------------------------- step 4
say "4. operator components over the master document order  (free, BGE-heavy)"
if [[ -f "$KG/data/$GRAPH/operator_components.npz" ]]; then echo "   already built"
elif ready "$NODES" "step 3 (master graph)" && [[ -f "$PROBF" ]]; then
  cd "$KG" && python3 precompute/precompute_operator_components.py \
      --graph $GRAPH --split test
else
  [[ -f "$PROBF" ]] || skip "waiting on step 2 (probes)"
fi

# ---------------------------------------------------------------- step 5
# BGE is a reported arm in its own right AND defines score_sir4's similar/dissimilar
# axis, so it is not optional.
say "5. BGE baseline  (free, BGE-heavy)"
if [[ -f "$S4/data/predictions_bge_${DS}_test.json" ]]; then echo "   already built"
elif ready "$KG/data/${DS}_test/raw/documents.json" "step 0"; then
  cd "$S4" && python3 eval/bge_sir4.py --dataset $DS --split test --topk 300
fi

# ---------------------------------------------------------------- step 6
say "6. audit  (free; exits non-zero on dangling edges or zero-seed queries)"
if [[ -f "$NODES" ]]; then
  cd "$S4" && python3 eval/audit_graph.py --dataset $DS --split test --sample 40
else
  skip "waiting on step 3 (master graph)"
fi

# ---------------------------------------------------------------- where are we
say "state"
for pair in "$DOCF|document frames" "$QRYF|query frames" "$PROBF|probes" \
            "$NODES|master graph" "$KG/data/$GRAPH/operator_components.npz|operator components" \
            "$S4/data/predictions_bge_${DS}_test.json|BGE baseline"; do
  f="${pair%%|*}"; label="${pair##*|}"
  if [[ -f "$f" ]]; then
    n=$([[ "$f" == *.jsonl || "$f" == *.csv ]] && wc -l < "$f" | tr -d ' ' || echo "")
    printf '   [x] %-20s %s\n' "$label" "${n:+$n lines}"
  else
    printf '   [ ] %-20s\n' "$label"
  fi
done
cat <<'EOF'

   expected: 20,322 doc frames | 1,367 query frames | 1,367 probes
   next once all six are [x]:
     python3 transfer/build_candidate_views.py      (Phase 2, local, free)
     then the Colab notebook                        (Phase 3, GPU)
EOF
