#!/usr/bin/env bash
# Run the chosen directions against the same Ollama server.
# Usage: ./run_all.sh                             (full run into output/)
#        PARALLEL=1 ./run_all.sh                  (all directions at once)
#        DIRS=srbh_to_a ./run_all.sh              (choose directions)
set -uo pipefail
cd "$(dirname "$0")"

LIMIT="${LIMIT:-0}"
OUT="${OUT:-output}"
HOSTS="${HOSTS:-http://127.0.0.1:11500}"
SRBH="${SRBH:-data/improved/SR-BH 2020.csv}"
PARALLEL="${PARALLEL:-0}"
DIRS="${DIRS:-b_to_a srbh_to_b srbh_to_a}"
mkdir -p "$OUT"

run_one() {
  local d="$1"
  echo "[$(date '+%F %T')] start $d (limit=$LIMIT)"
  PYTHONPATH=src .venv/bin/python src/benchmark.py \
    --direction "$d" --srbh "$SRBH" --hosts "$HOSTS" \
    --limit "$LIMIT" --output "$OUT" > "$OUT/$d.log" 2>&1
  local rc=$?
  echo "[$(date '+%F %T')] end $d exit=$rc"
  return $rc
}

status=0
if [ "$PARALLEL" = 1 ]; then
  pids=()
  for d in $DIRS; do run_one "$d" & pids+=($!); done
  for p in "${pids[@]}"; do wait "$p" || status=1; done
else
  for d in $DIRS; do run_one "$d" || status=1; done
fi
echo "[$(date '+%F %T')] all done status=$status"
exit $status
