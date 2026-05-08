#!/bin/bash
# Watcher that submits the next chunk of the 410M sweep as soon as the
# per-user submit-limit allows. Designed to be backgrounded:
#
#   nohup ./exps_410m/auto_submit.sh > exps_410m/slurm_logs/auto_submit.log 2>&1 &
#
# It exits once all chunks are submitted.

set -euo pipefail
cd /shared/home/arman.bolatov/plainLM

# Chunks left to submit (as --array specs). Chunk A is already queued by hand.
chunks=(
  "38-57%8"
  "58-77%8"
  "78-79"
)

# We need each chunk to be small enough that (queued+running) stays <= 20.
# Trigger: submit next chunk when current queue has <= (20 - chunk_size) jobs.
get_queue_count() { squeue -u "$USER" -h -t PD,R -o "%i" | wc -l; }

prev_jid=""
for spec in "${chunks[@]}"; do
  # parse chunk size: "38-57%8" -> 20, "78-79" -> 2
  range="${spec%%\%*}"
  lo="${range%%-*}"; hi="${range##*-}"
  size=$(( hi - lo + 1 ))
  threshold=$(( 20 - size ))

  echo "[$(date -Iseconds)] waiting for queue <= $threshold to submit chunk $spec ($size jobs)"
  while true; do
    n=$(get_queue_count)
    [ "$n" -le "$threshold" ] && break
    sleep 120
  done

  if [ -n "$prev_jid" ]; then
    jid=$(sbatch --parsable --dependency=afterany:"$prev_jid" --array="$spec" exps_410m/slurm_array.sh)
  else
    jid=$(sbatch --parsable --array="$spec" exps_410m/slurm_array.sh)
  fi
  echo "[$(date -Iseconds)] submitted chunk $spec as job $jid"
  prev_jid="$jid"
done

echo "[$(date -Iseconds)] all chunks submitted. final queue:"
squeue -u "$USER" -o "%.10i %.2t %.10M %.10l %.20R %E"
