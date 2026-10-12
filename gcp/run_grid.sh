#!/usr/bin/env bash
# Launch the experiment grid as N parallel worker processes in one tmux session.
#   bash gcp/run_grid.sh 3            # 3 workers
#   tmux attach -t grid               # watch (Ctrl-b n = next window, Ctrl-b d = detach)
# Re-running the script after a preemption resumes: finished runs are skipped.
set -euo pipefail
N="${1:-3}"
EXTRA="${EXTRA:---ablations}"            # e.g. EXTRA="--ablations --eps none 3" bash gcp/run_grid.sh 3
cd "$(dirname "$0")/.."
mkdir -p logs
CPUS=$(( $(nproc) / N )); (( CPUS < 1 )) && CPUS=1   # split cores between workers
tmux kill-session -t grid 2>/dev/null || true
tmux new-session -d -s grid -n w0
for ((i=0; i<N; i++)); do
  [[ $i -gt 0 ]] && tmux new-window -t grid -n "w$i"
  tmux send-keys -t "grid:w$i" "source .venv/bin/activate && PYTHONUNBUFFERED=1 OMP_NUM_THREADS=$CPUS LOKY_MAX_CPU_COUNT=$CPUS python experiments/run_experiments.py --shard $i/$N $EXTRA 2>&1 | tee -a logs/worker_$i.log" C-m
done
echo "Started $N workers in tmux session 'grid'. Attach with: tmux attach -t grid"
