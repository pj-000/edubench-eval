#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 BASE_TMUX_SESSION SEED [SEED ...]" >&2
  exit 2
fi

base_session=$1
shift
seeds=("$@")

for seed in "${seeds[@]}"; do
  summary="thesis_exp/outputs/exp63_same_state_counterfactual/base/seed_${seed}/run_summary.json"
  while [[ ! -s "$summary" ]]; do
    if ! tmux has-session -t "$base_session" 2>/dev/null; then
      echo "base session $base_session stopped before seed $seed completed" >&2
      exit 1
    fi
    sleep 30
  done
done

while tmux has-session -t "$base_session" 2>/dev/null; do
  sleep 5
done

bash thesis_exp/scripts/run_exp63_counterfactual_worker.sh "${seeds[@]}"
