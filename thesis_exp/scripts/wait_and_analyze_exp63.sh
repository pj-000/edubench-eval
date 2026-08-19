#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=/home/jpang/miniconda3/envs/llama_factory/bin/python

for seed in 67 68 69 70 71; do
  summary="thesis_exp/outputs/exp63_same_state_counterfactual/counterfactual/seed_${seed}/run_summary.json"
  while [[ ! -s "$summary" ]]; do
    active=0
    for session in exp63_cf4 exp63_cf6 exp63_cf7; do
      if tmux has-session -t "$session" 2>/dev/null; then
        active=1
      fi
    done
    if [[ $active -eq 0 ]]; then
      echo "counterfactual workers stopped before seed $seed completed" >&2
      exit 1
    fi
    sleep 30
  done
done

"$PYTHON_BIN" -m thesis_exp.exp63_same_state_counterfactual.analyze
