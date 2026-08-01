#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "${repo_root}"

arms=(S0 R1 R2 R3)
seeds=(42 43 44)
epochs=(1 2 3)

for arm in "${arms[@]}"; do
  for seed in "${seeds[@]}"; do
    for epoch in "${epochs[@]}"; do
      python -m thesis_exp.exp54_rar_sft.run_dev_inference_v2 \
        --arm "${arm}" \
        --seed "${seed}" \
        --epoch "${epoch}"
    done
  done
done
