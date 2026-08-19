#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "usage: $0 WORKER_ID CUDA_INDEX CUDA_UUID" >&2
  exit 2
fi

worker_id="$1"
cuda_index="$2"
cuda_uuid="$3"
repo_root="${EDUBENCH_REPO_ROOT:-$PWD}"
python_bin="${EDUBENCH_PYTHON:-$HOME/miniconda3/envs/llama_factory/bin/python}"
log_root="$repo_root/thesis_exp/outputs/exp54_rar_sft/rar_v2/preference_formal_runs/logs"
mkdir -p "$log_root"

case "$worker_id" in
  worker1)
    runs=(
      "P3_JOINT_SORC:42"
      "P1_FIELD_DPO:42"
      "P2_SORC_SCORE:42"
      "P1_SYN_SEED42:42"
    )
    ;;
  worker2)
    runs=(
      "P3_JOINT_SORC:43"
      "P1_FIELD_DPO:43"
      "P2_SORC_SCORE:43"
    )
    ;;
  worker3)
    runs=(
      "P3_JOINT_SORC:44"
      "P1_FIELD_DPO:44"
      "P2_SORC_SCORE:44"
    )
    ;;
  *)
    echo "unknown worker: $worker_id" >&2
    exit 2
    ;;
esac

cd "$repo_root"
export CUDA_VISIBLE_DEVICES="$cuda_index"
export PYTHONUNBUFFERED=1

for specification in "${runs[@]}"; do
  arm="${specification%%:*}"
  seed="${specification##*:}"
  log_path="$log_root/${arm,,}_seed_${seed}.log"
  echo "START arm=$arm seed=$seed gpu=$cuda_index uuid=$cuda_uuid"
  "$python_bin" -m thesis_exp.exp54_rar_sft.train_sorc_dpo_formal \
    --arm "$arm" \
    --seed "$seed" \
    --cuda-device-uuid "$cuda_uuid" \
    --execute 2>&1 | tee "$log_path"
  echo "COMPLETE arm=$arm seed=$seed"
done

echo "WORKER_COMPLETE worker=$worker_id"
