#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/jpang/edubench-eval-exp2"
PYTHON="/home/jpang/miniconda3/envs/llama_factory/bin/python"
OUTPUT_ROOT="${REPO_DIR}/thesis_exp/outputs/exp54_rar_sft/rar_v2/preference_lr5e6_followup"
LOG_ROOT="${OUTPUT_ROOT}/logs"
MODULE="thesis_exp.exp54_rar_sft.train_sorc_dpo_lr_followup"
SEEDS=(42 43 44)
GPUS=(0 1 2)
UUIDS=(
  "GPU-a8c16b60-d3f9-0f99-91f2-06e73154119e"
  "GPU-6094556a-56f8-5784-14ad-6369b980d4ca"
  "GPU-353fd55a-1293-802a-1fd5-29f37dcd1fc3"
)
ARMS=(
  "P1_FIELD_DPO"
  "P2_SORC_SCORE"
  "P3_JOINT_SORC"
)

mkdir -p "${LOG_ROOT}"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}"

verify_result() {
  local arm="$1"
  local seed="$2"
  local arm_lower="${arm,,}"
  "${PYTHON}" - "${OUTPUT_ROOT}/train/${arm_lower}/seed_${seed}/result.json" \
    "${arm}" "${seed}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
arm = sys.argv[2]
seed = int(sys.argv[3])
value = json.loads(path.read_text(encoding="utf-8"))
expected = {
    "schema_version": "exp54-sorc-dpo-lr5e6-multiseed-result-v1",
    "status": "SORC_DPO_FORMAL_TRAINING_COMPLETE",
    "arm": arm,
    "seed": seed,
    "learning_rate": 5e-6,
    "optimizer_steps": 27,
    "preference_epochs": 1,
    "dev_accessed": False,
    "test_accessed": False,
}
for field, expected_value in expected.items():
    if value.get(field) != expected_value:
        raise SystemExit(f"{path}: {field} differs")
PY
}

for arm in "${ARMS[@]}"; do
  arm_lower="${arm,,}"
  pids=()
  for index in 0 1 2; do
    seed="${SEEDS[${index}]}"
    gpu="${GPUS[${index}]}"
    uuid="${UUIDS[${index}]}"
    output_dir="${OUTPUT_ROOT}/train/${arm_lower}/seed_${seed}"
    log_path="${LOG_ROOT}/${arm_lower}_seed${seed}.log"
    if [[ -e "${output_dir}" ]]; then
      printf "Output already exists: %s\n" "${output_dir}" >&2
      exit 1
    fi
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -m "${MODULE}" \
      --arm "${arm}" \
      --seed "${seed}" \
      --cuda-device-uuid "${uuid}" \
      >"${log_path}" 2>&1 &
    pids+=("$!")
  done

  stage_failed=0
  for index in 0 1 2; do
    if ! wait "${pids[${index}]}"; then
      stage_failed=1
    fi
  done
  if [[ "${stage_failed}" -ne 0 ]]; then
    printf "LR5e-6 stage failed: %s\n" "${arm}" >&2
    exit 1
  fi
  for seed in "${SEEDS[@]}"; do
    verify_result "${arm}" "${seed}"
  done
  printf "LR5e-6 stage complete: %s\n" "${arm}"
done

printf "EXP54_LR5E6_MULTI_SEED_TRAINING_COMPLETE\n"
