#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${EDUBENCH_REPO_ROOT:-/home/jpang/edubench-eval-exp2}"
PYTHON="${EDUBENCH_VLLM_PYTHON:-/home/jpang/exp54_vllm063_shared/bin/python}"
ROOT="${REPO_DIR}/thesis_exp/outputs/exp54_rar_sft/rar_v2"
CAMPAIGN_ROOT="${EDUBENCH_TEST_CAMPAIGN_ROOT:-${ROOT}/sorc_dpo_one_time_test_v1}"
TEST_PATH="${CAMPAIGN_ROOT}/isolated_input/test.jsonl"
RUN_ROOT="${CAMPAIGN_ROOT}/runs"
RESULT_ROOT="${CAMPAIGN_ROOT}/final_results"
LOG_ROOT="${CAMPAIGN_ROOT}/logs"
CONFIRMATION="${EDUBENCH_ONE_TIME_TEST_CONFIRMATION:-}"
read -r -a GPUS <<<"${EDUBENCH_TEST_GPUS:-0 1 2 3}"

if [[ "${CONFIRMATION}" != "EXECUTE_FROZEN_EXP54_ONE_TIME_TEST_ONCE" ]]; then
  printf "Missing exact one-time test confirmation.\n" >&2
  exit 40
fi
if (( ${#GPUS[@]} != 4 )); then
  printf "Exactly four idle GPU IDs are required.\n" >&2
  exit 41
fi
if [[ -e "${CAMPAIGN_ROOT}" ]]; then
  printf "One-time test campaign path already exists: %s\n" \
    "${CAMPAIGN_ROOT}" >&2
  exit 42
fi

cd "${REPO_DIR}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_USE_V1=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

check_gpu_idle() {
  local gpu="$1"
  local values memory utilization
  values="$(nvidia-smi --query-gpu=memory.used,utilization.gpu \
    --format=csv,noheader,nounits -i "${gpu}")"
  memory="${values%%,*}"
  utilization="${values##*,}"
  memory="${memory// /}"
  utilization="${utilization// /}"
  if (( memory > 1024 || utilization > 5 )); then
    printf "GPU %s is not idle: memory=%s MiB utilization=%s%%\n" \
      "${gpu}" "${memory}" "${utilization}" >&2
    exit 43
  fi
}

for gpu in "${GPUS[@]}"; do
  check_gpu_idle "${gpu}"
done

"${PYTHON}" -m \
  thesis_exp.exp54_rar_sft.sorc_dpo_test_execution_contract \
  preflight \
  --repo-root "${REPO_DIR}"

"${PYTHON}" -m \
  thesis_exp.exp54_rar_sft.sorc_dpo_test_execution_contract \
  materialize \
  --repo-root "${REPO_DIR}" \
  --destination "${TEST_PATH}"

mkdir -p "${LOG_ROOT}"

run_arm() {
  local arm="$1"
  local gpu="$2"
  local seed
  export CUDA_VISIBLE_DEVICES="${gpu}"
  export TORCHINDUCTOR_CACHE_DIR="/tmp/exp54_test_vllm_gpu${gpu}"
  export TRITON_CACHE_DIR="/tmp/exp54_test_vllm_triton_gpu${gpu}"
  mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"
  for seed in 42 43 44; do
    printf "START arm=%s seed=%s time=%s\n" \
      "${arm}" "${seed}" "$(date --iso-8601=seconds)"
    "${PYTHON}" -u -m \
      thesis_exp.exp54_rar_sft.run_sorc_dpo_test_inference_vllm \
      --arm "${arm}" \
      --seed "${seed}" \
      --repo-root "${REPO_DIR}" \
      --test-path "${TEST_PATH}" \
      --output-root "${RUN_ROOT}" \
      --max-num-seqs 128 \
      --gpu-memory-utilization 0.94
    printf "COMPLETE arm=%s seed=%s time=%s\n" \
      "${arm}" "${seed}" "$(date --iso-8601=seconds)"
  done
}

arms=(P0_R3_SFT P1_FIELD_DPO P2_SORC_SCORE P3_JOINT_SORC)
pids=()
for slot in 0 1 2 3; do
  gpu="${GPUS[$slot]}"
  arm="${arms[$slot]}"
  (
    run_arm "${arm}" "${gpu}"
  ) >"${LOG_ROOT}/worker_gpu${gpu}_${arm}.log" 2>&1 &
  pids+=("$!")
done

status=0
set +e
for pid in "${pids[@]}"; do
  wait "${pid}"
  worker_status="$?"
  if (( worker_status != 0 )); then
    status="${worker_status}"
  fi
done
set -e
if (( status != 0 )); then
  printf "EXP54_ONE_TIME_TEST_CAMPAIGN_FAILED status=%s\n" \
    "${status}" >&2
  printf "Do not inspect partial predictions and do not retry this campaign.\n" \
    >&2
  exit "${status}"
fi

"${PYTHON}" -m \
  thesis_exp.exp54_rar_sft.sorc_dpo_test_execution_contract \
  receipts \
  --output-root "${RUN_ROOT}"

"${PYTHON}" -m \
  thesis_exp.exp54_rar_sft.collect_sorc_dpo_test_results \
  --test-root "${RUN_ROOT}" \
  --output-dir "${RESULT_ROOT}"

printf "EXP54_ONE_TIME_TEST_AND_FINAL_AGGREGATION_COMPLETE time=%s\n" \
  "$(date --iso-8601=seconds)"
