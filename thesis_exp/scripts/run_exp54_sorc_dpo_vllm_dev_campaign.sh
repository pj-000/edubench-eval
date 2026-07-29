#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${EDUBENCH_REPO_ROOT:-/home/jpang/edubench-eval-exp2}"
PYTHON="${EDUBENCH_VLLM_PYTHON:-/home/jpang/exp54_vllm063_shared/bin/python}"
ROOT="${REPO_DIR}/thesis_exp/outputs/exp54_rar_sft/rar_v2"
DEV_ROOT="${ROOT}/preference_dev_runs_vllm"
LOG_ROOT="${ROOT}/preference_dev_logs_vllm"
PROGRESS_LOG="${LOG_ROOT}/progress.log"
GPUS=(1 2 3)
TOTAL=10

cd "${REPO_DIR}"
mkdir -p "${LOG_ROOT}"
: >"${PROGRESS_LOG}"

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
    exit 20
  fi
}

for gpu in "${GPUS[@]}"; do
  check_gpu_idle "${gpu}"
done

report_progress() {
  local completed percent
  completed="$(find "${DEV_ROOT}" -name metrics.json -type f 2>/dev/null |
    wc -l)"
  percent="$("${PYTHON}" -c \
    "print(f'{int(${completed}) / ${TOTAL} * 100:.1f}')")"
  printf "EXP54_SORC_DPO_DEV_PROGRESS completed=%s/%s percent=%s time=%s\n" \
    "${completed}" "${TOTAL}" "${percent}" \
    "$(date --iso-8601=seconds)" | tee -a "${PROGRESS_LOG}"
}

run_one() {
  local gpu="$1"
  local arm="$2"
  local seed="$3"
  local output_dir="${DEV_ROOT}/${arm,,}/seed_${seed}"
  if [[ -f "${output_dir}/metrics.json" ]]; then
    printf "SKIP arm=%s seed=%s gpu=%s\n" "${arm}" "${seed}" "${gpu}"
    return
  fi
  if [[ -e "${output_dir}" ]]; then
    printf "Partial output exists: %s\n" "${output_dir}" >&2
    return 21
  fi
  printf "START arm=%s seed=%s gpu=%s time=%s\n" \
    "${arm}" "${seed}" "${gpu}" "$(date --iso-8601=seconds)"
  "${PYTHON}" -u -m \
    thesis_exp.exp54_rar_sft.run_sorc_dpo_dev_inference_vllm \
    --arm "${arm}" \
    --seed "${seed}" \
    --output-root "${DEV_ROOT}" \
    --gpu-memory-utilization 0.90
  printf "COMPLETE arm=%s seed=%s gpu=%s time=%s\n" \
    "${arm}" "${seed}" "${gpu}" "$(date --iso-8601=seconds)"
  report_progress
}

run_worker1() {
  run_one 1 P1_FIELD_DPO 42
  run_one 1 P2_SORC_SCORE 42
  run_one 1 P3_JOINT_SORC 42
  run_one 1 P1_SYN_SEED42 42
}

run_worker2() {
  run_one 2 P1_FIELD_DPO 43
  run_one 2 P2_SORC_SCORE 43
  run_one 2 P3_JOINT_SORC 43
}

run_worker3() {
  run_one 3 P1_FIELD_DPO 44
  run_one 3 P2_SORC_SCORE 44
  run_one 3 P3_JOINT_SORC 44
}

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_USE_V1=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

pids=()
for worker in 1 2 3; do
  (
    export CUDA_VISIBLE_DEVICES="${worker}"
    export TORCHINDUCTOR_CACHE_DIR="/tmp/exp54_sorc_vllm_gpu${worker}"
    export TRITON_CACHE_DIR="/tmp/exp54_sorc_vllm_triton_gpu${worker}"
    mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"
    "run_worker${worker}"
  ) >"${LOG_ROOT}/worker_gpu${worker}.log" 2>&1 &
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
  printf "EXP54_SORC_DPO_DEV_CAMPAIGN_FAILED status=%s\n" "${status}" >&2
  exit "${status}"
fi

report_progress
printf "EXP54_SORC_DPO_DEV_CAMPAIGN_COMPLETE time=%s\n" \
  "$(date --iso-8601=seconds)"
