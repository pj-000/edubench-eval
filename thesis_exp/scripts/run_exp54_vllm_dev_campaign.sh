#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/jpang/edubench-eval-exp2"
PYTHON="/home/jpang/exp54_vllm063_shared/bin/python"
ROOT="${REPO_DIR}/thesis_exp/outputs/exp54_rar_sft/rar_v2"
TRAINING_ROOT="${ROOT}/formal_runs"
DEV_ROOT="${ROOT}/dev_runs_vllm"
LOG_ROOT="${ROOT}/dev_logs_vllm"
SUMMARY_DIR="${ROOT}/dev_summary_vllm"
PROGRESS_LOG="${LOG_ROOT}/progress.log"
ARMS=(S0 R1 R2 R3)
SEEDS=(42 43 44)
EPOCHS=(1 2 3)
GPUS=(4 6 7)
TOTAL=36

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
  printf "EXP54_VLLM_PROGRESS completed=%s/%s percent=%s time=%s\n" \
    "${completed}" "${TOTAL}" "${percent}" \
    "$(date --iso-8601=seconds)" | tee -a "${PROGRESS_LOG}"
}

run_worker() {
  local gpu="$1"
  local worker_index="$2"
  local task_index=0
  local arm seed epoch output_dir
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES="${gpu}"
  export VLLM_USE_V1=1
  export TOKENIZERS_PARALLELISM=false
  export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
  export TORCHINDUCTOR_CACHE_DIR="/tmp/exp54_vllm_inductor_gpu${gpu}"
  export TRITON_CACHE_DIR="/tmp/exp54_vllm_triton_gpu${gpu}"
  mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"
  for arm in "${ARMS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      for epoch in "${EPOCHS[@]}"; do
        if (( task_index % ${#GPUS[@]} != worker_index )); then
          task_index=$((task_index + 1))
          continue
        fi
        task_index=$((task_index + 1))
        output_dir="${DEV_ROOT}/${arm,,}/seed${seed}/epoch${epoch}"
        if [[ -f "${output_dir}/metrics.json" ]]; then
          printf "EXP54_VLLM_SKIP arm=%s seed=%s epoch=%s gpu=%s\n" \
            "${arm}" "${seed}" "${epoch}" "${gpu}"
          continue
        fi
        if [[ -e "${output_dir}" ]]; then
          printf "Partial output exists: %s\n" "${output_dir}" >&2
          return 21
        fi
        printf "EXP54_VLLM_START arm=%s seed=%s epoch=%s gpu=%s time=%s\n" \
          "${arm}" "${seed}" "${epoch}" "${gpu}" \
          "$(date --iso-8601=seconds)"
        "${PYTHON}" -u -m \
          thesis_exp.exp54_rar_sft.run_dev_inference_vllm \
          --arm "${arm}" \
          --seed "${seed}" \
          --epoch "${epoch}" \
          --training-root "${TRAINING_ROOT}" \
          --output-root "${DEV_ROOT}"
        printf "EXP54_VLLM_COMPLETE arm=%s seed=%s epoch=%s gpu=%s time=%s\n" \
          "${arm}" "${seed}" "${epoch}" "${gpu}" \
          "$(date --iso-8601=seconds)"
        report_progress
      done
    done
  done
}

pids=()
for index in 0 1 2; do
  gpu="${GPUS[${index}]}"
  run_worker "${gpu}" "${index}" \
    >"${LOG_ROOT}/worker_gpu${gpu}.log" 2>&1 &
  worker_pid="$!"
  pids+=("${worker_pid}")
  printf "EXP54_VLLM_WORKER_STARTED gpu=%s pid=%s\n" \
    "${gpu}" "${worker_pid}"
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
  printf "EXP54_VLLM_CAMPAIGN_FAILED status=%s\n" "${status}" >&2
  exit "${status}"
fi

report_progress
if [[ -e "${SUMMARY_DIR}" ]]; then
  printf "Summary directory already exists: %s\n" "${SUMMARY_DIR}" >&2
  exit 22
fi
"${PYTHON}" -m thesis_exp.exp54_rar_sft.collect_dev_results \
  --dev-root "${DEV_ROOT}" \
  --output-dir "${SUMMARY_DIR}"
printf "EXP54_VLLM_CAMPAIGN_COMPLETE report=%s time=%s\n" \
  "${SUMMARY_DIR}/report.md" "$(date --iso-8601=seconds)"
