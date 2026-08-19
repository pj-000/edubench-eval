#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${EDUBENCH_REPO_ROOT:-/home/jpang/edubench-eval-exp2}"
PYTHON="${EDUBENCH_VLLM_PYTHON:-/home/jpang/exp54_vllm063_shared/bin/python}"
ROOT="${REPO_DIR}/thesis_exp/outputs/exp54_rar_sft/rar_v2"
FOLLOWUP_ROOT="${ROOT}/preference_lr5e6_followup"
TRAIN_ROOT="${FOLLOWUP_ROOT}/train"
AUDIT_REPORT="${FOLLOWUP_ROOT}/training_audit_report.json"
DEV_ROOT="${FOLLOWUP_ROOT}/dev"
LOG_ROOT="${FOLLOWUP_ROOT}/dev_logs"
PROGRESS_LOG="${LOG_ROOT}/progress.log"
GPUS=(1 2 3)
TOTAL=9

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
  printf "EXP54_LR5E6_DEV_PROGRESS completed=%s/%s percent=%s time=%s\n" \
    "${completed}" "${TOTAL}" "${percent}" \
    "$(date --iso-8601=seconds)" | tee -a "${PROGRESS_LOG}"
}

run_one() {
  local arm="$1"
  local seed="$2"
  local output_dir="${DEV_ROOT}/${arm,,}/seed_${seed}"
  if [[ -f "${output_dir}/metrics.json" ]]; then
    printf "SKIP arm=%s seed=%s\n" "${arm}" "${seed}"
    return
  fi
  if [[ -e "${output_dir}" ]]; then
    printf "Partial output exists: %s\n" "${output_dir}" >&2
    return 21
  fi
  printf "START arm=%s seed=%s time=%s\n" \
    "${arm}" "${seed}" "$(date --iso-8601=seconds)"
  "${PYTHON}" -u -m \
    thesis_exp.exp54_rar_sft.run_sorc_dpo_dev_inference_vllm \
    --arm "${arm}" \
    --seed "${seed}" \
    --training-root "${TRAIN_ROOT}" \
    --audit-report "${AUDIT_REPORT}" \
    --audit-contract lr5e6_followup \
    --output-root "${DEV_ROOT}" \
    --max-num-seqs 128 \
    --gpu-memory-utilization 0.94
  printf "COMPLETE arm=%s seed=%s time=%s\n" \
    "${arm}" "${seed}" "$(date --iso-8601=seconds)"
  report_progress
}

run_worker() {
  local seed="$1"
  run_one P1_FIELD_DPO "${seed}"
  run_one P2_SORC_SCORE "${seed}"
  run_one P3_JOINT_SORC "${seed}"
}

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_USE_V1=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

pids=()
for slot in 0 1 2; do
  gpu="${GPUS[$slot]}"
  seed="$((42 + slot))"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export TORCHINDUCTOR_CACHE_DIR="/tmp/exp54_lr5e6_vllm_gpu${gpu}"
    export TRITON_CACHE_DIR="/tmp/exp54_lr5e6_vllm_triton_gpu${gpu}"
    mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"
    run_worker "${seed}"
  ) >"${LOG_ROOT}/worker_gpu${gpu}.log" 2>&1 &
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
  printf "EXP54_LR5E6_DEV_CAMPAIGN_FAILED status=%s\n" "${status}" >&2
  exit "${status}"
fi

report_progress
printf "EXP54_LR5E6_DEV_CAMPAIGN_COMPLETE time=%s\n" \
  "$(date --iso-8601=seconds)"
