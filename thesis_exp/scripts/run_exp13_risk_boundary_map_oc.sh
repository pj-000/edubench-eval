#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-llama_factory}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"
QD_B1_CHECKPOINT_DIR="${QD_B1_CHECKPOINT_DIR:-thesis_exp/artifacts/exp06_question_disjoint_baselines/checkpoints/QD-B1_human_only_L1_weighted_ordinal/best}"
CONFIG_DIR="${CONFIG_DIR:-thesis_exp/configs/exp13_risk_boundary_map_oc}"
SEEDS="${SEEDS:-42}"
GPU_LIST="${GPU_LIST:-6 7}"
EPOCHS="${EPOCHS:-3}"
EXP13_RUNS="${EXP13_RUNS:-point_pair_proj_l2h_lam0p10 point_pair_proj_l2h_lam0p20 point_pair_proj_l2h_lam0p40 point_pair_proj_l2h_label2_w1p5_lam0p20 point_pair_proj_l2h_lam0p20_no_mono score_proj_l2h_lam0p20 map_oc_full_l2h_lam0p20 score_proj_t3_brier_lam0p05}"
EXP13_PARALLEL="${EXP13_PARALLEL:-1}"
MODE="${MODE:-scout}"
if [[ -z "${EVAL_TEST+x}" ]]; then
  if [[ "${MODE}" == "formal" ]]; then
    EVAL_TEST="1"
  else
    EVAL_TEST="0"
  fi
fi
FORMAL_RUN="${FORMAL_RUN:-1}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
RESET_RUN_DIR="${RESET_RUN_DIR:-0}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-32}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
BF16="${BF16:-auto}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
LOG_STEPS="${LOG_STEPS:-5}"
NO_PROGRESS_BAR="${NO_PROGRESS_BAR:-0}"
SELECTION_DELTA="${SELECTION_DELTA:-0.005}"
GAMMA="${GAMMA:-4.0}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ "${MODE}" != "scout" && "${MODE}" != "formal" ]]; then
  echo "ERROR: MODE must be scout or formal, got ${MODE}" >&2
  exit 1
fi
if [[ "${MODE}" == "scout" && "${EVAL_TEST}" != "0" ]]; then
  echo "ERROR: scout mode must run with EVAL_TEST=0." >&2
  exit 1
fi
if [[ "${MODE}" == "formal" && "${EVAL_TEST}" != "1" ]]; then
  echo "ERROR: formal mode must run with EVAL_TEST=1." >&2
  exit 1
fi
if [[ "${FORMAL_RUN}" != "1" ]]; then
  echo "ERROR: Exp13 training script must run with FORMAL_RUN=1." >&2
  exit 1
fi
if [[ -n "${MAX_TRAIN_SAMPLES:-}" || -n "${MAX_EVAL_SAMPLES:-}" || -n "${MAX_TRAIN_PAIRS:-}" || -n "${MAX_DEV_PAIRS:-}" ]]; then
  echo "ERROR: Exp13 formal/scout runs cannot use max sample or max pair limits." >&2
  exit 1
fi

if [[ -f "${HOME}/miniconda3/bin/activate" ]]; then
  if [[ "${CONDA_ENV}" != "base" && ! -d "${HOME}/miniconda3/envs/${CONDA_ENV}" ]]; then
    echo "WARNING: conda env ${CONDA_ENV} was not found; using current shell." >&2
  elif ! source "${HOME}/miniconda3/bin/activate" "${CONDA_ENV}"; then
    echo "WARNING: could not activate conda env ${CONDA_ENV}; using current shell." >&2
  fi
else
  echo "WARNING: ${HOME}/miniconda3/bin/activate was not found; using current shell." >&2
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF
export FORMAL_RUN
export REQUIRE_CUDA

read -r -a SEED_ARRAY <<< "${SEEDS}"
read -r -a GPU_ARRAY <<< "${GPU_LIST//,/ }"
read -r -a RUN_ARRAY <<< "${EXP13_RUNS}"
if [[ "${#SEED_ARRAY[@]}" -lt 1 ]]; then
  echo "ERROR: SEEDS must contain at least one seed." >&2
  exit 1
fi
if [[ "${#GPU_ARRAY[@]}" -lt 1 ]]; then
  echo "ERROR: GPU_LIST must contain at least one GPU id." >&2
  exit 1
fi

EFFECTIVE_BATCH_SIZE=$((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
if [[ "${EFFECTIVE_BATCH_SIZE}" != "128" ]]; then
  echo "ERROR: Exp13 formal effective batch size must remain 128, got ${EFFECTIVE_BATCH_SIZE}." >&2
  exit 1
fi
if [[ ! -f "${QD_B1_CHECKPOINT_DIR}/state_dict.pt" ]]; then
  echo "BLOCKED_MISSING_QDB1_CHECKPOINT: ${QD_B1_CHECKPOINT_DIR}" >&2
  exit 1
fi

cat <<CONFIG
Exp13 risk-boundary MAP-OC
MODE=${MODE}
EVAL_TEST=${EVAL_TEST}
FORMAL_RUN=${FORMAL_RUN}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}
QD_B1_CHECKPOINT_DIR=${QD_B1_CHECKPOINT_DIR}
SEEDS=${SEEDS}
GPU_LIST=${GPU_LIST}
EPOCHS=${EPOCHS}
EXP13_RUNS=${EXP13_RUNS}
SKIP_COMPLETED=${SKIP_COMPLETED}
RESET_RUN_DIR=${RESET_RUN_DIR}
EXP13_PARALLEL=${EXP13_PARALLEL}
SELECTION_RULE=mae_guard_p_gt_3_low_mean
SELECTION_DELTA=${SELECTION_DELTA}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE}
PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}
effective batch size=${EFFECTIVE_BATCH_SIZE}
CONFIG

python -m thesis_exp.src.edujudge.exp13_risk_boundary_map_oc.preflight_exp13 \
  --seeds "${SEEDS}" \
  --gpu_list "${GPU_LIST}" \
  --runs "${EXP13_RUNS}" \
  --epochs "${EPOCHS}" \
  --mode "${MODE}" \
  --eval_test "${EVAL_TEST}" \
  --selection_delta "${SELECTION_DELTA}" \
  --qd_b1_checkpoint_dir "${QD_B1_CHECKPOINT_DIR}"

progress_args=()
if [[ "${NO_PROGRESS_BAR}" == "1" ]]; then
  progress_args+=(--no_progress_bar)
fi
gc_args=()
if [[ "${GRADIENT_CHECKPOINTING}" == "1" ]]; then
  gc_args+=(--gradient_checkpointing)
fi
eval_test_args=(--no_eval_test)
if [[ "${EVAL_TEST}" == "1" ]]; then
  eval_test_args=(--eval_test --eval_each_epoch)
fi

config_for_run() {
  case "$1" in
    point_pair_proj_l2h_lam0p10) echo "${CONFIG_DIR}/exp13_point_pair_proj_l2h_lam0p10.yaml" ;;
    point_pair_proj_l2h_lam0p20) echo "${CONFIG_DIR}/exp13_point_pair_proj_l2h_lam0p20.yaml" ;;
    point_pair_proj_l2h_lam0p40) echo "${CONFIG_DIR}/exp13_point_pair_proj_l2h_lam0p40.yaml" ;;
    point_pair_proj_l2h_label2_w1p5_lam0p20) echo "${CONFIG_DIR}/exp13_point_pair_proj_l2h_label2_w1p5_lam0p20.yaml" ;;
    point_pair_proj_l2h_lam0p20_no_mono) echo "${CONFIG_DIR}/exp13_point_pair_proj_l2h_lam0p20_no_mono.yaml" ;;
    score_proj_l2h_lam0p20) echo "${CONFIG_DIR}/exp13_score_proj_l2h_lam0p20.yaml" ;;
    map_oc_full_l2h_lam0p20) echo "${CONFIG_DIR}/exp13_map_oc_full_l2h_lam0p20.yaml" ;;
    score_proj_t3_brier_lam0p05) echo "${CONFIG_DIR}/exp13_score_proj_t3_brier_lam0p05.yaml" ;;
    *) echo "ERROR: unknown Exp13 run '$1'" >&2; return 1 ;;
  esac
}

run_one() {
  local run_name="$1"
  local seed="$2"
  local gpu="$3"
  local config_path
  config_path="$(config_for_run "${run_name}")"
  local base_dir="thesis_exp/runs/exp13_risk_boundary_map_oc/${MODE}/${run_name}/seed_${seed}"
  local output_dir="${base_dir}/run"
  local checkpoint_dir="${base_dir}/checkpoints"
  local log_dir="${base_dir}/logs"
  local log_path="${log_dir}/train_exp13_${MODE}_${run_name}_seed_${seed}_gpu${gpu}.log"
  mkdir -p "${log_dir}"
  if [[ "${SKIP_COMPLETED}" == "1" && "${RESET_RUN_DIR}" != "1" && -f "${output_dir}/run_metadata.json" ]]; then
    if grep -q '"status": "completed"' "${output_dir}/run_metadata.json"; then
      echo "Skipping ${MODE}/${run_name} seed ${seed}: completed run found at ${output_dir}"
      return 0
    fi
  fi
  if [[ "${RESET_RUN_DIR}" == "1" ]]; then
    rm -rf "${base_dir}"
    mkdir -p "${log_dir}"
  fi
  echo "Starting Exp13 ${MODE}/${run_name} seed ${seed} on GPU ${gpu}; log=${log_path}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    python -m thesis_exp.src.edujudge.exp09_pairwise_ordinal.train_qdpr2_anchored_pairwise \
      --config_path "${config_path}" \
      --model_name_or_path "${MODEL_NAME_OR_PATH}" \
      --qd_b1_checkpoint_dir "${QD_B1_CHECKPOINT_DIR}" \
      --output_dir "${output_dir}" \
      --checkpoint_output_dir "${checkpoint_dir}" \
      --max_length "${MAX_LENGTH}" \
      --num_train_epochs "${EPOCHS}" \
      --learning_rate "${LEARNING_RATE}" \
      --weight_decay "${WEIGHT_DECAY}" \
      --warmup_ratio "${WARMUP_RATIO}" \
      --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
      --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
      --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
      --seed "${seed}" \
      --bf16 "${BF16}" \
      --log_steps "${LOG_STEPS}" \
      --save_each_epoch \
      --keep_epoch_checkpoints_local \
      --selection_rules_enabled \
      --soft_risk_gamma "${GAMMA}" \
      "${eval_test_args[@]}" \
      "${progress_args[@]}" \
      "${gc_args[@]}"
  ) 2>&1 | tee "${log_path}"
}

jobs=()
for run_name in "${RUN_ARRAY[@]}"; do
  for seed in "${SEED_ARRAY[@]}"; do
    jobs+=("${run_name}:${seed}")
  done
done

run_gpu_queue() {
  local gpu="$1"
  shift
  local queue=("$@")
  if [[ "${#queue[@]}" -eq 0 ]]; then
    echo "GPU ${gpu} queue is empty."
    return 0
  fi
  echo "GPU ${gpu} queue: ${queue[*]}"
  for item in "${queue[@]}"; do
    local run_name="${item%%:*}"
    local seed="${item##*:}"
    run_one "${run_name}" "${seed}" "${gpu}"
  done
}

pids=()
names=()
if [[ "${EXP13_PARALLEL}" == "1" ]]; then
  for gpu_index in "${!GPU_ARRAY[@]}"; do
    gpu="${GPU_ARRAY[gpu_index]}"
    queue=()
    for ((job_index=gpu_index; job_index<${#jobs[@]}; job_index+=${#GPU_ARRAY[@]})); do
      queue+=("${jobs[job_index]}")
    done
    run_gpu_queue "${gpu}" "${queue[@]}" &
    pids+=("$!")
    names+=("gpu_${gpu}")
  done

  for idx in "${!pids[@]}"; do
    if ! wait "${pids[idx]}"; then
      echo "Exp13 GPU queue failed: ${names[idx]}" >&2
      exit 1
    fi
  done
elif [[ "${EXP13_PARALLEL}" == "0" ]]; then
  echo "Running Exp13 queue sequentially with round-robin GPU assignment."
  for job_index in "${!jobs[@]}"; do
    item="${jobs[job_index]}"
    gpu="${GPU_ARRAY[$((job_index % ${#GPU_ARRAY[@]}))]}"
    run_name="${item%%:*}"
    seed="${item##*:}"
    run_one "${run_name}" "${seed}" "${gpu}"
  done
else
  echo "ERROR: EXP13_PARALLEL must be 0 or 1, got ${EXP13_PARALLEL}" >&2
  exit 1
fi

python -m thesis_exp.src.edujudge.exp13_risk_boundary_map_oc.collect_exp13_results \
  --mode "${MODE}" \
  --runs "${EXP13_RUNS}" \
  --delta "${SELECTION_DELTA}"
python -m thesis_exp.src.edujudge.exp13_risk_boundary_map_oc.readability_check_exp13
cat thesis_exp/outputs/exp13_risk_boundary_map_oc/reports/exp13_risk_boundary_map_oc_report.md
