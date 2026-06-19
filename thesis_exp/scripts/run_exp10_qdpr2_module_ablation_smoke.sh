#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-llama_factory}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"
QD_B1_CHECKPOINT_DIR="${QD_B1_CHECKPOINT_DIR:-thesis_exp/artifacts/exp06_question_disjoint_baselines/checkpoints/QD-B1_human_only_L1_weighted_ordinal/best}"
EXP10_SMOKE_GPU="${EXP10_SMOKE_GPU:-6}"
EXP10_SMOKE_ABLATIONS="${EXP10_SMOKE_ABLATIONS:-full_qdpr2 no_pair no_pair_same_pair_batches no_anchor}"
EXP10_SMOKE_MAX_TRAIN_SAMPLES="${EXP10_SMOKE_MAX_TRAIN_SAMPLES:-8}"
EXP10_SMOKE_MAX_EVAL_SAMPLES="${EXP10_SMOKE_MAX_EVAL_SAMPLES:-8}"
EXP10_SMOKE_MAX_TRAIN_PAIRS="${EXP10_SMOKE_MAX_TRAIN_PAIRS:-8}"
EXP10_SMOKE_MAX_DEV_PAIRS="${EXP10_SMOKE_MAX_DEV_PAIRS:-8}"
EXP10_SMOKE_EPOCHS="${EXP10_SMOKE_EPOCHS:-0.05}"
EXP10_SMOKE_BATCH_SIZE="${EXP10_SMOKE_BATCH_SIZE:-2}"
EXP10_SMOKE_EVAL_BATCH_SIZE="${EXP10_SMOKE_EVAL_BATCH_SIZE:-2}"
EXP10_SMOKE_MAX_LENGTH="${EXP10_SMOKE_MAX_LENGTH:-512}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f "${HOME}/miniconda3/bin/activate" ]]; then
  if [[ "${CONDA_ENV}" != "base" && ! -d "${HOME}/miniconda3/envs/${CONDA_ENV}" ]]; then
    echo "WARNING: conda env ${CONDA_ENV} was not found; using current shell." >&2
  elif ! source "${HOME}/miniconda3/bin/activate" "${CONDA_ENV}"; then
    echo "WARNING: could not activate conda env ${CONDA_ENV}; using current shell." >&2
  fi
else
  echo "WARNING: ${HOME}/miniconda3/bin/activate was not found; using current shell." >&2
fi

if [[ ! -f "${QD_B1_CHECKPOINT_DIR}/state_dict.pt" ]]; then
  echo "BLOCKED_MISSING_QDB1_CHECKPOINT: ${QD_B1_CHECKPOINT_DIR}" >&2
  exit 1
fi

export FORMAL_RUN=0
export REQUIRE_CUDA
export MODEL_NAME_OR_PATH
export QD_B1_CHECKPOINT_DIR
export EXP10_SMOKE_ABLATIONS
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF

cat <<CONFIG
Exp10 QD-PR2 module ablation smoke
FORMAL_RUN=${FORMAL_RUN}
REQUIRE_CUDA=${REQUIRE_CUDA}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}
QD_B1_CHECKPOINT_DIR=${QD_B1_CHECKPOINT_DIR}
EXP10_SMOKE_GPU=${EXP10_SMOKE_GPU}
EXP10_SMOKE_ABLATIONS=${EXP10_SMOKE_ABLATIONS}
EXP10_SMOKE_MAX_TRAIN_SAMPLES=${EXP10_SMOKE_MAX_TRAIN_SAMPLES}
EXP10_SMOKE_MAX_EVAL_SAMPLES=${EXP10_SMOKE_MAX_EVAL_SAMPLES}
EXP10_SMOKE_MAX_TRAIN_PAIRS=${EXP10_SMOKE_MAX_TRAIN_PAIRS}
EXP10_SMOKE_MAX_DEV_PAIRS=${EXP10_SMOKE_MAX_DEV_PAIRS}
CONFIG

python -m thesis_exp.src.edujudge.exp10_qdpr2_module_ablation.sanity_check_exp10_setup

read -r -a ABLATIONS <<< "${EXP10_SMOKE_ABLATIONS}"
mkdir -p thesis_exp/outputs/exp10_qdpr2_module_ablation/logs
for ablation in "${ABLATIONS[@]}"; do
  config_path="thesis_exp/configs/exp10_qdpr2_module_ablation/exp10_${ablation}.yaml"
  output_dir="thesis_exp/outputs/exp10_qdpr2_module_ablation/smoke_test/${ablation}"
  checkpoint_dir="thesis_exp/artifacts/exp10_qdpr2_module_ablation/smoke_test/checkpoints/${ablation}"
  log_path="thesis_exp/outputs/exp10_qdpr2_module_ablation/logs/smoke_EXP10_${ablation}_gpu${EXP10_SMOKE_GPU}.log"
  if [[ ! -f "${config_path}" ]]; then
    echo "Missing config: ${config_path}" >&2
    exit 1
  fi
  rm -rf "${output_dir}" "${checkpoint_dir}"
  echo "Starting smoke ${ablation} on GPU ${EXP10_SMOKE_GPU}; log=${log_path}"
  (
    export CUDA_VISIBLE_DEVICES="${EXP10_SMOKE_GPU}"
    python -m thesis_exp.src.edujudge.exp09_pairwise_ordinal.train_qdpr2_anchored_pairwise \
      --config_path "${config_path}" \
      --model_name_or_path "${MODEL_NAME_OR_PATH}" \
      --qd_b1_checkpoint_dir "${QD_B1_CHECKPOINT_DIR}" \
      --output_dir "${output_dir}" \
      --checkpoint_output_dir "${checkpoint_dir}" \
      --smoke \
      --max_train_samples "${EXP10_SMOKE_MAX_TRAIN_SAMPLES}" \
      --max_eval_samples "${EXP10_SMOKE_MAX_EVAL_SAMPLES}" \
      --max_train_pairs "${EXP10_SMOKE_MAX_TRAIN_PAIRS}" \
      --max_dev_pairs "${EXP10_SMOKE_MAX_DEV_PAIRS}" \
      --num_train_epochs "${EXP10_SMOKE_EPOCHS}" \
      --per_device_train_batch_size "${EXP10_SMOKE_BATCH_SIZE}" \
      --per_device_eval_batch_size "${EXP10_SMOKE_EVAL_BATCH_SIZE}" \
      --gradient_accumulation_steps 1 \
      --max_length "${EXP10_SMOKE_MAX_LENGTH}" \
      --bf16 false \
      --no_progress_bar
  ) 2>&1 | tee "${log_path}"
done

python -m thesis_exp.src.edujudge.exp10_qdpr2_module_ablation.smoke_check_exp10
cat thesis_exp/outputs/exp10_qdpr2_module_ablation/reports/exp10_smoke_check.md
