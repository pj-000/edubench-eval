#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-llama_factory}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"
VARIANT="${1:-${VARIANT:-qmr_meta}}"
SEED="${SEED:-42}"
GPU="${GPU:-6}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
MAX_LENGTH_QUALITY="${MAX_LENGTH_QUALITY:-2048}"
MAX_LENGTH_BOUNDARY="${MAX_LENGTH_BOUNDARY:-768}"
SAVE_BEST_BY="${SAVE_BEST_BY:-dev_mae}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

case "${VARIANT}" in
  global|metric_rubric|qmr|qmr_meta) ;;
  *)
    echo "ERROR: unknown Exp16A variant '${VARIANT}'" >&2
    echo "Allowed: global metric_rubric qmr qmr_meta" >&2
    exit 1
    ;;
esac

if [[ -f "${HOME}/miniconda3/bin/activate" ]]; then
  if [[ "${CONDA_ENV}" != "base" && ! -d "${HOME}/miniconda3/envs/${CONDA_ENV}" ]]; then
    echo "WARNING: conda env ${CONDA_ENV} was not found; using current shell." >&2
  elif ! source "${HOME}/miniconda3/bin/activate" "${CONDA_ENV}"; then
    echo "WARNING: could not activate conda env ${CONDA_ENV}; using current shell." >&2
  fi
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_VISIBLE_DEVICES="${GPU}"

OUTPUT_DIR="thesis_exp/outputs/exp16_boundary_linking/runs/${VARIANT}/seed_${SEED}"

cat <<CONFIG
Exp16A boundary linking scout
VARIANT=${VARIANT}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}
SEED=${SEED}
GPU=${GPU}
EPOCHS=${EPOCHS}
BATCH_SIZE=${BATCH_SIZE}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS}
OUTPUT_DIR=${OUTPUT_DIR}
CONFIG

python -m thesis_exp.src.edujudge.exp16_boundary_linking.train_boundary_linking \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --train_path thesis_exp/data/splits/question_seed42/train.jsonl \
  --dev_path thesis_exp/data/splits/question_seed42/dev.jsonl \
  --test_path thesis_exp/data/splits/question_seed42/test.jsonl \
  --output_dir "${OUTPUT_DIR}" \
  --variant "${VARIANT}" \
  --max_length_quality "${MAX_LENGTH_QUALITY}" \
  --max_length_boundary "${MAX_LENGTH_BOUNDARY}" \
  --batch_size "${BATCH_SIZE}" \
  --grad_accum_steps "${GRAD_ACCUM_STEPS}" \
  --epochs "${EPOCHS}" \
  --learning_rate "${LEARNING_RATE}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --seed "${SEED}" \
  --freeze_encoder false \
  --eval_every_epoch \
  --save_best_by "${SAVE_BEST_BY}" \
  --trust_remote_code

python -m thesis_exp.src.edujudge.exp16_boundary_linking.analyze_boundaries \
  --predictions_path "${OUTPUT_DIR}/predictions_dev.jsonl" \
  --output_dir "${OUTPUT_DIR}/analysis_dev"

# To run the other scout variants explicitly:
# ./thesis_exp/scripts/run_exp16a_boundary_linking_scout.sh global
# ./thesis_exp/scripts/run_exp16a_boundary_linking_scout.sh metric_rubric
# ./thesis_exp/scripts/run_exp16a_boundary_linking_scout.sh qmr
# ./thesis_exp/scripts/run_exp16a_boundary_linking_scout.sh qmr_meta
