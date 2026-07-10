#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f "${HOME}/miniconda3/bin/activate" ]]; then
  source "${HOME}/miniconda3/bin/activate" "${CONDA_ENV:-llama_factory}"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"
GPU_LIST="${GPU_LIST:-0}"
GPU="${GPU_LIST%% *}"
OUT_DIR="thesis_exp/exp17_low_score_evidence/outputs/exp27p_soft_target_reranker_seed42"
EXP27O_DIR="thesis_exp/exp17_low_score_evidence/outputs/exp27o_361_in_place_pilot_seed42"
RUN_ROOT="thesis_exp/runs/exp27p_soft_target_reranker_smoke"
ARTIFACT_ROOT="thesis_exp/artifacts/exp27p_soft_target_reranker_smoke"
LOG_DIR="${OUT_DIR}/logs_private/smoke"
VARIANTS=(
  v3_selective_soft_audit
  v0_original_unweighted
  v1_original_label_matched_weight
  v2_selective_hard_relabel
)

"${PYTHON_BIN}" -m thesis_exp.src.edujudge.exp27p.validate_exp27p_training_setup \
  --exp27o-dir "${EXP27O_DIR}" \
  --output-dir "${OUT_DIR}"

mkdir -p "${LOG_DIR}"
for variant in "${VARIANTS[@]}"; do
  train_jsonl="${EXP27O_DIR}/private/data/exp27o_${variant}_train.jsonl"
  run_dir="${RUN_ROOT}/${variant}/seed_42"
  artifact_dir="${ARTIFACT_ROOT}/${variant}/seed_42"
  rm -rf "${run_dir}" "${artifact_dir}"
  echo "Starting Exp27P smoke ${variant} on GPU ${GPU}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -m thesis_exp.src.edujudge.exp27p.train_exp27p_soft_target_reranker \
    --variant "${variant}" \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --train_jsonl "${train_jsonl}" \
    --run_dir "${run_dir}" \
    --checkpoint_output_dir "${artifact_dir}" \
    --num_train_epochs 0.05 \
    --max_train_samples 32 \
    --max_eval_samples 32 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --log_steps 1 \
    2>&1 | tee "${LOG_DIR}/${variant}.log"
done

"${PYTHON_BIN}" -m thesis_exp.src.edujudge.exp27p.validate_exp27p_training_setup \
  --exp27o-dir "${EXP27O_DIR}" \
  --output-dir "${OUT_DIR}" \
  --smoke-run-root "${RUN_ROOT}"

echo "Exp27P four-variant smoke completed on GPU ${GPU}."
