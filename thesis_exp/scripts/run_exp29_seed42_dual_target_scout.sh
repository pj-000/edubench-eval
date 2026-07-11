#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
if [[ -f "${HOME}/miniconda3/bin/activate" ]]; then source "${HOME}/miniconda3/bin/activate" "${CONDA_ENV:-llama_factory}"; fi
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"
DATA_ROOT="thesis_exp/exp17_low_score_evidence/outputs/exp29_dual_target_ce_seed42/private/datasets"
OUT_ROOT="thesis_exp/exp17_low_score_evidence/outputs/exp29_dual_target_ce_seed42"
CKPT_ROOT="thesis_exp/artifacts/exp29_dual_target_ce_seed42"
VARIANTS=(c1_audited_dual_target c2_selected_exposure_control c3_random_dual_target_control)
GPUS=(0 1 2)
pids=()
for i in "${!VARIANTS[@]}"; do
  variant="${VARIANTS[$i]}"; gpu="${GPUS[$i]}"; out="${OUT_ROOT}/runs/${variant}/seed_42"; ckpt="${CKPT_ROOT}/${variant}/seed_42"
  [[ -f "${DATA_ROOT}/${variant}/train.jsonl" && -f "${DATA_ROOT}/${variant}/dev.jsonl" ]] || { echo "Missing ${variant} data" >&2; exit 2; }
  [[ ! -f "${DATA_ROOT}/${variant}/test.jsonl" ]] || { echo "Test data forbidden" >&2; exit 2; }
  (
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -m thesis_exp.src.edujudge.exp02.train_ce_baseline \
      --model_name_or_path "${MODEL_NAME_OR_PATH}" --data_dir "${DATA_ROOT}/${variant}" \
      --output_dir "${out}" --checkpoint_output_dir "${ckpt}" --max_length 2048 \
      --num_train_epochs 10 --learning_rate 2e-5 --weight_decay 0.01 --warmup_ratio 0.05 \
      --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --gradient_accumulation_steps 32 \
      --max_grad_norm 1.0 --seed 42 --bf16 auto --local_files_only --log_steps 5 --dev_only \
      >"${OUT_ROOT}/${variant}_seed42.log" 2>&1
  ) & pids+=("$!")
done
failed=0; for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
[[ "${failed}" == 0 ]] || exit 1
"${PYTHON_BIN}" thesis_exp/exp17_low_score_evidence/collect_exp29_seed42_scout.py
