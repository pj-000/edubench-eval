#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f "${HOME}/miniconda3/bin/activate" ]]; then
  source "${HOME}/miniconda3/bin/activate" "${CONDA_ENV:-llama_factory}"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"
SEEDS="${SEEDS:-42 43 44}"
GPU_LIST="${GPU_LIST:-0 1 2 3}"
VARIANTS="${VARIANTS:-b0_original_human b1_primary_teacher_all b2_selective_dual_teacher b3_filter_unresolved b4_random_transition_control}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
RESET_RUN_DIR="${RESET_RUN_DIR:-0}"

DATA_ROOT="thesis_exp/exp17_low_score_evidence/outputs/exp28e_paper_ce_training_variants_seed42/private/datasets"
RUN_ROOT="thesis_exp/runs/exp28e_paper_reranker_ce"
ARTIFACT_ROOT="thesis_exp/artifacts/exp28e_paper_reranker_ce"
OUTPUT_ROOT="thesis_exp/exp17_low_score_evidence/outputs/exp28e_paper_reranker_multiseed_dev"

read -r -a gpus <<<"${GPU_LIST}"
read -r -a variants <<<"${VARIANTS}"
read -r -a seeds <<<"${SEEDS}"
[[ "${#gpus[@]}" -gt 0 ]] || { echo "GPU_LIST is empty" >&2; exit 2; }

for variant in "${variants[@]}"; do
  [[ -f "${DATA_ROOT}/${variant}/train.jsonl" ]] || { echo "Missing ${variant} train dataset" >&2; exit 2; }
  [[ -f "${DATA_ROOT}/${variant}/dev.jsonl" ]] || { echo "Missing ${variant} dev dataset" >&2; exit 2; }
  [[ ! -f "${DATA_ROOT}/${variant}/test.jsonl" ]] || { echo "Dev campaign must not contain test.jsonl" >&2; exit 2; }
done

run_one() {
  local seed="$1"
  local variant="$2"
  local gpu="$3"
  local data_dir="${DATA_ROOT}/${variant}"
  local output_dir="${OUTPUT_ROOT}/runs/${variant}/seed_${seed}"
  local checkpoint_dir="${ARTIFACT_ROOT}/${variant}/seed_${seed}"
  local summary="${output_dir}/metrics.json"
  local log_dir="${OUTPUT_ROOT}/logs_private/${variant}/seed_${seed}"
  if [[ "${SKIP_COMPLETED}" == "1" && -f "${summary}" ]]; then
    echo "Skipping completed Exp28E ${variant} seed ${seed}"
    return 0
  fi
  if [[ "${RESET_RUN_DIR}" == "1" ]]; then
    rm -rf "${output_dir}" "${checkpoint_dir}"
  fi
  mkdir -p "${log_dir}"
  echo "Starting Exp28E ${variant} seed ${seed} on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -m thesis_exp.src.edujudge.exp02.train_ce_baseline \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --data_dir "${data_dir}" \
    --output_dir "${output_dir}" \
    --checkpoint_output_dir "${checkpoint_dir}" \
    --max_length 2048 \
    --num_train_epochs 10 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --max_grad_norm 1.0 \
    --seed "${seed}" \
    --bf16 auto \
    --local_files_only \
    --log_steps 5 \
    --dev_only \
    2>&1 | tee "${log_dir}/train.log"
}

jobs=()
for seed in "${seeds[@]}"; do
  for variant in "${variants[@]}"; do
    jobs+=("${seed}|${variant}")
  done
done

pids=()
for worker in "${!gpus[@]}"; do
  (
    for index in "${!jobs[@]}"; do
      if (( index % ${#gpus[@]} == worker )); then
        IFS='|' read -r seed variant <<<"${jobs[$index]}"
        run_one "${seed}" "${variant}" "${gpus[$worker]}"
      fi
    done
  ) &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done
[[ "${failed}" == "0" ]] || { echo "At least one Exp28E GPU queue failed" >&2; exit 1; }

"${PYTHON_BIN}" thesis_exp/exp17_low_score_evidence/collect_exp28e_multiseed_dev_results.py
"${PYTHON_BIN}" thesis_exp/exp17_low_score_evidence/bootstrap_exp28f_dev_differences.py
"${PYTHON_BIN}" thesis_exp/exp17_low_score_evidence/plot_exp28_paper_results.py
echo "Exp28E three-seed dev-only campaign completed. Test was not read."
