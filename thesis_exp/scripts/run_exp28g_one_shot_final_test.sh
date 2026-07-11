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
GPU_LIST="${GPU_LIST:-0 1 2 3}"
DATA_ROOT="thesis_exp/exp17_low_score_evidence/outputs/exp28e_paper_ce_training_variants_seed42/private/datasets"
CHECKPOINT_ROOT="thesis_exp/artifacts/exp28e_paper_reranker_ce"
OUTPUT_ROOT="thesis_exp/exp17_low_score_evidence/outputs/exp28g_one_shot_final_test"
VARIANTS=(b0_original_human b2_selective_dual_teacher)
SEEDS=(42 43 44)

"${PYTHON_BIN}" thesis_exp/exp17_low_score_evidence/prepare_exp28g_one_shot_test.py

read -r -a gpus <<<"${GPU_LIST}"
jobs=()
for variant in "${VARIANTS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    jobs+=("${variant}|${seed}")
  done
done

run_one() {
  local variant="$1"
  local seed="$2"
  local gpu="$3"
  local checkpoint="${CHECKPOINT_ROOT}/${variant}/seed_${seed}/best"
  local output="${OUTPUT_ROOT}/runs/${variant}/seed_${seed}"
  [[ -d "${checkpoint}" ]] || { echo "Missing checkpoint: ${checkpoint}" >&2; return 2; }
  if [[ -f "${output}/metrics.json" ]] && \
    "${PYTHON_BIN}" -c 'import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if any(x.get("split")=="test" for x in r) else 1)' "${output}/metrics.json"; then
    echo "Skipping completed final test ${variant} seed ${seed}"
    return 0
  fi
  echo "Starting final test ${variant} seed ${seed} on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -m thesis_exp.src.edujudge.exp02.train_ce_baseline \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --data_dir "${DATA_ROOT}/${variant}" \
    --output_dir "${output}" \
    --checkpoint_output_dir "${checkpoint}" \
    --checkpoint_dir "${checkpoint}" \
    --eval_only \
    --max_length 2048 \
    --per_device_eval_batch_size 4 \
    --seed "${seed}" \
    --bf16 auto \
    --local_files_only
}

pids=()
for worker in "${!gpus[@]}"; do
  (
    for index in "${!jobs[@]}"; do
      if (( index % ${#gpus[@]} == worker )); then
        IFS='|' read -r variant seed <<<"${jobs[$index]}"
        run_one "${variant}" "${seed}" "${gpus[$worker]}"
      fi
    done
  ) &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done
[[ "${failed}" == "0" ]] || exit 1

"${PYTHON_BIN}" thesis_exp/exp17_low_score_evidence/collect_exp28g_final_test_results.py
"${PYTHON_BIN}" thesis_exp/exp17_low_score_evidence/bootstrap_exp28h_final_test.py
"${PYTHON_BIN}" thesis_exp/exp17_low_score_evidence/plot_exp28_paper_results.py --include-test
echo "Exp28G one-shot final test completed."
