#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f "${HOME}/miniconda3/bin/activate" ]]; then
  source "${HOME}/miniconda3/bin/activate" "${CONDA_ENV:-llama_factory}"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_FORMAL="${RUN_FORMAL:-1}"
SEED="${SEED:-42}"
GPU_LIST="${GPU_LIST:-0 1 2 3}"
RUN_VARIANTS="${RUN_VARIANTS:-v0_original_unweighted v1_original_label_matched_weight v2_selective_hard_relabel v3_selective_soft_audit}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
RESET_RUN_DIR="${RESET_RUN_DIR:-0}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"
OUT_DIR="thesis_exp/exp17_low_score_evidence/outputs/exp27p_soft_target_reranker_seed42"
EXP27O_DIR="thesis_exp/exp17_low_score_evidence/outputs/exp27o_361_in_place_pilot_seed42"
RUN_ROOT="thesis_exp/runs/exp27p_soft_target_reranker"
ARTIFACT_ROOT="thesis_exp/artifacts/exp27p_soft_target_reranker"
LOG_DIR="${OUT_DIR}/logs_private/seed_${SEED}"

if [[ "${RUN_FORMAL}" != "1" ]]; then
  echo "Exp27P formal gate is closed. Set RUN_FORMAL=1." >&2
  exit 2
fi
if [[ "${SEED}" != "42" ]]; then
  echo "This scout wrapper is locked to seed42. Use the multiseed plan only after decision PASS." >&2
  exit 2
fi

read -r -a GPUS <<<"${GPU_LIST}"
read -r -a VARIANTS <<<"${RUN_VARIANTS}"
if [[ "${#GPUS[@]}" -eq 0 || "${#VARIANTS[@]}" -eq 0 ]]; then
  echo "GPU_LIST and RUN_VARIANTS must not be empty." >&2
  exit 2
fi

"${PYTHON_BIN}" -m thesis_exp.src.edujudge.exp27p.validate_exp27p_training_setup \
  --exp27o-dir "${EXP27O_DIR}" \
  --output-dir "${OUT_DIR}"

smoke_decision="${OUT_DIR}/decision/exp27p_smoke_decision.json"
if [[ ! -f "${smoke_decision}" ]] || ! "${PYTHON_BIN}" -c 'import json,sys;sys.exit(0 if json.load(open(sys.argv[1]))["status"]=="PASS" else 1)' "${smoke_decision}"; then
  echo "Exp27P smoke PASS is required before seed42 scout: ${smoke_decision}" >&2
  exit 2
fi

mkdir -p "${LOG_DIR}"
run_variant() {
  local variant="$1"
  local gpu="$2"
  local train_jsonl="${EXP27O_DIR}/private/data/exp27o_${variant}_train.jsonl"
  local run_dir="${RUN_ROOT}/${variant}/seed_${SEED}"
  local artifact_dir="${ARTIFACT_ROOT}/${variant}/seed_${SEED}"
  local summary="${run_dir}/run_summary.json"
  if [[ "${SKIP_COMPLETED}" == "1" && -f "${summary}" ]] && \
    "${PYTHON_BIN}" -c 'import json,sys;sys.exit(0 if json.load(open(sys.argv[1])).get("status")=="COMPLETED" else 1)' "${summary}"; then
    echo "Skipping completed Exp27P ${variant} seed ${SEED}"
    return 0
  fi
  if [[ "${RESET_RUN_DIR}" == "1" ]]; then
    rm -rf "${run_dir}" "${artifact_dir}"
  fi
  echo "Starting Exp27P ${variant} seed ${SEED} on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -m thesis_exp.src.edujudge.exp27p.train_exp27p_soft_target_reranker \
    --variant "${variant}" \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --train_jsonl "${train_jsonl}" \
    --run_dir "${run_dir}" \
    --checkpoint_output_dir "${artifact_dir}" \
    --seed "${SEED}" \
    --max_length 2048 \
    --num_train_epochs 10 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --max_grad_norm 1.0 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --log_steps 5 \
    2>&1 | tee "${LOG_DIR}/${variant}.log"
}

pids=()
for worker in "${!GPUS[@]}"; do
  (
    for index in "${!VARIANTS[@]}"; do
      if (( index % ${#GPUS[@]} == worker )); then
        run_variant "${VARIANTS[$index]}" "${GPUS[$worker]}"
      fi
    done
  ) &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" != "0" ]]; then
  echo "At least one Exp27P GPU queue failed." >&2
  exit 1
fi

"${PYTHON_BIN}" -m thesis_exp.src.edujudge.exp27p.collect_exp27p_results \
  --run-root "${RUN_ROOT}" \
  --output-dir "${OUT_DIR}" \
  --seed "${SEED}" \
  --bootstrap-resamples 2000

echo "Exp27P seed42 scout and dev-only collection completed."
