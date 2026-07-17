#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
if [[ -f "${HOME}/miniconda3/bin/activate" ]]; then source "${HOME}/miniconda3/bin/activate" "${CONDA_ENV:-llama_factory}"; fi
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"
GPU_LIST="${GPU_LIST:-0 1 2 3}"
read -r -a GPUS <<<"${GPU_LIST}"
[[ "${#GPUS[@]}" -ge 1 ]] || { echo "GPU_LIST is empty" >&2; exit 2; }

# Increment before inference. A partial failure still consumes the one-shot access.
"${PYTHON_BIN}" -m thesis_exp.exp49_cphce.freeze_manifest --begin-test
jobs=("42|b0_hard_ce" "42|m1_human_soft" "43|b0_hard_ce" "43|m1_human_soft" "44|b0_hard_ce" "44|m1_human_soft")
pids=()
for worker in "${!GPUS[@]}"; do
  (
    for index in "${!jobs[@]}"; do
      if (( index % ${#GPUS[@]} == worker )); then
        IFS='|' read -r seed variant <<<"${jobs[$index]}"
        CUDA_VISIBLE_DEVICES="${GPUS[$worker]}" "${PYTHON_BIN}" -m thesis_exp.exp49_cphce.evaluate \
          --variant "${variant}" --seed "${seed}" --model-name-or-path "${MODEL_NAME_OR_PATH}"
      fi
    done
  ) &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
[[ "${failed}" == "0" ]] || { echo "Exp49 one-shot test failed and access remains consumed" >&2; exit 1; }
"${PYTHON_BIN}" -m thesis_exp.exp49_cphce.freeze_manifest --mark-tested
