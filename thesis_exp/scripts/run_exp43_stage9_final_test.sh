#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO_ROOT="${EDUBENCH_REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"; cd "${REPO_ROOT}"
PYTHON="${PYTHON:-/home/jpang/miniconda3/envs/llama_factory/bin/python}"; MODEL="${RUBIMOR_MODEL_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"; GPU_LIST="${GPU_LIST:-0 1 2 3}"; read -r -a GPUS <<< "${GPU_LIST}"
"${PYTHON}" -m thesis_exp.exp43_rubimor.lock_exp43_final_test
JOBS=(); for variant in E0 E3 E5 E6 E6N; do for seed in 42 43 44; do JOBS+=("${variant}:${seed}"); done; done
run_queue(){ local gpu="$1";shift;for job in "$@";do variant="${job%%:*}";seed="${job##*:}";CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -m thesis_exp.exp43_rubimor.evaluate_exp43_final_test --model-name-or-path "${MODEL}" --variant "${variant}" --seed "${seed}";done;}
PIDS=();for index in "${!GPUS[@]}";do queue=();for j in "${!JOBS[@]}";do (( j % ${#GPUS[@]} == index ))&&queue+=("${JOBS[$j]}");done;run_queue "${GPUS[$index]}" "${queue[@]}"&PIDS+=("$!");done
failed=0;for pid in "${PIDS[@]}";do wait "$pid"||failed=1;done;[[ $failed -eq 0 ]]||exit 1
"${PYTHON}" -m thesis_exp.exp43_rubimor.evaluate_exp43_final_test --collect

