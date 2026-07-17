#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ "${RUN_FINAL_TEST:-0}" != "1" ]]; then
  echo "Refusing Exp51 final test without RUN_FINAL_TEST=1" >&2
  exit 2
fi

MODEL_PATH="${MODEL_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"
[[ "${MODEL_PATH}" == "/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B" ]] || { echo "Final-test model path is locked" >&2; exit 2; }
GPU_LIST="${GPU_LIST:-0 1 2 3}"
read -r -a GPUS <<<"${GPU_LIST}"
[[ "${#GPUS[@]}" -ge 4 ]] || { echo "Need four GPUs" >&2; exit 2; }

PYTHONPATH=. python -m thesis_exp.exp51_hmsa.final_test --verify
STATE="thesis_exp/outputs/exp51_hmsa/final_test/campaign_state.json"
if [[ ! -e "${STATE}" ]]; then
  for path in \
    thesis_exp/outputs/exp51_hmsa/final_test/b0 \
    thesis_exp/outputs/exp51_hmsa/final_test/exp51 \
    thesis_exp/outputs/exp51_hmsa/final_test/final_test_summary.json; do
    [[ ! -e "${path}" ]] || { echo "Untracked final-test output exists: ${path}" >&2; exit 2; }
  done
  PYTHONPATH=. python -m thesis_exp.exp51_hmsa.final_test --begin
  PYTHONPATH=. python -m thesis_exp.exp51_hmsa.final_test --anchor-test
else
  PYTHONPATH=. python - <<'PY'
import json
from pathlib import Path
path = Path("thesis_exp/outputs/exp51_hmsa/final_test/campaign_state.json")
state = json.loads(path.read_text())
assert state["status"] == "TEST_IN_PROGRESS" and state["test_access_count"] == 1 and state["test_anchor"] is not None
PY
fi

LOG_DIR="thesis_exp/outputs/exp51_hmsa/final_test/logs_private"
mkdir -p "${LOG_DIR}"

run_one() {
  local gpu="$1"
  local arm="$2"
  local seed="$3"
  local output="thesis_exp/outputs/exp51_hmsa/final_test/${arm}/seed_${seed}/test_metrics.json"
  if [[ -f "${output}" ]]; then
    echo "Resuming campaign: keeping completed ${arm} seed${seed}"
    return 0
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=. python -m thesis_exp.exp51_hmsa.final_test \
    --evaluate --arm "${arm}" --seed "${seed}" --split test \
    > "${LOG_DIR}/${arm}_seed${seed}.log" 2>&1
}

run_wave() {
  local jobs=("$@")
  local pids=()
  for job in "${jobs[@]}"; do
    IFS='|' read -r gpu arm seed <<<"${job}"
    run_one "${gpu}" "${arm}" "${seed}" &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
  [[ "${failed}" == "0" ]] || { echo "Exp51 final-test wave failed; campaign remains TEST_IN_PROGRESS" >&2; exit 1; }
}

run_wave \
  "${GPUS[0]}|b0|42" \
  "${GPUS[1]}|exp51|42" \
  "${GPUS[2]}|b0|43" \
  "${GPUS[3]}|exp51|43"
run_wave \
  "${GPUS[0]}|b0|44" \
  "${GPUS[1]}|exp51|44"

PYTHONPATH=. python -m thesis_exp.exp51_hmsa.final_test --summarize
PYTHONPATH=. python -m thesis_exp.exp51_hmsa.final_test --mark-complete
