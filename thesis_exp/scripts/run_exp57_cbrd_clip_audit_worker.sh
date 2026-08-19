#!/usr/bin/env bash
# Wait for one confirmation trainer, then run frozen no-update clipping audits.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

GPU_ID="${GPU_ID:?Set an available RTX 3090 index}"
WAIT_SESSION="${WAIT_SESSION:?Set the tmux training session to wait for}"
ASSIGNMENT="${ASSIGNMENT:?Set gpu6 or gpu7}"
PYTHON_BIN="${PYTHON_BIN:-/home/jpang/miniconda3/envs/llama_factory/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"
AUDIT_DIR="${AUDIT_DIR:-thesis_exp/outputs/exp57_cbrd/audit/clip_gradient}"
LOG_DIR="thesis_exp/outputs/exp57_cbrd/logs_private"
LOG_PREFIX="${LOG_PREFIX:-clip}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
mkdir -p "${AUDIT_DIR}" "${LOG_DIR}"

while tmux has-session -t "${WAIT_SESSION}" 2>/dev/null; do
  sleep 15
done

run_audit() {
  local state_name="$1"
  local seed="$2"
  local trace_variant="$3"
  local checkpoint_variant="${4:-}"
  local output_path="${AUDIT_DIR}/${state_name}.json"
  local trace_path="thesis_exp/outputs/exp57_cbrd/runs/${trace_variant}/seed_${seed}/training_trace_first64.json"
  local checkpoint_args=()
  if [[ -n "${checkpoint_variant}" ]]; then
    checkpoint_args=(
      --checkpoint-path
      "thesis_exp/artifacts/exp57_cbrd/${checkpoint_variant}/seed_${seed}/best/state_dict.pt"
    )
  fi
  if [[ -e "${output_path}" ]]; then
    echo "Refusing to overwrite ${output_path}" >&2
    return 3
  fi
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -m thesis_exp.exp57_cbrd.clip_gradient_audit \
    --model-name-or-path "${MODEL_PATH}" \
    --trace-path "${trace_path}" \
    --output-path "${output_path}" \
    --seed "${seed}" \
    --microbatches 32 \
    "${checkpoint_args[@]}" \
    >"${LOG_DIR}/${LOG_PREFIX}_${state_name}.log" 2>&1
}

case "${ASSIGNMENT}" in
  gpu6)
    run_audit initial_seed42 42 consensus_only
    run_audit selected_consensus_seed42 42 consensus_only consensus_only
    run_audit selected_routed_seed42 42 routed_hmsa routed_hmsa
    run_audit initial_seed44 44 consensus_only
    run_audit selected_consensus_seed44 44 consensus_only consensus_only
    ;;
  gpu7)
    run_audit initial_seed43 43 consensus_only
    run_audit selected_consensus_seed43 43 consensus_only consensus_only
    run_audit selected_routed_seed43 43 routed_hmsa routed_hmsa
    run_audit selected_routed_seed44 44 routed_hmsa routed_hmsa
    ;;
  *)
    echo "Unknown audit assignment: ${ASSIGNMENT}" >&2
    exit 2
    ;;
esac
