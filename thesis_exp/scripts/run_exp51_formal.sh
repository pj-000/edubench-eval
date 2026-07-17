#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
if [[ "${RUN_FORMAL:-0}" != "1" ]]; then
  echo "Refusing Exp51 formal seeds without RUN_FORMAL=1" >&2
  exit 2
fi
PYTHONPATH=. python - <<'PY'
from thesis_exp.exp51_hmsa.formal_gate import verify_formal_lock
lock = verify_formal_lock()
print(f"Exp51 formal protocol lock verified: {lock['protocol_commit']} {lock['manifest_sha256']}")
PY
python - <<'PY'
import json
from pathlib import Path
p=Path('thesis_exp/outputs/exp51_hmsa/decision/seed42_decision.json')
d=json.loads(p.read_text())
assert d['status']=='EXP51_SEED42_PASS' and d['passed']
PY
MODEL_PATH="${MODEL_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"
[[ "${MODEL_PATH}" == "/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B" ]] || { echo "Formal model path is locked" >&2; exit 2; }
GPU_LIST="${GPU_LIST:-0 1 2 3}"
read -r -a GPUS <<<"${GPU_LIST}"
[[ "${#GPUS[@]}" -ge 4 ]] || { echo "Need four GPUs" >&2; exit 2; }
jobs=("43|b0" "43|exp51" "44|b0" "44|exp51")
for job in "${jobs[@]}"; do
  IFS='|' read -r seed arm <<<"${job}"
  if [[ "${arm}" == "b0" ]]; then
    output_dir="thesis_exp/outputs/exp49_cphce/runs/b0_hard_ce/seed_${seed}"
    checkpoint_dir="thesis_exp/artifacts/exp49_cphce/b0_hard_ce/seed_${seed}"
  else
    output_dir="thesis_exp/outputs/exp51_hmsa/runs/hmsa_lambda1/seed_${seed}"
    checkpoint_dir="thesis_exp/artifacts/exp51_hmsa/hmsa_lambda1/seed_${seed}"
  fi
  [[ ! -e "${output_dir}" && ! -e "${checkpoint_dir}" ]] || { echo "Refusing to overwrite ${arm} seed${seed}" >&2; exit 2; }
done
pids=()
for index in "${!jobs[@]}"; do
  IFS='|' read -r seed arm <<<"${jobs[$index]}"
  log_dir="thesis_exp/outputs/exp51_hmsa/logs_private/formal"
  mkdir -p "${log_dir}"
  if [[ "${arm}" == "b0" ]]; then
    (
      CUDA_VISIBLE_DEVICES="${GPUS[$index]}" PYTHONPATH=. python -m thesis_exp.exp49_cphce.train \
        --variant b0_hard_ce --model_name_or_path "${MODEL_PATH}" --seed "${seed}" \
        --gradient_checkpointing --local_files_only \
        2>&1 | tee "${log_dir}/b0_seed${seed}.log"
    ) &
  else
    (
      CUDA_VISIBLE_DEVICES="${GPUS[$index]}" EXP51_REQUIRE_SOURCE_LOCK=1 PYTHONPATH=. python -m thesis_exp.exp51_hmsa.train \
        --model_name_or_path "${MODEL_PATH}" --seed "${seed}" \
        --gradient_checkpointing --local_files_only \
        2>&1 | tee "${log_dir}/exp51_seed${seed}.log"
    ) &
  fi
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
[[ "${failed}" == "0" ]] || { echo "Exp51 formal training failed" >&2; exit 1; }
PYTHONPATH=. python -m thesis_exp.exp51_hmsa.formal_gate
