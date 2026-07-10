#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-python3}"
OUT_DIR="${OUT_DIR:-thesis_exp/exp17_low_score_evidence/outputs/exp27k_representative_teacher_validation_seed42}"
RUN_API="${RUN_API:-1}"
PREPARE_FIRST="${PREPARE_FIRST:-1}"
RESET_API_OUTPUTS="${RESET_API_OUTPUTS:-0}"
PARALLEL_PROVIDERS="${PARALLEL_PROVIDERS:-1}"
PROVIDER_SHARDS="${PROVIDER_SHARDS:-4}"
TIMEOUT="${TIMEOUT:-180}"
RETRIES="${RETRIES:-2}"
SCHEMA_REPAIR_RETRIES="${SCHEMA_REPAIR_RETRIES:-1}"
THINKING="${THINKING:-omit}"
SLEEP_SECONDS="${SLEEP_SECONDS:-0}"

cat <<CONFIG
Exp27K representative teacher validation
OUT_DIR=${OUT_DIR}
RUN_API=${RUN_API}
PREPARE_FIRST=${PREPARE_FIRST}
RESET_API_OUTPUTS=${RESET_API_OUTPUTS}
PARALLEL_PROVIDERS=${PARALLEL_PROVIDERS}
PROVIDER_SHARDS=${PROVIDER_SHARDS}
TIMEOUT=${TIMEOUT}
RETRIES=${RETRIES}
SCHEMA_REPAIR_RETRIES=${SCHEMA_REPAIR_RETRIES}
THINKING=${THINKING}

This command does not train and does not use GPU.
It fills missing Qwen/DeepSeek coverage for the 120-row representative audit,
then compares the locked protocol against the Exp27J silver reference.
CONFIG

if [[ "${PREPARE_FIRST}" == "1" ]]; then
  "${PYTHON}" thesis_exp/exp17_low_score_evidence/prepare_exp27k_representative_teacher_validation.py \
    --out-dir "${OUT_DIR}"
fi

"${PYTHON}" thesis_exp/exp17_low_score_evidence/validate_exp27k_representative_teacher_validation.py \
  --out-dir "${OUT_DIR}" \
  --allow-missing-api

if [[ "${RUN_API}" != "1" ]]; then
  echo "RUN_API=${RUN_API}; showing one blind request per provider without calling an API."
  for provider in qwen deepseek; do
    "${PYTHON}" thesis_exp/exp17_low_score_evidence/run_exp27d_teacher_audit_api.py \
      --provider "${provider}" \
      --stage blind \
      --out-dir "${OUT_DIR}" \
      --packets "${OUT_DIR}/packets/exp27d_v4_repilot_blind_packets.jsonl" \
      --audit-reference "${OUT_DIR}/packets/exp27d_v4_repilot_audit_reference_private.jsonl" \
      --dry-run
  done
  echo "Exp27K dry-run complete. Set QWEN_API_KEY and DEEPSEEK_API_KEY, then run this script normally."
  exit 0
fi

if [[ -z "${QWEN_API_KEY:-}" ]]; then
  echo "Missing QWEN_API_KEY. Export it in the current shell; never write it into the repository." >&2
  exit 2
fi
if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "Missing DEEPSEEK_API_KEY. Export it in the current shell; never write it into the repository." >&2
  exit 2
fi

if [[ "${RESET_API_OUTPUTS}" == "1" ]]; then
  echo "Resetting local Exp27K API outputs under ${OUT_DIR}/annotations"
  rm -rf "${OUT_DIR}/annotations"
fi

"${PYTHON}" - "${OUT_DIR}" "${PROVIDER_SHARDS}" <<'PY'
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
shards = int(sys.argv[2])
if shards <= 0:
    raise SystemExit("PROVIDER_SHARDS must be positive")
packets_path = out_dir / "packets" / "exp27d_v4_repilot_blind_packets.jsonl"
rows = [json.loads(line) for line in packets_path.open(encoding="utf-8") if line.strip()]
shard_dir = out_dir / "packets" / "shards"
shard_dir.mkdir(parents=True, exist_ok=True)
for idx in range(shards):
    shard_rows = rows[idx::shards]
    path = shard_dir / f"exp27k_packets_shard_{idx:02d}_of_{shards:02d}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in shard_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
print(json.dumps({"rows": len(rows), "shards": shards}, ensure_ascii=False, sort_keys=True))
PY

run_provider_stage() {
  local provider="$1"
  local stage="$2"
  local shard_index="$3"
  local suffix
  suffix="$(printf '_shard%02d_of_%02d' "${shard_index}" "${PROVIDER_SHARDS}")"
  local shard_packets
  shard_packets="${OUT_DIR}/packets/shards/exp27k_packets_shard_$(printf '%02d' "${shard_index}")_of_$(printf '%02d' "${PROVIDER_SHARDS}").jsonl"
  local extra_args=(
    --packets "${shard_packets}"
    --output-suffix "${suffix}"
  )
  if [[ "${stage}" == "audit" ]]; then
    extra_args+=(--blind-output "${OUT_DIR}/annotations/parsed/${provider}/exp27d_blind_outputs${suffix}.jsonl")
  fi
  "${PYTHON}" thesis_exp/exp17_low_score_evidence/run_exp27d_teacher_audit_api.py \
    --provider "${provider}" \
    --stage "${stage}" \
    --out-dir "${OUT_DIR}" \
    --audit-reference "${OUT_DIR}/packets/exp27d_v4_repilot_audit_reference_private.jsonl" \
    --resume \
    --temperature 0 \
    --sleep-seconds "${SLEEP_SECONDS}" \
    --timeout "${TIMEOUT}" \
    --retries "${RETRIES}" \
    --schema-repair-retries "${SCHEMA_REPAIR_RETRIES}" \
    --thinking "${THINKING}" \
    --run-api \
    "${extra_args[@]}"
}

merge_stage() {
  local stage="$1"
  "${PYTHON}" - "${OUT_DIR}" "${stage}" "${PROVIDER_SHARDS}" <<'PY'
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
stage = sys.argv[2]
shards = int(sys.argv[3])
for provider in ["qwen", "deepseek"]:
    parsed_dir = out_dir / "annotations" / "parsed" / provider
    merged = parsed_dir / f"exp27d_{stage}_outputs.jsonl"
    rows = []
    for idx in range(shards):
        suffix = f"_shard{idx:02d}_of_{shards:02d}"
        path = parsed_dir / f"exp27d_{stage}_outputs{suffix}.jsonl"
        if not path.exists():
            raise SystemExit(f"missing shard output: {path}")
        rows.extend(line for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    merged.parent.mkdir(parents=True, exist_ok=True)
    merged.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    print(json.dumps({"provider": provider, "stage": stage, "rows": len(rows)}, ensure_ascii=False, sort_keys=True))
PY
}

run_stage() {
  local stage="$1"
  local pids=()
  if [[ "${PARALLEL_PROVIDERS}" == "1" ]]; then
    echo "Starting ${stage} stage for Qwen and DeepSeek in parallel."
    for shard_index in $(seq 0 $((PROVIDER_SHARDS - 1))); do
      run_provider_stage qwen "${stage}" "${shard_index}" &
      pids+=("$!")
      run_provider_stage deepseek "${stage}" "${shard_index}" &
      pids+=("$!")
    done
    for pid in "${pids[@]}"; do
      wait "${pid}"
    done
  else
    for shard_index in $(seq 0 $((PROVIDER_SHARDS - 1))); do
      run_provider_stage qwen "${stage}" "${shard_index}"
      run_provider_stage deepseek "${stage}" "${shard_index}"
    done
  fi
  merge_stage "${stage}"
}

run_stage blind
run_stage audit

"${PYTHON}" thesis_exp/exp17_low_score_evidence/validate_exp27k_representative_teacher_validation.py \
  --out-dir "${OUT_DIR}"

"${PYTHON}" thesis_exp/exp17_low_score_evidence/analyze_exp27k_representative_teacher_validation.py \
  --out-dir "${OUT_DIR}"

echo "Exp27K completed: ${OUT_DIR}"
