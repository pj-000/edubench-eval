#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

OUT_DIR="${OUT_DIR:-thesis_exp/exp17_low_score_evidence/outputs/exp27i_target_aware_teacher_audit_361_seed42}"
MAX_ROWS="${MAX_ROWS:-0}"
SLEEP_SECONDS="${SLEEP_SECONDS:-0}"
TIMEOUT="${TIMEOUT:-180}"
RETRIES="${RETRIES:-2}"
SCHEMA_REPAIR_RETRIES="${SCHEMA_REPAIR_RETRIES:-1}"
THINKING="${THINKING:-omit}"
PREPARE_FIRST="${PREPARE_FIRST:-1}"
PARALLEL_PROVIDERS="${PARALLEL_PROVIDERS:-1}"
PROVIDER_SHARDS="${PROVIDER_SHARDS:-4}"
RESET_API_OUTPUTS="${RESET_API_OUTPUTS:-0}"
RUN_API="${RUN_API:-1}"

cat <<CONFIG
Exp27I target-aware teacher-audited 361 expansion
OUT_DIR=${OUT_DIR}
MAX_ROWS=${MAX_ROWS}
SLEEP_SECONDS=${SLEEP_SECONDS}
TIMEOUT=${TIMEOUT}
RETRIES=${RETRIES}
SCHEMA_REPAIR_RETRIES=${SCHEMA_REPAIR_RETRIES}
THINKING=${THINKING}
PREPARE_FIRST=${PREPARE_FIRST}
PARALLEL_PROVIDERS=${PARALLEL_PROVIDERS}
PROVIDER_SHARDS=${PROVIDER_SHARDS}
RESET_API_OUTPUTS=${RESET_API_OUTPUTS}
RUN_API=${RUN_API}

This step calls Qwen and DeepSeek teacher APIs with target-aware packets.
It does not train, does not use GPU, and does not read dev/test labels.
API keys are read only from QWEN_API_KEY and DEEPSEEK_API_KEY environment variables.
CONFIG

if [[ "${RUN_API}" != "1" ]]; then
  echo "RUN_API is not 1; only preparing packets and dry-running one message." >&2
fi

if [[ "${RUN_API}" == "1" ]]; then
  if [[ -z "${QWEN_API_KEY:-}" ]]; then
    echo "Missing QWEN_API_KEY" >&2
    exit 2
  fi
  if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
    echo "Missing DEEPSEEK_API_KEY" >&2
    exit 2
  fi
fi

if [[ "${PREPARE_FIRST}" == "1" ]]; then
  python thesis_exp/exp17_low_score_evidence/prepare_exp27i_target_aware_teacher_audit_361_packets.py \
    --out-dir "${OUT_DIR}"
fi

if [[ "${RESET_API_OUTPUTS}" == "1" ]]; then
  echo "Resetting Exp27I parsed/raw API outputs under ${OUT_DIR}/annotations"
  rm -rf "${OUT_DIR}/annotations"
fi

python - "${OUT_DIR}" "${PROVIDER_SHARDS}" <<'PY'
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
shards = int(sys.argv[2])
packets_path = out_dir / "packets" / "exp27d_v4_repilot_blind_packets.jsonl"
rows = [json.loads(line) for line in packets_path.open(encoding="utf-8") if line.strip()]
shard_dir = out_dir / "packets" / "shards"
shard_dir.mkdir(parents=True, exist_ok=True)
for idx in range(shards):
    shard_rows = rows[idx::shards]
    path = shard_dir / f"exp27i_packets_shard_{idx:02d}_of_{shards:02d}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in shard_rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
print(json.dumps({"shards": shards, "rows": len(rows)}, ensure_ascii=False, sort_keys=True))
PY

common_args=(
  --out-dir "${OUT_DIR}"
  --audit-reference "${OUT_DIR}/packets/exp27d_v4_repilot_audit_reference_private.jsonl"
  --max-rows "${MAX_ROWS}"
  --resume
  --temperature 0
  --sleep-seconds "${SLEEP_SECONDS}"
  --timeout "${TIMEOUT}"
  --retries "${RETRIES}"
  --schema-repair-retries "${SCHEMA_REPAIR_RETRIES}"
  --thinking "${THINKING}"
)

if [[ "${RUN_API}" == "1" ]]; then
  common_args+=(--run-api)
else
  common_args+=(--dry-run)
fi

run_provider_stage() {
  local provider="$1"
  local stage="$2"
  local shard_index="$3"
  local shard_suffix
  shard_suffix="$(printf '_shard%02d_of_%02d' "${shard_index}" "${PROVIDER_SHARDS}")"
  local shard_packets="${OUT_DIR}/packets/shards/exp27i_packets_shard_$(printf '%02d' "${shard_index}")_of_$(printf '%02d' "${PROVIDER_SHARDS}").jsonl"
  local extra_args=(
    --packets "${shard_packets}"
    --output-suffix "${shard_suffix}"
  )
  if [[ "${stage}" == "audit" ]]; then
    extra_args+=(--blind-output "${OUT_DIR}/annotations/parsed/${provider}/exp27d_blind_outputs${shard_suffix}.jsonl")
  fi
  python thesis_exp/exp17_low_score_evidence/run_exp27d_teacher_audit_api.py \
    --provider "${provider}" \
    --stage "${stage}" \
    "${common_args[@]}" \
    "${extra_args[@]}"
}

run_stage_for_both_providers() {
  local stage="$1"
  if [[ "${PARALLEL_PROVIDERS}" == "1" ]]; then
    echo "Starting Exp27I ${stage} stage for qwen and deepseek in parallel, shards=${PROVIDER_SHARDS}."
    local pids=()
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
    echo "Starting Exp27I ${stage} stage for qwen and deepseek sequentially, shards=${PROVIDER_SHARDS}."
    for shard_index in $(seq 0 $((PROVIDER_SHARDS - 1))); do
      run_provider_stage qwen "${stage}" "${shard_index}"
      run_provider_stage deepseek "${stage}" "${shard_index}"
    done
  fi
  if [[ "${RUN_API}" != "1" ]]; then
    return 0
  fi
  python - "${OUT_DIR}" "${stage}" "${PROVIDER_SHARDS}" <<'PY'
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
    print(json.dumps({"provider": provider, "stage": stage, "merged_rows": len(rows), "merged": str(merged)}, ensure_ascii=False, sort_keys=True))
PY
}

run_stage_for_both_providers blind

if [[ "${RUN_API}" != "1" ]]; then
  echo "Dry-run complete after blind message preview."
  exit 0
fi

run_stage_for_both_providers audit

python thesis_exp/exp17_low_score_evidence/collect_exp27i_target_aware_teacher_audit_results.py \
  --out-dir "${OUT_DIR}"

echo "Exp27I target-aware teacher-audited 361 expansion completed."
