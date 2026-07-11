#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f ".env.exp28.local" ]]; then
  set -a
  source ".env.exp28.local"
  set +a
fi

RUN_API="${RUN_API:-0}"
PROTOCOL_DECISION="thesis_exp/exp17_low_score_evidence/outputs/exp28c_teacher_protocol_evaluation_seed42/decision/exp28c_protocol_development_protocol_decision.json"
QUALIFICATION_DECISION="thesis_exp/exp17_low_score_evidence/outputs/exp28c_teacher_protocol_evaluation_seed42/decision/exp28c_sealed_qualification_protocol_decision.json"
PROTOCOL="${PROTOCOL:-}"

if [[ -z "${PROTOCOL}" ]]; then
  [[ -f "${PROTOCOL_DECISION}" ]] || { echo "Missing protocol decision: ${PROTOCOL_DECISION}" >&2; exit 2; }
  PROTOCOL="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_protocol"] or "")' "${PROTOCOL_DECISION}")"
fi
[[ -n "${PROTOCOL}" ]] || { echo "No protocol has been selected" >&2; exit 2; }
if [[ "${RUN_API}" == "1" ]]; then
  [[ -f "${QUALIFICATION_DECISION}" ]] || { echo "Missing sealed qualification decision" >&2; exit 2; }
  python -c 'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get("status")=="READY_FOR_FULL_TRAIN_ANNOTATION" else 1)' \
    "${QUALIFICATION_DECISION}" || { echo "Sealed qualification does not authorize full annotation" >&2; exit 2; }
fi

if [[ "${RUN_API}" == "1" ]]; then
  [[ -n "${QWEN_API_KEY:-}" ]] || { echo "Missing QWEN_API_KEY" >&2; exit 2; }
  [[ -n "${DEEPSEEK_API_KEY:-}" ]] || { echo "Missing DEEPSEEK_API_KEY" >&2; exit 2; }
fi

echo "Exp28 full teacher annotation; protocol=${PROTOCOL}; RUN_API=${RUN_API}"
echo "Stage 1: Qwen primary teacher on all 2,654 paper-train rows"
if [[ "${RUN_API}" == "1" ]]; then
  python thesis_exp/exp17_low_score_evidence/run_exp28b_teacher_protocol_api.py \
    --provider qwen \
    --protocol "${PROTOCOL}" \
    --subset all_train \
    --run-api \
    --resume
else
  python thesis_exp/exp17_low_score_evidence/run_exp28b_teacher_protocol_api.py \
    --provider qwen \
    --protocol "${PROTOCOL}" \
    --subset all_train
fi

if [[ "${RUN_API}" != "1" ]]; then
  echo "Dry run completed before route construction."
  exit 0
fi

echo "Stage 2: construct train-only selective secondary route"
python thesis_exp/exp17_low_score_evidence/prepare_exp28d_secondary_teacher_route.py \
  --protocol "${PROTOCOL}"

echo "Stage 3: DeepSeek secondary teacher on routed rows"
python thesis_exp/exp17_low_score_evidence/run_exp28b_teacher_protocol_api.py \
  --provider deepseek \
  --protocol "${PROTOCOL}" \
  --subset secondary_route \
  --packets thesis_exp/exp17_low_score_evidence/outputs/exp28d_selective_secondary_route_seed42/private/exp28d_secondary_teacher_packets.jsonl \
  --run-api \
  --resume

echo "Exp28 full primary plus selective secondary annotation completed."
