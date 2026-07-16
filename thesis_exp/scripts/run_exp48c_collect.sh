#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

OUT="thesis_exp/exp48_eduq_tail/outputs/exp48c_rubric_only_audit/private"
CODEX_OUT="${OUT}/codex_outputs/exp48c_codex_rubric_only_outputs.jsonl"
QWEN_OUT="${OUT}/qwen_outputs/exp48c_qwen_rubric_only_outputs.jsonl"

if [[ -f "${CODEX_OUT}" ]]; then
  python -m thesis_exp.exp48_eduq_tail.validate_exp48c_rubric_only_outputs --verifier codex
else
  echo "Codex output is not present; collector will report an awaiting status."
fi
if [[ -f "${QWEN_OUT}" ]]; then
  python -m thesis_exp.exp48_eduq_tail.validate_exp48c_rubric_only_outputs --verifier qwen
else
  echo "Qwen output is not present; collector will report an awaiting status."
fi

python -m thesis_exp.exp48_eduq_tail.analyze_exp48c_contract_dependence
