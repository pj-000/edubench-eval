#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ "${RUN_API:-0}" != "1" ]]; then
  echo "Exp48C Qwen is dry-run by default. Set RUN_API=1 and QWEN_API_KEY in the environment for real requests."
  python -m thesis_exp.exp48_eduq_tail.run_exp48c_qwen_rubric_only_api --max-rows 1 --dry-run
  exit 0
fi

if [[ -z "${QWEN_API_KEY:-}" ]]; then
  echo "Missing QWEN_API_KEY; provide it only through the environment." >&2
  exit 2
fi

python -m thesis_exp.exp48_eduq_tail.run_exp48c_qwen_rubric_only_api \
  --workers "${QWEN_WORKERS:-4}" \
  --model "${QWEN_MODEL:-qwen3.7-max}"
