#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
PROVIDER="${PROVIDER:-deepseek}"
WORKERS="${WORKERS:-2}"
MAX_ROWS="${MAX_ROWS:-60}"
python -m thesis_exp.exp48_eduq_tail.run_exp48a_generator_api \
  --provider "${PROVIDER}" --workers "${WORKERS}" --max-rows "${MAX_ROWS}"
