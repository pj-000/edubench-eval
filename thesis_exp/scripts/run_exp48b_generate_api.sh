#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
PROVIDER="${PROVIDER:-deepseek}"
WORKERS="${WORKERS:-2}"
MAX_ROWS="${MAX_ROWS:-}"
if [[ -n "${MAX_ROWS}" ]]; then
  python -m thesis_exp.exp48_eduq_tail.run_exp48b_generator_api --provider "${PROVIDER}" --workers "${WORKERS}" --max-rows "${MAX_ROWS}"
else
  python -m thesis_exp.exp48_eduq_tail.run_exp48b_generator_api --provider "${PROVIDER}" --workers "${WORKERS}"
fi
python -m thesis_exp.exp48_eduq_tail.validate_exp48b_generated_families
