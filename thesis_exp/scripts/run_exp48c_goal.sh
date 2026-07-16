#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

bash thesis_exp/scripts/run_exp48c_prepare.sh
if [[ "${RUN_CODEX:-0}" == "1" ]]; then
  python -m thesis_exp.exp48_eduq_tail.run_exp48c_codex_isolated \
    --workers "${CODEX_WORKERS:-4}"
else
  echo "Codex review is not run by default. Set RUN_CODEX=1 to launch 36 isolated contexts."
fi
bash thesis_exp/scripts/run_exp48c_qwen.sh
bash thesis_exp/scripts/run_exp48c_collect.sh

echo "Exp48C goal step completed. A final GO/NO-GO requires both isolated Codex and Qwen outputs."
