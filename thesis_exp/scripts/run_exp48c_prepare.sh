#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

python -m thesis_exp.exp48_eduq_tail.prepare_exp48c_rubric_only_packets
python -m thesis_exp.exp48_eduq_tail.prepare_exp48c_codex_review_bundle
python -m thesis_exp.exp48_eduq_tail.run_exp48c_qwen_rubric_only_api --max-rows 1 --dry-run
