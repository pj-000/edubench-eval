#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

python thesis_exp/exp39_educfa/prepare_exp39a_source_anchors.py
python thesis_exp/exp39_educfa/run_exp39a_qwen_counterfactual_generation.py --max-rows 1 --dry-run
python thesis_exp/exp39_educfa/run_exp39a_deepseek_blind_verification.py --max-rows 1 --dry-run

echo "Exp39A source lock and both API dry runs completed; no API call was made."
