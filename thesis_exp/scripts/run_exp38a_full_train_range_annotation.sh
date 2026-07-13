#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ "${RUN_API:-0}" != "1" ]]; then
  echo "Blocked: set RUN_API=1 only after Exp38A qualification is GO." >&2
  exit 2
fi
python thesis_exp/exp38_hails_score/prepare_exp38a_qwen_range_packets.py --scope all_train
python thesis_exp/exp38_hails_score/run_exp38a_qwen_range_api.py --split all_train
python thesis_exp/exp38_hails_score/validate_exp38a_qwen_range_outputs.py --split all_train
python thesis_exp/exp38_hails_score/build_exp38a_hails_supervision.py
python thesis_exp/exp38_hails_score/prepare_exp38a_groupcv_folds.py
