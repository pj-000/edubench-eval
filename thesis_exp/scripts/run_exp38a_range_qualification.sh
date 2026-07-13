#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PREPARE_ONLY=0
if [[ "${1:-}" == "--prepare-only" ]]; then
  PREPARE_ONLY=1
  shift
fi
if [[ "$#" -ne 0 ]]; then
  echo "Usage: $0 [--prepare-only]" >&2
  exit 2
fi

python thesis_exp/exp38_hails_score/prepare_exp38a_qwen_range_packets.py --scope qualification
python thesis_exp/exp38_hails_score/run_exp38a_qwen_range_api.py --split qualification --max-rows 1 --dry-run
if [[ "${PREPARE_ONLY}" == "1" ]]; then
  echo "Exp38A qualification packets prepared; no API call was made."
  exit 0
fi
if [[ "${RUN_API:-0}" != "1" ]]; then
  echo "Dry-run complete. Set RUN_API=1 to call Qwen for the frozen 196 rows."
  exit 0
fi
python thesis_exp/exp38_hails_score/run_exp38a_qwen_range_api.py --split qualification
python thesis_exp/exp38_hails_score/validate_exp38a_qwen_range_outputs.py --split qualification
python thesis_exp/exp38_hails_score/analyze_exp38a_range_qualification.py
