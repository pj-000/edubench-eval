#!/usr/bin/env bash
set -euo pipefail

SELECTION_DELTA="${SELECTION_DELTA:-0.005}"
ECE_BINS="${ECE_BINS:-10}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

cat <<CONFIG
Exp15 post-hoc ordinal calibration
GPU_REQUIRED=0
SELECTION_RULE=pava_mae_guard_low_to_high_then_label2_then_calibration
SELECTION_DELTA=${SELECTION_DELTA}
ECE_BINS=${ECE_BINS}
CONFIG

python -m thesis_exp.src.edujudge.exp15_posthoc_ordinal_calibration.smoke_check_exp15
python -m thesis_exp.src.edujudge.exp15_posthoc_ordinal_calibration.run_exp15_posthoc_ordinal_calibration \
  --selection_delta "${SELECTION_DELTA}" \
  --ece_bins "${ECE_BINS}"
python -m thesis_exp.src.edujudge.exp15_posthoc_ordinal_calibration.readability_check_exp15
cat thesis_exp/outputs/exp15_posthoc_ordinal_calibration/reports/exp15_posthoc_ordinal_calibration_report.md
