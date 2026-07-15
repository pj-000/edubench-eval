#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP46_VARIANTS="K1_standard_kd K2_hato_kd K3_shuffled_hato_control" bash "${SCRIPT_DIR}/run_exp46a_matrix.sh"
