#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP46_VARIANTS="T1_4B_teacher" bash "${SCRIPT_DIR}/run_exp46a_matrix.sh"
