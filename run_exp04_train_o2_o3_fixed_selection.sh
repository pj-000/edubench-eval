#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/thesis_exp/scripts/run_exp04_train_o2_o3_fixed_selection.sh" "$@"
