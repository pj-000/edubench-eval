#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

OUT_DIR="${EXP39B_R1_OUT_DIR:-thesis_exp/exp39b_educfa_rlcr/outputs/exp39b_r1_response_disjoint_pilot_seed44}"
python thesis_exp/exp39b_educfa_rlcr/prepare_exp39b_r1_response_disjoint_source_lock.py --out-dir "${OUT_DIR}"
python thesis_exp/exp39b_educfa_rlcr/validate_exp39b_r1_source_lock.py --out-dir "${OUT_DIR}"
python thesis_exp/exp39b_educfa_rlcr/audit_exp39b_r1_source_novelty.py --out-dir "${OUT_DIR}"
