#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
SERVER="${SERVER:-myserver}"
REMOTE_ROOT="${REMOTE_ROOT:-edubench-eval-exp2}"
rsync -a thesis_exp/exp51_hmsa/ "${SERVER}:${REMOTE_ROOT}/thesis_exp/exp51_hmsa/"
rsync -a thesis_exp/configs/exp51_hmsa/ "${SERVER}:${REMOTE_ROOT}/thesis_exp/configs/exp51_hmsa/"
rsync -a thesis_exp/tests/test_exp51_contract.py thesis_exp/tests/test_exp51_dual_head.py thesis_exp/tests/test_exp51_formal_gate.py "${SERVER}:${REMOTE_ROOT}/thesis_exp/tests/"
rsync -a thesis_exp/scripts/run_exp51_cpu_smoke.sh thesis_exp/scripts/run_exp51_gpu_smoke.sh thesis_exp/scripts/run_exp51_preflight.sh thesis_exp/scripts/run_exp51_determinism_smoke.sh thesis_exp/scripts/run_exp51_seed42.sh thesis_exp/scripts/run_exp51_formal.sh "${SERVER}:${REMOTE_ROOT}/thesis_exp/scripts/"
