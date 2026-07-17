#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python}"
"${PYTHON_BIN}" -m py_compile thesis_exp/exp50_cahs/*.py
"${PYTHON_BIN}" -m pytest -q \
  thesis_exp/tests/test_exp50_targets.py \
  thesis_exp/tests/test_exp50_loss_equivalence.py \
  thesis_exp/tests/test_exp50_contract.py
