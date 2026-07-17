#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -m py_compile thesis_exp/exp49_cphce/*.py
"${PYTHON_BIN}" -m pytest -q \
  thesis_exp/tests/test_exp49_metric_contract.py \
  thesis_exp/tests/test_exp49_soft_targets.py \
  thesis_exp/tests/test_exp49_input_parity.py \
  thesis_exp/tests/test_exp49_checkpoint_rule.py \
  thesis_exp/tests/test_exp49_no_test_access.py \
  thesis_exp/tests/test_exp49_loss_equivalence.py
"${PYTHON_BIN}" - <<'PY'
from thesis_exp.exp49_cphce.build_targets import aggregate_text_hash, load_split
train = load_split('train')
dev = load_split('dev')
print({'train': len(train), 'dev': len(dev), 'train_hash': aggregate_text_hash(train), 'dev_hash': aggregate_text_hash(dev)})
PY
