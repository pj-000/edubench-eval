#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
SERVER_HOST="${SERVER_HOST:-jpang@39.103.98.135}"
SERVER_PORT="${SERVER_PORT:-23722}"
SERVER_REPO="${SERVER_REPO:-~/edubench-eval-exp2}"
RSYNC_SSH="ssh -p ${SERVER_PORT} -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=2"
FILES=(
  thesis_exp/exp49_cphce
  thesis_exp/configs/exp49_cphce
  thesis_exp/tests/test_exp49_metric_contract.py
  thesis_exp/tests/test_exp49_soft_targets.py
  thesis_exp/tests/test_exp49_input_parity.py
  thesis_exp/tests/test_exp49_checkpoint_rule.py
  thesis_exp/tests/test_exp49_no_test_access.py
  thesis_exp/tests/test_exp49_loss_equivalence.py
  thesis_exp/scripts/run_exp49_metric_audit.sh
  thesis_exp/scripts/run_exp49_cpu_smoke.sh
  thesis_exp/scripts/run_exp49_seed42.sh
  thesis_exp/scripts/run_exp49_formal.sh
  thesis_exp/scripts/run_exp49_freeze.sh
  thesis_exp/scripts/run_exp49_test_once.sh
  thesis_exp/scripts/sync_exp49_to_server.sh
  thesis_exp/data/splits/paper_like_triple_seed42/train.jsonl
  thesis_exp/data/splits/paper_like_triple_seed42/dev.jsonl
  thesis_exp/outputs/exp49_cphce/audit
)
rsync -azR \
  --exclude='__pycache__/' --exclude='*.pyc' --exclude='*.pt' --exclude='*.pth' \
  --exclude='*.bin' --exclude='*.safetensors' --exclude='*.npy' --exclude='*.npz' --exclude='*.log' \
  -e "${RSYNC_SSH}" "${FILES[@]}" "${SERVER_HOST}:${SERVER_REPO%/}/"
ssh -p "${SERVER_PORT}" -o BatchMode=yes -o ConnectTimeout=10 "${SERVER_HOST}" \
  "bash -lc 'source ~/miniconda3/bin/activate llama_factory && cd ${SERVER_REPO} && export PYTHONPATH=\"\$(pwd):\${PYTHONPATH:-}\" && chmod +x thesis_exp/scripts/run_exp49_*.sh thesis_exp/scripts/sync_exp49_to_server.sh && python -m py_compile thesis_exp/exp49_cphce/*.py && bash -n thesis_exp/scripts/run_exp49_*.sh thesis_exp/scripts/sync_exp49_to_server.sh && if python -c \"import pytest\" >/dev/null 2>&1; then python -m pytest -q thesis_exp/tests/test_exp49_metric_contract.py thesis_exp/tests/test_exp49_soft_targets.py thesis_exp/tests/test_exp49_input_parity.py thesis_exp/tests/test_exp49_checkpoint_rule.py thesis_exp/tests/test_exp49_no_test_access.py thesis_exp/tests/test_exp49_loss_equivalence.py; else python - <<\"PY\"
import torch
from thesis_exp.exp49_cphce.build_targets import aggregate_text_hash, load_split
from thesis_exp.exp49_cphce.losses import hard_cross_entropy, soft_cross_entropy
train = load_split(\"train\")
dev = load_split(\"dev\")
assert (len(train), len(dev)) == (2654, 664)
assert aggregate_text_hash(train) == \"67a03a285ba0bde1458318d0d8ffc86409156c574bebffe6760b1fd200b5d961\"
assert aggregate_text_hash(dev) == \"888db0fef20586e339ee2b2a152a2ccd3ecfd53589ca10912793ebad00e8f679\"
logits = torch.tensor([[0.2, -0.1, 0.4, 0.9, -0.7], [1.0, 0.0, -0.5, 0.2, 0.1]])
labels = torch.tensor([3, 0])
targets = torch.nn.functional.one_hot(labels, num_classes=5).float()
assert torch.allclose(hard_cross_entropy(logits, labels), soft_cross_entropy(logits, targets), atol=1e-7)
print(\"Remote fallback preflight passed (pytest unavailable)\")
PY
fi'"
echo "Exp49 code and train/dev inputs synced to the GPU server."
