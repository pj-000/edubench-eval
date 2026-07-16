#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

cat <<'EOF'
Exp47A reuses existing logits and OOF predictions.
No model loading, new inference, training, API, dev access, or test access is performed.
EOF

python -m thesis_exp.exp47_label2_identifiability.evaluate_exp47_train_vs_heldout
python -m thesis_exp.exp47_label2_identifiability.analyze_exp47_class2_logits
