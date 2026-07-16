#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

python -m thesis_exp.exp47_label2_identifiability.resolve_exp47_inputs "$@"
python -m thesis_exp.exp47_label2_identifiability.audit_exp47_human_label_patterns
python -m thesis_exp.exp47_label2_identifiability.audit_exp47_label2_concentration
