#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
FAMILIES="${FAMILIES:-thesis_exp/exp48_eduq_tail/outputs/exp48a_qualification_pilot/private/generated_families/exp48a_generated_families.jsonl}"
OUT_DIR="${OUT_DIR:-thesis_exp/exp48_eduq_tail/outputs/exp48a_qualification_pilot}"
python -m thesis_exp.exp48_eduq_tail.prepare_exp48a_blind_verifier_packets --families "${FAMILIES}" --out-dir "${OUT_DIR}"
