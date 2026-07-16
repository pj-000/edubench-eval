#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

./thesis_exp/scripts/run_exp47a_prepare.sh
./thesis_exp/scripts/run_exp47a_inference.sh
./thesis_exp/scripts/run_exp47a_collect.sh

cat thesis_exp/exp47_label2_identifiability/outputs/exp47a_label2_audit/decision/exp47_label2_identifiability_decision.json
