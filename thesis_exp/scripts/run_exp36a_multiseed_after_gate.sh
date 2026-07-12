#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
RUN_MULTISEED="${RUN_MULTISEED:-0}"
DECISION="thesis_exp/exp36_safer_score/outputs/exp36a_safer_score_seed42/decision/exp36a_seed42_decision.json"
if [[ "${RUN_MULTISEED}" != "1" ]]; then
  echo "Exp36A multiseed is generated but locked. Set RUN_MULTISEED=1 only after seed42 GO."; exit 0
fi
python -c 'import json,sys;d=json.load(open(sys.argv[1]));sys.exit(0 if d.get("recommend_run_seeds_43_44") is True else 1)' "${DECISION}"
echo "Seed42 gate permits seeds43/44, but the formal multiseed execution implementation remains intentionally separate from the seed42 scout."

