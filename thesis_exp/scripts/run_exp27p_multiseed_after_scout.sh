#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DECISION="thesis_exp/exp17_low_score_evidence/outputs/exp27p_soft_target_reranker_seed42/decision/exp27p_seed42_scout_decision.json"

if [[ ! -f "${DECISION}" ]]; then
  echo "Seed42 decision is not available: ${DECISION}" >&2
  exit 2
fi
if ! python -c 'import json,sys;sys.exit(0 if json.load(open(sys.argv[1])).get("recommend_run_seeds_43_44") else 1)' "${DECISION}"; then
  echo "Seed42 does not authorize seeds 43/44." >&2
  exit 2
fi

cat <<'PLAN'
Exp27P multiseed is authorized but intentionally not auto-executed.
Locked future seeds: 43 44
Locked variants: v0 v1 v2 v3
Use the same four-GPU allocation and training configuration as seed42.
A dedicated multiseed collector must be reviewed before these runs are launched.
PLAN
