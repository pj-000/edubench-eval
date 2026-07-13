#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ "${RUN_API:-0}" != "1" ]]; then
  echo "Blocked: set RUN_API=1 to call Qwen and DeepSeek for the frozen 240 rows." >&2
  exit 2
fi

python thesis_exp/exp39_educfa/prepare_exp39a_source_anchors.py
python thesis_exp/exp39_educfa/run_exp39a_qwen_counterfactual_generation.py
python thesis_exp/exp39_educfa/run_exp39a_deepseek_blind_verification.py
python thesis_exp/exp39_educfa/validate_exp39a_counterfactuals.py

DECISION="thesis_exp/exp39_educfa/outputs/exp39a_educfa_seed42/decision/exp39a_data_qualification_decision.json"
if python - "${DECISION}" <<'PY'
import json, pathlib, sys
decision = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
raise SystemExit(0 if decision.get("recommend_groupcv_training") else 1)
PY
then
  python thesis_exp/exp39_educfa/build_exp39a_training_variants.py
  python thesis_exp/exp39_educfa/prepare_exp39a_groupcv_folds.py
  echo "Exp39A data qualification GO; V0H-V5 and GroupCV folds are ready."
else
  echo "Exp39A data qualification NO-GO; stopped before variant construction and GPU training."
fi
