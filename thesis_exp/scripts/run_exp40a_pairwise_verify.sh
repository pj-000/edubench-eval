#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

OUT_DIR="thesis_exp/exp40_edupair_cf/outputs/exp40a_edupair_cf_seed42"
if [[ ! -f "${OUT_DIR}/private/pair_packets/exp40a_pairwise_verification_packets.jsonl" ]]; then
  ./thesis_exp/scripts/run_exp40a_prepare_pairs.sh
fi

if [[ "${RUN_API:-0}" != "1" ]]; then
  python thesis_exp/exp40_edupair_cf/run_exp40a_deepseek_pairwise_verification.py --max-pairs 1 --dry-run
  echo "Dry-run only. Set RUN_API=1 and DEEPSEEK_API_KEY to execute all 480 judgments."
  exit 0
fi

python thesis_exp/exp40_edupair_cf/run_exp40a_deepseek_pairwise_verification.py
python thesis_exp/exp40_edupair_cf/validate_exp40a_pairwise_outputs.py --out-dir "${OUT_DIR}"

if python - "${OUT_DIR}/decision/exp40a_pairwise_qualification_decision.json" <<'PY'
import json, pathlib, sys
decision = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
raise SystemExit(0 if decision.get("recommend_groupcv_training") else 1)
PY
then
  python thesis_exp/exp40_edupair_cf/build_exp40a_pairwise_datasets.py --out-dir "${OUT_DIR}"
  python thesis_exp/exp40_edupair_cf/prepare_exp40a_groupcv_folds.py --out-dir "${OUT_DIR}"
  echo "Exp40A pairwise qualification GO; pair variants and GroupCV folds are ready."
else
  echo "Exp40A pairwise qualification NO-GO; GroupCV was not prepared or run."
fi
