#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ "${RUN_API:-0}" != "1" ]]; then
  echo "Blocked: set RUN_API=1 for the frozen Qwen rubric compiler." >&2
  exit 2
fi

OUT_DIR="thesis_exp/exp41_rubric_bridge/outputs/exp41a_rubric_bridge_groupcv_seed42"
TOKENIZER_NAME_OR_PATH="${TOKENIZER_NAME_OR_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"
python thesis_exp/exp41_rubric_bridge/prepare_exp41a_rubric_units.py --out-dir "${OUT_DIR}"
python thesis_exp/exp41_rubric_bridge/run_exp41a_qwen_rubric_compiler.py --out-dir "${OUT_DIR}" --workers "${API_WORKERS:-8}"
python thesis_exp/exp41_rubric_bridge/validate_exp41a_compiled_rubrics.py \
  --out-dir "${OUT_DIR}" --tokenizer-name-or-path "${TOKENIZER_NAME_OR_PATH}"

if ! python - "${OUT_DIR}/decision/exp41a_compiler_qualification_decision.json" <<'PY'
import json, pathlib, sys
decision = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if not decision.get("recommend_groupcv_training"):
    print("Exp41A compiler qualification NO-GO; downstream variant/GroupCV construction is correctly blocked.")
    raise SystemExit(1)
PY
then
  exit 0
fi

python thesis_exp/exp41_rubric_bridge/build_exp41a_training_variants.py \
  --out-dir "${OUT_DIR}" --tokenizer-name-or-path "${TOKENIZER_NAME_OR_PATH}"
python thesis_exp/exp41_rubric_bridge/prepare_exp41a_groupcv_folds.py --out-dir "${OUT_DIR}"
echo "Exp41A compiler qualification GO; six formal variants and five GroupCV folds are prepared."
