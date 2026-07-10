#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_DIR="${OUT_DIR:-thesis_exp/exp17_low_score_evidence/outputs/exp27n_selective_model_adjudication_seed42}"
ANNOTATIONS="${ANNOTATIONS:-${OUT_DIR}/private/exp27n_gpt56pro_selective_adjudication_54.raw.jsonl}"

if [[ ! -f "${ANNOTATIONS}" ]]; then
  echo "Missing returned Exp27N annotations: ${ANNOTATIONS}" >&2
  echo "Set ANNOTATIONS=/absolute/path/to/exp27n_gpt56pro_selective_adjudication_54.jsonl" >&2
  exit 2
fi

"${PYTHON_BIN}" -m thesis_exp.exp17_low_score_evidence.ingest_exp27n_selective_model_adjudication \
  --annotations "${ANNOTATIONS}" \
  --out-dir "${OUT_DIR}"
"${PYTHON_BIN}" -m thesis_exp.exp17_low_score_evidence.validate_exp27n_selective_model_adjudication_ingest \
  --out-dir "${OUT_DIR}" \
  --require-private

echo "Exp27N returned adjudications ingested: ${OUT_DIR}"
