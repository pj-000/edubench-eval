#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f "${HOME}/miniconda3/bin/activate" ]]; then
  source "${HOME}/miniconda3/bin/activate" "${CONDA_ENV:-llama_factory}"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_FINAL_TEST="${RUN_FINAL_TEST:-0}"
RESUME_INCOMPLETE="${RESUME_INCOMPLETE:-0}"
GPU_LIST="${GPU_LIST:-0 1 2}"
OUT_DIR="thesis_exp/exp17_low_score_evidence/outputs/exp27r_final_test_campaign_seed42_44"
TEST_JSONL="thesis_exp/data/splits/question_seed42/test.jsonl"
ACCESS_MANIFEST="${OUT_DIR}/private/exp27r_test_access_manifest.json"
LOCK_MANIFEST="${OUT_DIR}/configs/exp27r_final_lock_manifest.json"
EXP27R_SOURCE_COMMIT="$("${PYTHON_BIN}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["locked_source_commit"])' "${LOCK_MANIFEST}")"
export EXP27R_SOURCE_COMMIT
VARIANTS="v0_original_unweighted v1_original_label_matched_weight v2_selective_hard_relabel v3_selective_soft_audit v3_safe16_original_low_anchor"
SEEDS="42 43 44"

if [[ "${RUN_FINAL_TEST}" != "1" ]]; then
  echo "RUN_FINAL_TEST=1 is required for the one-shot campaign." >&2
  exit 2
fi
"${PYTHON_BIN}" thesis_exp/exp17_low_score_evidence/validate_exp27r_final_test_lock.py --out-dir "${OUT_DIR}"

if [[ -f "${ACCESS_MANIFEST}" ]]; then
  status="$("${PYTHON_BIN}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["completion_status"])' "${ACCESS_MANIFEST}")"
  if [[ "${status}" == "completed" ]]; then
    echo "Final test campaign is already completed and permanently closed." >&2
    exit 2
  fi
  if [[ "${RESUME_INCOMPLETE}" != "1" ]]; then
    echo "An incomplete campaign exists; set RESUME_INCOMPLETE=1 to resume only it." >&2
    exit 2
  fi
else
  mkdir -p "$(dirname "${ACCESS_MANIFEST}")"
  RUN_FINAL_TEST=1 "${PYTHON_BIN}" - "${TEST_JSONL}" "${LOCK_MANIFEST}" "${ACCESS_MANIFEST}" "${VARIANTS}" "${SEEDS}" <<'PY'
import datetime, hashlib, json, subprocess, sys
from pathlib import Path
test, lock, output, variants, seeds = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4].split(), [int(x) for x in sys.argv[5].split()]
def sha(path):
 d=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''): d.update(chunk)
 return d.hexdigest()
payload={
 'test_file_sha256':sha(test),'commit_sha':json.load(open(lock,encoding='utf-8'))['locked_source_commit'],
 'lock_manifest_sha256':sha(lock),'campaign_start_timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'campaign_end_timestamp':None,'variants':variants,'seeds':seeds,'command':'run_exp27r_final_test_campaign.sh',
 'completion_status':'incomplete'
}
output.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n',encoding='utf-8')
PY
fi

read -r -a GPUS <<<"${GPU_LIST}"
read -r -a SEED_ARRAY <<<"${SEEDS}"
read -r -a VARIANT_ARRAY <<<"${VARIANTS}"
if [[ "${#GPUS[@]}" -lt 3 ]]; then
  echo "Provide at least three GPU IDs for the locked campaign." >&2
  exit 2
fi
PURE_ENABLED="$("${PYTHON_BIN}" -c 'import json,sys;print(1 if json.load(open(sys.argv[1]))["pure_min_sensitivity_enabled"] else 0)' "${LOCK_MANIFEST}")"
mkdir -p "${OUT_DIR}/logs_private"

run_seed() {
  local seed="$1" gpu="$2"
  for variant in "${VARIANT_ARRAY[@]}"; do
    kinds=(selected)
    if [[ "${PURE_ENABLED}" == "1" ]]; then kinds+=(pure_min_mae); fi
    for kind in "${kinds[@]}"; do
      output="${OUT_DIR}/predictions_private/${kind}/${variant}/seed_${seed}.jsonl"
      if [[ "${RESUME_INCOMPLETE}" == "1" && -f "${output}" ]]; then
        echo "Resume: skipping existing ${kind} ${variant} seed ${seed}"
        continue
      fi
      echo "Exp27R test inference ${kind} ${variant} seed ${seed} on GPU ${gpu}"
      RUN_FINAL_TEST=1 CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -m thesis_exp.src.edujudge.exp27r.evaluate_exp27r_frozen_checkpoints \
        --variant "${variant}" --seed "${seed}" --checkpoint-kind "${kind}" \
        --test-jsonl "${TEST_JSONL}" --out-dir "${OUT_DIR}" --batch-size 4
    done
  done
}

pids=()
for index in "${!SEED_ARRAY[@]}"; do
  run_seed "${SEED_ARRAY[$index]}" "${GPUS[$index]}" >"${OUT_DIR}/logs_private/seed_${SEED_ARRAY[$index]}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do if ! wait "${pid}"; then failed=1; fi; done
if [[ "${failed}" != "0" ]]; then
  echo "Final test inference failed; access manifest remains incomplete." >&2
  exit 1
fi

"${PYTHON_BIN}" -m thesis_exp.src.edujudge.exp27r.collect_exp27r_final_test \
  --out-dir "${OUT_DIR}" --cluster-resamples 2000 --crossed-resamples 5000
"${PYTHON_BIN}" -m thesis_exp.src.edujudge.exp27r.validate_exp27r_final_test --out-dir "${OUT_DIR}"

"${PYTHON_BIN}" - "${ACCESS_MANIFEST}" <<'PY'
import datetime,json,sys
p=sys.argv[1]; d=json.load(open(p)); d['completion_status']='completed'; d['campaign_end_timestamp']=datetime.datetime.now(datetime.timezone.utc).isoformat(); open(p,'w').write(json.dumps(d,indent=2,sort_keys=True)+'\n')
PY
echo "Exp27R one-shot final test campaign completed and closed."
