#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${EDUBENCH_REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}"
PYTHON="${PYTHON:-/home/jpang/miniconda3/envs/llama_factory/bin/python}"
OUT="thesis_exp/exp46_hato_kd/outputs/exp46a_hato_seed42"
mkdir -p "${OUT}/state" "${OUT}/logs_private"
trap 'code=$?; printf "%s\n" "${code}" >"'"${OUT}"'/state/goal_exit_code.txt"' EXIT

"${PYTHON}" -m compileall -q thesis_exp/exp46_hato_kd
for script in thesis_exp/scripts/run_exp46a_*.sh; do bash -n "${script}"; done
"${PYTHON}" -c 'from thesis_exp.exp46_hato_kd.common import write_protocol_locks; write_protocol_locks()'

bash thesis_exp/scripts/run_exp46a_teacher_smoke.sh
bash thesis_exp/scripts/run_exp46a_student_smoke.sh
bash thesis_exp/scripts/run_exp46a_teacher_groupcv.sh
bash thesis_exp/scripts/run_exp46a_teacher_collect.sh
teacher_status=$("${PYTHON}" -c 'import json; print(json.load(open("thesis_exp/exp46_hato_kd/outputs/exp46a_hato_seed42/decision/exp46a_teacher_capacity_decision.json"))["status"])')
printf "%s\n" "${teacher_status}" >"${OUT}/state/teacher_status.txt"
if [[ "${teacher_status}" != "TEACHER_CAPACITY_GO" ]]; then
  printf "%s\n" "${teacher_status}" >"${OUT}/state/final_status.txt"
  echo "Exp46A stopped at preregistered teacher gate: ${teacher_status}"
  exit 0
fi

bash thesis_exp/scripts/run_exp46a_student_groupcv.sh
bash thesis_exp/scripts/run_exp46a_student_collect.sh
student_status=$("${PYTHON}" -c 'import json; print(json.load(open("thesis_exp/exp46_hato_kd/outputs/exp46a_hato_seed42/decision/exp46a_student_transfer_decision.json"))["status"])')
printf "%s\n" "${student_status}" >"${OUT}/state/final_status.txt"
echo "Exp46A completed: ${student_status}"
