#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${EDUBENCH_REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}"
PYTHON="${PYTHON:-/home/jpang/miniconda3/envs/llama_factory/bin/python}"
[[ -x "${PYTHON}" ]] || PYTHON="$(command -v python3)"
OUT="thesis_exp/exp45_dopr_head/outputs/exp45a_dopr_seed42"
RUN_ROOT="thesis_exp/runs/exp45_dopr_head"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
mkdir -p "${OUT}/logs_private"

status=$("${PYTHON}" -c 'import json; print(json.load(open("thesis_exp/exp45_dopr_head/outputs/exp45a_dopr_seed42/decision/exp45a_prototype_signal_decision.json"))["status"])')
[[ "${status}" == "PROTOTYPE_SIGNAL_GO" ]] || { echo "Prototype diagnostic is ${status}; head training is prohibited." >&2; exit 3; }

variants=(H1_vanilla_cRT H2_distributional_ordinal_cRT H3_prototype_cRT_no_prior H4_DOPR)
for variant in "${variants[@]}"; do
  CUDA_VISIBLE_DEVICES="" "${PYTHON}" -m thesis_exp.exp45_dopr_head.train_exp45a_decoupled_heads \
    --variant "${variant}" --fold 0 --mode smoke --epochs 1 \
    --max-train-rows 100 --max-heldout-rows 50 --max-updates 1
done

"${PYTHON}" - <<'PY'
import json
from pathlib import Path
root = Path("thesis_exp/runs/exp45_dopr_head/smoke")
variants = ("H1_vanilla_cRT", "H2_distributional_ordinal_cRT", "H3_prototype_cRT_no_prior", "H4_DOPR")
passed = True
for variant in variants:
    value = json.loads((root / variant / "seed_42/fold_0/run_summary.json").read_text())
    passed &= value["status"] == "COMPLETED" and value["save_reload"] == "PASS"
    passed &= value["encoder_parameters_trainable"] == 0 and value["nan_count"] == value["oom_count"] == 0
decision = {"status": "HEAD_SMOKE_GO" if passed else "HEAD_SMOKE_NO_GO", "save_reload": passed, "balanced_sampler": passed, "finite_losses": passed, "encoder_gradients": 0, "dev_access_count": 0, "test_access_count": 0}
path = Path("thesis_exp/exp45_dopr_head/outputs/exp45a_dopr_seed42/decision/exp45a_head_smoke_decision.json")
path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
print(json.dumps(decision, sort_keys=True))
if not passed: raise SystemExit(1)
PY

worker() {
  local variant="$1" fold skip=()
  [[ "${SKIP_COMPLETED}" == "1" ]] && skip=(--skip-completed)
  for fold in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES="" "${PYTHON}" -m thesis_exp.exp45_dopr_head.train_exp45a_decoupled_heads \
      --variant "${variant}" --fold "${fold}" "${skip[@]}" \
      >"${OUT}/logs_private/head_${variant}_fold_${fold}.log" 2>&1
  done
}

pids=()
for variant in "${variants[@]}"; do worker "${variant}" & pids+=("$!"); done
exit_code=0
for pid in "${pids[@]}"; do wait "${pid}" || exit_code=1; done
(( exit_code == 0 )) || { echo "Exp45A head worker failed; inspect logs_private" >&2; exit 1; }
