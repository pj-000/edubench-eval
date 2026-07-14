#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO_ROOT="${EDUBENCH_REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"; cd "${REPO_ROOT}"
PYTHON="${PYTHON:-/home/jpang/miniconda3/envs/llama_factory/bin/python}"; export PYTHON RUBIMOR_MODEL_PATH="${RUBIMOR_MODEL_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}" GPU_LIST="${GPU_LIST:-0 1 2 3}" SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
OUT="thesis_exp/exp43_rubimor/outputs/exp43_rubimor_preregistered"; mkdir -p "${OUT}/state"
git merge-base --is-ancestor 4843230 HEAD || { echo "HEAD does not descend from locked 4843230" >&2; exit 2; }
state(){ "${PYTHON}" - "$OUT/state/$1.json" "$2" <<'PY'
import json,pathlib,sys,tempfile,os,datetime
p=pathlib.Path(sys.argv[1]);p.parent.mkdir(parents=True,exist_ok=True);v={"status":sys.argv[2],"timestamp":datetime.datetime.now(datetime.timezone.utc).isoformat()};tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+"\n");os.replace(tmp,p)
PY
}
decision(){ "${PYTHON}" - "$1" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["status"])
PY
}
run_stage(){ local name="$1" script="$2";state "$name" STARTED;bash "$script";state "$name" COMPLETED;}
run_stage stage0 thesis_exp/scripts/run_exp43_stage0_audit.sh
[[ "$(decision "$OUT/decision/exp43_stage0_decision.json")" == GO && "$(decision "$OUT/decision/exp43_pair_data_decision.json")" == GO && "$(decision "$OUT/decision/exp43_loss_scale_decision.json")" == GO ]] || { "${PYTHON}" -m thesis_exp.exp43_rubimor.build_exp43_final_report; exit 0; }
run_stage stage1 thesis_exp/scripts/run_exp43_stage1_smoke.sh; [[ "$(decision "$OUT/decision/exp43_smoke_decision.json")" == GO ]] || { "${PYTHON}" -m thesis_exp.exp43_rubimor.build_exp43_final_report; exit 0; }
run_stage stage2 thesis_exp/scripts/run_exp43_stage2_baselines_seed42.sh; [[ "$(decision "$OUT/decision/exp43_baseline_pipeline_decision.json")" == GO ]] || { "${PYTHON}" -m thesis_exp.exp43_rubimor.build_exp43_final_report; exit 0; }
run_stage stage3 thesis_exp/scripts/run_exp43_stage3_ordinal_seed42.sh; [[ "$(decision "$OUT/decision/exp43_ordinal_decision.json")" == GO ]] || { "${PYTHON}" -m thesis_exp.exp43_rubimor.build_exp43_final_report; exit 0; }
run_stage stage4 thesis_exp/scripts/run_exp43_stage4_metric_head_seed42.sh; [[ "$(decision "$OUT/decision/exp43_metric_head_decision.json")" == GO ]] || { "${PYTHON}" -m thesis_exp.exp43_rubimor.build_exp43_final_report; exit 0; }
run_stage stage5 thesis_exp/scripts/run_exp43_stage5_pairwise_seed42.sh; [[ "$(decision "$OUT/decision/exp43_pairwise_decision.json")" == GO ]] || { "${PYTHON}" -m thesis_exp.exp43_rubimor.build_exp43_final_report; exit 0; }
run_stage stage6 thesis_exp/scripts/run_exp43_stage6_multiseed_groupcv.sh
group="$(decision "$OUT/decision/exp43_groupcv_decision.json")"; [[ "$group" == RUBIMOR_FULL_GROUPCV_GO || "$group" == RUBIMOR_OVERALL_GROUPCV_GO ]] || { "${PYTHON}" -m thesis_exp.exp43_rubimor.build_exp43_final_report; exit 0; }
"${PYTHON}" - "$OUT/reports/exp43_question_disjoint_robustness.md" "$group" <<'PY'
import pathlib,sys
pathlib.Path(sys.argv[1]).write_text(f"# Exp43 Question-Disjoint Robustness\n\nGroupCV decision: **{sys.argv[2]}**. Five outer folds isolate question keys; no dev/test data were used.\n")
PY
run_stage stage8 thesis_exp/scripts/run_exp43_stage8_headline_dev.sh; [[ "$(decision "$OUT/decision/exp43_headline_dev_decision.json")" == HEADLINE_DEV_GO ]] || { "${PYTHON}" -m thesis_exp.exp43_rubimor.build_exp43_final_report; exit 0; }
if [[ "${AUTO_FINAL_TEST:-1}" == 1 ]];then run_stage stage9 thesis_exp/scripts/run_exp43_stage9_final_test.sh;fi
"${PYTHON}" -m thesis_exp.exp43_rubimor.build_exp43_final_report

