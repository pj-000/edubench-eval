#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${EDUBENCH_REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-/home/jpang/miniconda3/envs/llama_factory/bin/python}"
MODEL="${RUBIMOR_MODEL_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"
GPU_LIST="${GPU_LIST:-0 1 2 3}"
OUT="thesis_exp/exp44_taco_score/outputs/exp44a_taco_seed42"
RUN_ROOT="thesis_exp/runs/exp44_taco_score"
ARTIFACT_ROOT="thesis_exp/artifacts/exp44_taco_score"
read -r -a GPUS <<< "${GPU_LIST}"
VARIANTS=(C0_E4_baseline C1_balanced_plain_contrastive C2_TACO C3_shuffled_margin_control)
[[ ${#GPUS[@]} -gt 0 ]] || { echo "GPU_LIST is empty" >&2; exit 2; }

CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${PYTHON}" -m thesis_exp.exp44_taco_score.audit_exp44a_loss_scales \
  --model-name-or-path "${MODEL}"

PIDS=()
for index in "${!VARIANTS[@]}"; do
  variant="${VARIANTS[$index]}"
  gpu="${GPUS[$((index % ${#GPUS[@]}))]}"
  log="${OUT}/logs_private/smoke_${variant}_gpu${gpu}.log"
  mkdir -p "$(dirname "${log}")"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -m thesis_exp.exp44_taco_score.train_exp44a_groupcv \
    --variant "${variant}" --mode smoke --fold 0 --seed 42 \
    --model-name-or-path "${MODEL}" --out-dir "${OUT}" --run-root "${RUN_ROOT}" --artifact-root "${ARTIFACT_ROOT}" \
    --epochs 1 --learning-rate 2e-5 --weight-decay .01 --warmup-ratio .05 \
    --batch-size 1 --eval-batch-size 1 --gradient-accumulation 1 --max-length 2048 \
    --max-train-rows 32 --max-eval-rows 32 --max-updates 1 >"${log}" 2>&1 &
  PIDS+=("$!")
done
failed=0
for pid in "${PIDS[@]}"; do wait "${pid}" || failed=1; done

"${PYTHON}" - "${OUT}" "${RUN_ROOT}" "${failed}" <<'PY'
import json,pathlib,sys
out=pathlib.Path(sys.argv[1]); root=pathlib.Path(sys.argv[2]); failed=int(sys.argv[3])
variants=("C0_E4_baseline","C1_balanced_plain_contrastive","C2_TACO","C3_shuffled_margin_control")
rows=[]
for variant in variants:
    path=root/"smoke"/variant/"seed_42"/"fold_0"/"run_summary.json"
    row=json.loads(path.read_text()) if path.exists() else {}
    rows.append({"variant":variant,"exists":path.exists(),"status":row.get("status"),"save_reload":row.get("smoke_save_reload"),"nan_count":row.get("nan_count",0),"oom_count":row.get("oom_count",0),"dev_access_count":row.get("dev_access_count",0),"test_access_count":row.get("test_access_count",0)})
passed=not failed and all(r["status"]=="COMPLETED" and r["save_reload"]=="PASS" and r["nan_count"]==r["oom_count"]==r["dev_access_count"]==r["test_access_count"]==0 for r in rows)
decision={"status":"SMOKE_GO" if passed else "SMOKE_NO_GO","runs":rows,"dev_access_count":0,"test_access_count":0}
path=out/"decision/exp44a_smoke_decision.json"; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(decision,indent=2,sort_keys=True)+"\n")
print(json.dumps(decision,sort_keys=True))
raise SystemExit(0 if passed else 2)
PY

