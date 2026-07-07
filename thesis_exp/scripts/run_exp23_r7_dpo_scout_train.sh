#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-llama_factory}"
GPU_LIST="${GPU_LIST:-0 1 2}"
OUT_DIR="${OUT_DIR:-thesis_exp/exp17_low_score_evidence/outputs/exp23_r7_dpo_scout}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}/logs}"
TABLE_DIR="${TABLE_DIR:-${OUT_DIR}/tables}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
DRY_RUN="${DRY_RUN:-0}"
CLEAN_CHECKPOINTS_AFTER_TRAIN="${CLEAN_CHECKPOINTS_AFTER_TRAIN:-1}"

RUN_NAMES=(
  "r7d_reason_real_s100_b0p03_lr5em6"
  "r7e_matched_score_only_s100_b0p03_lr5em6"
  "r7f_score_reason_consistency_s100_b0p03_lr5em6"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export LOG_DIR TABLE_DIR

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1: skipping conda activation for CONDA_ENV=${CONDA_ENV}"
elif [[ -f "${HOME}/miniconda3/bin/activate" ]]; then
  source "${HOME}/miniconda3/bin/activate" "${CONDA_ENV}"
fi

mkdir -p "${LOG_DIR}" "${TABLE_DIR}"

if [[ "${DRY_RUN}" != "1" ]] && ! command -v llamafactory-cli >/dev/null 2>&1; then
  echo "ERROR: llamafactory-cli not found in CONDA_ENV=${CONDA_ENV}" >&2
  exit 1
fi

python -m py_compile thesis_exp/exp17_low_score_evidence/prepare_exp23_r7_dpo_scout.py
python thesis_exp/exp17_low_score_evidence/prepare_exp23_r7_dpo_scout.py --out-dir "${OUT_DIR}"

IFS=' ' read -r -a GPUS <<< "${GPU_LIST}"
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "ERROR: GPU_LIST is empty" >&2
  exit 1
fi

cat <<CONFIG
Exp23 R7 DPO scout training
CONDA_ENV=${CONDA_ENV}
GPU_LIST=${GPU_LIST}
OUT_DIR=${OUT_DIR}
LOG_DIR=${LOG_DIR}
SKIP_COMPLETED=${SKIP_COMPLETED}
DRY_RUN=${DRY_RUN}
CLEAN_CHECKPOINTS_AFTER_TRAIN=${CLEAN_CHECKPOINTS_AFTER_TRAIN}

Runs:
  R7D: human-reason chosen real-error DPO
  R7E: matched score-only real-error DPO control
  R7F: score-reason consistency auxiliary scout
CONFIG

config_for_run() {
  local run_name="$1"
  echo "${OUT_DIR}/configs/llamafactory_qwen3_4b_${run_name}.yaml"
}

config_value() {
  local config_path="$1"
  local key="$2"
  python - "${config_path}" "${key}" <<'PY'
import sys
from pathlib import Path
import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(config.get(sys.argv[2], ""))
PY
}

check_dataset() {
  local config_path="$1"
  python - "${config_path}" <<'PY'
import json
import sys
from pathlib import Path
import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
dataset_dir = Path(config["dataset_dir"])
dataset_name = config["dataset"]
info_path = dataset_dir / "dataset_info.json"
info = json.loads(info_path.read_text(encoding="utf-8"))
if dataset_name not in info:
    raise SystemExit(f"dataset {dataset_name} missing from {info_path}")
data_file = dataset_dir / info[dataset_name]["file_name"]
if not data_file.exists():
    raise SystemExit(f"DPO data missing: {data_file}")
print(f"DATASET_READY dataset={dataset_name} data_file={data_file}")
PY
}

run_one() {
  local run_name="$1"
  local gpu_id="$2"
  local config_path
  local output_dir
  local init_adapter
  local log_path
  config_path="$(config_for_run "${run_name}")"

  if [[ ! -f "${config_path}" ]]; then
    echo "ERROR: missing config for ${run_name}: ${config_path}" >&2
    exit 1
  fi
  check_dataset "${config_path}"
  output_dir="$(config_value "${config_path}" output_dir)"
  init_adapter="$(config_value "${config_path}" adapter_name_or_path)"
  log_path="${LOG_DIR}/train_${run_name}_gpu${gpu_id}.log"

  if [[ ! -f "${init_adapter}/adapter_config.json" ]]; then
    echo "ERROR: missing init adapter for ${run_name}: ${init_adapter}/adapter_config.json" >&2
    exit 1
  fi
  if [[ "${SKIP_COMPLETED}" == "1" && -f "${output_dir}/adapter_config.json" ]]; then
    echo "Skipping ${run_name}: completed adapter exists (${output_dir}/adapter_config.json)"
    return 0
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN ${run_name}: CUDA_VISIBLE_DEVICES=${gpu_id} llamafactory-cli train ${config_path}"
    return 0
  fi

  echo "Starting ${run_name} on GPU ${gpu_id}; config=${config_path}; log=${log_path}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" llamafactory-cli train "${config_path}" 2>&1 | tee "${log_path}"
  if [[ "${CLEAN_CHECKPOINTS_AFTER_TRAIN}" == "1" ]]; then
    find "${output_dir}" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' -print -exec rm -rf {} +
  fi
  echo "Completed ${run_name} on GPU ${gpu_id}"
}

run_queue() {
  local gpu_id="$1"
  shift
  local idx
  for idx in "$@"; do
    run_one "${RUN_NAMES[$idx]}" "${gpu_id}"
  done
}

queues=()
for _ in "${GPUS[@]}"; do
  queues+=("")
done
for idx in "${!RUN_NAMES[@]}"; do
  gpu_slot=$((idx % ${#GPUS[@]}))
  queues[$gpu_slot]="${queues[$gpu_slot]} ${idx}"
done

pids=()
for slot in "${!GPUS[@]}"; do
  read -r -a queue_indices <<< "${queues[$slot]}"
  if [[ "${#queue_indices[@]}" -eq 0 ]]; then
    continue
  fi
  echo "GPU ${GPUS[$slot]} queue:${queues[$slot]}"
  run_queue "${GPUS[$slot]}" "${queue_indices[@]}" &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

python - "${OUT_DIR}" "${LOG_DIR}" "${TABLE_DIR}" <<'PY'
import csv
import json
import re
import sys
from pathlib import Path
import yaml

out_dir = Path(sys.argv[1])
log_dir = Path(sys.argv[2])
table_dir = Path(sys.argv[3])
run_matrix = list(csv.DictReader((out_dir / "tables" / "exp23_run_matrix.csv").open("r", encoding="utf-8")))
rows = []
for run in run_matrix:
    run_name = run["run_name"]
    config_path = out_dir / "configs" / f"llamafactory_qwen3_4b_{run_name}.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = Path(str(config["output_dir"]))
    logs = sorted(log_dir.glob(f"train_{run_name}_gpu*.log"))
    log_text = logs[-1].read_text(errors="ignore") if logs else ""
    losses = re.findall(r"'loss': ([0-9.]+)|loss=([0-9.]+)", log_text)
    flat_losses = [next(item for item in pair if item) for pair in losses]
    rows.append(
        {
            "run_name": run_name,
            "dataset_family": run.get("dataset_family", ""),
            "dataset": config.get("dataset", ""),
            "completed": (output_dir / "adapter_config.json").exists(),
            "output_dir": str(output_dir),
            "log_file": str(logs[-1]) if logs else "",
            "max_steps": config.get("max_steps", ""),
            "pref_beta": config.get("pref_beta", ""),
            "pref_ftx": config.get("pref_ftx", ""),
            "learning_rate": config.get("learning_rate", ""),
            "last_logged_loss": flat_losses[-1] if flat_losses else "",
        }
    )
path = table_dir / "exp23_r7_dpo_scout_training_summary.csv"
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "run_name",
            "dataset_family",
            "dataset",
            "completed",
            "output_dir",
            "log_file",
            "max_steps",
            "pref_beta",
            "pref_ftx",
            "learning_rate",
            "last_logged_loss",
        ],
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps({"summary_csv": str(path), "runs": rows}, ensure_ascii=False))
PY
