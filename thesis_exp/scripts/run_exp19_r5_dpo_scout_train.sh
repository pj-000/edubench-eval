#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-llama_factory}"
GPU_LIST="${GPU_LIST:-0 1 2 3}"
OUT_DIR="${OUT_DIR:-thesis_exp/exp17_low_score_evidence/outputs/exp19_r5_dpo_scout}"
CONFIG_ROOT="${CONFIG_ROOT:-${OUT_DIR}/train_configs}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}/logs}"
TABLE_DIR="${TABLE_DIR:-${OUT_DIR}/tables}"
MAX_STEPS="${MAX_STEPS:-100}"
PREF_BETA="${PREF_BETA:-0.05}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
DRY_RUN="${DRY_RUN:-0}"
CLEAN_CHECKPOINTS_AFTER_TRAIN="${CLEAN_CHECKPOINTS_AFTER_TRAIN:-1}"

RUN_NAMES=(
  "r5c_from_r2c"
  "r5c_from_r1b"
  "r5d_from_r2c"
  "r5e_from_r2c"
)
RUN_LABELS=(
  "R5C score-risk DPO from R2c"
  "R5C score-risk DPO from R1b"
  "R5D evidence-consistency DPO from R2c"
  "R5E hard-synthetic DPO control from R2c"
)
RUN_CONFIGS=(
  "thesis_exp/exp17_low_score_evidence/outputs/exp19_r5c_score_risk_dpo_seed42/configs/llamafactory_qwen3_4b_r5c_dpo_from_r2c.yaml"
  "thesis_exp/exp17_low_score_evidence/outputs/exp19_r5c_score_risk_dpo_seed42/configs/llamafactory_qwen3_4b_r5c_dpo_from_r1b.yaml"
  "thesis_exp/exp17_low_score_evidence/outputs/exp19_r5d_evidence_consistency_dpo_seed42/configs/llamafactory_qwen3_4b_r5d_dpo_from_r2c.yaml"
  "thesis_exp/exp17_low_score_evidence/outputs/exp19_r5e_hard_synthetic_dpo_control_seed42/configs/llamafactory_qwen3_4b_r5e_dpo_control_from_r2c.yaml"
)
RUN_OUTPUTS=(
  "saves/edubench/qwen3-4b/r5c_dpo_scout_from_r2c_maxsteps${MAX_STEPS}_lora"
  "saves/edubench/qwen3-4b/r5c_dpo_scout_from_r1b_maxsteps${MAX_STEPS}_lora"
  "saves/edubench/qwen3-4b/r5d_dpo_scout_from_r2c_maxsteps${MAX_STEPS}_lora"
  "saves/edubench/qwen3-4b/r5e_dpo_control_scout_from_r2c_maxsteps${MAX_STEPS}_lora"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export LOG_DIR TABLE_DIR MAX_STEPS PREF_BETA

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1: skipping conda activation for CONDA_ENV=${CONDA_ENV}"
elif [[ -f "${HOME}/miniconda3/bin/activate" ]]; then
  source "${HOME}/miniconda3/bin/activate" "${CONDA_ENV}"
fi

mkdir -p "${CONFIG_ROOT}" "${LOG_DIR}" "${TABLE_DIR}"

if [[ "${DRY_RUN}" != "1" ]] && ! command -v llamafactory-cli >/dev/null 2>&1; then
  echo "ERROR: llamafactory-cli not found in CONDA_ENV=${CONDA_ENV}" >&2
  exit 1
fi

python -m py_compile thesis_exp/exp17_low_score_evidence/validate_exp19_dpo_configs.py
python thesis_exp/exp17_low_score_evidence/validate_exp19_dpo_configs.py --out-dir "${OUT_DIR}"

IFS=' ' read -r -a GPUS <<< "${GPU_LIST}"
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "ERROR: GPU_LIST is empty" >&2
  exit 1
fi

cat <<CONFIG
Exp19-R5 small-step DPO scout training
CONDA_ENV=${CONDA_ENV}
GPU_LIST=${GPU_LIST}
OUT_DIR=${OUT_DIR}
CONFIG_ROOT=${CONFIG_ROOT}
LOG_DIR=${LOG_DIR}
MAX_STEPS=${MAX_STEPS}
PREF_BETA=${PREF_BETA}
SKIP_COMPLETED=${SKIP_COMPLETED}
DRY_RUN=${DRY_RUN}
CLEAN_CHECKPOINTS_AFTER_TRAIN=${CLEAN_CHECKPOINTS_AFTER_TRAIN}

Runs:
  r5c_from_r2c: R5C score-risk DPO from R2c
  r5c_from_r1b: R5C score-risk DPO from R1b
  r5d_from_r2c: R5D evidence-consistency DPO from R2c
  r5e_from_r2c: R5E hard-synthetic DPO control from R2c
CONFIG

ensure_dataset_info() {
  local config_path="$1"
  python - "${config_path}" <<'PY'
import json
import sys
from pathlib import Path

import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
dataset_dir = Path(config["dataset_dir"])
dataset = config["dataset"]
merged = {}
for path in sorted(dataset_dir.glob("dataset_info*.json")):
    if path.name == "dataset_info.json":
        continue
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"dataset info snippet is not an object: {path}")
    merged.update(data)
if dataset not in merged:
    raise SystemExit(f"dataset `{dataset}` not found in dataset_info snippets under {dataset_dir}")
target = dataset_dir / "dataset_info.json"
target.write_text(json.dumps(merged, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"DATASET_INFO_READY {target} dataset={dataset}")
PY
}

write_runtime_config() {
  local base_config="$1"
  local runtime_config="$2"
  local output_dir="$3"
  python - "${base_config}" "${runtime_config}" "${output_dir}" "${MAX_STEPS}" "${PREF_BETA}" <<'PY'
import sys
from pathlib import Path

import yaml

base_path = Path(sys.argv[1])
runtime_path = Path(sys.argv[2])
output_dir = sys.argv[3]
max_steps = int(sys.argv[4])
pref_beta = float(sys.argv[5])

config = yaml.safe_load(base_path.read_text(encoding="utf-8"))
if not isinstance(config, dict):
    raise SystemExit(f"{base_path} did not parse to a dict")
config["max_steps"] = max_steps
config["pref_beta"] = pref_beta
config["output_dir"] = output_dir
config["logging_steps"] = 5
config["save_steps"] = max_steps
config["save_only_model"] = True
config["plot_loss"] = True
config["report_to"] = "none"
config["overwrite_output_dir"] = True
runtime_path.parent.mkdir(parents=True, exist_ok=True)
runtime_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
print(f"RUNTIME_CONFIG_READY {runtime_path} output_dir={output_dir} max_steps={max_steps} pref_beta={pref_beta}")
PY
}

run_one() {
  local run_name="$1"
  local run_label="$2"
  local base_config="$3"
  local output_dir="$4"
  local gpu_id="$5"
  local runtime_config="${CONFIG_ROOT}/${run_name}_maxsteps${MAX_STEPS}_beta${PREF_BETA//./p}.yaml"
  local log_path="${LOG_DIR}/train_${run_name}_gpu${gpu_id}.log"

  if [[ ! -f "${base_config}" ]]; then
    echo "ERROR: missing base config for ${run_name}: ${base_config}" >&2
    exit 1
  fi

  write_runtime_config "${base_config}" "${runtime_config}" "${output_dir}"

  if [[ "${SKIP_COMPLETED}" == "1" && -f "${output_dir}/adapter_config.json" ]]; then
    echo "Skipping ${run_name}: completed adapter exists (${output_dir}/adapter_config.json)"
    return 0
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN ${run_label}: CUDA_VISIBLE_DEVICES=${gpu_id} llamafactory-cli train ${runtime_config}"
    return 0
  fi

  echo "Starting ${run_label} on GPU ${gpu_id}; config=${runtime_config}; log=${log_path}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" llamafactory-cli train "${runtime_config}" 2>&1 | tee "${log_path}"
  if [[ "${CLEAN_CHECKPOINTS_AFTER_TRAIN}" == "1" ]]; then
    find "${output_dir}" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' -print -exec rm -rf {} +
  fi
  echo "Completed ${run_label} on GPU ${gpu_id}"
}

run_queue() {
  local gpu_id="$1"
  shift
  local idx
  for idx in "$@"; do
    run_one "${RUN_NAMES[$idx]}" "${RUN_LABELS[$idx]}" "${RUN_CONFIGS[$idx]}" "${RUN_OUTPUTS[$idx]}" "${gpu_id}"
  done
}

for config_path in "${RUN_CONFIGS[@]}"; do
  ensure_dataset_info "${config_path}"
done

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

python - <<'PY'
import csv
import json
import os
import re
from pathlib import Path

run_names = ["r5c_from_r2c", "r5c_from_r1b", "r5d_from_r2c", "r5e_from_r2c"]
run_labels = [
    "R5C score-risk DPO from R2c",
    "R5C score-risk DPO from R1b",
    "R5D evidence-consistency DPO from R2c",
    "R5E hard-synthetic DPO control from R2c",
]
outputs = [
    f"saves/edubench/qwen3-4b/r5c_dpo_scout_from_r2c_maxsteps{os.environ.get('MAX_STEPS', '100')}_lora",
    f"saves/edubench/qwen3-4b/r5c_dpo_scout_from_r1b_maxsteps{os.environ.get('MAX_STEPS', '100')}_lora",
    f"saves/edubench/qwen3-4b/r5d_dpo_scout_from_r2c_maxsteps{os.environ.get('MAX_STEPS', '100')}_lora",
    f"saves/edubench/qwen3-4b/r5e_dpo_control_scout_from_r2c_maxsteps{os.environ.get('MAX_STEPS', '100')}_lora",
]
log_dir = Path(os.environ["LOG_DIR"])
table_dir = Path(os.environ["TABLE_DIR"])
table_dir.mkdir(parents=True, exist_ok=True)
rows = []
for run_name, run_label, output_dir in zip(run_names, run_labels, outputs):
    out = Path(output_dir)
    logs = sorted(log_dir.glob(f"train_{run_name}_gpu*.log"))
    log_text = logs[-1].read_text(errors="ignore") if logs else ""
    losses = re.findall(r"'loss': ([0-9.]+)|loss=([0-9.]+)", log_text)
    flat_losses = [next(item for item in pair if item) for pair in losses]
    rows.append(
        {
            "run_name": run_name,
            "run_label": run_label,
            "completed": (out / "adapter_config.json").exists(),
            "output_dir": str(out),
            "log_file": str(logs[-1]) if logs else "",
            "max_steps": os.environ.get("MAX_STEPS", "100"),
            "pref_beta": os.environ.get("PREF_BETA", "0.05"),
            "last_logged_loss": flat_losses[-1] if flat_losses else "",
        }
    )
path = table_dir / "exp19_r5_dpo_scout_training_summary.csv"
with path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "run_name",
            "run_label",
            "completed",
            "output_dir",
            "log_file",
            "max_steps",
            "pref_beta",
            "last_logged_loss",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps({"summary_csv": str(path), "runs": rows}, ensure_ascii=False))
PY

echo "Exp19-R5 DPO scout training script completed."
