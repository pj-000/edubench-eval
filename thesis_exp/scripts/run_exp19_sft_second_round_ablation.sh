#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-llama_factory}"
GPU_LIST="${GPU_LIST:-0 1 2 3}"
DATASET_DIR="${DATASET_DIR:-thesis_exp/exp17_low_score_evidence/outputs/exp19_sft_dpo_datasets_seed42}"
CONFIG_DIR="${CONFIG_DIR:-${DATASET_DIR}/configs}"
OUT_DIR="${OUT_DIR:-thesis_exp/exp17_low_score_evidence/outputs/exp19_sft_second_round}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}/logs}"
TABLE_DIR="${TABLE_DIR:-${OUT_DIR}/tables}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

RUN_NAMES=(
  "r1b_score_only_balanced"
  "r2n_reason_score_natural"
  "r2c_clean_reason_score_balanced"
  "r4b_shuffled_reason_balanced"
)
RUN_LABELS=(
  "R1b score-only balanced"
  "R2n reason-score natural"
  "R2c clean reason-score balanced"
  "R4b shuffled reason balanced"
)
RUN_CONFIGS=(
  "${CONFIG_DIR}/llamafactory_qwen3_4b_r1_score_only_balanced_lora.yaml"
  "${CONFIG_DIR}/llamafactory_qwen3_4b_r2_reason_score_natural_lora.yaml"
  "${CONFIG_DIR}/llamafactory_qwen3_4b_r2_clean_reason_score_balanced_lora.yaml"
  "${CONFIG_DIR}/llamafactory_qwen3_4b_r4_shuffled_reason_balanced_lora.yaml"
)
RUN_OUTPUTS=(
  "saves/edubench/qwen3-4b/r1_score_only_balanced_lora"
  "saves/edubench/qwen3-4b/r2_reason_score_natural_lora"
  "saves/edubench/qwen3-4b/r2_clean_reason_score_balanced_lora"
  "saves/edubench/qwen3-4b/r4_shuffled_reason_balanced_lora"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

if [[ -f "${HOME}/miniconda3/bin/activate" ]]; then
  source "${HOME}/miniconda3/bin/activate" "${CONDA_ENV}"
fi

mkdir -p "${LOG_DIR}" "${TABLE_DIR}"
export LOG_DIR TABLE_DIR

if ! command -v llamafactory-cli >/dev/null 2>&1; then
  echo "ERROR: llamafactory-cli not found in CONDA_ENV=${CONDA_ENV}" >&2
  exit 1
fi

python -m py_compile \
  thesis_exp/exp17_low_score_evidence/prepare_exp19_sft_dpo_datasets.py \
  thesis_exp/exp17_low_score_evidence/validate_exp19_llamafactory_data.py

python thesis_exp/exp17_low_score_evidence/validate_exp19_llamafactory_data.py \
  --dataset-dir "${DATASET_DIR}"

export EXP19_CONFIG_DIR_FOR_CHECK="${CONFIG_DIR}"
python - <<'PY'
from pathlib import Path
import os
import yaml

config_dir = Path(os.environ["EXP19_CONFIG_DIR_FOR_CHECK"])
configs = [
    config_dir / "llamafactory_qwen3_4b_r1_score_only_balanced_lora.yaml",
    config_dir / "llamafactory_qwen3_4b_r2_reason_score_natural_lora.yaml",
    config_dir / "llamafactory_qwen3_4b_r2_clean_reason_score_balanced_lora.yaml",
    config_dir / "llamafactory_qwen3_4b_r4_shuffled_reason_balanced_lora.yaml",
]
for path in configs:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise SystemExit(f"{path} did not parse to a dict")
    required = ["model_name_or_path", "dataset_dir", "dataset", "output_dir", "learning_rate", "num_train_epochs"]
    missing = [key for key in required if key not in data]
    if missing:
        raise SystemExit(f"{path} missing {missing}")
    print(
        "YAML_OK "
        f"{path.name} dataset={data['dataset']} lr={data['learning_rate']} "
        f"epochs={data['num_train_epochs']} output={data['output_dir']}"
    )
PY

IFS=' ' read -r -a GPUS <<< "${GPU_LIST}"
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "ERROR: GPU_LIST is empty" >&2
  exit 1
fi

cat <<CONFIG
Exp19 second-round Qwen3-4B LoRA SFT ablation
CONDA_ENV=${CONDA_ENV}
GPU_LIST=${GPU_LIST}
DATASET_DIR=${DATASET_DIR}
CONFIG_DIR=${CONFIG_DIR}
OUT_DIR=${OUT_DIR}
LOG_DIR=${LOG_DIR}
SKIP_COMPLETED=${SKIP_COMPLETED}

Runs:
  R1b: score-only balanced
  R2n: reason-score natural
  R2c: clean reason-score balanced
  R4b: shuffled reason balanced

Shared hyperparameters from YAML:
  model_name_or_path=/home/jpang/models/modelscope/Qwen/Qwen3-4B
  stage=sft
  finetuning_type=lora
  lora_rank=16
  lora_alpha=32
  lora_dropout=0.05
  lora_target=all
  cutoff_len=4096
  per_device_train_batch_size=2
  gradient_accumulation_steps=4
  learning_rate=1.0e-4
  num_train_epochs=3
  lr_scheduler_type=cosine
  warmup_ratio=0.05
  bf16=true
  gradient_checkpointing=true
CONFIG

run_one() {
  local run_name="$1"
  local run_label="$2"
  local config_path="$3"
  local output_dir="$4"
  local gpu_id="$5"
  local log_path="${LOG_DIR}/train_${run_name}_gpu${gpu_id}.log"

  if [[ ! -f "${config_path}" ]]; then
    echo "ERROR: missing config for ${run_name}: ${config_path}" >&2
    exit 1
  fi

  if [[ "${SKIP_COMPLETED}" == "1" && -f "${output_dir}/adapter_config.json" ]]; then
    echo "Skipping ${run_name}: completed adapter exists (${output_dir}/adapter_config.json)"
    return 0
  fi

  echo "Starting ${run_label} on GPU ${gpu_id}; config=${config_path}; log=${log_path}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" llamafactory-cli train "${config_path}" 2>&1 | tee "${log_path}"
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

run_names = [
    "r1b_score_only_balanced",
    "r2n_reason_score_natural",
    "r2c_clean_reason_score_balanced",
    "r4b_shuffled_reason_balanced",
]
run_labels = [
    "R1b score-only balanced",
    "R2n reason-score natural",
    "R2c clean reason-score balanced",
    "R4b shuffled reason balanced",
]
outputs = [
    "saves/edubench/qwen3-4b/r1_score_only_balanced_lora",
    "saves/edubench/qwen3-4b/r2_reason_score_natural_lora",
    "saves/edubench/qwen3-4b/r2_clean_reason_score_balanced_lora",
    "saves/edubench/qwen3-4b/r4_shuffled_reason_balanced_lora",
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
    trainer_state = out / "trainer_state.json"
    trainer_steps = ""
    if trainer_state.exists():
        try:
            trainer_steps = json.loads(trainer_state.read_text()).get("global_step", "")
        except Exception:
            trainer_steps = ""
    rows.append(
        {
            "run_name": run_name,
            "run_label": run_label,
            "completed": (out / "adapter_config.json").exists(),
            "output_dir": str(out),
            "log_file": str(logs[-1]) if logs else "",
            "trainer_global_step": trainer_steps,
            "last_logged_loss": flat_losses[-1] if flat_losses else "",
        }
    )

with (table_dir / "exp19_sft_second_round_training_summary.csv").open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "run_name",
            "run_label",
            "completed",
            "output_dir",
            "log_file",
            "trainer_global_step",
            "last_logged_loss",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps({"summary_csv": str(table_dir / "exp19_sft_second_round_training_summary.csv"), "runs": rows}, ensure_ascii=False))
PY

echo "Exp19 second-round SFT ablation completed."
