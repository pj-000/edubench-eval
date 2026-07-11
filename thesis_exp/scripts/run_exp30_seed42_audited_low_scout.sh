#!/usr/bin/env bash
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
[[ -f "${HOME}/miniconda3/bin/activate" ]] && source "${HOME}/miniconda3/bin/activate" "${CONDA_ENV:-llama_factory}"
ROOT=thesis_exp/exp17_low_score_evidence/outputs/exp30_audited_low_resampling_seed42
MODEL=${MODEL_NAME_OR_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}
VS=(d1_teacher_confirmed_low d2_random_low_control); pids=()
for i in 0 1; do v=${VS[$i]}; [[ ! -f "$ROOT/private/datasets/$v/test.jsonl" ]] || exit 2; (
 CUDA_VISIBLE_DEVICES=$i python -m thesis_exp.src.edujudge.exp02.train_ce_baseline --model_name_or_path "$MODEL" --data_dir "$ROOT/private/datasets/$v" --output_dir "$ROOT/runs/$v/seed_42" --checkpoint_output_dir "thesis_exp/artifacts/exp30/$v/seed_42" --max_length 2048 --num_train_epochs 10 --learning_rate 2e-5 --weight_decay .01 --warmup_ratio .05 --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --gradient_accumulation_steps 32 --max_grad_norm 1 --seed 42 --bf16 auto --local_files_only --log_steps 5 --dev_only >"$ROOT/${v}_seed42.log" 2>&1
 ) & pids+=("$!"); done
bad=0; for p in "${pids[@]}"; do wait "$p" || bad=1; done; [[ $bad == 0 ]] || exit 1
python thesis_exp/exp17_low_score_evidence/collect_exp30_seed42_scout.py
