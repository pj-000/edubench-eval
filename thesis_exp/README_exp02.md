# Exp2: EduBenchEvaluator 0.6B CE Baseline

Exp2 trains a 5-class cross-entropy baseline for EduBenchEvaluator 0.6B on the locked Exp0.1
paper-like split. It does not use synthetic/sample data and does not use existing automatic judge
predictions as targets.

## Framework Choice

Use the custom Hugging Face Transformers classifier in
`thesis_exp/src/edujudge/exp02/train_ce_baseline.py`.

This is preferred over LLaMA-Factory for Exp2 because the experiment is a classification CE
baseline: the model should directly optimize logits for labels 1-5 and report MAE, Exact Match,
Kendall tau, low-score overestimation, and Acc@5 without score-string parsing. LLaMA-Factory is
better for generative SFT; it can be used later if we want a score-generation baseline, but that is a
different protocol.

## Server Setup

On the GPU server:

```bash
ssh -p 23722 jpang@39.103.98.135

git clone https://github.com/pj-000/edubench-eval.git ~/edubench-eval-exp2
cd ~/edubench-eval-exp2
git checkout main
```

Create or reuse a Python environment with CUDA PyTorch and Transformers:

```bash
conda create -n edubench-exp2 python=3.10 -y
conda activate edubench-exp2
pip install -r thesis_exp/requirements-exp02.txt
```

If the server already has a suitable environment, activate it instead.

## Build Data

```bash
bash thesis_exp/scripts/run_exp02_build_data.sh
```

Expected split counts:

- train: 2654
- dev: 664
- test: 2218

Generated data:

- `thesis_exp/outputs/exp02_ce_baseline/data/train.jsonl`
- `thesis_exp/outputs/exp02_ce_baseline/data/dev.jsonl`
- `thesis_exp/outputs/exp02_ce_baseline/data/test.jsonl`

## Train

Set the 0.6B model path when you have it:

```bash
export MODEL_NAME_OR_PATH=/path/to/your/0.6b/model
export CUDA_VISIBLE_DEVICES=4
bash thesis_exp/scripts/run_exp02_train_ce_0_6b.sh
```

Useful overrides:

```bash
export CUDA_VISIBLE_DEVICES=6
export MAX_LENGTH=2048
export PER_DEVICE_TRAIN_BATCH_SIZE=4
export GRADIENT_ACCUMULATION_STEPS=32
export NUM_TRAIN_EPOCHS=10
export LEARNING_RATE=2e-5
bash thesis_exp/scripts/run_exp02_train_ce_0_6b.sh
```

Outputs:

- best checkpoint: `thesis_exp/outputs/exp02_ce_baseline/models/edubench_evaluator_0_6b_ce/best/`
- metrics: `thesis_exp/outputs/exp02_ce_baseline/models/edubench_evaluator_0_6b_ce/metrics.csv`
- test predictions: `thesis_exp/outputs/exp02_ce_baseline/models/edubench_evaluator_0_6b_ce/predictions_test.jsonl`
- run summary: `thesis_exp/outputs/exp02_ce_baseline/models/edubench_evaluator_0_6b_ce/run_summary.md`

## Notes

- Exp2 uses unweighted CE. Low-score-sensitive loss belongs to Exp5.
- The default training settings mirror the audit-paper protocol where it is specified:
  Qwen3-Reranker-0.6B, 5-class sequence classification, rounded human mean target, train-pool
  split into train/dev, CE + AdamW, 10 epochs, effective batch size 128, and best checkpoint
  selected by validation accuracy. On a 24GB 3090 this is implemented as batch size 4 with
  gradient accumulation 32.
- The train/dev/test files are fixed from Exp0.1 and should not be regenerated with a new split.
- Do not run on the old dirty `~/edubench-eval` server checkout; use a clean `~/edubench-eval-exp2`
  clone.
- If the 0.6B model lacks a native sequence-classification class, the script falls back to
  `AutoModel + last-token linear classifier`.
