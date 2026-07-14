# Exp43 RubiMOR Final Report

- Current final status: **ORDINAL_ONLY_SIGNAL**
- Final test consumed: **False**

## Stage decisions

- Stage 0: `GO`
- Stage 1: `GO`
- Stage 2: `GO`
- Stage 3: `GO`
- Stage 4: `METRIC_HEAD_STOP`
- Stage 5: `NOT_RUN_AFTER_GATE_STOP`
- Stage 6: `NOT_RUN_AFTER_GATE_STOP`
- Stage 8: `NOT_RUN_AFTER_GATE_STOP`
- Stage 9: `NOT_RUN_TEST_SEALED`

## Protocol integrity

- Qwen3-Reranker-0.6B only; full fine-tuning.
- No teacher API, teacher relabeling, or teacher reason supervision.
- No test-driven method tuning.
- Runtime checkpoints, raw predictions, pair JSONL, and logs remain private/ignored.
