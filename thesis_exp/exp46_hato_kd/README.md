# Exp46A HATO-KD

Exp46A tests whether a task-trained Qwen3-Reranker-4B can recover transferable
label-2 structure under the same human supervision as the compact 0.6B model.
Only a passing teacher may be distilled back to the 0.6B student.

## Locked data and supervision

- Input: `question + answer + metric + rubric + metadata` from the existing
  Exp43 E4 prepared train data.
- Split: the existing five-fold question-key GroupCV assignment over 2,654
  paper-like training rows.
- Target: the three-human score distribution plus 0.5 ordinal CDF MSE.
- The teacher does not relabel examples and does not receive API labels,
  reasons, dev data, or test data.

## Stages

1. `T1_4B_teacher`: Qwen3-Reranker-4B LoRA capacity qualification.
2. `K1_standard_kd`: natural human anchor plus standard KD.
3. `K2_hato_kd`: natural human anchor plus class-balanced KD and ordinal KD.
4. `K3_shuffled_hato_control`: K2 with teacher logits shuffled within the same
   hard score and language.

The goal script stops after Stage 1 unless the preregistered teacher capacity
gate passes. The test split remains sealed in every stage.

## Server command

```bash
cd ~/edubench-eval-exp2
./thesis_exp/scripts/run_exp46a_goal.sh
```

The script defaults to A6000 GPUs 0-3, the locally deployed 4B teacher at
`/home/jpang/models/modelscope/Qwen/Qwen3-Reranker-4B`, and the existing 0.6B
student at `/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B`.
