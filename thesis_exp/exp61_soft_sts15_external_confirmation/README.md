# Exp61: Soft-STS-15 external confirmation

Exp61 is a cross-task confirmation study, not a replacement for the EduBench
12-criterion benchmark. Soft-STS-15 contains English sentence pairs with five
published ordinal ratings on a single semantic-similarity scale from 0 to 5.

The only intended scientific question is whether a sample-aligned non-collinear
residual improves point-score accuracy over a geometry-matched, label-conditional
misaligned residual on an independent multi-rater ordinal task.

## Stage 0 rules

- No model training and no GPU use.
- Pin the upstream repository and data hash.
- Treat the five values in `score_list` as the complete empirical target used by
  this experiment. Do not mix their mean with the upstream `gs_score`, because
  the original SemEval preprocessing retains only the first five ratings in
  `score_list` when more ratings were collected.
- Use a deterministic sentence-connected-component-disjoint 60/20/20 split.
- Call the hard target a quantized-mean main target. In this dataset it is not
  generally equal to the unique mode, median, or an observed rating.
- Do not inspect any model outcome on the sealed test split before the protocol,
  trainer, analysis code, and source lock are frozen.

## Reproduce the data audit

```bash
git clone https://github.com/ale0xb/sts_beyond_averages.git /tmp/sts_beyond_averages
git -C /tmp/sts_beyond_averages checkout ca754a21e58437c4b843d10161d2838f39230e7f
python -m thesis_exp.exp61_soft_sts15_external_confirmation.audit_dataset \
  --source-repo /tmp/sts_beyond_averages \
  --official-archive /tmp/sts2015-en-post.zip

python -m thesis_exp.exp61_soft_sts15_external_confirmation.audit_token_lengths \
  --source-repo /tmp/sts_beyond_averages \
  --tokenizer-path /path/to/frozen/Qwen3-Reranker-0.6B

python -m thesis_exp.exp61_soft_sts15_external_confirmation.finalize_stage0
```

Stage 0 may authorize protocol design, but it cannot authorize formal training.

## Protocol-freeze implementation state

The three frozen arms are `quantized_mean_only`,
`aligned_orthogonal_only`, and `matched_shuffled_orthogonal_only`. The main
point prediction is the expectation of the hard-head six-class distribution;
the auxiliary head is discarded at inference.

The train/dev loader rejects `test`, the maximum-mismatch mapping uses train
targets only, and the formal trainer refuses all optimizer steps until an
independent final protocol/preflight review updates both the protocol and
source-lock authorization. The next permitted action is the real-model
no-update preflight for seeds 61/62/63.
