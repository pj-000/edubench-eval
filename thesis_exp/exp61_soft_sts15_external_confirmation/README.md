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
- Do not inspect any model outcome on the sealed test split before the protocol,
  trainer, analysis code, and source lock are frozen.

## Reproduce the data audit

```bash
git clone https://github.com/ale0xb/sts_beyond_averages.git /tmp/sts_beyond_averages
git -C /tmp/sts_beyond_averages checkout ca754a21e58437c4b843d10161d2838f39230e7f
python -m thesis_exp.exp61_soft_sts15_external_confirmation.audit_dataset \
  --source-repo /tmp/sts_beyond_averages
```

Stage 0 may authorize protocol design, but it cannot authorize formal training.

