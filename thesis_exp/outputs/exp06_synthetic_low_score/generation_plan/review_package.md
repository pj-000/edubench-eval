# Exp6-2 Generation Plan Review Package

Can generation start? **YES, after human review and API approval**

Can training start? **NO**

Required manual confirmations:

- Approve `deepseek-v4-pro` or replace it with another generation model.
- Confirm API budget and rate limits.
- Review prompt templates for accidental dev/test leakage or label ambiguity.
- Confirm first-run target: 384 low-score/boundary rows.
- Confirm optional D1/D4 full-score diagnostic should be generated separately.
- Confirm generated data remains train-only and pseudo-labeled.

API/model needed: `deepseek-v4-pro`.

Estimated generation count: **384** for first low-score augmentation; optional
diagnostic matrix count is **336**.

Risks:

- Synthetic pseudo labels may not match human scoring.
- Low-score answers may be too artificial or too easy.
- Full-score synthetic-only diagnostic can introduce model-style distribution bias.
- Any dev/test overlap must block affected rows.

Next step: run a reviewed dry-run prompt sample, then implement an approved API runner, generate a
small batch, and audit generated rows before training.
