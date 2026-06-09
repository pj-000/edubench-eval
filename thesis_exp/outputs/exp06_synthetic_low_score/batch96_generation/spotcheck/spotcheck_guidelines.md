# Exp6-6 Batch96 Spotcheck Guidelines

Samples to review: **36**

If filtered samples are 80 or fewer, review all. Otherwise review the stratified sample here,
covering labels, languages, error types, and metrics.

For every row, confirm:

- The answer is natural and plausible in the requested language.
- The synthetic pseudo-label is plausible for the target metric and rubric.
- The error type aligns with the visible failure.
- The answer has no artifact phrases such as "low-score answer", "synthetic", or "故意错误".
- The answer does not copy source or dev/test content.

Rows remain `synthetic_design` pseudo-labels, not human labels. Full 384 generation and Exp6
training remain blocked until batch96 spotcheck is reviewed.
