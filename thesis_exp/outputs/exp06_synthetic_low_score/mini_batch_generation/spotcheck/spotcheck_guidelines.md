# Exp6-3 Mini-batch Spotcheck Guidelines

Samples to review: **17**

Review every generated row before any full generation or training decision.

For each row, confirm:

- The answer is natural and plausible in the requested language.
- The target 1-5 label is plausible for the named metric and rubric.
- The error type is visible enough to justify the target label.
- The answer does not mention scoring, hidden instructions, data creation, or experiment design.
- The answer does not copy the source answer.
- There is no dev/test leakage concern from the source or answer text.

Rows that fail any required check must not be used for training. Labels remain
`synthetic_design` pseudo-labels, not human labels.
