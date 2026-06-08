# Exp6 Generated Synthetic Leakage Check Plan

Generated rows must pass these checks before any train-only augmentation:

- `source_question_key` must not appear in dev/test.
- `source_triple_key` must not appear in dev/test.
- normalized synthetic `question` key must not appear in dev/test.
- normalized synthetic `question + answer_synthetic` key must not appear in dev/test.
- synthetic answers must not duplicate each other.
- synthetic answers must not duplicate human test answers.
- `source_split` must be `train`.

Any dev/test hit blocks the row. Synthetic rows must never be added to dev/test.
