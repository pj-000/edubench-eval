# Exp27L Question-Key Cross-Fitted Calibration Protocol

Exp27L is a CPU-only calibration audit over the locked 180 train-only Exp27J rows.
Each question key belongs to one outer fold, so no answer from the same question can appear in both fit and held-out evaluation.

- Outer folds: 5
- Inner folds: 3, grouped by question key
- Score/risk fitting rows: representative view only (120 rows)
- Risk-stress rows: held-out OOF stress evaluation only (60 rows)
- Risk target: severe human-silver conflict, defined as an absolute score difference of at least 2
- No teacher API, no GPU, no train/dev/test model training

Exp27J silver fields are evaluation targets only. They are not allowed in the score fusion inputs, risk features, or external blind-review packets.
