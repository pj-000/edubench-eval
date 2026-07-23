# Exp54 / RAR-SFT

RAR-SFT stands for **Rubric-Aligned, Reliability-Gated Rationale Supervised Fine-Tuning**.

> Protocol update (2026-07-23): the historical structured five-field design has been superseded by
> [`PREREGISTRATION_V2.md`](PREREGISTRATION_V2.md), which defines **Rater-Aligned Multi-Reference
> Rationale SFT** with S0/R1/R2/R3 controls. RAR-0A remains the valid provenance-alignment audit;
> its old field gates do not authorize the superseded criterion/evidence conversion pipeline.

The experiment starts with RAR-0, a train-only supervision qualification audit. RAR-0 does not
call an API, load a model, use a GPU, train, or read dev/test. Its first executable slice answers:

1. which original human reasons can be joined uniquely to the paper-like train rows;
2. whether each reason score agrees with its originating human rater score;
3. whether it also agrees with the aggregate five-point target used by SFT;
4. how coverage varies by score, metric, and language.

Run the deterministic alignment audit:

```bash
python -m thesis_exp.exp54_rar_sft.audit_rar0_alignment
```

Run tests:

```bash
pytest -q thesis_exp/tests/test_exp54_rar0_alignment.py
```

The audit intentionally separates three statuses:

- `aligned_eligible`: exact unique alignment, non-empty reason, and agreement with that rater's
  original score;
- `aggregate_score_consistent`: the reason's score also equals `label_5`;
- `semantically_qualified`: always false in this first slice. Rubric/evidence/rationale fields
  become eligible only after structured conversion and independent verification.

The historical Exp53 score-only SFT is not a matched baseline because it excludes the rubric.
RAR experiments must therefore add a rubric-aware score-only baseline with the same input as R1,
R2, and R3.
