# Exp35A EduDART-Cal Qualification

Exp35A is a CPU-only label-method qualification stage. It uses Exp33's
representative train model-reviewed silver as calibration-development data and
constructs a new source-blind qualification view from the locked 2,654-row
paper train split. It does not train the student model and does not read dev or
test.

Because Exp33 already consumed every one of the 76 train-low sample IDs, the
requested fresh low=40 quota is mathematically infeasible. The frozen protocol
therefore separates:

- 120 sample-disjoint fresh general rows (20 mid, 100 high), used for the
  independent qualification gate;
- all 76 train-low rows, reviewed in new independent blind runs and used only
  as a repeated-sample low-tail safety stress test.

All references are described as independent model-reviewed silver. They are
not human expert gold. Reviewer model identities are provenance, not the method
contribution.

Prepare packets:

```bash
./thesis_exp/scripts/run_exp35a_prepare_model_reviewed_qualification.sh
```

After two blind reviews and third-model conflict adjudication, run:

```bash
python thesis_exp/exp35_edudart_cal/analyze_exp35a_blind_reviews.py
```
