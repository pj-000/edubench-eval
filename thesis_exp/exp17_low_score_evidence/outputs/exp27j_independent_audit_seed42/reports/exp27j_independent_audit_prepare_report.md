# Exp27J Independent Audit Preparation

Exp27J prepares a train-only independent blind-review benchmark. It does not call APIs, train, or
read dev/test labels.

## Sampling

- representative rows: 120
- risk-enriched rows: 60
- unique sample ids: 180
- unique question keys: 84 of 180 rows
- representative sampling is row-level disproportionate stratified probability sampling.
- repeated question keys are retained as analysis clusters because train has only 118 unique question keys, making 180 unique question keys impossible.
- risk-enriched rows are a separate stress view and are never used for population prevalence estimates.

## Blindness

- blind packets contain no original, individual-human, teacher, or calibrated labels.
- human rationales are private and do not enter reviewer packets.
- dev/test are used only for sample-id and question-key overlap guards.

## Review Status

- reviewer templates were generated but semantic decisions were not fabricated during preparation.
- downstream training remains blocked until dual reviews, adjudication, and tier validation are complete.
