# Exp54 dev execution attempt V1: format-only failure

## Status

The first Exp54 dev execution is frozen as:

`DEV_EXECUTION_ATTEMPT_V1_FORMAT_EXECUTION_FAILURE`

No complete cross-arm scientific metrics were produced, no checkpoint was
selected, and test was not accessed. The three formal attempts were limited
to S0/seed42 logical epochs 1, 2, and 3. A subsequent read-only 16-row
diagnostic established that the same serialization failure crosses R1, R2,
and R3; it did not write score or rationale semantic metrics.

The raw predictions, row identifiers, and row-level failure evidence remain
private on the training server. The public freeze contains only aggregate
counts and SHA-256 bindings.

## Observed failure

The strict parser correctly rejected outputs containing repeated score
digits, incomplete JSON strings, repeated rationale content, multiple JSON
objects, or trailing text. The formal failure counts were:

| Arm | Seed | Logical epoch | Rows | Strict failures |
|---|---:|---:|---:|---:|
| S0 | 42 | 1 | 664 | 664 |
| S0 | 42 | 2 | 664 | 576 |
| S0 | 42 | 3 | 664 | 664 |

No metric file was written for these attempts.

## Root cause

The V1 training objective supervises only score-value tokens and active
rationale-content tokens. Fixed JSON field names, punctuation, rationale
boundaries, the assistant suffix, and EOS receive zero task-loss weight.
Teacher forcing supplies those structural transitions during training, while
unconstrained greedy inference requires the model to generate them. This is
a train-inference serialization mismatch.

The failure therefore does not by itself establish that the checkpoints
failed to learn score or rationale content. It establishes that the
field-content objective is not an end-to-end autonomous JSON objective and
requires a shared structured serialization layer.

## Scientific boundary

The V1 outputs may not be repaired with a lenient parser, regex extraction,
first-object selection, first-score extraction, automatic quote/bracket
completion, or trailing-text deletion. They may not be used to compute
scientific metrics.

The failed artifacts must remain intact. A prospective V2 execution protocol
may be developed only from the frozen schema, tokenizer, 256-token budget,
train-only/synthetic probes, and the format-failure categories. It may not
use apparent dev score correctness, rationale quality, arm rankings, epoch
rankings, or any semantic dev metric.

Formal dev may be rerun only after the V2 decoder, configuration, tests,
train-only smoke, source closure, and public hashes receive independent
review. Test remains sealed.
