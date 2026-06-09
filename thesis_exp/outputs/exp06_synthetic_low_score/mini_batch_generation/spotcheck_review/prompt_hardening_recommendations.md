# Exp6 Mini-batch Prompt Hardening Recommendations

## Scope

These recommendations apply after manual review of the 17 filtered synthetic low-score samples.
They are intended for the next 96-row batch generation step. No API call, new synthetic generation,
or model training was run for this review.

## Required Hardening

1. Add a label plausibility check.
   - After generation, require an explicit self-check that the answer quality visibly matches the
     requested target label.
   - If the answer is substantially better than the target label, mark it for relabel or reject.

2. Add an error_type alignment check.
   - Require the generated failure mode to match the requested `error_type`.
   - If the sample has the right target label but the wrong failure type, mark it for
     `revise_error_type` instead of accepting it as-is.

3. For target labels 1 and 2, require a clear rubric-relevant failure.
   - Label 1 must contain a severe and obvious defect tied to the target metric.
   - Label 2 must contain a major defect that keeps the response unreliable even if some surface
     structure is present.

4. For target label 3, require boundary quality, not clearly good quality.
   - Label 3 samples should be partial, mixed, or borderline.
   - They should not read as mostly correct answers with only minor style issues.

5. Avoid answers whose actual quality is higher than the target label.
   - Prompt text should explicitly forbid producing a polished answer that only weakly gestures at
     the requested failure.
   - The filter should flag samples that over-satisfy the rubric.

6. If a response partially satisfies the rubric too well, mark it for relabel or reject.
   - Use `revise_label` when the sample is useful but the target label is too low.
   - Use `reject_from_low_score_pool` when the sample is not suitable for low-score augmentation.

## Manual-review Findings Driving These Changes

- `mb008` appears too strong for the requested low-score pool and should be rejected from that pool
  or relabeled upward.
- `mb018` should be relabeled from 1 to 2.
- `mb004` and `mb023` are usable after revising the error type.

## Next-batch Gate

The next 96-row batch can start only after the prompt and post-filter explicitly implement the
label plausibility and error_type alignment checks above.
