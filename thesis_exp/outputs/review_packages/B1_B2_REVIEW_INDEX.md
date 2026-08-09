# B1 + B2 independent-review index

This index contains the minimum evidence needed to audit the two experiments
added after the current manuscript PDF.  The PDF is intentionally a
pre-integration draft: the reviewer is asked to decide how the new evidence
should change its claims and organization.

## Current manuscript

- [Chinese manuscript PDF](../../../output/latex/Residual_Geometry_Ordinal_Scoring_Overleaf_zh/build/main.pdf)

## B1 / Exp62: independent SummEval replication

- [Plain-language final result](../../exp62_summeval_routing_confirmation/RESULTS.md)
- [Frozen formal protocol](../../exp62_summeval_routing_confirmation/configs/protocol.json)
- [Mechanical training-integrity audit](../exp62_summeval_routing_confirmation/decision/formal_training_integrity.json)
- [One-time test results](../exp62_summeval_routing_confirmation/test_once/test_results.json)
- [Test-access record](../exp62_summeval_routing_confirmation/decision/test_access_record.json)
- [Transitive source-lock compatibility note](../../exp62_summeval_routing_confirmation/SOURCE_LOCK_NOTE.md)
- [Implementation directory](../../exp62_summeval_routing_confirmation/)

Frozen conclusion: favorable external directions for Full and
Orthogonal-only routing, but both cluster intervals cross zero, so neither is
an independent confirmatory pass.

## B2 / Exp63: same-state counterfactual update

- [Plain-language final result](../../exp63_same_state_counterfactual/RESULTS.md)
- [Frozen protocol](../../configs/exp63_same_state_counterfactual/protocol.json)
- [Transitive source lock](../../configs/exp63_same_state_counterfactual/source_lock.json)
- [Real-model no-update preflight](../exp63_same_state_counterfactual/preflight/real_model.json)
- [Canonical five-seed decision](../exp63_same_state_counterfactual/decision/canonical_results.json)
- [Implementation directory](../../exp63_same_state_counterfactual/)

Frozen conclusion: Orthogonal-only versus Blocked fails the preregistered
4-of-5 seed rule (3/5; mean hard-CE delta -0.000678).  Full residual versus
Blocked meets the frozen directional rule (4/5; mean -0.001437).  The isolated
orthogonal-direction claim is therefore not authorized.

## Invalid-run disclosure

The first Exp63 counterfactual execution restored AdamW before constructing
the scheduler.  Scheduler construction silently reset the learning rate to
zero, so all four branches were no-op updates.  These invalid records were
detected from exact equality to the pre-update probe, stopped, and retained
under `thesis_exp/outputs/exp63_same_state_counterfactual/failed_v1_lr_zero/`.
They are excluded from the canonical result.  The corrected implementation
enforces a positive checkpoint/optimizer/scheduler learning-rate equality
gate before every arm.

## Review prompt

- [Chinese GPT-5.6 Pro prompt](B1_B2_GPT56_REVIEW_PROMPT.md)
