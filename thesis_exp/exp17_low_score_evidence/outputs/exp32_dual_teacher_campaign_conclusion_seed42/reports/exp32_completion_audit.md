# Exp32 Objective Completion Audit

## Protocol coverage

The data audit verifies the original-paper 2,654/664 train/dev split, zero train-dev identity and triple-key overlap, identical dev data across variants, and no written test dataset. Exp28 completed the fixed Qwen3-Reranker-0.6B ordinary-CE comparison for seeds 42, 43, and 44, including the original labels, all-primary-teacher labels, selective dual-teacher labels, unresolved-example filtering, and a transition-matched random control.

The dev analysis reports all requested overall, rank, bin, tail-risk, and per-label metrics. A 2,000-resample hierarchical bootstrap tested the predeclared guards. Lightweight result tables and figures are available for the confirmatory experiment and the Exp29-31 diagnostic scouts.

## Unmet terminal condition

The objective's final-test clause is conditional on a dev-selected teacher method. That condition is not met: selective dual-teacher relabeling is worse than the original-label baseline and does not beat the matched random control. Consequently, the final-test script correctly remains locked, and no held-out test labels or predictions were read.

This is a protocol-compliant negative result, not a completed positive-method claim. Running test despite the failed dev gate would violate the experiment design. The scientifically defensible terminal state is therefore to freeze the current campaign and either reframe the paper around selective audit reliability and negative supervision findings, or obtain independent expert-adjudicated labels before defining a new campaign.

## Completion decision

- Experimental design, annotation audit, three-seed dev training, inference, controls, statistical testing, tables, figures, GitHub push, and server synchronization: complete.
- Evidence that dual-teacher selective relabeling improves the student: absent.
- Authorization for one-shot test: absent.
- Test read: false.
- Additional dev-driven teacher-label variants: not recommended.
