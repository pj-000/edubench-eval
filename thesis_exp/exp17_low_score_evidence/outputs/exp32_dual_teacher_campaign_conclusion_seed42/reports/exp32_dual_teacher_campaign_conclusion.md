# Exp32 Dual-Teacher Campaign Conclusion

## Scope

This report closes the dev-only Qwen3.7-Max and DeepSeek-V4-Pro teacher-label campaign under the original-paper split: 2,654 train, 664 dev, and 2,218 held-out test examples. The student, input fields, CE objective, optimizer, learning rate, batch settings, epochs, checkpoint metric, and seeds were held fixed. Teacher outputs are model-generated silver annotations, not human adjudications.

## Main result

The dual-teacher audit is useful for locating label disagreement, but the resulting teacher-selected targets are not supported as improved training labels. The locked three-seed original-label baseline remains the best supported student model.

Exp28 provides the confirmatory comparison. Selective dual-teacher hard relabeling increased mean MAE from 0.3951 to 0.4645, reduced Exact Match from 0.7244 to 0.6396, reduced Kendall's tau from 0.5852 to 0.4656, and increased low-to-high from 0.5167 to 0.6167. It did not beat the matched random transition control. A 2,000-resample hierarchical bootstrap passed none of the predeclared gates, so test evaluation was not authorized.

## Follow-up diagnoses

Exp29 preserved original labels and added a second teacher target for audited disagreements. It did not improve the seed-42 baseline or beat the exposure and random controls. This rejects the explanation that hard replacement alone caused the failure.

Exp30 repeated 38 teacher-confirmed low-score examples while retaining their original labels. Its MAE was 0.3931 and low-to-high was 0.45, while the matched random low-score control achieved MAE 0.3745 and low-to-high 0.40. The teacher criterion therefore did not explain the gain.

Exp31 repeated 38 teacher-disputed low-score examples while retaining their original labels. It achieved MAE 0.3941 and low-to-high 0.45. The matched random control achieved MAE 0.3770 and stronger Exact Match, Kendall's tau, Bin Agreement, and label-5 accuracy, although its low-to-high was 0.50. The teacher-disagreement criterion trades some tail risk for worse overall performance and does not improve low-to-high over the original seed-42 baseline of 0.45.

## Interpretation

The evidence supports a distinction between audit utility and supervision utility. Qwen3.7-Max and DeepSeek-V4-Pro can expose uncertain or disputed records, but agreement or disagreement with the original label does not reliably identify examples whose relabeling or resampling improves the student. Random controls matching the same transition or class exposure frequently match or outperform teacher-targeted variants.

The negative result is not evidence that every form of model-assisted annotation is ineffective. It specifically rejects the evaluated zero-shot dual-teacher hard-label, dual-target, confirmed-low, and disputed-low strategies under this fixed protocol. A positive relabeling claim would require independent expert adjudication or a newly cleaned validation reference rather than additional dev-driven variants.

## Decision

- Freeze Exp28-31; do not add more ad hoc teacher-label variants.
- Keep `b0_original_human` as the locked best-supported model.
- Do not run seeds 43/44 for Exp29-31 because their seed-42 method-specific gates failed against controls.
- Do not open the held-out test. No teacher method passed the predeclared dev gate.
- A defensible paper can study selective teacher auditing and its failure modes, but the current evidence cannot support the claim that teacher relabeling improves the evaluator.
- To pursue a positive data-quality paper, obtain independent expert adjudication for a representative stratified subset and use it to validate teacher confidence before retraining.

## Integrity statement

All training modifications used train data only. Dev labels were used for evaluation and predeclared selection. Test data were not read by these campaigns. No model arbitration is described as human review.
