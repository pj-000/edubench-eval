# Exp28I Teacher Relabeling Diagnosis

- decision: **SELECTIVE_HARD_RELABELING_NOT_SUPPORTED**
- dev campaign: **DEV_SUCCESS_CRITERIA_NOT_MET**
- B2 overall worse than B0: True
- B2 low-to-high better than B0: False
- B2 low-to-high better than matched random relabeling: False
- test read: no

This diagnosis separates two questions: whether the selected rows contain useful risk information,
and whether replacing their hard labels is a valid training intervention. A lower low-to-high rate
with worse MAE, Exact Match, or Kendall tau supports the former but rejects the latter. Teacher
scores remain model-generated silver supervision and are not treated as corrected human gold.
