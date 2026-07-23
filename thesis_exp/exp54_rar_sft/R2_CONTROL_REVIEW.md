# R2 semantic-control review checkpoint

## Review status

- External review required: **yes**
- Formal R2 donor map frozen: **no**
- Training manifests built: **no**
- Dev/test accessed: **no**
- Model/API/training used: **no**

## Frozen scientific purpose

R2 is intended to control extra supervised tokens, output format, and generic multitask
regularization. It must preserve R3's score target and rationale-supervision locations while
breaking the real answer-rationale semantic relationship.

The preregistered first-choice donor stratum is:

```text
label_5 × metric_id × language
```

Within each stratum, the implementation builds a one-to-one, no-fixed-point assignment minimizing
rationale-length difference. Different references from the same sample are not valid donors.

## Provisional train-only feasibility result

The diagnostic uses Unicode character length because the exact frozen Qwen tokenizer is not
available on this machine. Character length affects which donor is closest, but the existence of a
same-stratum derangement is determined by the reference/sample structure.

| Quantity | Result |
|---|---:|
| R3 source references | 3,934 |
| Strictly matched active references | 3,904 |
| Deactivated references | 30 |
| Reference coverage | 99.2374% |
| Source reason-covered rows | 1,612 |
| Active reason rows after strict R2 eligibility | 1,601 |
| Fully deactivated rows | 11 |
| Strata | 95 |
| Strata with deactivation | 14 |

Low-score row impact:

| Label | Source reason rows | Active rows | Fully deactivated | Row coverage |
|---:|---:|---:|---:|---:|
| 1 | 21 | 18 | 3 | 85.71% |
| 2 | 40 | 34 | 6 | 85.00% |

The loss is small globally but concentrated in the sparse low-score tail that RAR-SFT is intended
to improve.

## Why this is a scientific decision

Keeping the strict stratum gives the cleanest negative control, but R2 and R3 must both deactivate
these rationale positions. Relaxing the rule may preserve low-score rationale coverage but change
what R2 controls:

1. **Strict deactivation**: keep `label × metric × language`; deactivate unmatched R2/R3
   positions.
2. **Donor reuse**: keep the strict stratum and allow a donor reference to serve more than one
   recipient.
3. **Drop language as last resort**: keep `label × metric`, but some R2 rationales switch
   language.
4. **Drop metric as last resort**: keep `label × language`, weakening metric matching.
5. **Two-tier analysis**: strict R2 is primary; a relaxed full-coverage R2 is a sensitivity
   analysis.

The selected option affects causal interpretation, low-tail coverage, and experimental workload.
It must be decided before the exact-tokenizer donor map and training manifests are frozen.

## Files for review

- `PREREGISTRATION_V2.md`
- `build_r2_donor_map.py`
- `tests/test_exp54_r2_donor_map.py`
- `outputs/exp54_rar_sft/rar_v2/audit/r2_character_length_feasibility_report.json`

The row-level diagnostic map contains derived reference identifiers and remains local.

## Required reviewer decision

The reviewer should:

1. verify that the optional-node Hungarian construction implements a valid one-to-one
   derangement over the active subset;
2. decide whether strict deactivation is preferable to donor reuse or relaxed strata;
3. assess whether losing 3/21 Label-1 and 6/40 Label-2 reason-covered rows materially undermines
   the low-tail hypothesis;
4. define the permitted R3-versus-R2 claim under the chosen control;
5. provide one final rule that can be frozen before tokenizer-based matching.
