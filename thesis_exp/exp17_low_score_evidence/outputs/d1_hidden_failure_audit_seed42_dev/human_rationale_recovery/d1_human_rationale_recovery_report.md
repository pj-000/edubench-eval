# Exp17-D1 Human Rationale Recovery

This is a dev-only provenance diagnostic. It only recovers original 5-grade human rating rationales for the existing D1 cases. It does not train, load checkpoints, read test labels, or write raw predictions.

## Source

- Source root: `.`
- Current fork: `https://github.com/pj-000/edubench-eval`
- Upstream source lineage: `https://github.com/danieglofsmi/edubench-eval/tree/main`
- Reason files:
  - `5-grades/5_merge_human_metric_en.jsonl`
  - `5-grades/5_merge_human_metric_zh.jsonl`
  - `5-grades/5_human_1.jsonl`
  - `5-grades/5_human_2.jsonl`
  - `5-grades/5_human_3.jsonl`
- Missing reason files: `none`

## Recovery Summary

- D1 cases: `27`
- Question-answer matched cases: `21/27` = `0.7778`
- Metric-level rationale recovered cases: `21/27` = `0.7778`
- Non-recovered case numbers: `4; 6; 9; 16; 20; 24`

| match_status | n |
|---|---:|
| metric_rationale_recovered | 21 |
| question_answer_unmatched | 6 |

## Concentration

| question_group_id | n |
|---|---:|
| `14ba3cb00f998348fe1c491eab066379d3bf192b` | 20 |
| `9d1179a873f8e7454e4075453ea96fee9e73ecff` | 5 |
| `9dcad11d15cc245e5dabc70ec2358208f1139f70` | 1 |
| `1bbfb9a5f532b875aaa1b5a1500fb88535b21a51` | 1 |

## Interpretation

- The earlier D1 audit used the merged modeling table, which keeps human scores but not the original human rationale text.
- The original 5-grade files contain rationale text for most D1 cases. Therefore, several apparently unexplained label-2 cases should be re-interpreted with these recovered rationales before deciding whether they are label conflicts.
- The Annales cases are not merely arbitrary label conflicts in the recovered rationales: the human reasons treat `Marc Bloch` as a wrong answer for the long-duration-history wording and point to `Fernand Braudel` as the intended figure.
- The marketing-manager cases mostly have recoverable rubric-linked reasons: incomplete corrected answer, missing key duties, shallow error explanation, weak scenario adaptation, or weak clarity/inspiration.
- Case 23 should be handled carefully: recovered human rationales emphasize task/format and scoring-design mismatch in the answer, while external expert review may additionally identify domain factual mismatch. These should be separated rather than collapsed into one label.

## Next Step

Before training Exp17-A, rerun the D1 annotation summary using this recovered rationale table as evidence. Do not train directly from dev annotations; use recovered rationales to design train-side weak-label expansion or pairwise evidence checks.
