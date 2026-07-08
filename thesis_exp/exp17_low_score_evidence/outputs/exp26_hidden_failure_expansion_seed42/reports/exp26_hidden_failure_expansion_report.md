# Exp26 Hidden-Failure Evidence Expansion Data

Exp26A prepares train-only data for the next evidence-aware SFT and field-masked ORC/SRC-DPO round.

## Outputs

- SFT dataset: `data/edubench_exp26a_evidence_aware_sft_train.json` (564 rows)
- DPO dataset: `data/edubench_exp26b_hidden_failure_dpo_train.json` (837 pairs)
- annotation manifest: `annotation/exp26_hidden_failure_annotation_manifest.csv` (538 rows)

## Important Scope

- This step does not train a model and does not require GPU.
- Dev/test are read only for sample_id/question_key leakage guards.
- Human rationales are included only in assistant targets, never in user prompts.
- `evidence_span` is intentionally null in this first data asset; the manifest marks rows needing teacher/human annotation.
- Counterfactual rejected outputs are explicitly marked as train-only non-human negatives.

## Dataset Counts

- train rows: 3326
- train low-label rows: 111
- train low-label rows with recovered reason: 84
- SFT risk distribution: {'hidden_low_failure': 84, 'clean_high': 360, 'mid_borderline': 120}
- DPO pair type distribution: {'failure_erasure_negative': 84, 'matched_high_protection_pair': 240, 'score_mismatch_same_evidence': 429, 'real_model_error_score_pair': 84}

## Recommendation

Review this data before training. The next likely step is teacher/human annotation for high-priority
low-label rows, then evidence-aware SFT, then field-masked ORC/SRC-DPO.
