# Exp19-R5F D1-like Generation Input QC

This train-only dataset selects low-score samples that resemble hidden-failure cases.
Human rationale is not included in the model prompt.

- train source: `thesis_exp/data/splits/question_seed42/train.jsonl`
- candidates selected: 63
- language filter: `all`
- min answer chars: 40
- require model-high D2 prediction: `False`
- D2 prediction map available for samples: 80

## Failure Modes

- insufficient_evidence: 18
- missing_key_point: 20
- surface_fluent_but_hidden_defect: 25

## Next Step

Run `run_exp19_r5f_d1_like_generate.sh` on server to sample rejected outputs.
Then run the same preparation script with `--stage build_dpo` to construct R5F DPO pairs.

No test split is read.
