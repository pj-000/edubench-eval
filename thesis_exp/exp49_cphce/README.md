# Exp49-CPHCE

Exp49 is a paired contract-repair experiment. It reruns the paper-like 0.6B
five-class baseline and changes exactly one training variable: the target passed
to cross-entropy.

- `b0_hard_ce`: rounded three-rater mean as a one-hot class.
- `m1_human_soft`: the empirical distribution of the three five-point ratings.

Both arms use the fixed 2654/664/2218 split, the exact Exp02
`qa_metric_baseline` prompt, the same model class, cosine scheduler, sampler,
optimizer, seed, and raw-logit argmax inference. Training and formal selection
are dev-only. The existing project test split has historical prior use; Exp49
therefore records both the legacy status and a separate Exp49 access counter.

Run order:

1. `bash thesis_exp/scripts/run_exp49_metric_audit.sh`
2. `bash thesis_exp/scripts/run_exp49_cpu_smoke.sh`
3. `bash thesis_exp/scripts/run_exp49_seed42.sh`
4. Only after `SEED42_PASS`: `bash thesis_exp/scripts/run_exp49_formal.sh`
5. Only after `FORMAL_PASS`: freeze and run the one-shot test scripts.

Runtime weights, checkpoints, raw predictions, arrays, and logs remain ignored.
