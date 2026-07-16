# Exp47A Label-2 Identifiability and Generalization Audit

Exp47A is an aggregate-only audit of existing paper-like train GroupCV artifacts. It does not train a model, call an API, or read paper-like dev/test.

The audit separates:

- human-label ambiguity behind rounded hard label 2;
- question/metric/subject concentration;
- 4B outer-train versus unseen-question behavior;
- existing 0.6B and 4B OOF behavior;
- class-2 probability rank and competing classes;
- optimization history and LoRA adaptation limits.

The historical 0.6B fold checkpoints are unavailable. Exp47A therefore reports 0.6B outer-train metrics as `MISSING_CHECKPOINT_NOT_RECOMPUTED`; it does not retrain or substitute a leaking checkpoint.

Run from the repository root:

```bash
./thesis_exp/scripts/run_exp47a_goal.sh
```

Only aggregate CSV, reports, configs, hashes, and the decision JSON may be committed. Existing row-level predictions, logits, summaries, checkpoints, and sample IDs remain private.
