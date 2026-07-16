# Exp47A Optimization-versus-Data Diagnosis

Primary diagnosis: **QUESTION_GENERALIZATION_LIMIT**

The audit separates human-label ambiguity, question-key generalization, and train-side adaptation. The 0.6B outer-train result is deliberately marked unavailable rather than reconstructed after checkpoint deletion.

Secondary flags:

- label2_target_ambiguous: **False**
- question_generalization_limit: **True**
- 4b_outer_train_adaptation_limit: **False**
- objective_or_adaptation_limit_both_models: **False**
- 0.6b_outer_train_unavailable: **True**

This diagnosis does not establish that full-fine-tuned 4B models are incapable. It only characterizes the existing locked 4B LoRA runs and existing 0.6B OOF runs.

No post-hoc hyperparameter search, student distillation, or test evaluation is authorized.
