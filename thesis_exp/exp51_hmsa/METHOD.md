# Exp51-HMSA locked method

Exp51 tests one structural hypothesis after Exp49 and Exp50: keep the paper's hard five-class decision in its own head and move the three-human empirical distribution to an independent auxiliary head over the same Qwen3 backbone.

The fixed loss is `CE(hard_logits, label_5) + 1.0 * CE(soft_logits, human_distribution)`. Only hard logits are used for checkpoint selection, predictions, and all paper metrics. The auxiliary weight is not searched.

The actual Qwen3 path is preserved: each bias-free linear head is applied tokenwise and the logits at the rightmost non-padding token are selected. No pooling layer, dropout, MLP, normalization, adapter, separate learning rate, curriculum, label weighting, or inference mixing is added. The soft head is a deep copy of the initialized hard head, so no additional random draw occurs, while its parameter storage is independent.

The scheduler is `cosine_with_warmup`, matching the executed Exp49/Exp50 code. Seed42/formal code refuses to load test. A previously frozen manifest may be inspected for test identity, but test contents are not re-read before formal authorization.

`lambda=1.0` does not isolate the shared backbone: hard and soft gradients still meet there. It is a fixed equal-scale multi-task test, not a claim that the main representation is protected or that this weight is optimal.
