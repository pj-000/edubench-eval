# Exp54 RAR-SFT materialized-manifest checkpoint

## Scope

This checkpoint converts the reviewed train-only base events and event-level
R2 control into the exact candidate sequences and two loss masks consumed by
SFT. It does not freeze manifests, authorize smoke/formal training, read
dev/test, or select any hyperparameter from evaluation data.

Each formal seed has 7,962 base events in logical-epoch-major, original-row
order. The four arms share the same prompt, base event ID, row order, score
target, score token mask, cutoff, padding mode, and optimizer schedule:

- **S0:** empty rationale; score block only.
- **R1:** deterministic all-rater rationale schedule; all available human
  reasons are used once across the three logical epochs.
- **R2:** event-level strict donor rationale on the reviewed active subset.
- **R3:** label-consistent aligned rationale on exactly the same active subset.

## Exact visible target and token boundaries

The candidate visible target is compact JSON:

```text
{"score":<1-5>,"rationale":"<rationale> "}
```

For active rationales, the final space inside the JSON string is a fixed,
unsupervised BPE boundary. It prevents a token such as `."` from combining the
last supervised rationale character with the unsupervised closing quote.
An exhaustive probe over all 4,836 human references found zero remaining
score/rationale boundary crossings. Inactive targets remain:

```text
{"score":<1-5>,"rationale":""}
```

Only the score value token(s) enter the score block. Only the original
rationale content tokens enter the rationale block. Prompt tokens, JSON keys
and punctuation, the boundary space, assistant closing tokens, and padding
enter neither block.

The prompt contains only question, answer, evaluation metric, and the full
five-level rubric. Human references and gold scores are not prompt inputs.

## Two-block SFT loss

For sample \(i\), the implementation computes shifted causal-token
cross-entropy and then normalizes each active field independently:

\[
L_i =
\frac{1}{|M_i^s|}\sum_{t \in M_i^s}\ell_{i,t}
+
\lambda_r \mathbf{1}[M_i^r \ne \varnothing]
\frac{1}{|M_i^r|}\sum_{t \in M_i^r}\ell_{i,t},
\qquad \lambda_r=1.
\]

The batch objective is the mean of \(L_i\). Thus a long rationale does not
automatically outweigh the score block merely because it has more tokens.
Inactive rationale events have an empty rationale mask and retain the full
score block.

The private manifest loader reconstructs each sequence from the shared prompt
cache, target tokens, and assistant suffix, then verifies its full token-ID
hash. The fixed collator right-pads to 2,048 without truncation and emits
separate score/rationale masks. The loss applies the standard causal shift:
the logit at position \(t-1\) predicts the supervised token at position \(t\).

## Formal aggregate result

All three seeds passed the candidate audit:

- 7,962 identical ordered base events per arm and seed;
- 7,962 score-active events per arm and seed;
- 4,803 R2/R3 rationale-active events per seed, exactly 1,601 per epoch;
- R2/R3 rationale-active vectors are identical;
- per individual epoch and checkpoint prefixes 1, 1–2, and 1–3, R2/R3 have
  zero L1 difference for rationale bytes, contextual rationale token IDs,
  complete target bytes, and complete target token IDs;
- supervised R2/R3 rationale-token totals are identical at every checkpoint;
- no sequence exceeds the 2,048-token cutoff and no truncation occurs;
- fixed padded input budget is 16,306,176 tokens per arm and seed;
- micro-batch size 2 and gradient accumulation 4 produce 332 optimizer steps
  per logical epoch and 996 total, with flush/reset at logical epoch
  boundaries;
- packing and shuffle are disabled for this controlled experiment.

Rationale-supervised totals intentionally differ between S0, R1, and R2/R3;
that difference is the intervention. Event count, padded compute, score
supervision, ordering, and optimizer steps are held fixed. Within the causal
R2-versus-R3 comparison, both padded and unpadded token totals are identical.

## Privacy and gate status

Private prompt caches, manifests, human rationale text, row-level donor maps,
token lists, record/reference/event IDs, and reversible mappings remain under
the ignored `rar_v2/data/` directory. Public artifacts contain only aggregate
counts, booleans, configuration, and cryptographic hashes.

This checkpoint remains `CANDIDATE_NOT_FROZEN`. It does not authorize smoke
training or formal training. The next reviewer must decide whether the
materialization, BPE boundary, two-block loss, checkpoint-prefix controls, and
budget contract are sufficient. Full base-model weights, LoRA/trainer
configuration, optimizer/scheduler precision settings, and executable
checkpoint/evaluation code remain a later freeze gate.
