# RAR-SFT inference protocol V2 candidate

## Amendment status

Protocol identifier:

`RAR_SFT_INFERENCE_PROTOCOL_V2`

This is a prospective execution-protocol amendment following the frozen
V1 format-only failure. It does not change model weights, training data,
losses, checkpoints, score targets, rationale targets, or scientific
hypotheses.

The first dev execution and its read-only cross-arm diagnostic are retained
under `DEV_EXECUTION_ATTEMPT_V1_FORMAT_EXECUTION_FAILURE`. They produced no
valid cross-arm scientific comparison and may not be used for model,
checkpoint, decoder, or hyperparameter selection. Test remains sealed.

V2 is not authorized for formal dev until its exact decoder source,
configuration, dependency wheel, grammar, tokenizer binding, train-only
smoke, and public audit report pass independent review.

## Revised system definition

The formal system is:

```text
LoRA checkpoint
    +
shared tokenizer-aware grammar-constrained serializer
```

The model chooses the five-level score and visible rationale content. The
shared serializer restricts generation to the previously frozen JSON
contract and stops at completion of one object. The method does not claim
that the model independently learned JSON field names, punctuation, or EOS.

The field-content training objective remains:

\[
L_i=L_{i,\mathrm{score}}+q_iL_{i,\mathrm{rationale}}.
\]

V2 adds no format loss and does not retrain any checkpoint.

## Legal output language

The only accepted decoded form is:

```json
{"score":2,"rationale":"visible rationale"}
```

The exact tokenizer-aware EBNF is stored in
`configs/rar_sft_output_v2.ebnf`. It enforces:

- exactly one object;
- exact key order `score`, then `rationale`;
- no additional or duplicate keys;
- score is one integer character from 1 through 5;
- rationale is a valid JSON string and may be empty;
- standard JSON escapes and Unicode content;
- no whitespace outside the rationale string;
- no second object or trailing text.

The grammar is identical for S0, R1, R2, and R3. It does not force S0 to
emit an empty rationale and does not force any rationale arm to emit a
non-empty rationale.

## Frozen decoding candidate

- Backend: XGrammar, exact version and wheel hash recorded in the candidate
  configuration before smoke.
- Runtime wheels: `xgrammar==0.2.5` and its direct compiled dependency
  `apache-tvm-ffi==0.1.9`; both exact Linux wheel filenames and SHA-256
  digests are frozen.
- Integration: direct batched `GrammarMatcher` masks over the frozen
  Hugging Face tokenizer and model-logit vocabulary.
- Search: greedy argmax after the grammar mask.
- Sampling: disabled.
- Beams: one.
- Maximum generated tokens: 256.
- Repetition penalty: 1.
- No-repeat n-gram: disabled.
- Temperature, top-p, top-k, length penalty, semantic repetition removal,
  and arm-specific rules: prohibited.
- Completion: matcher terminates immediately when the root grammar is
  complete; no trailing EOS token is required in the decoded payload.
- Parser: the unchanged strict V1 parser validates the completed payload.
- Post-processing repair: prohibited.

The direct loop, rather than a lenient parser, is required so that the same
matcher state supplies the token mask, termination state, and execution
diagnostics.

## Budget-aware completion

At every generation step, the decoder determines whether the current
matcher state has a tokenizer-valid shortest structural completion. When
the remaining 256-token budget is equal to that completion length, the
decoder masks generation to the deterministic completion path.

Budget completion:

- may only finish an in-progress JSON escape, close the rationale string,
  and close the object;
- may not add semantic rationale content;
- may not remove or deduplicate existing content;
- may not change score;
- uses no arm, seed, epoch, label, language, metric, or dev-dependent rule;
- records `forced_completion=true`;
- remains subject to the unchanged strict parser.

Failure to find a legal completion within the remaining budget is an
execution error and may not fall back to truncation or repair.

## Uniform application

The following must be byte-identical or hash-identical across all 36
arm/seed/epoch evaluations:

- grammar;
- decoder source;
- decoder configuration;
- backend and version;
- tokenizer and tokenizer files;
- model vocabulary size;
- maximum token budget;
- matcher stop mode;
- parser;
- prompt template;
- generation search mode.

No parse rate, intervention diagnostic, score metric, or rationale metric
may select a decoder variant.

## Required execution diagnostics

For each row:

- strict parse success;
- generated token count;
- rationale token count;
- grammar intervention steps;
- total active generation steps;
- unconstrained top-1 blocked steps;
- removed probability mass mean;
- forced completion;
- completion at the 256-token boundary;
- empty rationale.

For each arm/seed/epoch:

- `strict_parse_rate`;
- `single_object_rate`;
- `valid_score_rate`;
- `rationale_string_rate`;
- `empty_rationale_rate`;
- mean, p95, and maximum rationale tokens;
- `max_token_hit_rate`;
- `forced_completion_rate`;
- `grammar_intervention_step_rate`;
- `illegal_unconstrained_top1_rate`;
- `removed_probability_mass_mean`.

These are execution diagnostics and may not select an arm, checkpoint,
decoder, or hyperparameter. If V2 yields high forced-completion or semantic
repetition rates, that is a model outcome and does not authorize another
decoder amendment.

One `grammar_intervention_step` means the emitted token differs from the
unconstrained raw-logit argmax because of the grammar mask or the frozen
budget-completion path. `unconstrained_top-1 blocked` is the narrower event
in which the raw argmax itself is illegal under the grammar. Removed
probability mass is computed before greedy selection as one minus the
model probability assigned to all currently legal tokens.

## Required pre-dev tests

1. Grammar acceptance and rejection tests for one object, exact keys,
   score enum, rationale string, duplicate keys, second objects, and
   trailing text.
2. Unicode and escaping tests covering Chinese, English quotes,
   backslashes, newlines, tabs, emoji, mixed language, and full-width
   characters, including a tokenizer token ending in an incomplete UTF-8
   byte prefix that must be completed before structural closure.
3. Adversarial-logit tests in which illegal continuation tokens always
   have the highest raw logit.
4. Tokenizer tests binding the immutable Qwen tokenizer and model
   vocabulary; no character is assumed to be one token.
5. Budget tests with a mock policy that always prefers continuing the
   rationale, proving valid completion within 256 tokens and
   `forced_completion=true`.
6. Root-completion tests proving that a second object and trailing text
   are unreachable.
7. Repeated-run determinism proving byte-identical output hashes.
8. Train-only execution smoke covering every arm, seed, logical epoch,
   language, label, and active/inactive rationale state without semantic
   metric inspection.
9. Privacy audit proving no train prompt, human rationale, row identifier,
   donor edge, or generated row text enters the public report.

## V1 and dev boundary

The V1 raw outputs and hashes must not be deleted, overwritten, or silently
reclassified. V2 development may use only:

- the pre-existing output schema;
- the locked tokenizer;
- the pre-existing 256-token limit;
- train-only prompts and synthetic strings;
- the known format-failure categories;
- the deterministic fact that structural transitions received no task
  loss.

V2 development may not use apparent dev score correctness, rationale
quality, arm ranking, epoch ranking, semantic evaluator output, or any
scientific dev metric.

After independent review passes, all 36 checkpoints must be evaluated once
under the single frozen V2 protocol. Test remains sealed until the complete
SFT and later preference-optimization protocol reaches its final one-shot
test gate.

## Forbidden recovery behavior

The formal path must never:

- extract the first score digit from malformed output;
- select the first or last of multiple JSON objects;
- delete trailing text;
- add quotes, braces, commas, or keys after generation;
- repair escapes;
- deduplicate rationale text;
- fall back to unconstrained generation;
- use a lenient parser;
- change grammar or stopping behavior by arm;
- rerun dev before the V2 review gate passes.
