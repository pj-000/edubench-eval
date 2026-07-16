# Exp48C Codex isolated pointwise audit dispatcher

Audit all 36 records in the supplied private JSONL, but preserve strict pointwise isolation:

- create one fresh reviewer context per record;
- pass only that one record plus `exp48c_rubric_only_pointwise_verifier_prompt.md` and the output schema;
- never pass previous outputs, sibling records, the private mapping, or intended scores into a reviewer context;
- use `gpt-5.5`, matching the Exp48B contract-aware verifier model family;
- use deterministic/high-rigor review;
- combine the 36 returned JSON objects in the original packet order into the requested private JSONL;
- do not repair or alter a returned score during collection.

Each output provenance must use:

- `verifier_id`: `codex`
- `model_family`: `gpt-5.5`
- `model_version`: the exact available GPT-5.5 identifier
- `session_id`: a distinct reviewer-context ID for that packet

Do not read the private answer mapping. Do not read dev/test. Do not modify the frozen answers.
