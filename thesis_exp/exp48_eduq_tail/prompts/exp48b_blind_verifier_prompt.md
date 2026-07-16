# Exp48B independent blind metric-contract verifier

You are an independent verifier. You do not know how the answers were constructed and must not infer, output, or discuss an intended score. Use only the synthetic question, the original metric-specific 1-5 rubric, the three metric-specific contract assertions, and each anonymous answer.

For each answer, independently judge all three assertions:

- `entailed`: the answer contains explicit evidence that the assertion is true;
- `contradicted`: the answer contains explicit evidence incompatible with the assertion;
- `absent`: the answer contains no relevant evidence for resolving the assertion;
- `unclear`: the available text does not permit a reliable decision.

For `entailed` and `contradicted`, `evidence_span` must be a non-empty exact contiguous substring copied from that anonymous answer. For `absent` and `unclear`, leave `evidence_span` empty and provide a concise `missing_reason`. Evaluate D2, D3, and H4 as metric-specific assertions grounded in the supplied original rubric; do not replace them with a generic global standard.

Do not output a direct score, target label, edit description, answer ranking, or construction guess. Evaluate every answer before comparing answers. Return exactly one JSON object conforming to `schemas/exp48b_blind_verification_schema.json`.
