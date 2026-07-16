# Exp48A blind verifier session

You are one independent criterion verifier. Read one private verifier packet. Do not infer, output, or discuss target/intended scores. Do not use external dev/test examples or any hidden construction trace.

For every anonymous answer and every rubric criterion, output exactly one status:

- `satisfied`: the answer explicitly and substantively fulfills the requirement;
- `partial`: relevant evidence exists but is incomplete;
- `violated`: the required content is absent/contradicted, or a prohibited condition occurs;
- `unclear`: the criterion cannot be resolved from the answer and question.

For `satisfied` or `partial`, copy a short exact evidence span. For `violated` or `unclear`, provide a concise missing-evidence reason. Set answer uncertainty to `low`, `medium`, or `high`. Do not emit any score field. Evaluate each answer independently before comparing the three.

Return JSONL only, conforming to `schemas/exp48a_criterion_verification_schema.json`. Include your real model family and a unique independent session ID in `verifier_provenance`.
