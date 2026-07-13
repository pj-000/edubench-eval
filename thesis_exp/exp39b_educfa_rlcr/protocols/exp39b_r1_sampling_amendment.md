# Exp39B-R1 Sampling Amendment

Exp39B originally required both source-row and question-key disjointness from Exp39A. Its train-only preparation established that Exp39A's 240 source rows cover all 196 paper-like train question keys, leaving no eligible question key.

This amendment was registered before any Exp39B API call, generation, critique, revision, or verification result existed. It changes only the sampling unit:

- Source sample IDs remain disjoint from Exp39A.
- Exact and near-duplicate answer text remains disjoint from Exp39A.
- Exact assessment keys remain disjoint from Exp39A.
- Question clusters may be reused, with exactly one R1 source per question key.
- The frozen RLCR prompts, schemas, target bands, edit budgets, revision rules, and success gates remain unchanged.

The amended estimand is **response-level protocol feasibility within previously seen question clusters**. Exp39B-R1 does not test unseen-question generalization. Any downstream GroupCV must remain question-key disjoint, and paper-like dev/test remain sealed.
