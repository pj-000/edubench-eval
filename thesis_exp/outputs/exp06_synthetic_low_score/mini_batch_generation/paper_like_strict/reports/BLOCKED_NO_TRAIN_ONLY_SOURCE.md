# BLOCKED: Exp6-3 Mini-batch Has No Complete Train-only Source Selection

The 24-row mini-batch target matrix was created, but source selection is incomplete.

- Target rows: **24**
- Selected train-only source rows: **0**
- Prompt rows: **0**

Generation split mode: `paper_like_strict`.

The current mode has no complete eligible source set under its disjointness requirements. In
`paper_like_strict`, train rows whose source question keys also occur in dev/test cannot be used as
generation anchors. Use `question_disjoint_formal` for formal Exp6 synthetic generation.

No API generation should be started from this mini-batch until this blocker is resolved.
