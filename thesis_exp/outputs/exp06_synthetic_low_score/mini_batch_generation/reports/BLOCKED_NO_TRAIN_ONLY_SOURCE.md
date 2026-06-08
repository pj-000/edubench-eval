# BLOCKED: Exp6-3 Mini-batch Has No Complete Train-only Source Selection

The 24-row mini-batch target matrix was created, but source selection is incomplete.

- Target rows: **24**
- Selected train-only source rows: **0**
- Prompt rows: **0**

The current split has train rows whose source question keys also occur in dev/test. Under the Exp6-3
rule that source question/triple keys must not occur in dev/test, these rows cannot be used as
generation anchors without manual approval or a new train-only-by-question source split.

No API generation should be started from this mini-batch until this blocker is resolved.
