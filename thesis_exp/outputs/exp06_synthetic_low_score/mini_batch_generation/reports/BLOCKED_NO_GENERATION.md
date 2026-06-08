# BLOCKED: Exp6-3 Mini-batch Generation Did Not Run

Generation mode was requested, but no synthetic answers were generated.

Reason: **missing GENERATION_MODEL**

Required environment:

- `EXP6_RUN_GENERATION=1`
- `GENERATION_MODEL`
- `GENERATION_API_KEY` or `DEEPSEEK_API_KEY`, unless `GENERATION_ENDPOINT` is a local endpoint that
  needs no key

The runner is capped at 24 prompt rows and never logs API keys.
