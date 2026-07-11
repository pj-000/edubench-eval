# Exp28C Cost-Aware Qwen Candidate Qualification

- scope: paper-train sealed qualification only (120 rows)
- paper dev/test read: no
- decision: **KEEP_QWEN3_7_MAX_THINKING_PRIMARY**
- selected primary teacher: `qwen3.7-max-thinking`

Quality guards were locked before collecting the candidate results. Cost is considered only
after all quality and Max-agreement guards pass. Teacher outputs remain model annotations,
not human review.
