# Exp48C rubric-only pointwise audit

- Status: **EXP48C_RUBRIC_ONLY_AUDIT_NO_GO**
- Frozen input: 12 families / 36 answers; no regeneration or editing.
- Every review used one answer only. Codex packet contexts were isolated.
- Codex same-model ablation: `true`.
- Codex provenance: gpt-5.5, 36 isolated contexts.
- Qwen provenance: qwen3.7-max, 36 independent API requests.
- Codex exact/QWK/score2: 4/36 / 0.4946 / 2/12.
- Qwen exact/QWK/score2: 7/36 / 0.4301 / 1/12.
- Cross-verifier exact/QWK: 27/36 / 0.8063.
- Joint fully confirmed families/metrics: 0/12 / 0/12.
- Contract-aware to rubric-only exact delta: -0.8889.
- Contract-aware to rubric-only QWK delta: -0.5054.
- McNemar exact p: 0.000000.
- recommend_chinese_replication: `false`.
- stop_synthetic_low_tail_route_permanently: `true`.
- No new generation, no training, no GPU, and no dev/test access.
