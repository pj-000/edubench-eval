# Exp28E Ordinary-CE Training Variants

- protocol: `p0_holistic_zero_shot`
- paper train/dev: 2654/664
- accepted selective changes: 518
- unresolved conflicts filtered in B3: 787
- primary teacher: Qwen3.7-Max on all 2,654 train rows
- secondary teacher: DeepSeek-V4-Pro on the locked selective route only
- student input: question + answer + evaluation dimension
- loss: ordinary cross-entropy
- test read: no

B0 preserves the original benchmark labels. B1 replaces every target with the primary teacher
score. B2 changes labels only when Qwen and DeepSeek agree with high confidence on an adjacent
label transition. Risky transitions are never relabelled automatically.
B3 removes unresolved teacher conflicts. B4 applies the same transition multiset to randomly
matched rows and is a negative control for selective targeting.

Both teachers receive question, answer, metric, rubric, and metadata and return a rubric-grounded
structured audit. Their reasons, failure tags, confidence, and score caps are used only for audit
and routing. The fixed Qwen3-Reranker-0.6B student receives the original paper input format and is
trained only on the resulting 1-5 class target with ordinary cross-entropy. All teacher-derived
targets are model-generated silver supervision, not human adjudication or replacement gold.
