# Exp28D Selective Secondary-Teacher Route

- selected protocol: `p0_holistic_zero_shot`
- paper-train rows: 2654
- primary Qwen rows: 2654
- routed to DeepSeek: 1552 (58.48%)
- confidence threshold: 0.75
- locked high-control rate: 10%
- dev/test read: no

Routing is triggered by original low scores, primary/original disagreement, large score gaps,
low primary confidence, detected failures, score caps, and a deterministic sample of confirmed
high-score controls. The secondary teacher receives the same blind input and never sees the
original label or Qwen output.
