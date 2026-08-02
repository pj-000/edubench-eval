# Exp58: Common-Scale-Matched Residual Control

Exp58 is the single frozen control required after the Exp57 clipping audit.
It is not a new prediction architecture.  It asks whether the aligned human-
disagreement residual remains useful when the hard, soft-head, and common
backbone gradients receive exactly the Consensus-only clipping scale.

For one accumulated pre-clip window,

`G_R = G_C + R`

and the matched update is

`U_M = alpha_C * G_C + beta * R`.

`beta` is `alpha_R` whenever that update respects the original maximum norm of
1.  Otherwise it is the largest non-negative value no greater than `alpha_R`
that satisfies the same norm bound.  The update is written directly to
parameter gradients before AdamW; ordinary global clipping must not be called
again.

Only train/dev are permitted.  The historical EduBench test is prohibited.
Existing five-seed Consensus-only and Routed-HMSA runs are reused; the only new
formal arm is the matched control at seeds 42--46.

Before formal training, the frozen v2 protocol requires two no-update checks on
the same complete accumulation window.  FP32 is the strict algebraic identity
gate.  BF16 is the actual-runtime safety and direction-preservation gate, with
a tolerance derived from BF16 unit roundoff.  The first BF16 failure remains
preserved as an audit artifact; no optimizer step preceded this revision.
