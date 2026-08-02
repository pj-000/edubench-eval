# Exp59: Residual Geometry Ablation under Standard Global Clipping

Exp59 follows the formal Exp58 treatment-integrity stop.  It does not revise
the Exp58 five-percent cap gate and does not claim that the capped seed-42 run
confirmed a full residual effect.

At each complete accumulation window, the backbone-only residual is decomposed
relative to the Consensus-only backbone gradient into parallel and orthogonal
components.  The two new arms add exactly one component to the complete common
gradient and then use the original global `clip_grad_norm_` rule.  Both heads
retain their original hard-CE and empirical soft-CE updates.  There is no beta,
safety cap, component norm matching, or residual-weight search.

The unique primary comparison is Orthogonal-only versus Consensus-only across
seeds 42--46.  Parallel-only is a secondary mechanism comparison.  Only
train/dev are allowed; the historical test split is prohibited.
