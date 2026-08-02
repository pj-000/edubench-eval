# Exp57 CBRD Stage 1 frozen protocol

Stage 1 is a train/dev-only causal intervention study.  It asks whether the
sample-aligned signed adjacent residual entering the shared backbone has an
effect beyond a second consensus task.  It does not treat the CE identity or
the dual-head architecture as a new method by itself.

The unique primary comparison is **Routed-HMSA versus Consensus-only**.  Both
train the auxiliary head with the same empirical soft CE; only the residual
component returned to the backbone differs.  DualHard, Residual-only,
Sign-flipped and Shuffled-residual answer distinct secondary questions.

All six arms first run seed 42.  Seed 42 can reject only broken, unstable or
catastrophically degraded implementations; a small metric difference cannot
be used to prune an arm.  If integrity gates pass, all six run seeds 43 and
44.  Seeds 45 and 46 are reserved for the primary pair and may be run only
after its frozen three-seed development gate passes.

Detached-soft is a seed-42 technical parity run, not a scientific multi-seed
arm.  The fixed Exp55 within-label mapping is reused only for the residual
entering the backbone.  The auxiliary-head target always remains the original
sample-aligned empirical distribution.

Training settings, metric direction, stopping rules, route definitions and
interpretation limits are machine-readable in `stage1_protocol.json`.  The
historical test split is prohibited throughout Stage 1.
