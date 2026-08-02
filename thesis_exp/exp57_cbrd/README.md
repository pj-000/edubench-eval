# Exp57: Consensus--Boundary Residual Decomposition (CBRD)

This is a mechanism audit, not a relabeling experiment and not a new test-set
claim.  All development work is restricted to the frozen 2,654/664 train/dev
splits.  The historical 2,218-row test set is prohibited during CBRD method
development.

For a consensus label \(y\), empirical three-rater distribution \(d\), and
auxiliary logits \(z\), CBRD begins from the exact identity:

\[
\operatorname{CE}(d,z)=\operatorname{CE}(e_y,z)+(e_y-d)^\top z.
\]

The first term is **consensus supervision**.  The second is a signed,
adjacent-boundary residual.  For example, `[3,4,4]` has hard label 4 and the
residual route lowers the auxiliary 4-vs-3 logit gap; `[4,4,5]` lowers the
4-vs-5 gap.  A unanimous target has a zero residual.

This identity alone is not the paper contribution.  The empirical question is
whether the residual route causes an effect beyond a second hard-label loss.
Stage 0 must pass before any GPU work:

1. resolve and archive the exact Exp49/50/51 source blobs;
2. audit the 13 observed target vectors and the fixed Exp55 permutation;
3. verify scalar and gradient identities, including the single-soft-CE hidden
   gradient-routing implementation.  A preliminary detached two-branch
   implementation was rejected because BF16 accumulation changed shared
   backbone gradients; the production route uses one original auxiliary CE.

Only then can the pre-registered train/dev intervention ladder begin:
DualHard, CBRD Consensus-only, Routed-HMSA, Residual-only, Sign-flipped, and
Shuffled-residual.  The primary comparison is Routed-HMSA versus
Consensus-only because it holds the auxiliary-head target and trajectory fixed
and changes only the backbone residual route.
