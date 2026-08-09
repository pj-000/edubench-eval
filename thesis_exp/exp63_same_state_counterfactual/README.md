# Exp63: same-state counterfactual residual update

Exp63 is a train/dev-only mechanism control for Exp57--Exp59. It regenerates
five outcome-independent Consensus-only trajectories, freezes the complete
model/AdamW/scheduler/RNG state after epochs 2, 5 and 8, and applies four
one-step counterfactual gradients from each exact state.

The four candidates share the same samples and are matched to a full-parameter
gradient norm of 0.95 before a common global clipping threshold of 1.0. Thus
no candidate is actually clipped. The complete dev split is the frozen probe;
the historical test split is forbidden.

The primary unit is a seed: the three fixed stage contrasts are averaged
within seed before the five-seed decision is calculated.

