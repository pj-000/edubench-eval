# Exp29 Uncertainty-Preserving Dual-Target CE Data

Every original benchmark target is retained. For the 518 high-confidence adjacent disagreements
accepted by the locked Qwen/DeepSeek audit, C1 adds a second observation carrying the teacher
target. C2 repeats the same selected rows with the original target, controlling for exposure.
C3 adds the identical transition multiset at matched random positions, controlling for targeting.
All variants use ordinary cross-entropy; dev is identical and test is neither read nor written.
