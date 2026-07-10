# Exp27L-R1 Balanced Group Crossfit Preparation

Exp27L-R1 uses sklearn StratifiedGroupKFold with question-key groups. It fails fast on any outer-fold balance violation and has no fallback to Exp27L's retired greedy allocator. All 180 rows are train-only; dev/test identifiers are used only for zero-overlap guards.
