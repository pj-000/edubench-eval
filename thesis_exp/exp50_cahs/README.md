# Exp50-CAHS-0.5

Exp50 tests one fixed internal point between the Exp49 hard and full-human-soft
targets. It does not reopen Exp49 and does not search alpha.

`target = 0.5 * one_hot(label_5) + 0.5 * human_empirical_distribution`

Only the CAHS seed42 arm is new. The frozen Exp49 B0 seed42 run is the
comparator. Both use the fixed 2654/664 split, Exp02 question+answer+metric
prompt, Qwen3-Reranker-0.6B, cosine scheduler, and strict highest-Exact
checkpoint selection with earlier-epoch ties. Scout/formal paths refuse test.

Run order:

1. Mechanism/reproducibility audit.
2. CPU tests.
3. Two-run deterministic B0 GPU smoke.
4. CAHS seed42.
5. Run the locked gate; stop on `SEED42_NO_GO`.
