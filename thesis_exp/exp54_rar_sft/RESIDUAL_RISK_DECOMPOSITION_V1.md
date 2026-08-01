# Residual Risk Decomposition V1

This is the final no-training direction-selection diagnostic for Exp54. It
uses only the locked train/dev split, frozen epoch-3 S0/R2/R3 predictions,
frozen LR-5e-6 P1 Field-DPO predictions, and already extracted canonical
five-score probabilities for R3/P1.

It separates three observationally entangled explanations for the remaining
low-score risk:

1. response exposure: dev reuses a response observed under another metric;
2. local ordinal support: adjacent score boundaries are weakly represented in
   the matching metric-language train stratum;
3. correction landing: P1 removes probability from scores 4/5 without moving
   it to the observed gold score.

For a low-score record, define

\[
\Delta H_i=-\{[p_{P1}(4)+p_{P1}(5)]-[p_{R3}(4)+p_{R3}(5)]\},
\]

\[
\Delta G_i=p_{P1}(y_i)-p_{R3}(y_i),\qquad
\Delta M_i=p_{P1}(3)-p_{R3}(3).
\]

On records with positive \(\Delta H_i\), the aggregate gold landing
efficiency and middle capture rate are

\[
\mathrm{GLE}=\frac{\sum_i\Delta G_i}{\sum_i\Delta H_i},\qquad
\mathrm{MCR}=\frac{\sum_i\Delta M_i}{\sum_i\Delta H_i}.
\]

The exact gates are frozen in
`configs/residual_risk_decomposition_v1.json`. Exactly one result is emitted:

- A: build a response-disjoint evaluation/benchmark study;
- B: study local ordinal support and identifiability;
- C: authorize a matched single-seed relative-to-absolute pilot;
- D: stop method expansion and write the bounded empirical thesis.

If more than one of A/B/C passes, D is selected because there is no uniquely
identified direction. No test artifact, new model call, or new training is
allowed.
