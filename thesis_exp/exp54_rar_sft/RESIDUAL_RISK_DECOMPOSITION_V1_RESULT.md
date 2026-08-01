# Residual Risk Decomposition V1 Result

## Decision

`D_STOP_METHOD_EXPANSION`

None of the three preregistered A/B/C direction gates passed. This result does
not say that response exposure, sparse boundaries, or relative preference
objectives never matter. It says that the present locked train/dev evidence
does not uniquely support one of them as the next method-development target.

## Main evidence

- The dev set has 615 response-seen rows and 49 response-unseen rows, with zero
  exact response-metric edges shared with train.
- Response-unseen rows were easier, not harder, for R3. Unseen-minus-seen MAE
  was -0.259, -0.249, and -0.243 for seeds 42/43/44; every 95% clustered
  bootstrap interval was strictly below zero. RAR/P1 benefits were not mainly
  concentrated in response-seen rows. Gate A failed.
- The binary adjacent-boundary rule marks 286 rows unsupported and 378 rows
  supported. All score-1, score-2, and score-3 dev rows are unsupported, while
  supported rows contain only scores 4/5. The large raw unsupported-minus-
  supported MAE gap (0.308, 0.275, 0.286) is therefore label-composition
  confounded.
- After aggregating by metric-language-boundary cell and controlling boundary
  identity, support was positively rather than negatively associated with
  cumulative-boundary error in both response-seen and response-unseen strata.
  The preregistered adverse continuous-support condition failed in all seeds.
  Gate B failed.
- The strict landing population requires score <=2, response seen, supported
  adjacent boundaries, and H0/H1 ambiguity. It contains zero rows because no
  score-1/2 dev row meets the support threshold. GLE/MCR are therefore not
  identified under the preregistered control population; they are not treated
  as zero. Gate C was not testable and did not pass.
- R2-minus-R3 stratum MAE contrasts changed sign across seeds/support strata,
  so the rationale semantic contrast remains non-decisive.

## Scientific interpretation

The analysis rejects the simple claim that current dev performance is inflated
because unseen responses are harder. It also shows why the apparent binary
support result cannot be used as evidence for a support-aware method: support
status is structurally entangled with the gold label. Finally, the current data
contain no support-controlled low-score population on which to isolate a
relative-to-absolute landing mismatch.

Per the frozen decision rule, the project should not introduce another loss,
preference-pair family, reasoning module, or rubric intervention on this
evidence. The defensible next action is to write the completed RAR-SFT and
actual-error Field-DPO work as a bounded empirical master-thesis contribution.
A future CCF-A method claim would require genuinely new identification data
(for example, supported low-score boundary cells), not another transformation
of the same train/dev artifacts.

No GPU, new training, API call, test artifact, or row-level public output was
used.
