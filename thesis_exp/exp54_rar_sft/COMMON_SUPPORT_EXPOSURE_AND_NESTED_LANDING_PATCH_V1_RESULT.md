# Common-Support Exposure and Nested-Landing Patch V1 Result

## Decision

`WRITE_THESIS_NO_NEW_METHOD_ON_CURRENT_DATA`

Scientific direction label:

`D_NO_METHOD_DIRECTION_IDENTIFIED_ON_CURRENT_DATA`

This finite patch corrects the composition-confounded interpretation of the
first residual-risk analysis. It does not claim that response exposure,
boundary support, or relative-to-absolute mismatch are ineffective.

## Exposure

The actual three-level inventory is:

- E2, exact QA response seen: 615 rows;
- E1, question seen but answer/response unseen: 49 rows;
- E0, question unseen: 0 rows.

E1 contains only labels 3/4/5. Labels 1/2 contain 20 rows, all E2. Therefore
neither truly unseen-question generalization nor low-score exposure effects are
identified.

Within the 27 metric-language-label-ambiguity strata shared by E1 and E2, R3
E1-minus-E2 standardized MAE was -0.086, -0.074, and -0.072 for seeds
42/43/44. All three question-cluster multiplier-bootstrap 95% intervals were
below zero. This supports only the bounded statement that E1 was not harder
than E2 among common-support labels 3/4/5. It says nothing about labels 1/2 or
E0.

S0-to-R3 and R3-to-P1 paired benefits did not show a stable E1 advantage.

## Support

The original measure is renamed cumulative-threshold marginal support. The
patch also reports direct adjacent-class support
`min(count(Y=k), count(Y=k+1))`.

Under both definitions, every label-1/2/3 dev row is unsupported. The low-score
support effect remains a positivity failure and is not estimated.

## Descriptive low-score probability transport

For all 20 label-1/2 dev rows, restricted per seed to records with positive
removed score-4/5 mass, P1 showed:

| Seed | Positive DeltaH rows | GLE | MCR | Other-low capture |
|---|---:|---:|---:|---:|
| 42 | 18/20 | 0.808 | -0.092 | 0.284 |
| 43 | 20/20 | 0.680 | 0.016 | 0.304 |
| 44 | 20/20 | 0.719 | 0.096 | 0.185 |

The removed high-score mass moved predominantly toward the observed gold score
and the other low score, not toward score 3. Thus the current probabilities do
not support the proposed high-MCR/low-GLE middle-capture signature. This is a
descriptive result: because all low-score rows lack support overlap, it cannot
separate data support from preference-objective mechanisms.

The strict cumulative-supported and direct-adjacent-supported low-score
populations both contain zero rows, and their GLE/MCR remain null.

## Final scope

No new loss, preference family, reasoning module, rubric intervention, or new
training is authorized on the current artifacts. The completed RAR-SFT and
actual-error Field-DPO chain can now be written as a bounded empirical master
thesis. New identification data remain a possible future study, but the
specific middle-capture hypothesis did not receive the descriptive signal
required by this patch to prioritize that collection.

No test artifact, GPU, new model call, or new training was used.
