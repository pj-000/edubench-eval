# Training-manifest schedule compatibility review

## Newly discovered interaction

The frozen R2 donor map is a valid strict permutation over active references. The preregistered
trainer does not use every reference on every row occurrence, however. It selects one reference
per sample and epoch using:

```text
j(seed, sample_id, epoch) =
    (SHA256(seed | sample_id) offset + epoch) mod reference_count
```

Consequently, a reference attached to a two-reference row and a reference attached to a
three-reference row have different realized selection frequencies during a three-epoch run. When
the R2 donor permutation crosses those row types, the R2 and R3 reference inventories remain the
same but their realized rationale-text frequencies do not.

## Current formal-map diagnostic

- Active references: 3,904 / 3,934.
- Active donor edges crossing 2-reference and 3-reference rows: 1,742 / 3,904.
- Active rationale events per arm over three epochs: 4,800.
- R2/R3 frequency multiset identical: `false` for seeds 42, 43, and 44.
- Absolute per-text frequency delta:
  - seed 42: 1,302;
  - seed 43: 1,302;
  - seed 44: 1,284.

The total number of active rationale events is matched, but the empirical frequency of individual
rationale texts is not. This weakens the claim that answer-rationale pairing is the only R2/R3
difference at training time.

## Candidate A: reference-count strata plus six epochs

Add `reference_count` to the donor strata and train for six epochs, the least common multiple of
two and three.

- Active references: 3,866 / 3,934 = 98.27%.
- R2/R3 frequency multiset identical for all three seeds: `true`.
- Label 1 active samples: 16 / 21.
- Label 2 active samples: 28 / 40.
- Active rationale events per arm: 9,510.

This restores exact frequency matching but changes the frozen donor map, reduces low-score
coverage, and doubles row exposures relative to the candidate three-epoch configuration.

## Candidate B: three-epoch schedule-signature strata

Add `reference_count` and the reference's three-epoch selection-frequency signature for seeds
42/43/44 to the donor strata.

- Active references: 3,764 / 3,934 = 95.68%.
- R2/R3 frequency multiset identical for all three seeds: `true`.
- Label 1 active samples: 16 / 21.
- Label 2 active samples: 24 / 40.
- Active rationale events per arm: 4,602.

This preserves three epochs and exact frequency matching, but causes a larger overall and
low-score coverage loss.

## Current gate

Status: `TRAINING_MANIFEST_SCHEDULE_BLOCKED`.

The existing reference-level donor map remains a valid strict permutation, but the four training
manifests must not be frozen under the current schedule until the scientific contract specifies
whether exact realized rationale-frequency matching is required and, if so, which correction is
accepted.
