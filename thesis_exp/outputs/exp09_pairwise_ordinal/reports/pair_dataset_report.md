# Exp9 Pair Dataset Report

Status: `ready_for_smoke`

- Train pairs: `20000`
- Dev diagnostic pairs: `5000`
- low_high train pairs: `8000`
- Pair shards: `10`
- Max low-record reuse: `208`
- Mean low-record reuse: `108.14414414414415`

## Train Pair Type Distribution

| pair_type | count | rate |
| --- | ---: | ---: |
| low_high | 8000 | 0.4000 |
| low_mid | 4000 | 0.2000 |
| adjacent | 6000 | 0.3000 |
| random_ordinal | 2000 | 0.1000 |

## Dev Pair Type Distribution

| pair_type | count | rate |
| --- | ---: | ---: |
| low_high | 2000 | 0.4000 |
| low_mid | 1000 | 0.2000 |
| adjacent | 1500 | 0.3000 |
| random_ordinal | 500 | 0.1000 |

## Pair Comparability Audit

Pair construction follows priority `same_question > same_metric_language > same_metric`; an
`any_valid` fallback is used only if needed to fill the configured pair budget.
Actual comparability rates are reported below. Formal training results should be interpreted in
light of these rates.

| split | pair_type | pairs | same_question | same_metric | same_language | same_metric_language |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| train | low_high | 8000 | 0.2604 | 0.7550 | 1.0000 | 0.7550 |
| train | low_mid | 4000 | 0.0895 | 0.8260 | 0.5225 | 0.3970 |
| train | adjacent | 6000 | 1.0000 | 0.1208 | 1.0000 | 0.1208 |
| train | random_ordinal | 2000 | 1.0000 | 0.0795 | 1.0000 | 0.0795 |
| dev | low_high | 2000 | 0.2850 | 0.7355 | 1.0000 | 0.7355 |
| dev | low_mid | 1000 | 0.2780 | 0.6050 | 0.6770 | 0.3580 |
| dev | adjacent | 1500 | 1.0000 | 0.1067 | 1.0000 | 0.1067 |
| dev | random_ordinal | 500 | 1.0000 | 0.0800 | 1.0000 | 0.0800 |

No synthetic rows are used. Dev pairs are diagnostic only and are not used to tune test thresholds.
