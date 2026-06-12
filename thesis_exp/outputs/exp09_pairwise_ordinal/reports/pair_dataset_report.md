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

No synthetic rows are used. Dev pairs are diagnostic only and are not used to tune test thresholds.
