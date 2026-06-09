# Exp6 Batch96 Plan

| item | value |
| --- | --- |
| Curated usable count | 16 |
| Revised count | 3 |
| Rejected count | 1 |
| Batch96 planned count | 96 |
| Label distribution | `{'1': 40, '2': 40, '3': 16}` |
| Language distribution | `{'en': 48, 'zh': 48}` |
| Metric coverage | 12 |
| Error type coverage | 7 |
| API called | NO |
| Synthetic generated | NO |
| Batch96 generation can start | YES |
| Full 384 can start | NO |
| Training can start | NO |

## Required gates

- Use only `question_seed42/train` sources.
- Keep dev/test questions out of generation sources.
- Run only batch96 when API generation is explicitly enabled.
- Do not start full 384 or Exp6 training from this plan.
