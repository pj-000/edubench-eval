# Exp54 minimal rationale audit checkpoint

## Completed without evaluator calls

| Item | Result |
|---|---|
| Unique dev rows | 40 |
| Label 1 / 2 / 3 / 4 / 5 | 6 / 14 / 8 / 8 / 4 |
| Languages | English 20, Chinese 20 |
| Metric coverage | All 12 metrics |
| Primary comparison | R3 versus R2 |
| Secondary comparison | R3 versus R1 |
| Pair instances per comparison | 120 (40 rows × 3 seeds) |
| A/B orientations | 2 per pair |
| Score-blind presentations | 480 |
| Score-visible presentations | 480 |
| Test accessed | No |
| GPU used | No |

The private sample manifest, score-blind tasks, score-visible tasks, and
orientation answer key were generated on the server and are excluded from
version control. The public candidate report contains only counts and hashes.

The selector reads only dev record identity/order, label, metric, and language.
It selects all 20 Label-1/2 rows, 8 Label-3 rows, and 12 Label-4/5 rows. It
cannot inspect generated score, rationale, arm, seed, forced-completion status,
or evaluator preference.

## Current external blocker

No Qwen, DeepSeek, OpenAI, or Anthropic API key is configured in the local or
server environment. The server has only the Qwen model family locally, so it
cannot provide the required two independent evaluator families without an
additional model/API channel.

Both Qwen and DeepSeek runner paths pass dry-run validation and each sees the
expected 480 tasks per stage. Actual evaluator calls have not occurred.

To continue, configure two independent provider keys in the server process
environment without writing keys into the repository, then run both stages for
each provider. Exact model names must be recorded at execution time.

## Preference-data work completed in parallel

`SORC_DPO_PAIR_CONSTRUCTION_PROTOCOL.md` fixes the negative-sample design:

- one adjacent wrong-score pair for every train row;
- one severe 4/5 overestimate pair for every Label-1/2 row;
- one 1/2 underestimate protection pair for every Label-4/5 row;
- one R3-aligned versus R2-shuffled rationale pair for every eligible control
  event;
- score pairs change only score;
- rationale pairs change only rationale;
- invalid, truncated, or dev/test-derived negatives are forbidden.

Preference training has not started. The exact ODPO offset placement remains a
theory/implementation check before training.
