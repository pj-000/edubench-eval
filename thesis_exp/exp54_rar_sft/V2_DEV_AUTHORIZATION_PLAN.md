# Exp54 V2 dev execution plan

The earlier trust-anchor and one-use-claim design is retired from the active
execution path. It was an operational hardening proposal, not a requirement
of the scientific experiment.

## Normal execution workflow

1. Complete and test the V2 decoder and direct dev runner locally.
2. Commit the exact source and synchronize it to the server.
3. Confirm that the selected GPU is idle and allowed by the operator.
4. Start the fixed S0/R1/R2/R3 × seeds 42/43/44 × logical epochs 1/2/3
   campaign in tmux.
5. Retain stdout/stderr logs and one result directory per checkpoint.
6. Aggregate all 36 dev results only after every task has completed.

The direct runner uses runtime batch size 32 on the 24 GB RTX 3090 cards and
the frozen 256-token V2 grammar decoder. Batch size is an execution-throughput
parameter; the singleton/batched regression requires identical decoded token
IDs. It reads only the dev split; test remains inaccessible.

## Duplicate-run protection

Each task has a deterministic result path:

`dev_runs_v2/<arm>/seed<seed>/epoch<epoch>`

The runner atomically creates that ordinary result directory with
`exist_ok=False` before reading dev rows or loading the model. If the path
already exists, the task fails instead of overwriting or silently rerunning.
No separate claim directory, trust anchor, staged digest, root ownership,
append-only flag, special file mode, administrator action or `sudo` is used.

If a task fails, retain its result directory and log, stop the campaign, and
diagnose the failure before deciding whether to remove the failed output and
restart it. Codex must not delete or rerun a task without the operator's
knowledge.

## Scientific boundary

This simplification changes only operational launch control. It does not
change the model, adapters, dev rows, prompts, grammar, decoder, generation
budget, batch size, checkpoint set or metrics. Completion of dev still does
not authorize test access or permit protocol selection from test results.
