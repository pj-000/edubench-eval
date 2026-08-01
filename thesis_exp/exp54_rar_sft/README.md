# Exp54 / RAR-SFT

RAR-SFT stands for **Rubric-Aligned, Reliability-Gated Rationale Supervised Fine-Tuning**.

> Protocol update (2026-07-23): the historical structured five-field design has been superseded by
> [`PREREGISTRATION_V2.md`](PREREGISTRATION_V2.md), which defines **Rater-Aligned Multi-Reference
> Rationale SFT** with S0/R1/R2/R3 controls. RAR-0A remains the valid provenance-alignment audit;
> its old field gates do not authorize the superseded criterion/evidence conversion pipeline.

> Mechanism-follow-up update (2026-07-30): after completion of the frozen
> RAR-SFT and preference test, [`THESIS_SCIENTIFIC_CONTRACT_V1.md`](THESIS_SCIENTIFIC_CONTRACT_V1.md)
> freezes three post-hoc mechanism controls: R3-BLOCK versus R3-TOKENAVG,
> P1-FIELD versus P1-FULLSEQ, and P1-ACTUAL versus matched P1-SYN. The contract
> requires nine new training runs and changes no completed result.

The experiment starts with RAR-0, a train-only supervision qualification audit. RAR-0 does not
call an API, load a model, use a GPU, train, or read dev/test. Its first executable slice answers:

1. which original human reasons can be joined uniquely to the paper-like train rows;
2. whether each reason score agrees with its originating human rater score;
3. whether it also agrees with the aggregate five-point target used by SFT;
4. how coverage varies by score, metric, and language.

Run the deterministic alignment audit:

```bash
python -m thesis_exp.exp54_rar_sft.audit_rar0_alignment
```

Run tests:

```bash
pytest -q thesis_exp/tests/test_exp54_rar0_alignment.py
```

The audit intentionally separates three statuses:

- `aligned_eligible`: exact unique alignment, non-empty reason, and agreement with that rater's
  original score;
- `aggregate_score_consistent`: the reason's score also equals `label_5`;
- `semantically_qualified`: always false in this first slice. Rubric/evidence/rationale fields
  become eligible only after structured conversion and independent verification.

The historical Exp53 score-only SFT is not a matched baseline because it excludes the rubric.
RAR experiments must therefore add a rubric-aware score-only baseline with the same input as R1,
R2, and R3.

## R2 strict-control implementation checkpoint

The executable R2 stages are:

1. `build_rar_v2_reference_sets.py`: deterministic R1/R3 reference construction and normalized
   question-answer content keys;
2. `audit_r2_solver_oracle.py`: brute-force objective and input-order invariance audit;
3. `build_r2_donor_map.py`: character diagnostic or formal tokenizer-based strict donor map;
4. `build_r2_r3_active_mask.py`: byte-identical R2/R3 rationale-active masks and lock.

Current status:

- scientific rule: `PASS_STRICT`;
- solver audit: `R2_MATCHER_ORACLE_PASS`;
- tokenizer: `QWEN_TOKENIZER_REVISION_LOCKED`;
- formal map and active-mask gate: `R2_STRICT_DONOR_MAP_READY`;
- training-manifest construction: allowed;
- formal training: still locked until S0/R1/R2/R3 manifests and remaining protocol locks pass.

Run the local tokenizer-independent audit:

```bash
PYTHONPATH=. python3 thesis_exp/exp54_rar_sft/audit_r2_solver_oracle.py
```

Run formal token matching in the frozen training environment:

```bash
PYTHONPATH=. python3 thesis_exp/exp54_rar_sft/build_r2_donor_map.py \
  --length-mode tokenizer \
  --tokenizer-path /home/share/models/modelscope/Qwen/Qwen3-4B-Instruct-2507
```

The formal command refuses to run if the Oracle report is stale, the upstream revision is not an
immutable SHA, or the local `tokenizer.json` hash differs from the frozen official hash in
`configs/qwen_tokenizer_lock_spec.json`.

Freeze the matched active masks:

```bash
PYTHONPATH=. python3 thesis_exp/exp54_rar_sft/build_r2_r3_active_mask.py
```

Raw human rationale text, row-level donor mappings, and row-level masks remain local. GitHub may
contain only code, tests, hashes, and aggregate audit reports.
