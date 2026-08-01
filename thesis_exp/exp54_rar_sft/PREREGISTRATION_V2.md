# Exp54 RAR-SFT V2 preregistration

## Status and scope

- Protocol version: `2.1-event-control-amendment`
- Date: `2026-07-23`
- Method/data design status: **frozen for implementation**
- Training status: **locked**
- Dev access in RAR-0A: `false`
- Test access in RAR-0A: `false`
- Supersedes: `PREREGISTRATION.md` for all future Exp54 work
- Historical protocol preservation: the superseded file remains unchanged except for a pointer to
  this version.
- Pre-training amendment: after a train-only schedule audit and external review, the formal R2
  control is defined over scheduled training events rather than static reference identities. No
  dev/test outcome or training run informed this amendment.

Training remains locked until the exact model revision/snapshot, tokenizer/chat-template hashes,
runtime versions, final training configuration, and all deterministic data-readiness checks have
been recorded. This document freezes the scientific question and experimental design; it does not
authorize access to the test split.

The V2 design deliberately removes the former atomic criterion registry, structured failure field,
answer-span evidence field, score cap, model normalizer, training-data model verifier, and
train-time human adjudication. Those components require labels that are not present in the source
data and would change the study into an evaluation of model-created intermediate labels.

## 1. Scientific question

### RQ1

In five-level educational answer-quality scoring, can real multi-reference human scoring
rationales whose originating rater scores agree with the aggregate label reduce severe
low-to-high overestimation while preserving overall scoring performance and high-score
recognition; and is any benefit attributable to the real answer-rationale semantic relationship
rather than extra supervised tokens, output length, or generic multitask regularization?

### H1: score non-inferiority and low-tail improvement

Relative to a matched rubric-aware score-only SFT baseline, RAR-SFT will preserve overall scoring
performance and Label-5 recognition while improving at least one low-tail outcome: low-to-high
error count or Label-2 correct count/recall.

### H2: semantic increment

Relative to a matched shuffled-rationale control with the same score target, metric, language,
coverage mask, output schema, exact scheduled rationale-frequency multiset at every epoch prefix,
and approximately matched paired rationale length, answer-matched human rationales will provide
additional value in scoring and/or visible-rationale evaluation.

### H3: consensus alignment

Relative to using all three raters' rationales, selecting rationales whose originating rater score
matches the aggregate `label_5` will reduce score-rationale conflict for the aggregate-scoring
deployment task.

H1-H3 are experimental hypotheses, not assumptions. In particular, H3 may be rejected if
dissenting rationales contain useful information.

## 2. Intended contribution and claim boundary

The intended contribution is a supervision protocol and controlled empirical study:

1. **Rater-aligned human multi-reference supervision**: preserve original human rationales,
   construct all-rater and aggregate-label-consistent reference sets, and rotate references
   without duplicating rows.
2. **Score-rationale block-balanced SFT**: calculate token NLL within the score and rationale
   fields separately before combining the two task blocks.
3. **Matched semantic-mismatch control**: preserve the auxiliary-task structure and break only the
   answer-rationale relationship.
4. **Low-tail risk evaluation**: evaluate severe low-to-high errors and Label-2 recognition while
   protecting overall score metrics and Label-5 recognition.

This study does not claim:

- a new neural architecture;
- a universally novel SFT algorithm;
- gold, expert-verified, or human-adjudicated rationales;
- criterion-level interpretability or answer-span evidence grounding;
- faithful recovery of model-internal reasoning;
- that model-based rationale evaluation equals human correctness;
- universal effectiveness outside the frozen dataset and model setting.

The rationale is a **visible scoring rationale**, not hidden chain-of-thought.

## 3. Frozen data boundary

### 3.1 Split

- Split family: `paper_like_triple_seed42`
- Train: `2,654`
- Dev: `664`
- Test: `2,218`
- Train SHA-256:
  `0a1733b9209984c5c4291d205d1ac6057bed341717903b9de075d07de44a878e`
- Dev SHA-256 (from the pre-existing formal lock; not newly read by RAR-0A):
  `a18d6a27b9a524d4592a359658ae70c9348fe88e43c962971ba95f62d2b6cdf0`
- Test hash/read: deferred to the final frozen test campaign.

All SFT arms use the same 2,654 train row IDs in the same order. No row is removed because it lacks
a rationale.

### 3.2 Label distribution in train

| `label_5` | Rows |
|---:|---:|
| 1 | 24 |
| 2 | 52 |
| 3 | 251 |
| 4 | 946 |
| 5 | 1,381 |

The low-score tail is sparse. Percentage-only changes in Labels 1-2 must always be accompanied by
raw counts and confusion matrices.

### 3.3 Human-reason provenance

RAR-0A exact alignment used normalized question + answer + canonical metric + generator model,
with no fuzzy fallback.

| Property | Count |
|---|---:|
| Train rows | 2,654 |
| Rows with all three exactly aligned human reasons | 1,612 |
| Rows without an eligible exactly aligned human reason | 1,042 |
| Rows with three aggregate-label-consistent reasons | 710 |
| Rows with two aggregate-label-consistent reasons | 902 |
| Aggregate-label-consistent reasons | 3,934 |
| Aligned dissenting reasons | 902 |

Source hashes:

| Source | SHA-256 |
|---|---|
| Human 1 | `a22d5c36e771325f75df586fa334c4433035d41dc018043d82cea721245fb72f` |
| Human 2 | `796d0f6bf31225d57e198ac7f961e2cbbceb5ec3a71df62a3677d627fe605bf4` |
| Human 3 | `23fc472daa9e573d7b52b29c6e37ba3e0deac679abd7095203c880d7fa211bad` |

Reason coverage is not missing completely at random. It is higher for Labels 1-2 than for Label 5.
Coverage by label, metric, and language must be reported. The R2/R3 matched comparison is the
primary control for the location of auxiliary supervision.

## 4. Model boundary

- Model ID: `Qwen/Qwen3-4B-Instruct-2507`
- Model type: causal language model, non-thinking output contract
- Fine-tuning method: LoRA SFT
- Exact upstream revision: `cdbee75f17c01a7cc42f958dc650907174af0554`
- Training-server snapshot:
  `/home/share/models/modelscope/Qwen/Qwen3-4B-Instruct-2507`
- Tokenizer status: `QWEN_TOKENIZER_REVISION_LOCKED`
- `tokenizer.json` SHA-256:
  `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`
- Tokenizer lock SHA-256:
  `4c4fa5083c4b6da6097657bb15a304df06080a5692033c1f444288574acdf0b6`
- Full model safetensor lock: `UNRESOLVED_BEFORE_TRAINING_MANIFEST_FREEZE`

The historical Exp53 local path and candidate configuration may inform the implementation, but
Exp53 is not the matched scientific baseline because its input excludes the full rubric. No Exp54
training may start from an Exp53 adapter.

Before smoke training, freeze:

- exact immutable model revision and local snapshot;
- config, tokenizer, chat template, and safetensor hashes;
- `transformers`, `peft`, trainer framework, PyTorch, CUDA, and driver versions;
- LoRA rank, alpha, dropout, and target modules;
- cutoff length, optimizer, learning rate, scheduler, warmup, epochs, batch size, gradient
  accumulation, precision, padding, and packing policy.

All matched arms must inherit the same frozen values.

## 5. Model input and output

### 5.1 Input

All matched arms receive the same content:

```text
System:
依据给定问题、回答、评价维度和五级 rubric，
输出 JSON 格式的整数评分和简短可见评分依据。

Question:
{question}

Answer:
{answer}

Evaluation Metric:
{metric}

Five-level Rubric:
{rubric}
```

Language-specific wording may follow the sample language, but prompt templates must be frozen and
hashed before training. The input must never contain:

- `label_5`;
- individual human scores;
- human reasons;
- donor identities;
- model predictions;
- dev/test labels or metadata unavailable at inference.

### 5.2 Output schema

Rationale arms:

```json
{
  "score": 2,
  "rationale": "答案给出了基本结论，但缺少支持该结论所需的关键推导。"
}
```

The matched score-only arm uses the same serialized schema for decoding compatibility but has no
active rationale-content supervision:

```json
{
  "score": 2,
  "rationale": ""
}
```

For rows without an active rationale, the empty field must not become a negative target saying
"produce no rationale." Rationale content, rationale-associated closing/EOS positions, and any
other positions whose supervision would reward an empty rationale must be masked. Only the score
value contributes task loss for those rows.

At inference:

- use the same schema for all matched arms;
- use constrained JSON decoding if supported by the frozen stack;
- otherwise use one frozen deterministic decoding/parser contract;
- use greedy decoding or temperature `0`;
- parse score only from the `score` field;
- never revise score based on the generated rationale;
- never generate or expose a `<think>` field.

## 6. Rationale source construction

For train row \(i\):

- \(y_i\): aggregate `label_5`;
- \(r_{ij}\): individual score from rater \(j\);
- \(\rho_{ij}\): exactly aligned original reason from rater \(j\).

### 6.1 All-rater set

\[
R_i^{all}=\{\rho_{i1},\rho_{i2},\rho_{i3}\}
\]

The set is non-empty only for the 1,612 exactly aligned rows. It is used by R1.

### 6.2 Aggregate-label-consistent set

\[
R_i^{+}=\{\rho_{ij}: r_{ij}=y_i\}
\]

It is used by R3. In the current exact-alignment inventory, every reason-covered row has either two
or three references in \(R_i^{+}\).

The construction is train-label-conditioned auxiliary-target selection. It is not train/test
leakage, but it must be disclosed as a selection effect.

### 6.3 Rows without reasons

\[
q_i =
\begin{cases}
1,& R_i\neq\emptyset\\
0,& R_i=\emptyset
\end{cases}
\]

All 1,042 rows with no eligible reason remain in every arm and contribute score loss. No model
rationale is generated to fill missing supervision.

## 7. Deterministic reason cleaning

Allowed operations:

- Unicode NFKC normalization;
- whitespace normalization;
- removal of source-file metadata accidentally embedded in reason text;
- high-precision removal/replacement of explicit self-reported score phrases;
- removal of a reference only if it is empty after deterministic cleaning.

Forbidden operations:

- model rewriting, summarization, expansion, or normalization;
- rubric-guided completion;
- semantic filtering;
- manual case selection;
- fuzzy provenance recovery;
- addition of claims not present in the source reason.

High-precision score-redaction examples include:

```text
score: 2
rating = 2
I gave this response 2 points
我给这份回答 2 分
该回答得两分
```

Ordinary numbers in mathematical content, facts, dates, enumerations, or answer-relevant reasoning
must not be removed.

Every redaction record must include:

- `sample_id`;
- `rater_id`;
- original reason hash;
- original character span;
- replacement;
- rule ID;
- cleaned reason hash.

Any retained active reason containing an explicit score-report pattern is a deterministic
data-readiness failure.

## 8. Multi-reference schedule

Rows are not duplicated according to their number of references. Every original row occurs once
per epoch in every arm.

References are sorted by stable `rater_id`. For sample \(i\), seed \(s\), and zero-based epoch
\(e\), define:

\[
h_{i,s} =
\operatorname{int}\left(
\operatorname{SHA256}(\texttt{f"\{s\}|\{sample\_id\}"})[0:16], 16
\right)
\]

\[
j_{i,e,s}=(h_{i,s}+e)\bmod |R_i|
\]

The active reference is:

\[
\rho_{i,e,s}=R_i[j_{i,e,s}]
\]

The exact byte encoding and separator used by the implementation must be tested and recorded. The
same function is used by R1, R2, and R3. A saved schedule is an audit artifact; the source of truth
is the frozen deterministic function plus the reference-set hashes.

## 9. Experimental arms

| ID | Input | Score target | Rationale target | Purpose |
|---|---|---|---|---|
| H0 | Q+A+metric | `label_5` | none | Historical reference only |
| S0 | Q+A+metric+rubric | `label_5` | none | Matched score-only baseline |
| R1 | Same as S0 | `label_5` | \(R_i^{all}\) | Effect of including dissenting reasons |
| R2 | Same as S0 | `label_5` | matched donor references | Format/length/auxiliary-task control |
| R3 | Same as S0 | `label_5` | \(R_i^{+}\) | Main RAR-SFT method |

H0 is not a matched scientific baseline and does not determine RAR-SFT success.

R1 is required because label-consistent selection is a central design choice. R2 is required
because S0 alone cannot distinguish real rationale semantics from extra output supervision.

## 10. R2 matched semantic-mismatch control

### 10.1 Candidate strata

For each R3 reference, donors are restricted exactly to:

```text
label_5 × metric_id × language
```

No donor may cross label, metric, or language. A recipient-donor edge is legal only when:

- `recipient.sample_id != donor.sample_id`;
- `recipient.normalized_qa_key != donor.normalized_qa_key`;
- both references belong to the same frozen reference inventory and pass the same cleaning rules.

The normalized QA key is a SHA-256 over the NFKC and whitespace-normalized question-answer pair.
It prevents duplicate content with different row IDs from becoming a false semantic mismatch.

### 10.2 Strict active-subset permutation

For each stratum \(g\), jointly define active indicators \(a_i\) and assignment indicators
\(x_{ij}\). The assignment must satisfy:

\[
\sum_j x_{ij}=a_i,\qquad \sum_j x_{ji}=a_i.
\]

Therefore the active set \(A_g=\{i:a_i=1\}\) is identical on the recipient and donor sides.
Every active reference receives exactly one donor and is used exactly once as a donor. Donor
reuse is forbidden.

The optimization objective is lexicographic:

1. maximize \(|A_g|\);
2. conditional on maximum coverage, minimize
   \(\sum_{i,j}x_{ij}|T_i-T_j|\), where \(T_i\) is rationale content-token length under the
   frozen tokenizer;
3. conditional on the first two objectives, use stable reference IDs for deterministic
   tie-breaking.

The implementation may retain the optional-diagonal Hungarian construction only while a
deterministic brute-force Oracle audit confirms equivalence to this mathematical objective on
adversarial and random strata of size at most eight.

The donor map is created once and reused for all seeds. It records recipient/donor sample IDs,
normalized QA keys, reference indices, strata, token lengths, absolute token-length difference,
active status, inactive reason, cycle ID, solver version, tokenizer revision, and lock hashes.

### 10.3 Ineligible strata

If a reference cannot enter a legal strict permutation:

- deactivate the corresponding rationale position in both R2 and R3;
- retain score loss in both arms;
- record the reason for deactivation.

The final R2/R3 active rationale mask must be identical at reference level and hash level. No
fallback may reuse a donor, use another rater from the same sample, or cross label, metric, or
language.

### 10.4 Interpretation

Some donor rationales may accidentally remain applicable to the recipient. This weakens the
R3-R2 contrast and is treated as conservative contamination, not manually corrected.

The R2/R3 semantic comparison applies only to the control-eligible rationale subset that admits a
strict permutation. Ineligible rows remain in all arms for score supervision. A null R3-R2 result
cannot by itself establish that all low-score human rationales are ineffective because strict
control eligibility reduces rationale coverage in scarce low-score strata.

Character length is a feasibility diagnostic only. The formal donor map must be rebuilt with the
frozen `Qwen/Qwen3-4B-Instruct-2507` tokenizer, an immutable upstream revision, verified local
tokenizer-file hashes, and a frozen Oracle report covering the exact matcher source.

### 10.5 Pre-training event-level amendment

Sections 10.1-10.3 preserve the reviewed reference-level feasibility result, but their static
reference map is superseded as the authority for formal training. The formal R2 control keeps
three epochs and first expands the exact schedule from Section 8 for seeds 42, 43, and 44:

```text
2,654 ordered rows × 3 epochs = 7,962 row events per seed
```

Each event has a deterministic ID derived from the schedule schema version, seed, zero-based
epoch index, zero-based row position, record ID, and selected reference ID. The exact hash
serialization, the use of the first 16 hexadecimal SHA-256 characters for the schedule offset,
and both zero-/one-based epoch conventions must be recorded.

R2 donor matching is performed independently within:

```text
seed × epoch × label_5 × metric_id × language
```

For each event stratum, legal non-diagonal edges require different event IDs, record IDs, and
normalized QA keys. Active recipient events and used donor events must be the same set; each
active event has indegree and outdegree one and belongs to a nontrivial cycle. The objective
remains lexicographic: maximize active events, minimize frozen-tokenizer rationale-length
difference, then use a stable hash tie-break.

Donor non-reuse is defined at scheduled-event occurrence level. If one reference is selected
twice by the original R3 schedule, it produces two distinct events and must occur exactly twice
in R2 as well. The same reference ID is therefore allowed to appear in multiple epochs only when
the frozen base schedule already does so.

R2 and R3 must have byte-identical rationale-active vectors and identical rationale bytes/token-ID
counters separately for every seed and epoch. The same equality must hold for every cumulative
epoch prefix eligible for checkpoint selection. All inactive events retain active score
supervision. The static reference-level map remains a valid audited artifact but has status:

```text
R2_REFERENCE_LEVEL_MAP_VALID
SUPERSEDED_FOR_TRAINING_BY_EVENT_LEVEL_MAP
```

No event-level candidate artifact may be marked frozen or used for training until its matcher
Oracle, event permutation, epoch-prefix frequency equality, cross-arm manifest equality, and
training-budget audit pass a new review gate.

## 11. Token masks and block-balanced loss

For token NLL:

\[
n_{it}=-\log p_\theta(y_{it}\mid x_i,y_{i,<t})
\]

define:

- \(m^s_{it}\): score-value token mask;
- \(m^r_{it}\): rationale-content token mask.

Prompt tokens, padding, fixed schema field names, JSON punctuation, and inactive rationale
positions have task-loss weight zero. They may remain in teacher-forcing context.

### 11.1 Score block

\[
L_{i,s}
=
\frac{\sum_t m^s_{it} n_{it}}
{\max(1,\sum_t m^s_{it})}
\]

### 11.2 Rationale block

\[
L_{i,r}
=
\frac{\sum_t m^r_{it} n_{it}}
{\max(1,\sum_t m^r_{it})}
\]

### 11.3 Per-sample objective

\[
L_i=L_{i,s}+q_iL_{i,r}
\]

The batch loss is the arithmetic mean of \(L_i\) over original rows in the batch.

The rationale coefficient is fixed at:

\[
\lambda_r=1
\]

This is a neutral equal-task-block choice after within-field length normalization. It is not
claimed to be optimal. No coefficient search is allowed in the primary experiment. S0 is the
matched \(\lambda_r=0\) baseline.

The previously discussed agreement weights `0`, `2/3`, and `1` are not part of the V2 primary
method. They may not be added after observing primary results. A future sensitivity study would
require a separately preregistered question and matched control.

### 11.4 Required loss tests

Before smoke training, tests must verify:

- prompt and padding tokens never contribute task loss;
- score masks cover the complete score value under the actual tokenizer;
- rationale masks cover only rationale content;
- inactive rationale rows produce exactly score-only task loss;
- empty rationale and associated EOS/closing positions do not reward empty output;
- rationale length does not change the rationale block's per-sample coefficient;
- mixed active/inactive rows produce the intended batch mean;
- Chinese and English serialization boundaries are correct;
- R2 and R3 use identical active masks and loss coefficients.

## 12. Training fairness

S0, R1, R2, and R3 must start independently from the same frozen base snapshot. No rationale arm
may continue from S0 or a historical adapter.

The following must be identical:

- 2,654 row IDs and row order;
- score targets;
- base snapshot and tokenizer;
- LoRA configuration and initialization rule;
- optimizer, learning rate, scheduler, and warmup;
- epochs, batch size, and gradient accumulation;
- cutoff length and truncation policy;
- padding and packing policy;
- checkpoint evaluation schedule;
- decoding and parsing;
- random seeds `42`, `43`, and `44`.

Report separately:

1. optimizer steps and examples processed;
2. compute-token budget including padding;
3. active supervised score tokens;
4. active supervised rationale tokens.

S0 naturally has fewer active supervised tokens. The primary semantic isolation is R3 versus R2,
whose active masks and rationale-token distributions must be matched and reported.

## 13. Deterministic data-readiness gate

Training is locked unless all checks pass:

- model revision/snapshot and runtime lock complete;
- train and source hashes match Section 3;
- four arms contain the same 2,654 row IDs in the same order;
- four arms have identical score targets;
- R1/R3 reference provenance is complete;
- retained active reasons have zero explicit score-report leakage;
- reference choice is reproducible from sample ID, seed, and epoch;
- R2 contains no fixed points;
- every R2 donor matches recipient score, metric, and language;
- R2/R3 active rationale masks are identical;
- R2/R3 length and active-token diagnostics are generated;
- no train/dev/test overlap is introduced;
- the SFT data builder cannot open the test split;
- all loss-mask tests pass.

The following are reports, not arbitrary universal pass thresholds:

- coverage by label, metric, and language;
- number of references per row;
- rationale-length distributions;
- R2 donor coverage and length cost;
- unmatched/ambiguous rows;
- dissenting-reason counts;
- reference semantic-diversity descriptions;
- compute and active-token budgets.

## 14. Smoke test

Before formal training:

- use 8-32 train rows;
- run 1-5 optimizer steps or the minimum required to exercise all paths;
- include active and inactive rationale rows, Chinese and English, and two/three-reference rows;
- verify finite loss and expected mask accounting;
- verify deterministic reference rotation;
- verify JSON generation/parser behavior;
- verify that no test loader or test path is created.

Smoke outputs cannot be used to select the scientific method.

## 15. Seed-42 scout

After the data-readiness gate, run:

- S0 seed 42;
- R1 seed 42;
- R2 seed 42;
- R3 seed 42.

Checkpoint selection uses score metrics only:

1. maximum dev Exact;
2. lower dev MAE;
3. earlier epoch.

Rationale evaluation must not select checkpoints.

### 15.1 Scout score criteria

Relative to S0, R3 should satisfy:

- \(\Delta MAE\le +0.005\);
- \(\Delta Kendall\ge -0.005\);
- dev Exact loses no more than 2 correct rows;
- Label-5 correct count loses no more than 3 rows;
- at least one low-tail count improves:
  - L2H decreases by at least 1 row, or
  - Label-2 correct count increases by at least 1 row.

Relative to R2, R3 should satisfy:

- MAE is not worse by more than `0.005`;
- Exact loses no more than 2 correct rows;
- at least one of MAE, L2H count, or Label-2 correct count improves;
- both frozen rationale evaluators have positive net preference for R3.

These are scout continuation rules, not final statistical claims.

## 16. Formal multi-seed protocol

If seed 42 passes the frozen continuation rule, complete:

| Arm | Seeds |
|---|---|
| S0 | 42, 43, 44 |
| R1 | 42, 43, 44 |
| R2 | 42, 43, 44 |
| R3 | 42, 43, 44 |

Seed-42 runs are reused if and only if they were produced under the final frozen data/model/training
lock. The formal study therefore contains 12 unique LoRA runs, not 16.

Report:

- every seed result;
- mean and sample standard deviation;
- direction consistency;
- paired row-level bootstrap where applicable;
- raw low-tail counts and confusion matrices.

Formal interpretation requires:

- mean overall score metrics remain non-inferior to S0 under the frozen margins;
- Recall-5/high-score correct counts remain protected;
- at least 2/3 seeds improve L2H or Label-2 correct count in the same direction;
- R3 is not worse than R2 on mean score performance;
- R3 improves at least one primary score/risk outcome or the frozen rationale preference outcome
  relative to R2.

Exact formal statistical reporting code and bootstrap seed must be frozen before reading formal
results.

## 17. Evaluation

### 17.1 Score outcomes

Primary:

- MAE;
- Exact;
- Kendall;
- L2H raw count and rate;
- Label-2 correct count and Recall-2;
- Label-5 correct count and Recall-5.

Secondary:

- signed bias;
- full confusion matrix;
- per-label metrics;
- metric/language strata.

### 17.2 Deterministic rationale outcomes

- JSON/schema validity;
- score parse rate;
- non-empty rationale rate;
- explicit score leakage;
- rationale token length;
- language consistency;
- exact-copy and n-gram-copy rates;
- source-reason coverage.

### 17.3 Reference similarity

For rows with human references, maximum BERTScore or a frozen equivalent may be reported
descriptively. Text similarity is not rationale correctness and cannot be a primary success
criterion.

### 17.4 Frozen model-based blind preference

Use two evaluator model families not involved in training-data construction. Evaluate a fixed
120-row dev manifest covering:

- all available low-score dev rows;
- all 12 metrics;
- both languages;
- stratified completion from Labels 3 and 4-5.

Primary comparisons:

- R3 versus R2;
- R3 versus R1.

Dimensions:

- metric alignment;
- rubric relevance;
- answer grounding;
- score-rationale consistency;
- specificity;
- unsupported claims.

Randomize A/B order and repeat with swapped order. Order-inconsistent judgments are ties. Report
win/tie/loss and paired bootstrap intervals.

These results are named model-based agreement/preference, never accuracy, expert correctness, or
human validity.

Evaluator identities, revisions, prompts, schemas, decoding, sample manifest, and bootstrap seed
must be frozen before evaluation.

## 18. Result interpretation

| Observed pattern | Permitted conclusion |
|---|---|
| R3 > S0 and R3 > R2 | Real consensus-aligned rationale semantics have incremental value in this setting |
| R3 > S0 but R3 ≈ R2 | Benefit is consistent with extra auxiliary generation or regularization, not demonstrated rationale semantics |
| R3 > R2 but R3 ≈ S0 | Real semantics improve visible-rationale preference but do not improve scoring |
| R1 > R3 | Dissenting rationales may contain useful information removed by label-consistent selection |
| R3 > R1 | Score-rationale alignment is more useful for the aggregate-label task |
| R3 < S0 | The current rationale supervision harms the main scoring task |
| R3 rationale not preferred to R2 | No claim that the student used real rationale semantics |

No failed primary result may be repaired by selecting a new rationale coefficient, changing donor
rules, filtering examples, or revisiting the test set. Any follow-up becomes a separately labeled
study.

## 19. Test boundary and transition to preference optimization

After the formal dev-stage RAR-SFT study:

1. freeze the selected R3 checkpoint, prompt, schema, parser, and decoding;
2. do not access test;
3. use the frozen score+rationale contract as the initialization and data interface for
   SORC-DPO;
4. construct score-risk and rationale-consistency preference pairs from train-only sources;
5. run one final frozen test campaign after both SFT and preference optimization are complete.

The intended final comparison includes:

- S0;
- R2;
- R3;
- Standard DPO;
- SORC-DPO.

RAR-SFT is an independent RQ1 contribution and also supplies the model/output contract for RQ2. It
must not be described only as preprocessing for DPO.

The later stage should be called direct preference optimization or preference alignment unless it
actually introduces a reward model and PPO-style online reinforcement learning.

## 20. Required artifacts before seed 42

```text
thesis_exp/outputs/exp54_rar_sft/rar_v2/
├── protocol/
│   ├── model_snapshot_lock.json
│   ├── data_lock.json
│   ├── training_lock.json
│   └── prompt_and_schema_lock.json
├── data/
│   ├── aligned_reason_inventory.jsonl
│   ├── label_consistent_reference_sets.jsonl
│   ├── all_rater_reference_sets.jsonl
│   ├── shuffled_rationale_donor_map.json
│   ├── training_manifest_s0.jsonl
│   ├── training_manifest_r1.jsonl
│   ├── training_manifest_r2.jsonl
│   └── training_manifest_r3.jsonl
├── audit/
│   ├── explicit_score_redaction_audit.csv
│   ├── coverage_by_label_metric_language.csv
│   ├── rationale_length_statistics.csv
│   ├── donor_match_diagnostics.csv
│   ├── reference_schedule_audit.csv
│   ├── token_budget_report.csv
│   └── data_readiness_report.json
└── reports/
    └── data_readiness_report.md
```

## 21. Literature lock still required

Before seed-42 training, create a literature evidence matrix with verified primary citations for:

- rationale-supervised educational/essay scoring;
- human versus model-generated rationales;
- multi-reference generation/SFT and one-to-many interference;
- annotator disagreement and human-label variation;
- rationale leakage and faithfulness limitations;
- block/task-level loss normalization;
- semantic-mismatch or shuffled-rationale controls;
- low-tail risk in educational scoring.

The literature review may refine claim wording and citations but may not change RQ1, arms, data
selection, loss, or test boundary without creating V3 and explicitly documenting why V2 was
abandoned before training.

## 22. Immediate next implementation step

After this V2 preregistration is accepted, the next step is deterministic train-only data
construction:

1. redact explicit score-report phrases;
2. build R1 and R3 reference sets;
3. expand and hash the shared three-epoch reference schedule for every formal seed;
4. build and review the per-seed, per-epoch R2 event donor maps and masks;
5. build candidate S0/R1/R2/R3 manifests and prove epoch-prefix frequency and budget equality;
6. freeze artifacts only after the event-level review gate passes;
7. emit the deterministic data-readiness report.

No model API, GPU training, normalizer, verifier, dev outcome, or test data is used in that step.
