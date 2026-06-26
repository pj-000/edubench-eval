# Exp16A Rubric-Conditioned Ordinal Boundary Linking

## Scientific Question

Exp16A studies how a scorer should determine score boundaries on unseen
questions. The experiment tests whether a model should separate:

- answer quality, represented by a scalar `s`; and
- question/rubric-specific score boundaries, represented by ordered thresholds
  `tau1..tau4`.

## Method

The model uses one shared encoder in two passes:

- quality tower input: `question + answer + metric + rubric + metadata`;
- boundary tower input: `question + metric + rubric + metadata`.

The boundary tower must not receive `answer`. It represents how the current
question and rubric define the 1/2, 2/3, 3/4, and 4/5 score boundaries. The
quality tower represents where the answer sits on that scale.

The model computes:

```text
logits_k = alpha * (s - tau_k)
P(y > k) = sigmoid(logits_k)
pred = 1 + sum(P(y > k) > 0.5)
```

`tau1 < tau2 < tau3 < tau4` is enforced by a cumulative softplus parameterization,
so monotonic violation should be zero or extremely close to zero.

## Difference From A Standard Ordinal Head

A standard ordinal head directly predicts four threshold logits from the full
input. Exp16A instead decomposes the task:

- `s` is answer quality;
- `tau` is the scoring ruler generated from the question/rubric context;
- the final score is produced by comparing `s` with `tau`.

This is a structure test only. Exp16A intentionally does not include pair loss,
anchor loss, risk loss, or dynamic lambda.

## Boundary Variants

- `global`: uses learned global ordered thresholds.
- `metric_rubric`: boundary text uses metric and rubric.
- `qmr`: boundary text uses question, metric, and rubric.
- `qmr_meta`: boundary text uses question, metric, rubric, and metadata.

`--boundary_fields` can override the variant defaults, but it must not include
`answer`.

## Data Field Mapping

The default data path is:

```text
thesis_exp/data/splits/question_seed42/{train,dev,test}.jsonl
```

Exp16A uses:

- `record_id` or `id` as `sample_id`;
- `question_key` or a SHA1 hash of `question` as `question_key`;
- `metric_canonical` as `metric`;
- `rubric_text` as `rubric`;
- `label_5` as the 1-5 gold label;
- `scenario_canonical`, `subject_canonical`, `education_level_canonical`, and
  `language` as metadata when available.

## Sanity Check

```bash
./thesis_exp/scripts/run_exp16a_boundary_linking_sanity.sh
```

The sanity check uses a tiny random local encoder and verifies:

- logits shape is `[B,4]`;
- thresholds are strictly increasing;
- probabilities are monotonic;
- changing `answer` changes only the quality text, not the boundary text;
- a CPU dry-run writes metrics and predictions.

## Scout Template

```bash
./thesis_exp/scripts/run_exp16a_boundary_linking_scout.sh
```

The scout script defaults to the Qwen3-Reranker-0.6B path used by earlier thesis
experiments. By default, it queues all four variants across GPU 6 and GPU 7:
`global`, `metric_rubric`, `qmr`, and `qmr_meta`.

To run only one variant:

```bash
./thesis_exp/scripts/run_exp16a_boundary_linking_scout.sh qmr_meta
```

## Outputs

Each run writes under `thesis_exp/outputs/exp16_boundary_linking/`:

- `metrics_dev.json`
- `metrics_test.json`
- `predictions_dev.jsonl`
- `predictions_test.jsonl`
- `threshold_stats_dev.json`
- `threshold_stats_test.json`
- `threshold_by_metric_dev.csv`
- `threshold_by_metric_test.csv`
- `config.json`

Predictions include `quality_score_s`, `tau1..tau4`, `margin_tau2`,
`margin_tau3`, and `is_low_to_high`.

## Dev-Only Selection Rule

Do not use test labels for model or configuration selection.

Recommended dev-only rule:

1. monotonic violation must be zero or extremely close to zero;
2. dev MAE should not exceed the QD weighted ordinal baseline by more than 0.01;
3. dev QWK should not clearly decline;
4. among eligible runs, compare dev low-to-high;
5. test is for final reporting only.
