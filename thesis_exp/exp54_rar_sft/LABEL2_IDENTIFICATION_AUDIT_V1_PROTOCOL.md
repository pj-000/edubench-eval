# No-Training Label-2 Identification Audit v1

## Status and purpose

This protocol is frozen before inspecting row-level score probabilities or
performing the new human validity review. It does not authorize model training,
new preference-pair construction, or any access to test data.

The primary question is:

> Why do the frozen P1 Field-DPO checkpoints still fail to identify answers
> whose existing three-rater aggregate target is Label 2, and which of those
> mechanisms were already present in the seed-matched frozen R3-SFT models?

The target is deliberately called the **observed-consensus `label_5`**. This
audit does not identify latent student mastery or assume that the rounded
three-rater aggregate is a uniquely correct score. Rater ambiguity and rubric
incompleteness are themselves audited as competing explanations.

## Primary population

The primary units are P1 `seed × dev-record` occurrences satisfying:

- `label_5 == 2`;
- the score parsed by the frozen structured decoder is not 2;
- seed is 42, 43, or 44.

The locked dev set contains 14 unique Label-2 records. Results are reported
separately by seed. Question-level clustering is retained for uncertainty and
pooled results are descriptive, so three predictions of one record are not
treated as three independent human examples.

R3 is a paired comparator, not a second freely selected method. Its role is to
show whether preference optimization moves probability mass from scores 4/5
toward score 3 or toward the target score 2. The 7,962 train failure-bank
outputs are used only to audit training support, transition coverage, and the
preference graph; they are not generalization evidence.

## Two complementary attribution views

Mechanisms overlap in the real system. For example, sparse Label-2 support can
create a score-3 prior and poor classwise calibration. Reporting only one
label per error would therefore make the result depend on an arbitrary order.

The audit consequently reports both:

1. a frozen hierarchical primary attribution, in which every failure receives
   exactly one primary category; and
2. independent multi-label flags, in which every applicable mechanism remains
   visible as a sensitivity analysis.

The exclusive hierarchy is:

1. measurement ambiguity;
2. rubric incompleteness;
3. decoder failure;
4. prior-recoverable failure;
5. calibration-recoverable failure;
6. support deficiency;
7. preference-coverage deficiency;
8. residual.

The hierarchy is a project decision convention, not a claim that the mechanisms
are causally independent. Conclusions must be stable in the multi-label view
and after removing measurement-ambiguous cases.

## Frozen mechanism definitions

### Measurement ambiguity

At least two valid individual rater scores must be present. A record is marked
ambiguous when the rater range is at least two score points or when both scores
2 and 3 occur among the raters. This tests whether an apparent model error is
located on an unstable observed 2/3 boundary.

### Rubric incompleteness

Two independent reviewers inspect the answer and supplied rubric without model
identity or seed information. The flag is positive only when both agree—or an
adjudicator concludes—that a score-decisive criterion is absent or cannot be
executed reproducibly from the rubric. An LLM from the model family cannot be
the final arbiter.

### Decoder failure

For the identical frozen prompt and score-field prefix, compute the conditional
sequence log-probability of each canonical value 1–5. No single-token
assumption is allowed. Decoder failure means the five-way forced-choice argmax
is 2 while the already frozen structured generation produced a different
parsed score.

### Prior recoverability

Using only the locked train label frequencies, subtract the log empirical class
prior from each frozen score log-probability and use a uniform target prior. A
failure is prior-recoverable when this diagnostic changes the five-way argmax
to 2. It is not reported as a new deployed model.

### Calibration recoverability

Fit regularized multinomial vector scaling with nested, question-grouped
cross-fitting on dev. Every record is scored only by a calibrator that did not
fit on that question group. A failure is calibration-recoverable when its
outer-fold calibrated argmax is 2 after it was not already classified as
decoder- or prior-recoverable. This is diagnostic evidence rather than a new
checkpoint result.

### Support deficiency

Because the evaluation split is question-disjoint, exact-question support is
structurally zero and is not a useful discriminator. The frozen primary proxy
is the number of locked train Label-2 rows in the exact
`metric_id × language` stratum. A row is eligible for this category below five
examples, but support is called explanatory only if the preregistered adjusted
association has the adverse direction with question-clustered uncertainty.

### Preference-coverage deficiency

The frozen P1 pair graph is reconstructed from the private pair source. A row
is deficient when neither its record nor its `metric_id × language` stratum
contains a direct chosen-score-2 versus rejected-score-3 edge. This directly
tests the hypothesis that the current pairs teach “2 is better than 4/5” while
failing to teach “2 is better than 3.”

### Residual

No preceding exclusive rule is satisfied. Residual is a required result, not a
bucket that may be silently removed.

## Human review

All 14 unique dev Label-2 records receive the two-reviewer validity audit. A
secondary 100–150-record sample is drawn only from train: Label-1/2 actual
failures, prediction-3 controls, and severe-overestimate controls. This larger
sample evaluates whether the observed measurement pattern is broader than the
small dev Label-2 cohort; it cannot replace the primary dev estimand.

Reviewers answer whether score 2 is uniquely defensible, whether score 3 is
also defensible, whether a decisive criterion is missing or unusable, and
which answer evidence establishes the 2/3 boundary. Disagreement is
adjudicated and retained in the report.

## Statistics and decision gate

- Report all three seeds separately.
- Use 10,000 question-cluster bootstrap replicates with seed 20260731.
- Report exclusive attribution and overlapping sensitivity flags.
- Report MAE, Exact, QWK, Kendall's tau-b, L2H, H2L, Recall-1/2/5, NLL,
  multiclass Brier score, and ranked probability score.
- A mechanism is a project-level dominant candidate only when its exclusive
  fraction is at least 60%, the clustered 95% interval has lower bound above
  50%, at least two seeds pass, removing ambiguous cases does not reverse the
  direction, and the direction appears in at least two metric/language strata.

The 60% rule is a go/no-go standard for further method development, not a
universal hypothesis-test threshold. If no mechanism dominates, this project
will not invent another loss merely to preserve a CCF-A-method narrative.

## Permitted next action

The next action is a read-only artifact inventory. It must locate and hash:

- raw three-rater fields in locked train/dev;
- row-level frozen R3 and P1 dev generations;
- five canonical score-option log-probabilities for all three seeds;
- the private P1 score-pair graph and train failure bank;
- exact frozen checkpoint, tokenizer, prompt, and decoder identities.

If score-option probabilities are absent, the same frozen R3/P1 checkpoints
may be run once on train/dev to extract them. That is inference, not training.
No test or historical one-time-test row may be read by this audit.
