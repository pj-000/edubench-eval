# Exp27F Conflict Adjudication Pilot

Exp27F adjudicates the Exp27E top-40 conflict queue offline. It does not call
teacher APIs, does not train a model, and does not read dev/test labels. The
output is a pilot adjudication artifact for review, not final gold annotation.

## What This Step Does

- Reads the Exp27E top-40 provider/human conflict queue.
- Applies explicit case-level adjudications with the Exp27E schema.
- Checks whether original labels, Qwen labels, and DeepSeek labels are plausible.
- Attempts strict recovery of human reason snippets from `5-grades`.
- Produces a decision on whether 361-case teacher auditing should proceed.

## Main Counts

- Top40 adjudicated samples: 40
- Original label implausible count: 12
- High-weight usable count: 9
- Low-weight usable count: 26
- Review-only count: 5
- Exclude count: 0
- Samples with strictly recovered human reason snippets: 28/40

## Provider Reliability After Adjudication

- original: plausible=28/40, MAE_to_adjudicated=0.900
- qwen: plausible=35/40, MAE_to_adjudicated=0.550
- deepseek: plausible=30/40, MAE_to_adjudicated=0.925

## Interpretation

The top40 queue confirms that the original labels are not always trustworthy:
some low labels are likely wrong high-quality answers, while some teacher high
scores miss score-format or hidden rubric failures. Qwen is generally more
conservative on these conflicts, while DeepSeek is often more lenient; neither
provider is reliable enough to be used alone without a conflict policy.

The recovered human reasons are useful when they match exactly, but coverage is
not guaranteed because the processed split lost direct reason identifiers. The
pipeline therefore treats recovered snippets as audit evidence only, not as a
mandatory label source.

## Decision

- Proceed to 361-case teacher audit: True
- Proceed directly to full 3326 train relabeling: False
- Recommended primary teacher for 361: qwen
- Use dual teacher selectively: True

The next step should be a controlled 361-case expansion, not full-train
relabeling. Use dual teacher or second-teacher review on conflict-prone cases,
then adjudicate high-disagreement samples before using them for SFT/DPO data.

## Paper Claim Boundary

Allowed claim: The original labels and teacher labels both contain nontrivial conflict; a
teacher-audited protocol with selective adjudication is justified.

Not allowed claim: Do not claim that Codex top40 adjudication is final gold annotation.
