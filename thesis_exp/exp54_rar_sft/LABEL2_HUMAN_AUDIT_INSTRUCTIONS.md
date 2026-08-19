# Label-2 human validity audit instructions

## Purpose

This review asks whether the observed-consensus score 2 and the supplied
rubric define a reproducible 2-versus-3 boundary. It does not ask which model
is better. The packet hides human scores, aggregate labels, model names, model
predictions, seeds, probabilities, and automatic mechanism flags.

Two reviewers complete their packets independently. They must not compare
answers until both first-pass files are frozen. A third person adjudicates only
disagreements. An LLM or a model from the evaluated family cannot serve as a
human reviewer or final adjudicator.

## Responses

For the first four fields, enter exactly `YES`, `NO`, or `UNCERTAIN`:

1. `score_2_uniquely_defensible`: Does the answer and supplied rubric make
   score 2 the only defensible score?
2. `score_3_also_defensible`: Could a careful reviewer also assign score 3?
3. `decisive_criterion_absent`: Is a criterion needed to distinguish 2 from 3
   missing from the rubric?
4. `present_criterion_too_vague`: Is a relevant criterion present but too
   vague to apply reproducibly?

In `boundary_evidence_span`, quote or precisely identify the smallest answer
span that determines the 2-versus-3 boundary and connect it to the rubric.
Use `reviewer_notes` only for concise uncertainty or conflict explanations.

Do not infer the hidden aggregate label, search for the record elsewhere, or
use model output as evidence. Preserve the CSV columns and presentation IDs.

## Completion boundary

The two completed files must be validated for allowed categorical responses,
nonempty evidence spans, unchanged source columns, unique presentation IDs,
and exact packet hashes before unblinding. Human agreement, adjudication, and
the rubric-incompleteness category are not available until that validation is
complete.
