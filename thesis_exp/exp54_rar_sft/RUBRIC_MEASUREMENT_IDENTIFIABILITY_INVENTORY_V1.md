# Rubric Measurement Identifiability Inventory v1

## Purpose and boundary

This CPU-only inventory tests whether the frozen 2,654-row RAR train split can
support a crossed multi-trait measurement question. It does not fit a probe,
extract hidden states, train a model, call an API, or authorize a CCF-A claim.

The primary response node is the SHA-256 of the canonical JSON pair
`[question_key, answer_key]`. The primary rubric node is the SHA-256 of
`[metric_id, language, rubric_sha256]`. A unique edge is a response--rubric
observation with its record identity and three human scores. Duplicate rows are
audited before collapsing.

The inventory reports graph topology, shared-response structure, ordinal
boundary support, categorical confounding, split overlap, rubric-text
variation, and rater-provenance availability. Public output contains only
counts, canonical metric/language names, non-reversible hashes, matrices, and
file hashes; it contains no question, answer, rationale, or human identity.

`human_1/2/3` are treated as source slots, not persistent people, unless an
independent provenance artifact establishes stable natural-person identity.

## Interpretation

Crossed edges between different metrics are multi-trait observations, not
automatically common-item anchors for a single scalar scale. Graph connectivity
does not establish construct equivalence or threshold identifiability.

The direction is not authorized unless all of the following hold:

1. at least four metric pairs each share at least 30 train responses;
2. every target rubric node has at least 20 observations on both sides of each
   claimed ordinal boundary;
3. edge, response-node, rubric-node, and double holdouts are constructible;
4. the model is explicitly multi-trait rather than a scalar-quality model;
5. rater effects are either provenance-verified or removed;
6. a genuine held-out-rubric source exists beyond memorizing 24 canonical
   texts, via an external dataset or newly collected rubric nodes;
7. evaluation uses a newly frozen holdout rather than the previously accessed
   test split.

Failure of these gates stops the cold-start natural-language operator claim. A
descriptive, label-only multi-trait baseline may still be useful for the thesis,
but is not a new-method authorization.

## Access note

During preliminary file discovery on 2026-08-01, a command parsed the tracked
old test JSONL to count rows and hashed identifiers. No test text, label, or
metric result was printed or analyzed. The formal inventory excludes test
entirely and records this metadata-only access; any future method requires a
new Holdout-2.
